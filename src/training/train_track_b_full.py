#!/usr/bin/env python3
"""
Universal Track B Full Training & Evaluation Script.
Auto-detects Apple Silicon Metal (MPS) and NVIDIA CUDA (4-bit BitsAndBytes).
Trains Qwen2.5-VL-3B with LoRA for 3 full epochs with class-balanced oversampling.
"""
import os
import sys
import json
import random
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_SPLIT   = PROJECT_ROOT / "data/processed/track_b_train.jsonl"
BENCH_IDS     = PROJECT_ROOT / "results/track_b_v2/bench_eval_ids.json"
BENCH_IMAGES  = PROJECT_ROOT / "data/raw/disasterm3/DisasterM3_Bench/test_images"
TRAIN_IMAGES  = PROJECT_ROOT / "data/raw/disasterm3/DisasterM3_Instruct/train_images"
OUTPUT_DIR    = PROJECT_ROOT / "checkpoints/lora_track_b_v3_full"
PRED_OUTPUT   = PROJECT_ROOT / "results/track_b_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_OUTPUT, exist_ok=True)

BATCH_SIZE = 1
GRAD_ACCUM = 8          # Effective batch size = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 800         # ~3 full epochs over balanced DTR items
WARMUP_STEPS = 50
SAVE_STEPS = 100

# --- DEVICE DETECTION (Apple Silicon MPS vs NVIDIA CUDA) ---
if torch.backends.mps.is_available():
    device_type = "mps"
    device = torch.device("mps")
    print("[INIT] Device Detected: Apple Silicon Metal (MPS)")
    print("[INIT] Architecture: Unified Memory (24GB Pool)")
elif torch.cuda.is_available():
    device_type = "cuda"
    device = torch.device("cuda")
    print(f"[INIT] Device Detected: NVIDIA CUDA ({torch.cuda.get_device_name(0)})")
    print(f"[INIT] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device_type = "cpu"
    device = torch.device("cpu")
    print("[INIT] Device Detected: CPU (Warning: Training will be slow)")

# 1. Load and Balance Dataset
dtr_by_class = defaultdict(list)
caption_records = []

if not TRAIN_SPLIT.exists():
    raise FileNotFoundError(f"[ERROR] Could not find {TRAIN_SPLIT}. Ensure data/processed/ is in place.")

with open(TRAIN_SPLIT, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        post_p = r.get("post_image_path", "")
        abs_p = TRAIN_IMAGES / Path(post_p).name
        if not abs_p.exists():
            continue
        r["post_image_abs"] = str(abs_p)
        pre_p = r.get("pre_image_path", "")
        if pre_p:
            r["pre_image_abs"] = str(TRAIN_IMAGES / Path(pre_p).name)

        if r.get("task") in ["Disaster Type Recognition", "DTR"]:
            d_type = r.get("disaster_type", "unknown").lower()
            dtr_by_class[d_type].append(r)
        else:
            caption_records.append(r)

# Balance DTR classes (oversample rare classes to 150 items each)
target_per_class = 150
balanced_dtr = []
for d_type, items in dtr_by_class.items():
    if len(items) >= target_per_class:
        balanced_dtr.extend(random.sample(items, target_per_class))
    else:
        # Oversample rare tail (volcano, explosion, tsunami)
        mult = target_per_class // len(items)
        rem = target_per_class % len(items)
        balanced_dtr.extend(items * mult + random.sample(items, rem))

# Combine 1:1 balanced DTR with selected caption reports
sampled_captions = random.sample(caption_records, min(len(caption_records), len(balanced_dtr)))
combined_training_data = balanced_dtr + sampled_captions
random.shuffle(combined_training_data)

print(f"[DATA] Prepared {len(combined_training_data)} balanced training samples:")
print(f"       DTR: {len(balanced_dtr)} (balanced across {len(dtr_by_class)} classes) | Captions: {len(sampled_captions)}")

# 2. Load Base Model
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=512*28*28)

if device_type == "cuda":
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print(f"[MODEL] Loading {model_id} in 4-bit NF4 for CUDA...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model = prepare_model_for_kbit_training(model)
elif device_type == "mps":
    print(f"[MODEL] Loading {model_id} in Native 16-bit for Apple Silicon MPS (uses ~6GB of 24GB Unified RAM)...")
    # Native float16 runs with zero quantization loss on Apple Silicon Metal
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
else:
    print(f"[MODEL] Loading {model_id} on CPU...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Training Loop
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
lr_scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, MAX_STEPS)

model.train()
step = 0
running_loss = 0.0
progress_bar = tqdm(total=MAX_STEPS, desc="Training")

data_idx = 0
while step < MAX_STEPS:
    sample = combined_training_data[data_idx % len(combined_training_data)]
    data_idx += 1

    pre_path = sample.get("pre_image_abs")
    post_path = sample.get("post_image_abs")
    user_prompt = sample.get("prompts", ["Identify the disaster."])[0] if isinstance(sample.get("prompts"), list) else sample.get("prompts")
    gold_answer = sample.get("gold", "")

    content = []
    if pre_path and os.path.exists(pre_path):
        content.append({"type": "image", "image": f"file://{pre_path}"})
    if post_path and os.path.exists(post_path):
        content.append({"type": "image", "image": f"file://{post_path}"})
    content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": gold_answer}]}
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(device)

        # Mask loss to assistant tokens only
        labels = inputs.input_ids.clone()
        prompt_len = len(processor.tokenizer.encode(processor.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=True)))
        labels[:, :prompt_len] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM

        if data_idx % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{running_loss / GRAD_ACCUM:.4f}"})
            running_loss = 0.0

            if step % SAVE_STEPS == 0 or step == MAX_STEPS:
                ckpt_path = OUTPUT_DIR / f"checkpoint_step_{step}"
                model.save_pretrained(ckpt_path)
                processor.save_pretrained(ckpt_path)

    except Exception as e:
        print(f"Skipping error at sample {data_idx}: {e}")
        continue

print(f"\n[DONE] Training finished! Final model saved to {OUTPUT_DIR}")
