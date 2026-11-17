"""
DisasterM3 Track B: Base-Model Visual Sanity Verification
Deliverable 4: Verify that the base vision-language model (without LoRA adapter)
correctly perceives and describes real visual features (buildings, roads,
vegetation, water, debris, etc.) on genuine benchmark satellite images.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any
from PIL import Image

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Ensure src can be resolved if executed from within scripts
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.eval_trackb import ImageResolver, resolve_path


DOMAIN_KEYWORDS = {
    "buildings": [r"\bbuilding", r"\bstructures?", r"\bhouses?", r"\broofs?", r"\bresidential", r"\bfacilit"],
    "roads": [r"\broads?", r"\bstreets?", r"\bhighways?", r"\bpaths?", r"\bpavement", r"\btransit"],
    "vegetation": [r"\bvegetat", r"\btrees?", r"\bforests?", r"\bgreenery", r"\bfields?", r"\bcrops?"],
    "water": [r"\bwater", r"\briver", r"\bocean", r"\bsea\b", r"\bflooding", r"\bpond", r"\blake"],
    "damage_or_debris": [r"\bdamage", r"\bdebris", r"\brubble", r"\bdestr", r"\bcollaps", r"\bsmoke", r"\bburn"]
}


def parse_args():
    parser = argparse.ArgumentParser(description="DisasterM3 Base-Model Visual Sanity Verification")
    parser.add_argument("--eval_ids", type=str, default="results/track_b_v2/bench_eval_ids.json",
                        help="Path to benchmark evaluation manifest")
    parser.add_argument("--output_dir", type=str, default="results/track_b_v2",
                        help="Directory to save sanity check report")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace model identifier")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of diverse benchmark images to test")
    parser.add_argument("--device", type=str, default=None,
                        help="Target device (default: cuda if available else cpu)")
    return parser.parse_args()


def run_sanity_check():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    eval_ids_path = resolve_path(args.eval_ids)
    if not os.path.exists(eval_ids_path):
        # Fallback to track_b
        alt_path = resolve_path("results/track_b/bench_eval_ids.json")
        if os.path.exists(alt_path):
            eval_ids_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find benchmark evaluation manifest at {eval_ids_path}")

    print("=" * 70)
    print("DISASTERM3 BASE-MODEL VISUAL SANITY VERIFICATION")
    print(f"Manifest    : {eval_ids_path}")
    print(f"Model       : {args.model_name} (BASE ONLY - ZERO ADAPTERS)")
    print(f"Num Samples : {args.num_samples}")
    print("=" * 70)

    # 1. Load evaluation manifest
    with open(eval_ids_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("sample_records", []) if isinstance(data, dict) else data

    bench_root = resolve_path("data/raw/disasterm3/DisasterM3_Bench")
    resolver = ImageResolver(bench_root)

    # 2. Pick diverse samples across disaster types
    selected_samples: List[Dict[str, Any]] = []
    seen_types = set()
    for s in samples:
        dtype = s.get("disaster_type", "unknown")
        if dtype not in seen_types:
            selected_samples.append(s)
            seen_types.add(dtype)
        if len(selected_samples) >= args.num_samples:
            break

    # If fewer than num_samples, fill up
    if len(selected_samples) < args.num_samples:
        for s in samples:
            if s not in selected_samples:
                selected_samples.append(s)
            if len(selected_samples) >= args.num_samples:
                break

    print(f"[SANITY] Selected {len(selected_samples)} diverse benchmark images across {len(seen_types)} disaster types:")
    for idx, s in enumerate(selected_samples, 1):
        print(f"  {idx:2d}. ID: {s['id']} | Type: {s.get('disaster_type')} | Post-Img: {os.path.basename(s.get('post_image_path', ''))}")

    # 3. Load Base Model without adapters
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[MODEL] Loading base model '{args.model_name}' on {device}...")

    processor = AutoProcessor.from_pretrained(args.model_name)

    model_kwargs = {"device_map": "auto"}
    if torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["quantization_config"] = bnb_config
        except Exception:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    model.eval()

    # Verify adapter is definitely not loaded
    if hasattr(model, "peft_config"):
        print("[SANITY WARNING] PEFT config detected on model! Disabling adapter layers...")
        if hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()
    else:
        print("[SANITY VERIFIED] Pure base model loaded. No LoRA adapter present.")

    # 4. Prompt base model on each image
    prompt_text = "Describe this satellite image in detail. Specifically identify any buildings, roads, vegetation, water bodies, and visible damage or debris."
    results = []
    keyword_occurrences = {k: 0 for k in DOMAIN_KEYWORDS}

    print("\n[EVALUATION] Generating descriptions for benchmark images...")
    t0 = time.time()

    import re
    for idx, s in enumerate(selected_samples, 1):
        img_rel = s.get("post_image_path") or s.get("pre_image_path")
        pil_img = resolver.load_image(img_rel)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=250,
                do_sample=False,
                temperature=None,
                top_p=None
            )
            # Trim input tokens
            prompt_len = inputs.input_ids.shape[1]
            gen_tokens = gen_ids[0][prompt_len:]
            description = processor.decode(gen_tokens, skip_special_tokens=True).strip()

        # Keyword presence analysis
        detected_categories = []
        for cat, patterns in DOMAIN_KEYWORDS.items():
            found = False
            for pat in patterns:
                if re.search(pat, description, re.IGNORECASE):
                    found = True
                    break
            if found:
                keyword_occurrences[cat] += 1
                detected_categories.append(cat)

        res_entry = {
            "index": idx,
            "id": s["id"],
            "disaster_type": s.get("disaster_type", "unknown"),
            "image_filename": os.path.basename(img_rel),
            "image_size": list(pil_img.size),
            "description": description,
            "detected_categories": detected_categories
        }
        results.append(res_entry)

        print(f"\n--- [{idx}/{len(selected_samples)}] ID: {s['id']} ({s.get('disaster_type')}) ---")
        print(f"Image: {os.path.basename(img_rel)} ({pil_img.size[0]}x{pil_img.size[1]})")
        print(f"Detected: {', '.join(detected_categories) if detected_categories else 'None'}")
        print(f"Description:\n{description[:200]}...")

    elapsed = time.time() - t0
    resolver.close()

    # 5. Sanity Assertion: At least 3 of 5 visual categories must be detected across images
    total_cats_active = sum(1 for c, count in keyword_occurrences.items() if count > 0)
    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print(f"Total Images Evaluated : {len(results)}")
    print(f"Total Time             : {elapsed:.2f}s ({elapsed / len(results):.2f}s/img)")
    print("Domain Keyword Detections across all 10 images:")
    for cat, count in keyword_occurrences.items():
        print(f"  - {cat:<20}: {count}/{len(results)} images ({count / len(results) * 100:.1f}%)")

    passed = (total_cats_active >= 3)
    status_str = "PASSED" if passed else "FAILED"
    print(f"\nSanity Check Status: [{status_str}] (Detected {total_cats_active}/5 domain categories)")
    print("=" * 70)

    # 6. Save JSON and Markdown artifacts
    json_path = os.path.join(output_dir, "base_model_sanity_check.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "sanity_status": status_str,
            "categories_active": total_cats_active,
            "keyword_summary": keyword_occurrences,
            "samples": results
        }, f, indent=2)

    md_path = os.path.join(output_dir, "base_model_sanity_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DisasterM3 Base-Model Visual Sanity Report\n\n")
        f.write(f"- **Status**: `{status_str}` ({total_cats_active}/5 visual categories verified)\n")
        f.write(f"- **Base Model**: `{args.model_name}` (Zero LoRA adapters attached)\n")
        f.write(f"- **Samples Evaluated**: {len(results)} genuine benchmark satellite images\n\n")
        f.write("## Category Detection Frequency\n\n")
        f.write("| Visual Category | Occurrences in Descriptions | Frequency |\n")
        f.write("| :--- | :---: | :---: |\n")
        for cat, count in keyword_occurrences.items():
            f.write(f"| **{cat}** | {count} / {len(results)} | {count / len(results) * 100:.1f}% |\n")
        f.write("\n## Sample Descriptions\n\n")
        for r in results:
            f.write(f"### {r['index']}. {r['id']} ({r['disaster_type'].capitalize()})\n\n")
            f.write(f"- **File**: `{r['image_filename']}` ({r['image_size'][0]}x{r['image_size'][1]})\n")
            f.write(f"- **Visual Elements Detected**: {', '.join(r['detected_categories']) if r['detected_categories'] else 'None'}\n\n")
            f.write(f"> {r['description']}\n\n")

    print(f"[SANITY] Saved JSON results to: {json_path}")
    print(f"[SANITY] Saved Markdown report to: {md_path}")

    if not passed:
        raise RuntimeError("Base-model sanity check FAILED. The base model failed to recognize real visual features.")


if __name__ == "__main__":
    run_sanity_check()

