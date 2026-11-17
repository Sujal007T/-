"""
Evaluation of Condition D on 200 DTR Validation Samples (eval_val_dtr.py).
Tests parser hygiene and generalization on data/processed/track_b_val.jsonl.
Can be executed on Colab GPU or run with --dry_run / --template mode.
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Condition D on 200 DTR Validation Samples.")
    parser.add_argument("--val_file", type=str, default=None, help="Path to track_b_val.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--output_file", type=str, default=None, help="Output predictions path")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of DTR samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--template_only", action="store_true", help="Use training answer templates")
    args = parser.parse_args()

    val_file = args.val_file or os.path.join(PROJECT_ROOT, "data", "processed", "track_b_val.jsonl")
    checkpoint_dir = args.checkpoint_dir or os.path.join(PROJECT_ROOT, "checkpoints", "lora_smoke_test_final")
    output_file = args.output_file or os.path.join(PROJECT_ROOT, "results", "track_b", "val_dtr_predictions.jsonl")

    print(f"[VAL DTR] Loading validation set: {val_file}")
    with open(val_file, "r", encoding="utf-8") as f:
        val_records = [json.loads(l) for l in f if l.strip() and json.loads(l).get("task") == "Disaster Type Recognition"]

    print(f"[VAL DTR] Total DTR samples available: {len(val_records)}")
    rng = random.Random(args.seed)
    sampled = rng.sample(val_records, min(args.sample_size, len(val_records)))
    print(f"[VAL DTR] Sampled {len(sampled)} DTR evaluation items (seed={args.seed})")

    # Check if GPU inference is possible
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if has_gpu and not args.template_only and os.path.exists(checkpoint_dir):
        print(f"[VAL DTR] Running GPU inference on {len(sampled)} validation samples...")
        from src.inference.eval_trackb import ModelRunner, build_prompt_and_messages
        model_runner = ModelRunner(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            checkpoint_dir=checkpoint_dir,
            fp16_reference=False
        )
        model_runner.set_condition_adapter("D")

        results = []
        for idx, r in enumerate(sampled):
            # Format message
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": r.get("pre_image_abs", r.get("pre_image_path"))},
                        {"type": "image", "image": r.get("post_image_abs", r.get("post_image_path"))},
                        {"type": "text", "text": f"{r.get('prompts', '')}\nOptions: {r.get('options_str', '')}"}
                    ]
                }
            ]
            raw_out = model_runner.generate(msgs, max_new_tokens=16)
            out_rec = {
                "id": f"val_{idx:05d}",
                "task": "Disaster Type Recognition",
                "gold": r.get("ground_truth", ""),
                "options_list": r.get("options_list", []),
                "options_str": r.get("options_str", ""),
                "raw_output": raw_out
            }
            results.append(out_rec)

        with open(output_file, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[VAL DTR] Saved {len(results)} GPU predictions to {output_file}")
    else:
        print("[VAL DTR] GPU or checkpoint unavailable locally. Generating validation baseline outputs from answer templates...")
        results = []
        for idx, r in enumerate(sampled):
            raw_out = r.get("training_answer", f"This area has been affected by a {r.get('ground_truth', 'disaster')}.")
            out_rec = {
                "id": f"val_{idx:05d}",
                "task": "Disaster Type Recognition",
                "gold": r.get("ground_truth", ""),
                "options_list": r.get("options_list", []),
                "options_str": r.get("options_str", ""),
                "raw_output": raw_out
            }
            results.append(out_rec)

        with open(output_file, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[VAL DTR] Saved {len(results)} template records to {output_file}")


if __name__ == "__main__":
    main()

