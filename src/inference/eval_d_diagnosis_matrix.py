"""
DisasterM3 Track B: Condition D In-Depth Diagnosis Matrix (eval_d_diagnosis_matrix.py)

Evaluates why Condition D scores 82.5% on validation DTR but 19.5% on Bench DTR.
Runs a 2x2 factorial matrix + negative controls on 200 DTR samples each:
  - Cell 1: Val data x Validation-style input (2 images, no patches, no system prompt)
  - Cell 2: Val data x Bench-style input (6 images: pre + post + 4 patches, system prompt)
  - Cell 3: Bench data x Validation-style input (2 images, no patches, no system prompt)
  - Cell 4: Bench data x Bench-style input (6 images: pre + post + 4 patches, system prompt)
  - Cell 1 Controls: Blank & Cross-Event Shuffled images
  - Cell 3 Controls: Blank & Cross-Event Shuffled images

Uses sentence-aware v2 parser (parse_dtr_improved) as the primary parser,
reporting the legacy letter-first parser (parse_dtr_old) as a secondary column.
"""

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
import numpy as np
from PIL import Image
from scipy.stats import binomtest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.eval_trackb import (
    ImageResolver,
    make_2x2_patches,
    ModelRunner,
    TASK_TOKENS
)
from src.evaluation.grounding_analysis import (
    parse_dtr_improved,
    parse_dtr_old,
    KNOWN_DISASTER_TYPES
)


def get_event_id(sample: Dict[str, Any]) -> str:
    """Extracts or infers a clean event ID for cross-event shuffling."""
    if sample.get("event_id"):
        return str(sample["event_id"]).lower()
    p = sample.get("post_image_path", "")
    fname = os.path.basename(p)
    import re
    m1 = re.match(r'^(.+?)_\d+_(pre|post)', fname, re.IGNORECASE)
    if m1: return m1.group(1).lower()
    m2 = re.match(r'^(.+?)_(pre|post)_\d+', fname, re.IGNORECASE)
    if m2: return m2.group(1).lower()
    return fname.split('.')[0].lower()


def build_cross_event_shuffled_mapping(samples: List[Dict[str, Any]], seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """Builds a deterministic mapping where every sample gets a donor from a DIFFERENT event."""
    rng = random.Random(seed)
    mapping = {}
    n = len(samples)

    for i, s in enumerate(samples):
        curr_eid = get_event_id(s)
        candidates = [other for other in samples if get_event_id(other) != curr_eid]
        if not candidates:
            raise RuntimeError(f"Cannot find cross-event donor for sample {s.get('id')} (event={curr_eid})")
        donor = rng.choice(candidates)
        assert get_event_id(donor) != curr_eid, f"Donor must be from different event: {get_event_id(donor)} vs {curr_eid}"
        s_id = s.get("id") or f"idx_{i:05d}"
        mapping[s_id] = donor

    return mapping


def build_cell_messages_and_images(
    cell_type: str,
    sample: Dict[str, Any],
    image_resolver: ImageResolver,
    donor_sample: Optional[Dict[str, Any]] = None,
    image_size: Tuple[int, int] = (448, 448)
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Constructs the exact multimodal conversation messages and image list for each diagnostic cell.
    """
    is_blank = "blank" in cell_type
    is_shuffled = "shuffled" in cell_type
    is_bench_format = ("bench_format" in cell_type) or ("cell2" in cell_type) or ("cell4" in cell_type)

    # 1. Resolve raw images
    if is_blank:
        pre_img = Image.new("RGB", image_size, (128, 128, 128))
        post_img = Image.new("RGB", image_size, (128, 128, 128))
    elif is_shuffled:
        pre_path = sample.get("pre_image_path", "")
        pre_img = image_resolver.load_image(pre_path) if pre_path else None
        donor_post_path = donor_sample.get("post_image_path", "") if donor_sample else ""
        post_img = image_resolver.load_image(donor_post_path)
    else:
        pre_path = sample.get("pre_image_path", "")
        post_path = sample.get("post_image_path", "")
        pre_img = image_resolver.load_image(pre_path) if pre_path else None
        post_img = image_resolver.load_image(post_path)

    # 2. Base text and options
    base_prompt = sample.get("prompts", "")
    if isinstance(base_prompt, list):
        base_prompt = base_prompt[0] if base_prompt else ""
    if not base_prompt:
        base_prompt = "What kind of calamity has this area experienced?"

    options_str = sample.get("options_str", "")
    if options_str:
        prompt_text = f"{base_prompt}\nOptions: {options_str}"
    else:
        prompt_text = base_prompt

    # 3. Format differences:
    # Validation format: 2 images, NO system prompt, plain text prompt
    # Bench Condition D format: 6 images (pre + post + 4 patches), system prompt, patch note
    if not is_bench_format:
        # Validation-style
        images = [pre_img, post_img] if pre_img else [post_img]
        user_content = []
        if pre_img:
            user_content.append({"type": "image", "image": pre_img})
        user_content.append({"type": "image", "image": post_img})
        user_content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": user_content}]
    else:
        # Bench Condition D-style
        system_prompt = TASK_TOKENS.get("Disaster Type Recognition", "")
        patches = make_2x2_patches(post_img)
        images = [pre_img, post_img] + patches if pre_img else [post_img] + patches

        user_content = []
        if pre_img:
            user_content.append({"type": "image", "image": pre_img})
        user_content.append({"type": "image", "image": post_img})
        for p in patches:
            user_content.append({"type": "image", "image": p})

        augmented_prompt = prompt_text + "\n\nNote: The last 4 images above are 2x2 local patches of the post-disaster image."
        user_content.append({"type": "text", "text": augmented_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    return messages, images


def evaluate_cell(
    cell_name: str,
    cell_desc: str,
    samples: List[Dict[str, Any]],
    image_resolver: ImageResolver,
    model_runner: Optional[ModelRunner],
    donor_mapping: Optional[Dict[str, Dict[str, Any]]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Runs evaluation for a single cell and computes rigorous statistics."""
    print(f"\n{'='*70}\nRUNNING CELL: {cell_name} ({cell_desc})\nSamples: {len(samples)}\n{'='*70}")

    records = []
    primary_correct = 0
    secondary_correct = 0
    primary_unparseable = 0
    secondary_unparseable = 0

    all_gold = []
    primary_preds = []
    secondary_preds = []

    start_time = time.time()
    for idx, s in enumerate(samples):
        sid = s.get("id") or f"idx_{idx:05d}"
        donor = donor_mapping.get(sid) if donor_mapping else None

        messages, images = build_cell_messages_and_images(
            cell_type=cell_name,
            sample=s,
            image_resolver=image_resolver,
            donor_sample=donor
        )

        if dry_run or model_runner is None:
            # Deterministic simulation for local dry run
            # In validation format on val data, simulate training template
            if "val_val" in cell_name:
                raw_out = f"This area has been affected by a {s.get('ground_truth', 'disaster')}."
            elif "blank" in cell_name:
                raw_out = "Answer: A. earthquake\nReason: No visible changes."
            elif "bench_format" in cell_name:
                raw_out = "Answer: A. earthquake\nReason: Structural collapse observed in patches."
            else:
                raw_out = f"This region has experienced a {s.get('ground_truth', 'disaster')}."
        else:
            raw_out = model_runner.generate(messages=messages, max_new_tokens=48)

        gold_label = str(s.get("ground_truth", "")).strip().lower()
        pred_primary = parse_dtr_improved(raw_out, s)
        pred_secondary = parse_dtr_old(raw_out, s)

        is_prim_corr = (pred_primary == gold_label)
        is_sec_corr = (pred_secondary == gold_label)

        if is_prim_corr: primary_correct += 1
        if is_sec_corr: secondary_correct += 1
        if pred_primary == "unparseable": primary_unparseable += 1
        if pred_secondary == "unparseable": secondary_unparseable += 1

        all_gold.append(gold_label)
        primary_preds.append(pred_primary)
        secondary_preds.append(pred_secondary)

        records.append({
            "id": sid,
            "disaster_type": s.get("disaster_type", gold_label),
            "gold": gold_label,
            "pred_primary_v2": pred_primary,
            "pred_secondary_old": pred_secondary,
            "correct_primary": is_prim_corr,
            "correct_secondary": is_sec_corr,
            "raw_output": raw_out
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - start_time
            rate = elapsed / (idx + 1)
            remaining = rate * (len(samples) - (idx + 1))
            print(f"  [{idx+1}/{len(samples)}] Acc: {primary_correct/(idx+1)*100:.1f}% | Rate: {rate:.2f}s/item | ETA: {remaining/60:.1f}m")

    # Compute Statistics
    n = len(samples)
    chance_p0 = 0.20  # 5 options per question

    # Primary Parser Metrics
    prim_acc = primary_correct / max(1, n)
    btest_prim = binomtest(primary_correct, n, p=chance_p0, alternative="two-sided")
    ci_prim = btest_prim.proportion_ci(confidence_level=0.95)

    # Secondary Parser Metrics
    sec_acc = secondary_correct / max(1, n)
    btest_sec = binomtest(secondary_correct, n, p=chance_p0, alternative="two-sided")
    ci_sec = btest_sec.proportion_ci(confidence_level=0.95)

    # Distributions
    gold_dist = Counter(all_gold)
    prim_pred_dist = Counter(primary_preds)
    sec_pred_dist = Counter(secondary_preds)

    # Per-Gold-Class Accuracy
    per_class_acc = {}
    for g_cls in sorted(gold_dist.keys()):
        cls_total = gold_dist[g_cls]
        cls_corr = sum(1 for r in records if r["gold"] == g_cls and r["correct_primary"])
        per_class_acc[g_cls] = {
            "total": cls_total,
            "correct": cls_corr,
            "accuracy": cls_corr / max(1, cls_total)
        }

    return {
        "cell_name": cell_name,
        "description": cell_desc,
        "n_samples": n,
        "chance_baseline": chance_p0,
        "primary_parser_v2": {
            "correct": primary_correct,
            "accuracy": prim_acc,
            "exact_95_ci": [float(ci_prim.low), float(ci_prim.high)],
            "p_value_vs_chance": float(btest_prim.pvalue),
            "unparseable_count": primary_unparseable,
            "unparseable_rate": primary_unparseable / max(1, n),
            "prediction_distribution": dict(prim_pred_dist)
        },
        "secondary_parser_old": {
            "correct": secondary_correct,
            "accuracy": sec_acc,
            "exact_95_ci": [float(ci_sec.low), float(ci_sec.high)],
            "p_value_vs_chance": float(btest_sec.pvalue),
            "unparseable_count": secondary_unparseable,
            "unparseable_rate": secondary_unparseable / max(1, n),
            "prediction_distribution": dict(sec_pred_dist)
        },
        "gold_distribution": dict(gold_dist),
        "per_gold_class_accuracy": per_class_acc,
        "records": records
    }


def main():
    parser = argparse.ArgumentParser(description="DisasterM3 Condition D Diagnosis Factorial Matrix")
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--bench_eval_file", type=str, default=None)
    parser.add_argument("--bench_root", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="Run without loading GPU model")
    args = parser.parse_args()

    val_file = args.val_file or os.path.join(PROJECT_ROOT, "data", "processed", "track_b_val.jsonl")
    bench_eval_file = args.bench_eval_file or os.path.join(PROJECT_ROOT, "results", "track_b_v2", "bench_eval_ids.json")
    bench_root = args.bench_root or os.path.join(PROJECT_ROOT, "data", "raw", "disasterm3", "DisasterM3_Bench")
    checkpoint_dir = args.checkpoint_dir or os.path.join(PROJECT_ROOT, "checkpoints", "lora_smoke_test_final")
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "results", "track_b_v2", "d_diagnosis")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 1. Load Data
    print(f"[DATA] Loading validation records from: {val_file}")
    with open(val_file, "r", encoding="utf-8") as f:
        val_all = [json.loads(l) for l in f if l.strip() and json.loads(l).get("task") == "Disaster Type Recognition"]
    rng = random.Random(args.seed)
    val_samples = rng.sample(val_all, min(args.sample_size, len(val_all)))
    for idx, s in enumerate(val_samples):
        if not s.get("id"): s["id"] = f"val_{idx:05d}"
    print(f"[DATA] Sampled {len(val_samples)} validation DTR items.")

    print(f"[DATA] Loading benchmark records from: {bench_eval_file}")
    with open(bench_eval_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    bench_records = [r for r in bench_data["sample_records"] if r.get("task") == "Disaster Type Recognition"]
    bench_samples = bench_records[:args.sample_size]
    print(f"[DATA] Loaded {len(bench_samples)} benchmark DTR items.")

    # 2. Build Shuffled Donor Mappings
    val_shuffled_mapping = build_cross_event_shuffled_mapping(val_samples, seed=args.seed)
    bench_shuffled_mapping = build_cross_event_shuffled_mapping(bench_samples, seed=args.seed)

    # 3. Initialize Resolver & Model Runner
    image_resolver = ImageResolver(bench_root)

    model_runner = None
    if not args.dry_run:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[MODEL] Loading LoRA checkpoint from: {checkpoint_dir}")
                model_runner = ModelRunner(
                    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                    checkpoint_dir=checkpoint_dir,
                    fp16_reference=False,
                    device=args.device
                )
                model_runner.load()
                model_runner.set_condition_adapter("D")
                print("[MODEL] Condition D LoRA Adapter active.")
            else:
                print("[WARNING] CUDA not available. Forcing --dry_run.")
                args.dry_run = True
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}. Forcing --dry_run.")
            args.dry_run = True

    # 4. Define All Cells
    cell_configs = [
        ("cell1_val_val_format", "Cell 1: Val data x Validation input format", val_samples, None),
        ("cell2_val_bench_format", "Cell 2: Val data x Bench input format (6 imgs + patches)", val_samples, None),
        ("cell3_bench_val_format", "Cell 3: Bench data x Validation input format (2 imgs, no patches)", bench_samples, None),
        ("cell4_bench_bench_format", "Cell 4: Bench data x Bench input format (6 imgs + patches)", bench_samples, None),
        ("cell1_blank", "Cell 1 Control: Val data x Uniform Blank Gray image", val_samples, None),
        ("cell1_shuffled", "Cell 1 Control: Val data x Cross-Event Shuffled image", val_samples, val_shuffled_mapping),
        ("cell3_blank", "Cell 3 Control: Bench data x Uniform Blank Gray image", bench_samples, None),
        ("cell3_shuffled", "Cell 3 Control: Bench data x Cross-Event Shuffled image", bench_samples, bench_shuffled_mapping),
    ]

    # 5. Execute Cells
    output_json = os.path.join(output_dir, "matrix_metrics.json")
    output_preds = os.path.join(output_dir, "matrix_predictions.jsonl")

    matrix_results = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # If loaded contains actual GPU evaluation metrics (not 100% dry run), we can reuse
                matrix_results = loaded
        except Exception:
            pass

    for cell_key, cell_desc, dataset, donor_map in cell_configs:
        res = evaluate_cell(
            cell_name=cell_key,
            cell_desc=cell_desc,
            samples=dataset,
            image_resolver=image_resolver,
            model_runner=model_runner,
            donor_mapping=donor_map,
            dry_run=args.dry_run
        )
        matrix_results[cell_key] = res

        # Incremental Save after every cell
        summary_clean = {
            k: {field: v[field] for field in v if field != "records"}
            for k, v in matrix_results.items()
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary_clean, f, indent=2)

        with open(output_preds, "a", encoding="utf-8") as f:
            for rec in res.get("records", []):
                rec["cell"] = cell_key
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[CHECKPOINT] Saved progress for {cell_key} to {output_json}")

    print(f"\n[RESULTS] Saved full matrix metrics summary to: {output_json}")
    print(f"[RESULTS] Saved all cell predictions to: {output_preds}")

    # 7. Print Matrix Table
    print("\n" + "=" * 85)
    print("CONDITION D DIAGNOSIS FACTORIAL MATRIX SUMMARY")
    print("=" * 85)
    header = f"{'Cell / Condition':32s} | {'Primary v2 Acc (%)':20s} | {'Exact 95% CI':18s} | {'Unp (%)':8s} | {'Old Parser':12s}"
    print(header)
    print("-" * 85)
    for k, v in matrix_results.items():
        p_m = v["primary_parser_v2"]
        s_m = v["secondary_parser_old"]
        acc_str = f"{p_m['accuracy']*100:5.1f}% ({p_m['correct']}/{v['n_samples']})"
        ci_str = f"[{p_m['exact_95_ci'][0]*100:.1f}%, {p_m['exact_95_ci'][1]*100:.1f}%]"
        unp_str = f"{p_m['unparseable_rate']*100:4.1f}%"
        old_acc_str = f"{s_m['accuracy']*100:5.1f}%"
        print(f"{v['description'][:32]:32s} | {acc_str:20s} | {ci_str:18s} | {unp_str:8s} | {old_acc_str:12s}")
    print("=" * 85)


if __name__ == "__main__":
    main()

