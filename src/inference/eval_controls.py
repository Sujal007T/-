"""
DisasterM3 Track B Negative Grounding Controls Pipeline (eval_controls.py).
Evaluates Condition D under strict negative controls to verify visual grounding:

Controls:
  1. D_shuffled: Each sample receives the satellite images (pre, post, 4 patches)
     of a random sample from a DIFFERENT disaster event.
     Tests whether the model's damage descriptions degrade when visual context
     is uncorrelated with the ground truth.
  2. D_blank: Each sample receives a uniform mean-gray image (RGB 128, 128, 128)
     for pre, post, and 4 patches.
     Tests whether the model degrades when all visual signals are removed.

Key Features:
  - Evaluates on the EXACT same stratified benchmark sample IDs from bench_eval_ids.json.
  - Reuses trained LoRA adapter (lora_smoke_test_final) and greedy decoding.
  - Generates resumable predictions.jsonl per control condition.
  - Automatically invokes grounding_analysis.py to update grounding_report.md.
"""

import os
import sys
import re
import json
import time
import random
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
from PIL import Image

import torch

# Ensure disaSense_project is in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics_trackb import (
    compute_dtr_metrics,
    compute_caption_metrics,
    bootstrap_ci,
    export_manual_review_csv
)
from src.inference.eval_trackb import (
    resolve_path,
    load_config_and_schema,
    ImageResolver,
    make_2x2_patches,
    compute_sample_input_hash,
    ModelRunner,
    TASK_TOKENS
)
from src.evaluation.grounding_analysis import (
    parse_dtr_improved,
    analyze_training_mix
)


# =========================================================================
#  1. Event Shuffling Mapping
# =========================================================================

def build_shuffled_sample_mapping(
    eval_set: List[Dict[str, Any]],
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Builds a deterministic mapping where each sample is assigned the images
    of a random sample from a strictly DIFFERENT disaster event.
    """
    rng = random.Random(seed)

    # Group samples by event_id
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in eval_set:
        by_event[s.get("event_id", "unknown")].append(s)

    all_events = sorted(list(by_event.keys()))
    assert len(all_events) >= 2, f"Cross-event shuffling requires at least 2 distinct events, found {len(all_events)}."

    mapping: Dict[str, Dict[str, Any]] = {}

    for s in eval_set:
        curr_eid = s.get("event_id", "unknown")
        # Eligible events strictly different from current
        other_events = [e for e in all_events if e != curr_eid]
        assert len(other_events) > 0, f"Cross-event constraint violation: no different event available for {s['id']} (event: {curr_eid})"

        chosen_event = rng.choice(other_events)
        donor_sample = rng.choice(by_event[chosen_event])
        assert donor_sample.get("event_id") != curr_eid, f"Shuffled image must come from a DIFFERENT event: got {donor_sample.get('event_id')} vs {curr_eid}"
        mapping[s["id"]] = donor_sample

    return mapping


# =========================================================================
#  2. Control Input Builders
# =========================================================================

def build_shuffled_inputs(
    sample: Dict[str, Any],
    donor_sample: Dict[str, Any],
    image_resolver: ImageResolver
) -> Tuple[List[Dict[str, Any]], str, List[Image.Image]]:
    """Constructs prompt using sample text, but images from donor_sample."""
    task = sample.get("task", "")
    is_dtr = (task == "Disaster Type Recognition")

    # Load donor images
    donor_post = image_resolver.load_image(donor_sample.get("post_image_path", ""))
    donor_pre = image_resolver.load_image(donor_sample.get("pre_image_path", ""))
    donor_patches = make_2x2_patches(donor_post)
    images = [donor_pre, donor_post] + donor_patches

    # Prompt text from original sample
    base_prompt = sample.get("prompts", "")
    if isinstance(base_prompt, list):
        base_prompt = base_prompt[0] if base_prompt else ""
    if not base_prompt:
        base_prompt = "What kind of calamity has this area experienced?" if is_dtr else "Please describe a comprehensive damage situation based on pre- and post-disaster images."

    if is_dtr and sample.get("options_str"):
        prompt_text = f"{base_prompt}\nOptions: {sample.get('options_str')}"
    else:
        prompt_text = base_prompt

    system_prompt = TASK_TOKENS.get(task, "")
    user_content = [
        {"type": "image", "image": donor_pre},
        {"type": "image", "image": donor_post}
    ]
    for p in donor_patches:
        user_content.append({"type": "image", "image": p})

    augmented_prompt = prompt_text + "\n\nNote: The last 4 images above are 2x2 local patches of the post-disaster image."
    user_content.append({"type": "text", "text": augmented_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return messages, augmented_prompt, images


def build_blank_inputs(
    sample: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], str, List[Image.Image]]:
    """Constructs prompt using sample text, but uniform blank mean-gray images."""
    task = sample.get("task", "")
    is_dtr = (task == "Disaster Type Recognition")

    blank_img = Image.new("RGB", (448, 448), color=(128, 128, 128))
    blank_patches = [Image.new("RGB", (224, 224), color=(128, 128, 128)) for _ in range(4)]
    images = [blank_img, blank_img] + blank_patches

    base_prompt = sample.get("prompts", "")
    if isinstance(base_prompt, list):
        base_prompt = base_prompt[0] if base_prompt else ""
    if not base_prompt:
        base_prompt = "What kind of calamity has this area experienced?" if is_dtr else "Please describe a comprehensive damage situation based on pre- and post-disaster images."

    if is_dtr and sample.get("options_str"):
        prompt_text = f"{base_prompt}\nOptions: {sample.get('options_str')}"
    else:
        prompt_text = base_prompt

    system_prompt = TASK_TOKENS.get(task, "")
    user_content = [
        {"type": "image", "image": blank_img},
        {"type": "image", "image": blank_img}
    ]
    for p in blank_patches:
        user_content.append({"type": "image", "image": p})

    augmented_prompt = prompt_text + "\n\nNote: The last 4 images above are 2x2 local patches of the post-disaster image."
    user_content.append({"type": "text", "text": augmented_prompt})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    return messages, augmented_prompt, images


# =========================================================================
#  3. Negative Controls Runner
# =========================================================================

def run_grounding_controls(
    conditions: List[str],
    smoke: bool = False,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42
):
    """Executes negative control conditions (D_shuffled, D_blank) on GPU."""
    config, schema = load_config_and_schema()

    bench_file = resolve_path("data/raw/disasterm3/DisasterM3_Bench/benchmark_release.json")
    train_file = resolve_path("data/processed/track_b_train.jsonl")
    bench_root = os.path.dirname(bench_file)
    output_base = resolve_path(output_dir or "results/track_b_v2")
    os.makedirs(output_base, exist_ok=True)
    eval_ids_file = os.path.join(output_base, "bench_eval_ids.json")
    rules_file = resolve_path("configs/section_rules.yaml")

    if checkpoint_dir is None:
        checkpoint_dir = resolve_path("checkpoints/lora_smoke_test_final")
    else:
        checkpoint_dir = resolve_path(checkpoint_dir)

    model_name = config.get("model", {}).get("name", "Qwen/Qwen2.5-VL-3B-Instruct")
    max_tokens_dtr = schema.get("decoding", {}).get("max_new_tokens_dtr", 16)
    max_tokens_cap = schema.get("decoding", {}).get("max_new_tokens_caption", 400)

    # 1. Load evaluation set from existing bench_eval_ids.json
    if not os.path.exists(eval_ids_file):
        raise FileNotFoundError(f"Evaluation set {eval_ids_file} not found. Run eval_trackb.py first.")

    with open(eval_ids_file, "r", encoding="utf-8") as f:
        saved_manifest = json.load(f)
    eval_set = saved_manifest["sample_records"]

    if smoke:
        # 10 DTR and 10 Captioning
        dtr_s = [r for r in eval_set if r["task"] == "Disaster Type Recognition"][:10]
        cap_s = [r for r in eval_set if r["task"] in ["Disaster Report", "disaster caption"]][:10]
        eval_set = dtr_s + cap_s

    print("=" * 70)
    print("DISASTERM3 TRACK B: NEGATIVE GROUNDING CONTROLS (D_shuffled, D_blank)")
    print(f"Controls to evaluate: {conditions}")
    print(f"Total samples       : {len(eval_set)} ({sum(1 for r in eval_set if r['task'] == 'Disaster Type Recognition')} DTR, {sum(1 for r in eval_set if r['task'] != 'Disaster Type Recognition')} Caption)")
    print(f"Checkpoint Dir      : {checkpoint_dir}")
    print("=" * 70)

    # 2. Image resolver & shuffled mapping
    image_resolver = ImageResolver(bench_root)
    shuffled_mapping = build_shuffled_sample_mapping(eval_set, seed=seed)

    # Automated Preflight Gate: verify all images are present and pass variance checks
    print("\n[PREFLIGHT GATE] Automatically verifying image integrity before running controls...")
    from src.evaluation.preflight_images import run_preflight_verification
    try:
        run_preflight_verification(
            bench_eval_ids_path=eval_ids_file,
            val_jsonl_path=None,
            bench_root=bench_root,
            output_dir=output_base,
            contact_sheet_count=0,
            min_allowed_std=5.0,
            seed=seed
        )
        print("[PREFLIGHT GATE PASS] All control images verified authentic and present on disk.")
    except Exception as e:
        print(f"\n[CRITICAL PREFLIGHT FAILURE] {e}")
        raise RuntimeError(f"Automated preflight failed: {e}. Aborting controls evaluation immediately.")

    # 3. Load Model (Condition D uses fine-tuned adapter)
    model_runner = ModelRunner(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        device=device
    )
    model_runner.load()
    model_runner.set_condition_adapter("D")  # Ensure adapter is enabled

    # 4. Evaluate each control condition
    for cond in conditions:
        print("\n" + "=" * 60)
        print(f"RUNNING NEGATIVE CONTROL: {cond}")
        print("=" * 60)

        cond_dir = os.path.join(output_base, cond)
        os.makedirs(cond_dir, exist_ok=True)
        pred_file = os.path.join(cond_dir, "predictions.jsonl")
        metrics_file = os.path.join(cond_dir, "metrics.json")
        review_file = os.path.join(cond_dir, "manual_review.csv")

        # Load existing finished IDs for resumability
        finished_records: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        finished_records[r["id"]] = r
            print(f"[{cond}] Found {len(finished_records)} existing predictions. Resuming...")

        latencies = []
        with open(pred_file, "a", encoding="utf-8") as out_f:
            for idx, sample in enumerate(eval_set):
                sample_id = sample["id"]
                task = sample.get("task", "")
                is_dtr = (task == "Disaster Type Recognition")
                max_tokens = max_tokens_dtr if is_dtr else max_tokens_cap

                if sample_id in finished_records:
                    continue

                t0 = time.time()
                if cond == "D_shuffled":
                    donor = shuffled_mapping[sample_id]
                    messages, prompt_text, imgs = build_shuffled_inputs(sample, donor, image_resolver)
                    input_hash = compute_sample_input_hash(f"shuffled_from_{donor['event_id']}", prompt_text, imgs)
                elif cond == "D_blank":
                    messages, prompt_text, imgs = build_blank_inputs(sample)
                    input_hash = compute_sample_input_hash("blank_gray", prompt_text, imgs)
                else:
                    raise ValueError(f"Unknown control condition: {cond}")

                # Model generation
                raw_output = model_runner.generate(messages, max_new_tokens=max_tokens)
                latency = time.time() - t0
                latencies.append(latency)

                # Parse
                if is_dtr:
                    parsed = parse_dtr_improved(raw_output, sample)
                else:
                    parsed = raw_output

                pred_rec = {
                    "id": sample_id,
                    "bench_index": sample.get("bench_index"),
                    "task": task,
                    "disaster_type": sample.get("disaster_type", ""),
                    "event_id": sample.get("event_id", ""),
                    "donor_event_id": donor.get("event_id") if cond == "D_shuffled" else "none",
                    "gold": sample.get("ground_truth", ""),
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "input_hash": input_hash,
                    "latency_s": round(latency, 4)
                }

                out_f.write(json.dumps(pred_rec) + "\n")
                out_f.flush()
                finished_records[sample_id] = pred_rec

                if len(latencies) == 20:
                    avg_lat = sum(latencies) / len(latencies)
                    rem = len(eval_set) - (idx + 1)
                    print(f"[{cond}] ETA: ~{(avg_lat * rem)/60:.1f} minutes remaining (avg {avg_lat:.2f}s / sample)")

                if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_set):
                    print(f"[{cond}] Processed {idx + 1}/{len(eval_set)} samples...")

        # Compute standard metrics
        all_preds = [finished_records[r["id"]] for r in eval_set if r["id"] in finished_records]
        dtr_preds = [r for r in all_preds if r["task"] == "Disaster Type Recognition"]
        cap_preds = [r for r in all_preds if r["task"] in ["Disaster Report", "disaster caption"]]

        cond_metrics: Dict[str, Any] = {
            "condition": cond,
            "sample_count": len(all_preds),
            "dtr_count": len(dtr_preds),
            "caption_count": len(cap_preds)
        }

        if dtr_preds:
            y_true_dtr = [r["disaster_type"] for r in dtr_preds]
            y_pred_dtr = [r["parsed"] for r in dtr_preds]
            dtr_res = compute_dtr_metrics(y_true_dtr, y_pred_dtr)
            acc_ind = [1.0 if t == p else 0.0 for t, p in zip(y_true_dtr, y_pred_dtr)]
            _, a_lo, a_hi = bootstrap_ci(acc_ind, seed=seed)
            dtr_res["accuracy_ci_95"] = [a_lo, a_hi]
            cond_metrics["dtr"] = dtr_res

        if cap_preds:
            golds_cap = [r["gold"] for r in cap_preds]
            preds_cap = [r["parsed"] for r in cap_preds]
            dtypes_cap = [r["disaster_type"] for r in cap_preds]
            cap_res = compute_caption_metrics(golds_cap, preds_cap, disaster_types=dtypes_cap)
            _, r_lo, r_hi = bootstrap_ci(cap_res["raw_rouge_l"], seed=seed)
            _, b_lo, b_hi = bootstrap_ci(cap_res["raw_bleu_4"], seed=seed)
            _, c_lo, c_hi = bootstrap_ci(cap_res["raw_compliance"], seed=seed)
            cap_res["rouge_l_ci_95"] = [r_lo, r_hi]
            cap_res["bleu_4_ci_95"] = [b_lo, b_hi]
            cap_res["compliance_ci_95"] = [c_lo * 100.0, c_hi * 100.0]

            del cap_res["raw_rouge_l"]
            del cap_res["raw_bleu_4"]
            del cap_res["raw_compliance"]
            cond_metrics["captioning"] = cap_res

            export_manual_review_csv([r["id"] for r in cap_preds], golds_cap, preds_cap, review_file, n_samples=30, seed=seed)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(cond_metrics, f, indent=2)
        print(f"[{cond}] Saved metrics to {metrics_file}")

    image_resolver.close()

    # 5. Automatically invoke grounding_analysis to update report
    print("\n" + "=" * 70)
    print("UPDATING VISUAL GROUNDING REPORT (grounding_report_v2.md)...")
    print("=" * 70)

    try:
        from src.evaluation.grounding_analysis import main as run_grounding_suite
        sys_argv_backup = list(sys.argv)
        sys.argv = [
            "grounding_analysis.py",
            "--results_dir", output_base,
            "--bench_file", bench_file,
            "--train_file", train_file,
            "--rules_file", rules_file,
            "--seed", str(seed)
        ]
        run_grounding_suite()
        sys.argv = sys_argv_backup
        print(f"\n[ALL CONTROLS FINISHED] Visual Grounding Report generated in {output_base}")
    except Exception as e:
        print(f"[CONTROLS NOTICE] Grounding suite report notice: {e}")
        print(f"You can also run grounding analysis independently via:\n  python -m src.evaluation.grounding_analysis --results_dir {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Negative Visual Grounding Controls (D_shuffled, D_blank).")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["D_shuffled", "D_blank"],
        choices=["D_shuffled", "D_blank"],
        help="Controls to evaluate (default: D_shuffled D_blank)"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: evaluate 10 samples per task"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to trained LoRA checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/track_b_v2",
        help="Directory to save control results (default: results/track_b_v2)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for cross-event shuffling and bootstrapping"
    )

    args = parser.parse_args()

    run_grounding_controls(
        conditions=args.conditions,
        smoke=args.smoke,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

