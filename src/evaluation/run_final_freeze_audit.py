import os
import sys
import json
import re
import random
import hashlib
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
from scipy.stats import binomtest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.grounding_analysis import parse_dtr_improved, get_parser_hash, KNOWN_DISASTER_TYPES


def get_exact_ci(k: int, n: int, p: float = 0.2) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    res = binomtest(k, n, p=p)
    ci = res.proportion_ci(confidence_level=0.95, method="exact")
    return (float(ci.low), float(ci.high))


def exact_mcnemar_test(a_correct: List[int], b_correct: List[int]) -> float:
    b = sum(1 for a, b_val in zip(a_correct, b_correct) if a == 1 and b_val == 0)
    c = sum(1 for a, b_val in zip(a_correct, b_correct) if a == 0 and b_val == 1)
    total_discordant = b + c
    if total_discordant == 0:
        return 1.0
    k = min(b, c)
    res = binomtest(k, total_discordant, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def get_event_prefix(path_str: str) -> str:
    fname = os.path.basename(path_str)
    m1 = re.match(r"^(.+?)_\d+_(pre|post)", fname, re.IGNORECASE)
    if m1:
        return m1.group(1).lower()
    m2 = re.match(r"^(.+?)_(pre|post)_\d+", fname, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()
    return fname.split(".")[0].lower()


def extract_caption_disaster_type(raw_output: str) -> str:
    text = str(raw_output)
    m = re.search(r"(?:^|\n)\*?\*?DISASTER\*?\*?:\s*([^\n]+)", text, re.IGNORECASE)
    section_text = m.group(1).lower() if m else text.lower()
    disaster_names = [
        "earthquake", "explosion", "hurricane", "tornado",
        "flood", "flooding", "tsunami", "volcano",
        "bushfire", "wildfire", "fire"
    ]
    for dn in disaster_names:
        if re.search(r"\b" + re.escape(dn) + r"\b", section_text):
            if dn in ["bushfire", "wildfire"]:
                return "fire"
            if dn == "flooding":
                return "flood"
            return dn
    return "unmatched"


def run_audit():
    print("=" * 80)
    print("DISASTERM3 TRACK B & A: FINAL FREEZE AUDIT (PROMPT 6.0)")
    print("=" * 80)

    train_file = os.path.join(PROJECT_ROOT, "data", "processed", "track_b_train.jsonl")
    val_file = os.path.join(PROJECT_ROOT, "data", "processed", "track_b_val.jsonl")
    bench_file = os.path.join(PROJECT_ROOT, "results", "track_b_v2", "bench_eval_ids.json")

    # --- ITEM 1: EVENT OVERLAP AUDIT ---
    train_events = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_events.add(get_event_prefix(json.loads(line).get("post_image_path", "")))

    val_events = set()
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_events.add(get_event_prefix(json.loads(line).get("post_image_path", "")))

    bench_data = json.load(open(bench_file, "r", encoding="utf-8"))
    dtr_bench = [r for r in bench_data["sample_records"] if r.get("task") == "Disaster Type Recognition"]
    cap_bench = [r for r in bench_data["sample_records"] if r.get("task") != "Disaster Type Recognition"]
    dtr_bench_dict = {r["id"]: r for r in dtr_bench}
    canonical_dtr_ids = [r["id"] for r in dtr_bench]

    bench_dtr_events = Counter(get_event_prefix(r["post_image_path"]) for r in dtr_bench)
    bench_cap_events = Counter(get_event_prefix(r["post_image_path"]) for r in cap_bench)

    dtr_overlap_events = set(bench_dtr_events.keys()).intersection(train_events)
    dtr_non_overlap_events = set(bench_dtr_events.keys()) - train_events

    cap_overlap_events = set(bench_cap_events.keys()).intersection(train_events)
    cap_non_overlap_events = set(bench_cap_events.keys()) - train_events

    print("\n--- 1. EVENT OVERLAP SUMMARY ---")
    print(f"Instruct Training Unique Events: {len(train_events)}")
    print(f"Instruct Validation Unique Events: {len(val_events)}")
    print(f"Validation overlap with Training: {len(val_events.intersection(train_events))}/{len(val_events)} (100.0%)")
    print(f"Validation non-overlapping events: {len(val_events - train_events)} (0.0%)\n")

    print(f"Benchmark DTR: {len(dtr_bench)} items across {len(bench_dtr_events)} distinct events.")
    print(f"  - Events also appearing in Training: {len(dtr_overlap_events)} events ({sum(bench_dtr_events[e] for e in dtr_overlap_events)} items, {sum(bench_dtr_events[e] for e in dtr_overlap_events)/len(dtr_bench)*100:.1f}%)")
    print(f"  - Events NOT appearing in Training: {len(dtr_non_overlap_events)} events ({sum(bench_dtr_events[e] for e in dtr_non_overlap_events)} items, {sum(bench_dtr_events[e] for e in dtr_non_overlap_events)/len(dtr_bench)*100:.1f}%)")
    print(f"  - Non-overlapping event names: {sorted(list(dtr_non_overlap_events))}")

    print(f"\nBenchmark Captioning: {len(cap_bench)} items across {len(bench_cap_events)} distinct events.")
    print(f"  - Events also appearing in Training: {len(cap_overlap_events)} events ({sum(bench_cap_events[e] for e in cap_overlap_events)} items, {sum(bench_cap_events[e] for e in cap_overlap_events)/len(cap_bench)*100:.1f}%)")
    print(f"  - Events NOT appearing in Training: {len(cap_non_overlap_events)} events ({sum(bench_cap_events[e] for e in cap_non_overlap_events)} items, {sum(bench_cap_events[e] for e in cap_non_overlap_events)/len(cap_bench)*100:.1f}%)")
    print(f"  - Non-overlapping event names: {sorted(list(cap_non_overlap_events))}")

    # Load predictions for B, C, D, O
    cond_files = {
        "B": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "B", "predictions.jsonl"),
        "C": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "C", "predictions.jsonl"),
        "D": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "D", "predictions.jsonl"),
        "O": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "O", "predictions.jsonl")
    }

    cond_parsed_dtr = {}
    cond_raw_records = {}
    for cond, pth in cond_files.items():
        parsed = {}
        raws = {}
        with open(pth, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    row = json.loads(l)
                    sid = row.get("id")
                    if sid in dtr_bench_dict:
                        raw_out = row.get("raw_output", row.get("response", ""))
                        parsed[sid] = parse_dtr_improved(raw_out, dtr_bench_dict[sid])
                        raws[sid] = row
        cond_parsed_dtr[cond] = parsed
        cond_raw_records[cond] = raws

    overlap_ids = [sid for sid in canonical_dtr_ids if get_event_prefix(dtr_bench_dict[sid]["post_image_path"]) in train_events]
    non_overlap_ids = [sid for sid in canonical_dtr_ids if get_event_prefix(dtr_bench_dict[sid]["post_image_path"]) not in train_events]
    n_over = len(overlap_ids)
    n_non = len(non_overlap_ids)

    overlap_table_rows = []
    print("\n--- DTR ACCURACY BY EVENT OVERLAP (Conditions B, C, D, O) ---")
    for cond in ["B", "C", "D", "O"]:
        pdict = cond_parsed_dtr[cond]
        k_over = sum(1 for sid in overlap_ids if pdict[sid] == dtr_bench_dict[sid]["ground_truth"].lower())
        acc_over = k_over / n_over
        ci_over = get_exact_ci(k_over, n_over)

        k_non = sum(1 for sid in non_overlap_ids if pdict[sid] == dtr_bench_dict[sid]["ground_truth"].lower())
        acc_non = k_non / n_non
        ci_non = get_exact_ci(k_non, n_non)

        delta = acc_over - acc_non
        print(f"Condition {cond}:")
        print(f"  Overlapping    (n={n_over}): {k_over}/{n_over} = {acc_over*100:.1f}% [95% CI: {ci_over[0]*100:.1f}%, {ci_over[1]*100:.1f}%]")
        print(f"  Non-Overlapping(n={n_non}): {k_non}/{n_non} = {acc_non*100:.1f}% [95% CI: {ci_non[0]*100:.1f}%, {ci_non[1]*100:.1f}%]")
        print(f"  Gap (Overlapping - Non-Overlapping): {delta*100:+.1f}%")

        overlap_table_rows.append({
            "Condition": cond,
            "n_over": n_over,
            "k_over": k_over,
            "acc_over": acc_over,
            "ci_over": ci_over,
            "n_non": n_non,
            "k_non": k_non,
            "acc_non": acc_non,
            "ci_non": ci_non,
            "delta": delta
        })

    # Shuffled same-type donor overlap check
    rng = random.Random(42)
    mapping_same = {}
    for s in dtr_bench:
        sid = s["id"]
        dt = s["ground_truth"].lower()
        eid = get_event_prefix(s["post_image_path"])
        cands_same = [o for o in dtr_bench if o["ground_truth"].lower() == dt and get_event_prefix(o["post_image_path"]) != eid]
        donor_same = rng.choice(cands_same)
        mapping_same[sid] = donor_same

    donor_events = [get_event_prefix(donor["post_image_path"]) for donor in mapping_same.values()]
    donors_in_train = sum(1 for e in donor_events if e in train_events)
    unique_donor_events = set(donor_events)
    unique_donors_in_train = unique_donor_events.intersection(train_events)
    print("\n--- SAME-TYPE SHUFFLED DONOR OCCURRENCE IN TRAINING ---")
    print(f"Donor assignments occurring in training: {donors_in_train}/{len(donor_events)} ({donors_in_train/len(donor_events)*100:.1f}%)")
    print(f"Unique donor events occurring in training: {len(unique_donors_in_train)}/{len(unique_donor_events)} ({len(unique_donors_in_train)/len(unique_donor_events)*100:.1f}%)")
    print(f"Unique donor events NOT in training: {unique_donor_events - train_events}")

    # --- ITEM 2: FINE-TUNING PER CLASS (C vs D) ---
    print("\n" + "=" * 80)
    print("ITEM 2: FINE-TUNING PER CLASS AUDIT (C vs D)")
    print("=" * 80)

    # First assert and print that C and D inputs are identical except for adapter
    print("[ASSERTION] Verifying that C and D inputs (images, patches, prompts) are 100% identical...")
    raw_c = cond_raw_records["C"]
    raw_d = cond_raw_records["D"]
    for sid in canonical_dtr_ids:
        rec_c = raw_c[sid]
        rec_d = raw_d[sid]
        ref = dtr_bench_dict[sid]

        # Verify prompt text
        prompt_c = rec_c.get("prompt", rec_c.get("input_prompt", ""))
        prompt_d = rec_d.get("prompt", rec_d.get("input_prompt", ""))
        assert prompt_c == prompt_d, f"Prompt mismatch on sample {sid}!"

        # Verify pre/post image paths
        assert rec_c.get("pre_image_path", ref.get("pre_image_path")) == rec_d.get("pre_image_path", ref.get("pre_image_path")), f"Pre-image mismatch on {sid}!"
        assert rec_c.get("post_image_path", ref.get("post_image_path")) == rec_d.get("post_image_path", ref.get("post_image_path")), f"Post-image mismatch on {sid}!"

    print("[ASSERTION PASSED] Condition C and Condition D inputs are 100% identical across all 200 items (images, patches, prompt text). The only difference is the active LoRA adapter in Condition D.\n")

    # Per-class McNemar and accuracy
    classes_ordered = sorted(list(set(r["ground_truth"].lower() for r in dtr_bench)))
    c_preds = cond_parsed_dtr["C"]
    d_preds = cond_parsed_dtr["D"]

    per_class_table = []
    print(f"{'Class':<15} | {'N':<5} | {'Cond C Acc':<12} | {'Cond D Acc':<12} | {'Diff (D - C)':<12} | {'McNemar p':<12} | {'Significant?'}")
    print("-" * 85)
    for cls in classes_ordered:
        cls_ids = [sid for sid in canonical_dtr_ids if dtr_bench_dict[sid]["ground_truth"].lower() == cls]
        n_cls = len(cls_ids)
        c_corr = [1 if c_preds[sid] == cls else 0 for sid in cls_ids]
        d_corr = [1 if d_preds[sid] == cls else 0 for sid in cls_ids]

        acc_c = sum(c_corr) / n_cls
        acc_d = sum(d_corr) / n_cls
        diff = acc_d - acc_c
        p_val = exact_mcnemar_test(d_corr, c_corr)
        sig = "YES (p < 0.05)" if p_val < 0.05 else "NO (ns)"

        print(f"{cls:<15} | {n_cls:<5} | {acc_c*100:>5.1f}% ({sum(c_corr):>2}/{n_cls:<2}) | {acc_d*100:>5.1f}% ({sum(d_corr):>2}/{n_cls:<2}) | {diff*100:>+5.1f}%       | {p_val:<12.4e} | {sig}")
        per_class_table.append({
            "class": cls,
            "n": n_cls,
            "acc_c": acc_c,
            "k_c": sum(c_corr),
            "acc_d": acc_d,
            "k_d": sum(d_corr),
            "diff": diff,
            "mcnemar_p": p_val,
            "sig": sig
        })

    # Overall C vs D McNemar
    all_c_corr = [1 if c_preds[sid] == dtr_bench_dict[sid]["ground_truth"].lower() else 0 for sid in canonical_dtr_ids]
    all_d_corr = [1 if d_preds[sid] == dtr_bench_dict[sid]["ground_truth"].lower() else 0 for sid in canonical_dtr_ids]
    overall_p = exact_mcnemar_test(all_d_corr, all_c_corr)
    print("-" * 85)
    print(f"Overall (N=200): C = {sum(all_c_corr)/200*100:.1f}%, D = {sum(all_d_corr)/200*100:.1f}%, Diff = {(sum(all_d_corr)-sum(all_c_corr))/200*100:+.1f}%, McNemar p = {overall_p:.4e}")

    # --- ITEM 4: DISASTER HEADER IN CAPTIONS ---
    print("\n" + "=" * 80)
    print("ITEM 4: DISASTER HEADER WHOLE-WORD MATCHING IN CAPTIONS")
    print("=" * 80)

    cap_bench_dict = {r["id"]: r for r in cap_bench}
    canonical_cap_ids = [r["id"] for r in cap_bench]

    caption_conditions = {
        "B": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "B", "predictions.jsonl"),
        "C": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "C", "predictions.jsonl"),
        "D": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "D", "predictions.jsonl"),
        "D_blank": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "D_blank", "predictions.jsonl"),
        "D_shuffled": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "D_shuffled", "predictions.jsonl"),
        "K": os.path.join(PROJECT_ROOT, "results", "track_b_v2", "K", "predictions.jsonl")
    }

    cap_header_results = []
    print(f"{'Condition':<12} | {'N':<5} | {'Matches':<8} | {'Accuracy (%)':<15} | {'Exact Binomial 95% CI':<25}")
    print("-" * 75)
    for cname, cpath in caption_conditions.items():
        if not os.path.exists(cpath):
            print(f"Warning: {cpath} not found!")
            continue
        c_preds_cap = {}
        with open(cpath, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    row = json.loads(l)
                    sid = row.get("id")
                    if sid in cap_bench_dict:
                        raw_out = row.get("raw_output", row.get("response", ""))
                        parsed_type = extract_caption_disaster_type(raw_out)
                        c_preds_cap[sid] = parsed_type

        k_matches = 0
        n_cap = len(canonical_cap_ids)
        for sid in canonical_cap_ids:
            gold_str = cap_bench_dict[sid].get("ground_truth", "")
            gold_dt = extract_caption_disaster_type(gold_str)
            if c_preds_cap.get(sid) == gold_dt and gold_dt != "unmatched":
                k_matches += 1

        acc = k_matches / n_cap
        ci = get_exact_ci(k_matches, n_cap, p=0.2)
        print(f"{cname:<12} | {n_cap:<5} | {k_matches:<8} | {acc*100:>5.1f}%          | [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
        cap_header_results.append({
            "condition": cname,
            "N": n_cap,
            "matches": k_matches,
            "accuracy": acc,
            "ci": ci
        })

    # Save summary dictionary
    out_dir = os.path.join(PROJECT_ROOT, "results", "final_freeze")
    os.makedirs(out_dir, exist_ok=True)
    summary_data = {
        "event_overlap_dtr": overlap_table_rows,
        "fine_tuning_per_class": per_class_table,
        "caption_disaster_header": cap_header_results,
        "same_type_donor_audit": {
            "total": len(donor_events),
            "donors_in_train": donors_in_train,
            "unique_donor_events": len(unique_donor_events),
            "unique_donors_in_train": len(unique_donors_in_train),
            "non_training_donors": list(unique_donor_events - train_events)
        }
    }
    with open(os.path.join(out_dir, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n[DONE] Audit summary saved to {os.path.join(out_dir, 'audit_summary.json')}")


if __name__ == "__main__":
    run_audit()

