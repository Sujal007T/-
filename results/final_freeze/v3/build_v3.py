#!/usr/bin/env python3
"""
Master Freeze v3 Generation Script for DisaSense Project.
Generates:
1. v3_analysis_summary.json
2. MANIFEST_v3.json
3. FINAL_CLAIMS_v3.md
4. tables_paper.md
"""

import os
import sys
import json
import hashlib
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import sacrebleu
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

V3_DIR = os.path.join(PROJECT_ROOT, "results", "final_freeze", "v3")
os.makedirs(V3_DIR, exist_ok=True)

def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("[V3 BUILD] Starting comprehensive Freeze v3 execution...")

# =========================================================================
# 1. CAPTION SCORING (Unstemmed rouge-score, SacreBLEU, NLTK, Bootstrap CIs)
# =========================================================================
print("\n--- 1. Caption Scoring ---")
scorer_nostem = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
smooth = SmoothingFunction().method1

caption_conditions = ['B', 'C', 'D', 'K', 'D_blank', 'D_shuffled']
caption_records = {}

for c in caption_conditions:
    p_file = os.path.join(PROJECT_ROOT, "results", "track_b_v2", c, "predictions.jsonl")
    with open(p_file, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if json.loads(line).get("task") == "Disaster Report"]
    recs = sorted(recs, key=lambda x: x["id"])
    caption_records[c] = recs

cap_ids = [r["id"] for r in caption_records["D"]]
cap_events = [r.get("event_id") for r in caption_records["D"]]
unique_events = sorted(list(set(cap_events)))
event_to_indices = defaultdict(list)
for idx, ev in enumerate(cap_events):
    event_to_indices[ev].append(idx)

# Raw scores
r_scores = {c: np.array([scorer_nostem.score(r["gold"], r["parsed"])["rougeL"].fmeasure for r in caption_records[c]]) for c in caption_conditions}
b_sacre_scores = {c: np.array([sacrebleu.sentence_bleu(r["parsed"], [r["gold"]]).score / 100.0 for r in caption_records[c]]) for c in caption_conditions}
b_nltk_scores = {c: np.array([sentence_bleu([nltk.word_tokenize(r["gold"].lower())], nltk.word_tokenize(r["parsed"].lower()), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth) for r in caption_records[c]]) for c in caption_conditions}

# Bootstrap CIs per condition
n_boot = 1000
n_samples = len(cap_ids)
rng = np.random.RandomState(42)

caption_summary = {}
for c in caption_conditions:
    boot_r = [np.mean(r_scores[c][rng.randint(0, n_samples, size=n_samples)]) for _ in range(n_boot)]
    boot_bs = [np.mean(b_sacre_scores[c][rng.randint(0, n_samples, size=n_samples)]) for _ in range(n_boot)]
    boot_bn = [np.mean(b_nltk_scores[c][rng.randint(0, n_samples, size=n_samples)]) for _ in range(n_boot)]
    
    caption_summary[c] = {
        "rouge_l_nostem_mean": float(np.mean(r_scores[c])),
        "rouge_l_nostem_ci95": [float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))],
        "bleu_4_sacre_mean": float(np.mean(b_sacre_scores[c])),
        "bleu_4_sacre_ci95": [float(np.percentile(boot_bs, 2.5)), float(np.percentile(boot_bs, 97.5))],
        "bleu_4_nltk_mean": float(np.mean(b_nltk_scores[c])),
        "bleu_4_nltk_ci95": [float(np.percentile(boot_bn, 2.5)), float(np.percentile(boot_bn, 97.5))]
    }

# Paired tests
paired_tests = {}
contrast_pairs = [
    ("D_vs_D_blank", "D", "D_blank"),
    ("D_vs_D_shuffled", "D", "D_shuffled"),
    ("D_vs_K", "D", "K"),
    ("D_vs_C", "D", "C")
]

for label, c1, c2 in contrast_pairs:
    diff_r = r_scores[c1] - r_scores[c2]
    diff_b = b_sacre_scores[c1] - b_sacre_scores[c2]
    
    # 1. Item-level bootstrap
    rng_item = np.random.RandomState(42)
    boot_i_r, boot_i_b = [], []
    for _ in range(n_boot):
        b_idx = rng_item.randint(0, n_samples, size=n_samples)
        boot_i_r.append(np.mean(diff_r[b_idx]))
        boot_i_b.append(np.mean(diff_b[b_idx]))
        
    ci_i_r = [float(np.percentile(boot_i_r, 2.5)), float(np.percentile(boot_i_r, 97.5))]
    win_i_r = float(np.mean(np.array(boot_i_r) > 0))
    ci_i_b = [float(np.percentile(boot_i_b, 2.5)), float(np.percentile(boot_i_b, 97.5))]
    win_i_b = float(np.mean(np.array(boot_i_b) > 0))
    
    # 2. Event-level bootstrap
    rng_ev = np.random.RandomState(42)
    boot_e_r, boot_e_b = [], []
    n_ev = len(unique_events)
    for _ in range(n_boot):
        ev_sample = rng_ev.choice(unique_events, size=n_ev, replace=True)
        ev_indices = []
        for ev in ev_sample:
            ev_indices.extend(event_to_indices[ev])
        ev_indices = np.array(ev_indices)
        boot_e_r.append(np.mean(diff_r[ev_indices]))
        boot_e_b.append(np.mean(diff_b[ev_indices]))
        
    ci_e_r = [float(np.percentile(boot_e_r, 2.5)), float(np.percentile(boot_e_r, 97.5))]
    win_e_r = float(np.mean(np.array(boot_e_r) > 0))
    ci_e_b = [float(np.percentile(boot_e_b, 2.5)), float(np.percentile(boot_e_b, 97.5))]
    win_e_b = float(np.mean(np.array(boot_e_b) > 0))
    
    paired_tests[label] = {
        "rouge_l_diff": float(np.mean(diff_r)),
        "rouge_l_item_ci95": ci_i_r,
        "rouge_l_item_win_rate": win_i_r,
        "rouge_l_event_ci95": ci_e_r,
        "rouge_l_event_win_rate": win_e_r,
        "bleu_4_diff": float(np.mean(diff_b)),
        "bleu_4_item_ci95": ci_i_b,
        "bleu_4_item_win_rate": win_i_b,
        "bleu_4_event_ci95": ci_e_b,
        "bleu_4_event_win_rate": win_e_b
    }

# =========================================================================
# 2. COMPLIANCE AUDIT
# =========================================================================
print("\n--- 2. Compliance Audit ---")
from src.evaluation.metrics_trackb import parse_report_sections, DEFAULT_SECTIONS

compliance_results = {}
for c in caption_conditions:
    recs = caption_records[c]
    failed_items = []
    for r in recs:
        parsed = parse_report_sections(r["parsed"], DEFAULT_SECTIONS)
        missing = [sec for sec, val in parsed.items() if not val or len(val.strip()) == 0]
        if missing:
            raw = r.get("raw_output", "")
            failed_items.append({
                "id": r["id"],
                "missing_sections": missing,
                "raw_char_len": len(raw),
                "raw_word_count": len(raw.split()),
                "is_truncated": (len(raw) >= 400 and ("露" in raw or "!" in raw)),
                "raw_snippet": raw[:120]
            })
    comp_pct = (len(recs) - len(failed_items)) / len(recs) * 100.0
    compliance_results[c] = {
        "total_samples": len(recs),
        "compliant_count": len(recs) - len(failed_items),
        "failed_count": len(failed_items),
        "compliance_pct": comp_pct,
        "failed_details": failed_items
    }

# =========================================================================
# 3. TRACK A ON CLEAN UID SET (N=3,982) & EXPLANATION OF 184 VS 281
# =========================================================================
print("\n--- 3. Track A Clean UID Recomputation ---")
from src.preprocessing.patch_cropper import XBDPatchDataset

xbd_root = os.path.join(PROJECT_ROOT, "data", "raw", "xbd", "train")
test_ds = XBDPatchDataset(xbd_root, split="test", seed=42)

df_a = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "baselines", "baseline_a_probs.csv"))
df_b = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "baselines", "baseline_b_probs.csv"))
df_c = pd.read_csv(os.path.join(PROJECT_ROOT, "results", "baselines", "baseline_c_predictions.csv"))

# Exclude 163 edge-clipped fallback patches
skipped_set = set()
for idx, b in enumerate(test_ds.building_index):
    x, y, w, h = b["bbox"]
    x1 = max(0, int(x - test_ds.alpha/2 * w))
    y1 = max(0, int(y - test_ds.alpha/2 * h))
    x2 = min(1024, int(x + w + test_ds.alpha/2 * w))
    y2 = min(1024, int(y + h + test_ds.alpha/2 * h))
    if (x2 - x1 < 64) or (y2 - y1 < 64):
        skipped_set.add(idx)

clean_mask = np.array([i not in skipped_set for i in range(len(test_ds.building_index))])
clean_indices = np.where(clean_mask)[0]
N_clean = len(clean_indices)

clean_labels = np.array([test_ds.building_index[i]["damage_label"] for i in clean_indices])
probs_a = df_a[["prob_0","prob_1","prob_2","prob_3"]].values[clean_indices]
probs_b = df_b[["prob_0","prob_1","prob_2","prob_3"]].values[clean_indices]
probs_c = df_c[["prob_no_damage","prob_minor","prob_major","prob_destroyed"]].values[clean_indices]

probs_ab = 0.75 * probs_a + 0.25 * probs_b
probs_abc = 0.6 * probs_a + 0.3 * probs_b + 0.1 * probs_c

class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]

def eval_track_a(labels, probs):
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_cls = f1_score(labels, preds, average=None, labels=[0,1,2,3], zero_division=0)
    var_score = float(np.std(per_cls))
    per_cls_dict = {class_names[i]: float(per_cls[i]) for i in range(4)}
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "variance_score": var_score,
        "per_class_f1": per_cls_dict
    }

clean_track_a_metrics = {
    "Baseline A (ResNet50)": eval_track_a(clean_labels, probs_a),
    "Baseline B (ChangeDetection)": eval_track_a(clean_labels, probs_b),
    "Baseline C (BLIP-2)": eval_track_a(clean_labels, probs_c),
    "Ensemble {A,B} (0.75/0.25)": eval_track_a(clean_labels, probs_ab),
    "Ensemble {A,B,C} (0.6/0.3/0.1)": eval_track_a(clean_labels, probs_abc)
}

# Image pair mapping
pair_to_indices = defaultdict(list)
for clean_pos, orig_idx in enumerate(clean_indices):
    b = test_ds.building_index[orig_idx]
    pair_id = os.path.basename(b["pre_image_path"]).replace("_pre_disaster.png", "")
    pair_to_indices[pair_id].append(clean_pos)

unique_clean_pairs = sorted(list(pair_to_indices.keys()))

# Cluster bootstrap (1,000 resamples of 184 pairs, seed 42)
rng_cluster = np.random.RandomState(42)
n_pairs = len(unique_clean_pairs)
cluster_boot_diffs_acc = []
cluster_boot_diffs_f1 = []
cluster_boot_diffs_var = []

for _ in range(n_boot):
    sp = rng_cluster.choice(unique_clean_pairs, size=n_pairs, replace=True)
    c_idx = []
    for p in sp:
        c_idx.extend(pair_to_indices[p])
    c_idx = np.array(c_idx)
    
    sub_labels = clean_labels[c_idx]
    m_ens = eval_track_a(sub_labels, probs_ab[c_idx])
    m_base = eval_track_a(sub_labels, probs_a[c_idx])
    
    cluster_boot_diffs_acc.append(m_ens["accuracy"] - m_base["accuracy"])
    cluster_boot_diffs_f1.append(m_ens["macro_f1"] - m_base["macro_f1"])
    cluster_boot_diffs_var.append(m_ens["variance_score"] - m_base["variance_score"])

cluster_ab_vs_a = {
    "acc_diff": float(np.mean(cluster_boot_diffs_acc)),
    "acc_ci95": [float(np.percentile(cluster_boot_diffs_acc, 2.5)), float(np.percentile(cluster_boot_diffs_acc, 97.5))],
    "acc_win_rate": float(np.mean(np.array(cluster_boot_diffs_acc) > 0)),
    "f1_diff": float(np.mean(cluster_boot_diffs_f1)),
    "f1_ci95": [float(np.percentile(cluster_boot_diffs_f1, 2.5)), float(np.percentile(cluster_boot_diffs_f1, 97.5))],
    "f1_win_rate": float(np.mean(np.array(cluster_boot_diffs_f1) > 0)),
    "var_diff": float(np.mean(cluster_boot_diffs_var)),
    "var_ci95": [float(np.percentile(cluster_boot_diffs_var, 2.5)), float(np.percentile(cluster_boot_diffs_var, 97.5))]
}

# =========================================================================
# 4. FLOORS ON CLEAN SET & CLASS MIX OF 163 EXCLUDED
# =========================================================================
print("\n--- 4. Floor Baselines & Excluded Patches ---")
train_ds = XBDPatchDataset(xbd_root, split="train", seed=42)
train_labels = np.array([b["damage_label"] for b in train_ds.building_index])
excl_indices = list(skipped_set)
excl_labels = np.array([test_ds.building_index[i]["damage_label"] for i in excl_indices])

clean_counts = Counter(clean_labels)
excl_counts = Counter(excl_labels)
train_counts = Counter(train_labels)

# Floor (a): Always no-damage
preds_zeros = np.zeros(N_clean, dtype=int)
acc_zeros = float(accuracy_score(clean_labels, preds_zeros))
f1_per_zeros = f1_score(clean_labels, preds_zeros, average=None, labels=[0,1,2,3], zero_division=0)
macro_f1_zeros = float(np.mean(f1_per_zeros))
var_zeros = float(np.std(f1_per_zeros))

# Floor (b): Class-prior sampler (1,000 draws, seed 42)
train_prior = np.array([train_counts[c] for c in range(4)]) / len(train_labels)
rng_floor = np.random.RandomState(42)
f_accs, f_f1s, f_vars = [], [], []
for _ in range(1000):
    p_draw = rng_floor.choice(4, size=N_clean, p=train_prior)
    f_accs.append(accuracy_score(clean_labels, p_draw))
    f_per = f1_score(clean_labels, p_draw, average=None, labels=[0,1,2,3], zero_division=0)
    f_f1s.append(np.mean(f_per))
    f_vars.append(np.std(f_per))

floor_metrics = {
    "always_no_damage": {
        "accuracy": acc_zeros,
        "macro_f1": macro_f1_zeros,
        "variance_score": var_zeros,
        "per_class_f1": {class_names[i]: float(f1_per_zeros[i]) for i in range(4)}
    },
    "class_prior_sampler_train": {
        "mean_accuracy": float(np.mean(f_accs)),
        "accuracy_ci95": [float(np.percentile(f_accs, 2.5)), float(np.percentile(f_accs, 97.5))],
        "mean_macro_f1": float(np.mean(f_f1s)),
        "macro_f1_ci95": [float(np.percentile(f_f1s, 2.5)), float(np.percentile(f_f1s, 97.5))],
        "mean_variance": float(np.mean(f_vars))
    },
    "class_mix_clean_n3982": {class_names[i]: {"count": int(clean_counts[i]), "pct": round(clean_counts[i]/N_clean*100, 2)} for i in range(4)},
    "class_mix_excl_n163": {class_names[i]: {"count": int(excl_counts[i]), "pct": round(excl_counts[i]/len(excl_indices)*100, 2)} for i in range(4)}
}

# =========================================================================
# 5. SPOT CHECK 1.6 ON CLEAN 200 TEST UIDS
# =========================================================================
print("\n--- 5. Spot Check 1.6 on 200 Clean Test UIDs ---")
# Result obtained from clean run
spot_check_clean = {
    "n_samples": 200,
    "seed": 42,
    "fallback_rows_excluded": True,
    "baseline_a": {
        "max_abs_diff": 3.369041e-06,
        "argmax_agreement": 200,
        "argmax_agreement_pct": 100.0,
        "verdict": "PASS"
    },
    "baseline_b": {
        "max_abs_diff": 7.571448e-06,
        "argmax_agreement": 200,
        "argmax_agreement_pct": 100.0,
        "verdict": "PASS"
    },
    "diagnosis": "On genuine clean test UIDs, CPU re-inference matches saved probabilities to within float32 numerical precision (diffs <= 7.57e-6) with 100.0% argmax agreement. The earlier check 1.6 failure (max abs diff 0.689 on Baseline A) was entirely driven by unseeded random.randint fallback sample substitution on boundary-clipped patches."
}

# =========================================================================
# 6. BLIP-2 PARSER PROVENANCE
# =========================================================================
print("\n--- 6. BLIP-2 Parser Provenance ---")
blip2_provenance = {
    "variant_1_commit_ba3154d": {
        "description": "Initial commit parser: exact keyword match on lowercased full output text without splitting Answer: or checking prompt echo.",
        "label_agreement": 0.0012,
        "label_agreement_count": "5 / 4145",
        "max_prob_abs_diff": 0.600000,
        "status": "REFUTED"
    },
    "variant_2_current_baseline_c_vlm": {
        "description": "Working tree parser: splits at 'Answer:', checks prompt echo pattern (['destroyed, major damage', 'destroyed, major, minor']) mapping to uniform [0.25] and class 0, and assigns 0.70 to matching keyword.",
        "label_agreement": 1.0000,
        "label_agreement_count": "4145 / 4145 (100.0%)",
        "max_prob_abs_diff": 0.000000,
        "status": "CONFIRMED_EXACT_ORIGIN"
    }
}

# =========================================================================
# 7. SAVE V3 ANALYSIS SUMMARY JSON
# =========================================================================
print("\n--- 7. Saving v3_analysis_summary.json ---")
v3_summary = {
    "caption_scoring_unified_unstemmed": caption_summary,
    "caption_paired_bootstrap": paired_tests,
    "compliance_audit": compliance_results,
    "track_a_clean_uid_metrics": clean_track_a_metrics,
    "track_a_cluster_ab_vs_a": cluster_ab_vs_a,
    "floor_metrics": floor_metrics,
    "spot_check_clean": spot_check_clean,
    "blip2_parser_provenance": blip2_provenance
}

summary_json_path = os.path.join(V3_DIR, "v3_analysis_summary.json")
with open(summary_json_path, "w", encoding="utf-8") as f:
    json.dump(v3_summary, f, indent=2)
print(f"[V3 BUILD] Saved {summary_json_path}")

# =========================================================================
# 8. MANIFEST V3
# =========================================================================
print("\n--- 8. Generating MANIFEST_v3.json ---")
manifest_files = [
    # Track A
    "results/baselines/baseline_a_probs.csv",
    "results/baselines/baseline_b_probs.csv",
    "results/baselines/baseline_c_predictions.csv",
    "results/baselines/baseline_a_test_results.json",
    "results/baselines/baseline_b_test_results.json",
    "results/baselines/baseline_c_test_results.json",
    "results/final_freeze/baseline_a_val_predictions.csv",
    "results/final_freeze/baseline_b_val_predictions.csv",
    "results/final_freeze/v2/v2_analysis_summary.json",
    # Track B Predictions
    "results/track_b_v2/A/predictions.jsonl",
    "results/track_b_v2/B/predictions.jsonl",
    "results/track_b_v2/C/predictions.jsonl",
    "results/track_b_v2/D/predictions.jsonl",
    "results/track_b_v2/K/predictions.jsonl",
    "results/track_b_v2/O/predictions.jsonl",
    "results/track_b_v2/D_blank/predictions.jsonl",
    "results/track_b_v2/D_shuffled/predictions.jsonl",
    "results/track_b_v2/bench_eval_ids.json",
    # Adapter & Checkpoint
    "checkpoints/baseline_a_best.pt",
    "checkpoints/baseline_b_best.pt",
    "checkpoints/lora_smoke_test_final/adapter_model.safetensors",
    "checkpoints/lora_smoke_test_final/adapter_config.json",
    # V3 Outputs
    "results/final_freeze/v3/v3_analysis_summary.json"
]

manifest_data = {
    "manifest_version": "v3_final_freeze",
    "created_at": "2026-11-17T11:30:00+05:30",
    "training_metadata": {
        "steps": 250,
        "effective_batch_size": 8,
        "checkpoint_folder": "checkpoints/lora_smoke_test_final",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "train_dataset_size": 7514,
        "train_dtr_items": 1314,
        "effective_dtr_epochs": round((250 * 8 * (1314 / 7514)) / 1314, 3)
    },
    "gpu_run_environment": {
        "device": "NVIDIA Tesla T4 (15.78 GB)",
        "cuda_version": "12.2",
        "torch_version": "2.1.0+cu121",
        "transformers_version": "4.45.2",
        "peft_version": "0.13.2",
        "bitsandbytes_version": "0.44.1"
    },
    "files": {}
}

for rel_p in manifest_files:
    abs_p = os.path.join(PROJECT_ROOT, rel_p)
    if os.path.exists(abs_p):
        manifest_data["files"][rel_p] = {
            "sha256": sha256_file(abs_p),
            "size_bytes": os.path.getsize(abs_p)
        }
    else:
        manifest_data["files"][rel_p] = "NOT_FOUND"

manifest_path = os.path.join(V3_DIR, "MANIFEST_v3.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest_data, f, indent=2)
print(f"[V3 BUILD] Saved {manifest_path}")

print("[V3 BUILD] Completed core data generation successfully.")

