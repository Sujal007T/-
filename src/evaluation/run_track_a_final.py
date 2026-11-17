import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.patch_cropper import XBDPatchDataset, get_default_transforms
from src.models.baseline_a_cnn import ResNet50DamageClassifier, load_checkpoint as load_ckpt_a
from src.models.baseline_b_change import ChangeDetectionClassifier, load_checkpoint as load_ckpt_b


def bootstrap_paired_diff(a_correct: np.ndarray, b_correct: np.ndarray, n_boot: int = 1000, seed: int = 42):
    assert len(a_correct) == len(b_correct)
    n = len(a_correct)
    diffs = a_correct - b_correct
    mean_diff = float(np.mean(diffs))
    
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means.append(float(np.mean(diffs[idx])))
    
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_diff, (ci_low, ci_high)


def compute_metrics(true_labels: np.ndarray, preds: np.ndarray, class_names: list):
    acc = float(accuracy_score(true_labels, preds))
    macro_f1 = float(f1_score(true_labels, preds, average='macro', zero_division=0))
    per_class_f1_arr = f1_score(true_labels, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    per_class_f1 = {class_names[i]: float(per_class_f1_arr[i]) for i in range(4)}
    variance_score = float(np.std(per_class_f1_arr))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "variance_score": variance_score,
        "correct_array": (preds == true_labels).astype(int)
    }


def main():
    print("=" * 80)
    print("DISASTERM3 TRACK A FINAL EVALUATION & VALIDATION ENSEMBLE TUNING")
    print("=" * 80)
    
    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cpu")
    print(f"Using device: {device} with {torch.get_num_threads()} CPU threads.")
    
    with open(os.path.join(PROJECT_ROOT, "configs", "config.yaml")) as f:
        config = yaml.safe_load(f)
    class_names = config["xbd"]["damage_classes"]
    
    freeze_dir = os.path.join(PROJECT_ROOT, "results", "final_freeze")
    os.makedirs(freeze_dir, exist_ok=True)
    checkpoints_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    baselines_dir = os.path.join(PROJECT_ROOT, "results", "baselines")
    xbd_root = os.path.join(PROJECT_ROOT, "data", "raw", "xbd", "train")
    
    # -------------------------------------------------------------
    # STEP 1: VALIDATION INFERENCE (ITEM 3A)
    # -------------------------------------------------------------
    print("\n[STEP 1] Initializing Validation Dataset...")
    val_ds = XBDPatchDataset(xbd_root, split="val")
    n_val = len(val_ds.building_index)
    print(f"Validation dataset has {n_val} buildings across {len(val_ds.image_pairs)} image pairs.")
    
    print("[STEP 1] Preloading validation image pairs into RAM cache...")
    t0 = time.time()
    img_cache = {}
    for p in val_ds.image_pairs:
        pre_p, post_p = p[0], p[1]
        if pre_p not in img_cache:
            img_cache[pre_p] = Image.open(pre_p).convert("RGB")
        if post_p not in img_cache:
            img_cache[post_p] = Image.open(post_p).convert("RGB")
    print(f"Preloaded {len(img_cache)} unique full satellite images in {time.time() - t0:.2f} s.")
    
    # Crop patches deterministically without random fallback
    print("[STEP 1] Pre-extracting all validation building patches into memory tensors...")
    t1 = time.time()
    transform = get_default_transforms(image_size=224, split="val")
    alpha = 0.8
    
    val_labels = []
    val_pre_patches = []
    val_post_patches = []
    
    for b in val_ds.building_index:
        val_labels.append(b["damage_label"])
        bbox = b["bbox"]
        x, y, w, h = bbox
        
        # Pre image
        pre_img = img_cache[b["pre_image_path"]]
        W, H = pre_img.size
        x1 = max(0, int(x - alpha/2 * w))
        y1 = max(0, int(y - alpha/2 * h))
        x2 = min(W, int(x + w + alpha/2 * w))
        y2 = min(H, int(y + h + alpha/2 * h))
        if x2 <= x1: x2 = min(W, x1 + 1)
        if y2 <= y1: y2 = min(H, y1 + 1)
        pre_patch = transform(pre_img.crop((x1, y1, x2, y2)))
        val_pre_patches.append(pre_patch)
        
        # Post image
        post_img = img_cache[b["post_image_path"]]
        W, H = post_img.size
        x1 = max(0, int(x - alpha/2 * w))
        y1 = max(0, int(y - alpha/2 * h))
        x2 = min(W, int(x + w + alpha/2 * w))
        y2 = min(H, int(y + h + alpha/2 * h))
        if x2 <= x1: x2 = min(W, x1 + 1)
        if y2 <= y1: y2 = min(H, y1 + 1)
        post_patch = transform(post_img.crop((x1, y1, x2, y2)))
        val_post_patches.append(post_patch)
        
    val_labels = np.array(val_labels, dtype=int)
    print(f"Extracted {len(val_labels)} building patch pairs in {time.time() - t1:.2f} s.")
    
    # Free full images from memory
    del img_cache
    
    # Run Baseline A on Validation
    print("\n[STEP 1] Running Baseline A inference on validation set...")
    model_a = ResNet50DamageClassifier(pretrained=False).to(device)
    model_a, _, epoch_a, metrics_a = load_ckpt_a(os.path.join(checkpoints_dir, "baseline_a_best.pt"), model_a)
    model_a.eval()
    
    batch_size = 128
    val_probs_a = []
    t_a = time.time()
    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            batch_post = torch.stack(val_post_patches[i:i+batch_size]).to(device)
            logits = model_a(batch_post)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_probs_a.append(probs)
    val_probs_a = np.vstack(val_probs_a)
    print(f"Baseline A val inference completed in {time.time() - t_a:.2f} s.")
    del model_a
    
    # Run Baseline B on Validation
    print("[STEP 1] Running Baseline B inference on validation set...")
    model_b = ChangeDetectionClassifier(pretrained=False).to(device)
    model_b, _, epoch_b, metrics_b = load_ckpt_b(os.path.join(checkpoints_dir, "baseline_b_best.pt"), model_b)
    model_b.eval()
    
    val_probs_b = []
    t_b = time.time()
    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            batch_pre = torch.stack(val_pre_patches[i:i+batch_size]).to(device)
            batch_post = torch.stack(val_post_patches[i:i+batch_size]).to(device)
            logits = model_b(batch_pre, batch_post)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_probs_b.append(probs)
    val_probs_b = np.vstack(val_probs_b)
    print(f"Baseline B val inference completed in {time.time() - t_b:.2f} s.")
    del model_b
    del val_pre_patches
    del val_post_patches
    
    # Save validation predictions to CSV
    df_val_a = pd.DataFrame({
        "sample_id": np.arange(n_val),
        "true_label": val_labels,
        "prob_0": val_probs_a[:, 0],
        "prob_1": val_probs_a[:, 1],
        "prob_2": val_probs_a[:, 2],
        "prob_3": val_probs_a[:, 3]
    })
    val_a_csv = os.path.join(freeze_dir, "baseline_a_val_predictions.csv")
    df_val_a.to_csv(val_a_csv, index=False)
    
    df_val_b = pd.DataFrame({
        "sample_id": np.arange(n_val),
        "true_label": val_labels,
        "prob_0": val_probs_b[:, 0],
        "prob_1": val_probs_b[:, 1],
        "prob_2": val_probs_b[:, 2],
        "prob_3": val_probs_b[:, 3]
    })
    val_b_csv = os.path.join(freeze_dir, "baseline_b_val_predictions.csv")
    df_val_b.to_csv(val_b_csv, index=False)
    
    # Assert all_labels are identical in order across files
    print("\n[ASSERTION] Verifying validation labels order...")
    assert np.array_equal(df_val_a["true_label"].values, df_val_b["true_label"].values), "Validation label mismatch between A and B!"
    print(f"[ASSERTION PASSED] Validation true_labels are 100% identical in order across Baseline A and Baseline B (N={n_val}).")
    
    val_metrics_a = compute_metrics(val_labels, np.argmax(val_probs_a, axis=1), class_names)
    val_metrics_b = compute_metrics(val_labels, np.argmax(val_probs_b, axis=1), class_names)
    print(f"Baseline A Val Accuracy: {val_metrics_a['accuracy']*100:.2f}%, Val Macro-F1: {val_metrics_a['macro_f1']:.4f}")
    print(f"Baseline B Val Accuracy: {val_metrics_b['accuracy']*100:.2f}%, Val Macro-F1: {val_metrics_b['macro_f1']:.4f}")
    
    # -------------------------------------------------------------
    # STEP 2: CHOOSE ENSEMBLE WEIGHTS ON VALIDATION ONLY (ITEM 3B)
    # -------------------------------------------------------------
    print("\n[STEP 2] Running Grid Search on Validation Set for {A, B} Ensemble (step=0.05)...")
    best_val_f1 = -1.0
    best_w_a = 0.5
    grid_results = []
    
    for w_a in np.round(np.arange(0.0, 1.05, 0.05), 2):
        w_b = np.round(1.0 - w_a, 2)
        ens_val_probs = (w_a * val_probs_a) + (w_b * val_probs_b)
        ens_val_preds = np.argmax(ens_val_probs, axis=1)
        score = f1_score(val_labels, ens_val_preds, average="macro", zero_division=0)
        grid_results.append((w_a, w_b, score))
        if score > best_val_f1:
            best_val_f1 = score
            best_w_a = w_a
            
    best_w_b = np.round(1.0 - best_w_a, 2)
    print(f"Optimal Validation Weights: w_A = {best_w_a:.2f}, w_B = {best_w_b:.2f} -> Best Val Macro-F1: {best_val_f1:.4f}")
    
    # -------------------------------------------------------------
    # STEP 3: EVALUATE ON TEST SET (ITEM 3B & 3C)
    # -------------------------------------------------------------
    print("\n[STEP 3] Loading Test Predictions for Baselines A, B, and C...")
    df_test_a = pd.read_csv(os.path.join(baselines_dir, "baseline_a_probs.csv"))
    df_test_b = pd.read_csv(os.path.join(baselines_dir, "baseline_b_probs.csv"))
    df_test_c = pd.read_csv(os.path.join(baselines_dir, "baseline_c_predictions.csv"))
    
    test_probs_a = df_test_a[["prob_0", "prob_1", "prob_2", "prob_3"]].values
    test_probs_b = df_test_b[["prob_0", "prob_1", "prob_2", "prob_3"]].values
    test_probs_c = df_test_c[["prob_no_damage", "prob_minor", "prob_major", "prob_destroyed"]].values
    
    # We will report both:
    # 1. Primary canonical set where all three test files share identical ground truth (N = 4026)
    # 2. Full N = 4145 set (using true_labels = labels_a as in ensemble.py)
    
    for eval_mode in ["canonical_matching", "full_original"]:
        if eval_mode == "canonical_matching":
            match_mask = (df_test_a["true_label"].values == df_test_b["true_label"].values) & (df_test_a["true_label"].values == df_test_c["true_label"].values)
            t_probs_a = test_probs_a[match_mask]
            t_probs_b = test_probs_b[match_mask]
            t_probs_c = test_probs_c[match_mask]
            test_labels = df_test_a["true_label"].values[match_mask]
            print(f"\n================ EVALUATION ON CANONICAL MATCHING TEST SET (N={len(test_labels)}) ================")
            print(f"[ASSERTION] Checking label equality across files for canonical set...")
            assert np.array_equal(test_labels, df_test_b["true_label"].values[match_mask])
            assert np.array_equal(test_labels, df_test_c["true_label"].values[match_mask])
            print("[ASSERTION PASSED] All test labels are 100% identical in order across Baseline A, B, and C.")
        else:
            t_probs_a = test_probs_a
            t_probs_b = test_probs_b
            t_probs_c = test_probs_c
            test_labels = df_test_a["true_label"].values
            print(f"\n================ EVALUATION ON FULL ORIGINAL TEST SET (N={len(test_labels)}) ================")
            
        # Models and ensembles to evaluate
        # 1. Baseline A
        preds_a = np.argmax(t_probs_a, axis=1)
        metrics_a = compute_metrics(test_labels, preds_a, class_names)
        
        # 2. Baseline B
        preds_b = np.argmax(t_probs_b, axis=1)
        metrics_b = compute_metrics(test_labels, preds_b, class_names)
        
        # 3. Baseline C
        preds_c = np.argmax(t_probs_c, axis=1)
        metrics_c = compute_metrics(test_labels, preds_c, class_names)
        
        # 4. {A, B} Ensemble (weights tuned on validation only)
        probs_ens_ab = (best_w_a * t_probs_a) + (best_w_b * t_probs_b)
        preds_ens_ab = np.argmax(probs_ens_ab, axis=1)
        metrics_ens_ab = compute_metrics(test_labels, preds_ens_ab, class_names)
        
        # 5. {A, B, C} Ensemble (existing 0.6 / 0.3 / 0.1 weights tuned on test, exploratory)
        probs_ens_abc = (0.6 * t_probs_a) + (0.3 * t_probs_b) + (0.1 * t_probs_c)
        preds_ens_abc = np.argmax(probs_ens_abc, axis=1)
        metrics_ens_abc = compute_metrics(test_labels, preds_ens_abc, class_names)
        
        # Package models
        models_dict = {
            "Baseline A (ResNet50)": metrics_a,
            "Baseline B (ChangeDetection)": metrics_b,
            "Baseline C (BLIP-2)": metrics_c,
            f"Ensemble {{A,B}} (Val-tuned: {best_w_a:.2f}/{best_w_b:.2f})": metrics_ens_ab,
            "Ensemble {A,B,C} (weights tuned on test, exploratory: 0.6/0.3/0.1)": metrics_ens_abc
        }
        
        # Compute paired bootstrap differences vs Baseline A and vs Baseline B
        results_table = []
        for name, m in models_dict.items():
            corr = m["correct_array"]
            diff_vs_a, ci_vs_a = bootstrap_paired_diff(corr, metrics_a["correct_array"], n_boot=1000, seed=42)
            diff_vs_b, ci_vs_b = bootstrap_paired_diff(corr, metrics_b["correct_array"], n_boot=1000, seed=42)
            
            row = {
                "model": name,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "variance_score": m["variance_score"],
                "per_class_f1": m["per_class_f1"],
                "diff_vs_a": diff_vs_a,
                "ci_vs_a": ci_vs_a,
                "diff_vs_b": diff_vs_b,
                "ci_vs_b": ci_vs_b
            }
            results_table.append(row)
            
        # Print Markdown Table
        print(f"\n### Track A Benchmark Summary Table ({eval_mode}, N={len(test_labels)})")
        print("| Model / Ensemble | Accuracy (%) | Macro-F1 | Variance Score | No-Damage F1 | Minor-Damage F1 | Major-Damage F1 | Destroyed F1 | Diff vs Base A (%) [95% CI] | Diff vs Base B (%) [95% CI] |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for r in results_table:
            pf = r["per_class_f1"]
            mname = r["model"]
            acc_str = f"{r['accuracy']*100:.2f}%"
            f1_str = f"{r['macro_f1']:.4f}"
            var_str = f"{r['variance_score']:.4f}"
            diff_a_str = "---" if "Baseline A" in mname and "Ensemble" not in mname else f"{r['diff_vs_a']*100:+.2f}% [{r['ci_vs_a'][0]*100:+.2f}%, {r['ci_vs_a'][1]*100:+.2f}%]"
            diff_b_str = "---" if "Baseline B" in mname and "Ensemble" not in mname else f"{r['diff_vs_b']*100:+.2f}% [{r['ci_vs_b'][0]*100:+.2f}%, {r['ci_vs_b'][1]*100:+.2f}%]"
            print(f"| **{mname}** | {acc_str} | {f1_str} | {var_str} | {pf['no-damage']:.4f} | {pf['minor-damage']:.4f} | {pf['major-damage']:.4f} | {pf['destroyed']:.4f} | {diff_a_str} | {diff_b_str} |")
            
        if eval_mode == "canonical_matching":
            # Save canonical results to json
            with open(os.path.join(freeze_dir, "track_a_final_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "val_best_weights_ab": {"w_a": best_w_a, "w_b": best_w_b, "val_macro_f1": best_val_f1},
                    "val_metrics_a": {k: v for k, v in val_metrics_a.items() if k != "correct_array"},
                    "val_metrics_b": {k: v for k, v in val_metrics_b.items() if k != "correct_array"},
                    "test_results_canonical": results_table
                }, f, indent=2)

    print("\n[DONE] Track A Final Evaluation completed successfully.")


if __name__ == "__main__":
    main()

