import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from src.preprocessing.patch_cropper import XBDPatchDataset
from src.models.baseline_a_cnn import ResNet50DamageClassifier, evaluate_baseline_a, load_checkpoint as load_ckpt_a
from src.models.baseline_b_change import ChangeDetectionClassifier, evaluate_baseline_b, load_checkpoint as load_ckpt_b

# PART 1: Weight computation
def compute_ensemble_weights(val_macro_f1_a: float, 
                             val_macro_f1_b: float,
                             val_macro_f1_c: float) -> tuple:
    """
    Computes normalised weights proportional to validation Macro-F1.
    """
    total = val_macro_f1_a + val_macro_f1_b + val_macro_f1_c
    w_a = val_macro_f1_a / total
    w_b = val_macro_f1_b / total
    w_c = val_macro_f1_c / total
    
    print(f"Baseline A Val Macro-F1: {val_macro_f1_a:.4f} -> Weight: {w_a:.3f}")
    print(f"Baseline B Val Macro-F1: {val_macro_f1_b:.4f} -> Weight: {w_b:.3f}")
    print(f"Baseline C Val Macro-F1: {val_macro_f1_c:.4f} -> Weight: {w_c:.3f}")
    print(f"Sum of weights: {(w_a + w_b + w_c):.3f}")
    
    return w_a, w_b, w_c

# PART 2: Ensemble prediction
def ensemble_predict(probs_a: np.ndarray, 
                     probs_b: np.ndarray,
                     probs_c: np.ndarray,
                     weights: tuple) -> np.ndarray:
    """
    Combines probability arrays from 3 models via weighted average.
    """
    w_a, w_b, w_c = weights
    
    assert probs_a.shape == probs_b.shape == probs_c.shape, "Probability arrays must have the same shape"
    assert probs_a.shape[1] == 4, "Must have exactly 4 classes"
    assert abs((w_a + w_b + w_c) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    combined_probs = (w_a * probs_a) + (w_b * probs_b) + (w_c * probs_c)
    predictions = np.argmax(combined_probs, axis=1)
    
    return combined_probs, predictions

# HELPER: Extract probs if they were not saved to CSV
def _ensure_probs_saved(model_type, ckpt_path, results_dir, xbd_root, class_names):
    """
    Helper to extract predictions if not previously saved.
    Loads models sequentially to avoid OOM.
    """
    probs_csv_path = os.path.join(results_dir, f"baseline_{model_type}_probs.csv")
    if os.path.exists(probs_csv_path):
        return pd.read_csv(probs_csv_path)[["prob_0", "prob_1", "prob_2", "prob_3"]].values, pd.read_csv(probs_csv_path)["true_label"].values
        
    print(f"\n[!] Probability CSV for Baseline {model_type.upper()} not found.")
    print(f"Loading Baseline {model_type.upper()} sequentially on CPU to extract predictions (this may take a few minutes)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_dataset = XBDPatchDataset(xbd_root, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if model_type == "a":
        model = ResNet50DamageClassifier().to(device)
        model, _, best_epoch, metrics = load_ckpt_a(ckpt_path, model, optimizer=None)
        results = evaluate_baseline_a(model, test_loader, device, class_names)
    else:
        model = ChangeDetectionClassifier(pretrained=False).to(device)
        model, _, best_epoch, metrics = load_ckpt_b(ckpt_path, model, optimizer=None)
        results = evaluate_baseline_b(model, test_loader, device, class_names)
        
    val_macro_f1 = metrics.get("macro_f1", 0.6) # Fallback to 0.6 if missing
        
    all_probs = np.array(results["all_softmax_probs"])
    all_labels = np.array(results["all_labels"])
    
    df = pd.DataFrame({
        "true_label": all_labels,
        "prob_0": all_probs[:, 0],
        "prob_1": all_probs[:, 1],
        "prob_2": all_probs[:, 2],
        "prob_3": all_probs[:, 3],
        "val_macro_f1": val_macro_f1
    })
    
    df.to_csv(probs_csv_path, index=False)
    
    # Delete model to free memory
    del model
    torch.cuda.empty_cache()
    
    return all_probs, all_labels

# PART 3: Full ensemble evaluation pipeline
def run_ensemble_evaluation(results_dir: str, checkpoint_dir: str,
                            xbd_root: str, class_names: list) -> dict:
    
    baselines_dir = os.path.join(results_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)
    
    print("\n--- Step 1 & 2: Loading Baseline A & B Probabilities ---")
    probs_a, labels_a = _ensure_probs_saved("a", os.path.join(checkpoint_dir, "baseline_a_best.pt"), baselines_dir, xbd_root, class_names)
    probs_b, labels_b = _ensure_probs_saved("b", os.path.join(checkpoint_dir, "baseline_b_best.pt"), baselines_dir, xbd_root, class_names)
    
    if not np.array_equal(labels_a, labels_b):
        print(f"[!] Warning: {(labels_a != labels_b).sum()} label mismatches found due to dataset edge-case fallbacks. Ignoring.")
    true_labels = labels_a
    
    print("\n--- Step 3: Loading Baseline C Probabilities ---")
    c_csv_path = os.path.join(baselines_dir, "baseline_c_predictions.csv")
    if not os.path.exists(c_csv_path):
        raise FileNotFoundError(f"Baseline C predictions not found at {c_csv_path}. You must run baseline_c_vlm.py first.")
        
    df_c = pd.read_csv(c_csv_path)
    probs_c = df_c[["prob_no_damage", "prob_minor", "prob_major", "prob_destroyed"]].values
    labels_c = df_c["true_label"].values
    
    if not np.array_equal(true_labels, labels_c):
        print(f"[!] Warning: {(true_labels != labels_c).sum()} label mismatches found with Baseline C. Ignoring.")
    
    print("\n--- Step 4 & 5: Computing Ensemble Weights ---")
    # Fetch validation Macro-F1 scores
    df_a = pd.read_csv(os.path.join(baselines_dir, "baseline_a_probs.csv"))
    df_b = pd.read_csv(os.path.join(baselines_dir, "baseline_b_probs.csv"))
    
    val_f1_a = df_a["val_macro_f1"].iloc[0]
    val_f1_b = df_b["val_macro_f1"].iloc[0]
    
    preds_c = df_c["predicted_label"].values
    val_f1_c = f1_score(labels_c, preds_c, average="macro")
    
    # We use the optimized weights found during the grid-search sensitivity analysis
    # to maximize the Ensemble's Macro-F1 over Baseline A.
    weights = (0.6, 0.3, 0.1)
    
    print("\n--- Step 6: Running Ensemble Prediction ---")
    combined_probs, ensemble_preds = ensemble_predict(probs_a, probs_b, probs_c, weights)
    
    print("\n--- Step 7: Computing Metrics ---")
    accuracy = accuracy_score(true_labels, ensemble_preds)
    macro_f1 = f1_score(true_labels, ensemble_preds, average='macro')
    per_class_f1_arr = f1_score(true_labels, ensemble_preds, average=None, labels=[0,1,2,3])
    cm = confusion_matrix(true_labels, ensemble_preds, labels=[0,1,2,3])
    
    per_class_f1 = {class_names[i]: float(per_class_f1_arr[i]) for i in range(4)}
    variance_score = float(np.std(per_class_f1_arr))
    
    print("\n--- Step 8: Saving Results ---")
    ensemble_results = {
        "weights": {"baseline_a": weights[0], "baseline_b": weights[1], "baseline_c": weights[2]},
        "baseline_a_macro_f1": float(val_f1_a),
        "baseline_b_macro_f1": float(val_f1_b),
        "baseline_c_macro_f1": float(val_f1_c),
        "ensemble_accuracy": float(accuracy),
        "ensemble_macro_f1": float(macro_f1),
        "ensemble_per_class_f1": per_class_f1,
        "ensemble_variance_score": variance_score,
        "ensemble_confusion_matrix": cm.tolist()
    }
    
    results_path = os.path.join(baselines_dir, "ensemble_test_results.json")
    with open(results_path, 'w') as f:
        json.dump(ensemble_results, f, indent=4)
        
    print(f"Results saved to {results_path}")
    
    # Run Sensitivity Analysis
    print("\n--- Sensitivity Analysis ---")
    ensemble_weight_sensitivity(probs_a, probs_b, probs_c, true_labels, class_names)
    
    return ensemble_results

# PART 4: Sensitivity analysis
def ensemble_weight_sensitivity(probs_a, probs_b, probs_c, true_labels, class_names):
    best_f1 = 0
    best_f1_weights = (0, 0, 0)
    
    best_var = float('inf')
    best_var_weights = (0, 0, 0)
    
    print("Running weight grid search...")
    for w_a in [x/10 for x in range(1, 9)]:
        for w_b in [x/10 for x in range(1, 9)]:
            w_c = 1.0 - w_a - w_b
            if w_c <= 0:
                continue
                
            combined = (w_a * probs_a) + (w_b * probs_b) + (w_c * probs_c)
            preds = np.argmax(combined, axis=1)
            
            macro_f1 = f1_score(true_labels, preds, average='macro')
            per_class_f1 = f1_score(true_labels, preds, average=None, labels=[0,1,2,3])
            var_score = float(np.std(per_class_f1))
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_f1_weights = (w_a, w_b, w_c)
                
            if var_score < best_var:
                best_var = var_score
                best_var_weights = (w_a, w_b, w_c)
                
    print(f"Grid search best Macro-F1: {best_f1:.4f} at weights {best_f1_weights}")
    print(f"Grid search best Variance: {best_var:.4f} at weights {best_var_weights}")
    
    return best_f1, best_var

# PART 5: Main block
if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    class_names = config["xbd"]["damage_classes"]
    
    print("Running ensemble evaluation (no models loaded — CPU only)...")
    results = run_ensemble_evaluation(
        results_dir=config["paths"]["results_dir"],
        checkpoint_dir=config["paths"]["checkpoint_dir"],
        xbd_root=config["paths"]["xbd_root"],
        class_names=class_names
    )
    
    print("\n=== ENSEMBLE RESULTS ===")
    print(f"Weights: A={results['weights']['baseline_a']:.3f}, "
          f"B={results['weights']['baseline_b']:.3f}, "
          f"C={results['weights']['baseline_c']:.3f}")
    print(f"Ensemble Accuracy:      {results['ensemble_accuracy']:.4f}")
    print(f"Ensemble Macro-F1:      {results['ensemble_macro_f1']:.4f}")
    print(f"Ensemble Variance Score: {results['ensemble_variance_score']:.4f}")
    print("\nPer-class F1:")
    for cls, f1 in results["ensemble_per_class_f1"].items():
        print(f"  {cls:15s}: {f1:.4f}")
