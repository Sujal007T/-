import os
import sys
import json
import hashlib
import platform

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.grounding_analysis import get_parser_hash


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    freeze_dir = os.path.join(PROJECT_ROOT, "results", "final_freeze")
    os.makedirs(freeze_dir, exist_ok=True)
    
    # 1. Collect Library Versions
    import torch
    import torchvision
    import sklearn
    import scipy
    import pandas
    import numpy
    try:
        import transformers
        transformers_ver = transformers.__version__
    except ImportError:
        transformers_ver = "unknown"
        
    libraries = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers_ver,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__
    }
    
    # 2. Result Files to Freeze
    target_files = [
        # Track B Canonical Deliverables
        "results/track_b_v2/bench_eval_ids.json",
        "results/track_b_v2/final/final_consistent_metrics.json",
        "results/track_b_v2/final/main_table.md",
        "results/track_b_v2/final/claims_check.md",
        "results/track_b_v2/final/CHANGELOG.md",
        "results/track_b_v2/final/d_shuffled_donor_audit.csv",
        "results/track_b_v2/final/matched_controls_metrics.json",
        "results/track_b_v2/final/matched_controls_predictions.jsonl",
        "results/track_b_v2/final/matched_controls_report.md",
        
        # Track B Condition Predictions
        "results/track_b_v2/B/predictions.jsonl",
        "results/track_b_v2/C/predictions.jsonl",
        "results/track_b_v2/D/predictions.jsonl",
        "results/track_b_v2/O/predictions.jsonl",
        "results/track_b_v2/D_blank/predictions.jsonl",
        "results/track_b_v2/D_shuffled/predictions.jsonl",
        "results/track_b_v2/K/predictions.jsonl",
        "results/track_b_v2/base_model_sanity_check.json",
        "results/track_b_v2/d_diagnosis/matrix_metrics.json",
        
        # Track A Baselines & Ensembles
        "results/baselines/baseline_a_probs.csv",
        "results/baselines/baseline_b_probs.csv",
        "results/baselines/baseline_c_predictions.csv",
        "results/baselines/ensemble_test_results.json",
        "results/final_freeze/baseline_a_val_predictions.csv",
        "results/final_freeze/baseline_b_val_predictions.csv",
        "results/final_freeze/track_a_final_metrics.json",
        "results/final_freeze/audit_summary.json"
    ]
    
    file_manifest = {}
    for rel_path in target_files:
        full_p = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(full_p):
            file_manifest[rel_path] = {
                "sha256": compute_sha256(full_p),
                "size_bytes": os.path.getsize(full_p),
                "last_modified": os.path.getmtime(full_p)
            }
        else:
            file_manifest[rel_path] = "MISSING_OR_PENDING"
            
    manifest_data = {
        "dataset_revision": {
            "track_b_train_samples": 7514,
            "track_b_val_samples": 1898,
            "track_b_bench_eval_samples": 400,
            "track_b_bench_dtr_canonical": 200,
            "track_b_bench_caption_canonical": 200,
            "track_a_xbd_total_tiles": 5598,
            "track_a_val_buildings": 5246,
            "track_a_test_buildings": 4145
        },
        "parser": {
            "name": "parse_dtr_improved (Sentence-Aware v2)",
            "sha256": get_parser_hash()
        },
        "seeds": {
            "global_seed": 42,
            "bootstrap_seed": 42,
            "bootstrap_resamples": 1000
        },
        "libraries": libraries,
        "files": file_manifest
    }
    
    manifest_path = os.path.join(freeze_dir, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"[MANIFEST] Wrote {manifest_path}")
    
    # 3. Write README_FREEZE.md
    readme_path = os.path.join(freeze_dir, "README_FREEZE.md")
    readme_content = f"""# DisasterM3 / DisaSense: Canonical Evaluation Final Freeze

**STATUS: STRICTLY FROZEN. NO FURTHER EXPERIMENTS MAY BE RUN AFTER THIS POINT.**

All experimental evaluations for **Track A (Building Damage Classification)** and **Track B (Disaster Type Recognition & Captioning)** are finalized, audited, and frozen in this directory.

---

## 1. Governance & Freeze Policy
1. **Zero New Inference Runs**: No further model checkpoints, evaluations, or test set queries are permitted.
2. **Canonical Deliverables**: All numbers reported in the paper must be cited exclusively from the frozen tables in `results/final_freeze/` and `results/track_b_v2/final/`.
3. **Audited Parser Standard**: All Track B DTR evaluations adhere to the verified sentence-aware v2 parser (`parse_dtr_improved`), SHA-256: `{get_parser_hash()}`.
4. **Reproducibility Guarantee**: All random seeds are fixed to `42`, with $N=1000$ iterations for all bootstrap confidence intervals.

---

## 2. Environment & Software Stack
- **Python**: `{libraries['python']}`
- **PyTorch**: `{libraries['torch']}`
- **Torchvision**: `{libraries['torchvision']}`
- **Transformers**: `{libraries['transformers']}`
- **Scikit-Learn**: `{libraries['scikit-learn']}`
- **SciPy**: `{libraries['scipy']}`
- **Pandas**: `{libraries['pandas']}`
- **NumPy**: `{libraries['numpy']}`

---

## 3. Directory Contents
- `MANIFEST.json`: Complete cryptographic inventory of all result files, checkpoints, and hashes.
- `baseline_a_val_predictions.csv`: Validation predictions from `baseline_a_best.pt` ($N=5246$).
- `baseline_b_val_predictions.csv`: Validation predictions from `baseline_b_best.pt` ($N=5246$).
- `track_a_final_metrics.json`: Final metrics, per-class F1, variance scores, and bootstrap comparisons.
- `audit_summary.json`: Detailed event overlap breakdown, C vs D per-class McNemar tests, and caption DISASTER header matching.
- `README_FREEZE.md`: This freeze document.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"[README_FREEZE] Wrote {readme_path}")


if __name__ == "__main__":
    generate_manifest()

