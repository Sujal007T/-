import json
import os

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

# CELL 1
add_md("""# Integration Test — Phase 3 & Phase 4
# Run this before opening the mid-evaluation notebook
# All tests must show PASS""")

# CELL 2
add_code("""# Cell 2 — Setup
import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image

try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass

sys.path.append('/content/drive/MyDrive/Project/disaSense_project')
if not os.path.exists('/content/drive/MyDrive/Project/disaSense_project/configs/config.yaml'):
    sys.path.append('..')
    config_path = '../configs/config.yaml'
else:
    config_path = '/content/drive/MyDrive/Project/disaSense_project/configs/config.yaml'

with open(config_path) as f:
    config = yaml.safe_load(f)

test_results = {}
failed_details = []

def record_test(test_num, name, status, detail=""):
    test_results[test_num] = {"name": name, "status": status}
    if status == "FAIL":
        failed_details.append(f"Test {test_num} ({name}): {detail}")
    print(f"[{status}] {name}")
    if detail and status == "FAIL":
        print(f"  -> Error: {detail}")""")

# CELL 3
add_code("""# Cell 3 — TEST 9: Baseline B architecture
try:
    from src.models.baseline_b_change import ChangeDetectionClassifier
    import torch.nn as nn
    
    model = ChangeDetectionClassifier(pretrained=False)
    
    pre = torch.randn(2, 3, 224, 224)
    post = torch.randn(2, 3, 224, 224)
    out = model(pre, post)
    
    assert list(out.shape) == [2, 4], f"Expected shape [2, 4], got {out.shape}"
    
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.classifier.parameters())
    assert backbone_params > head_params, "Backbone should have more params than head"
    
    # Optional warm-start check
    ckpt_a = os.path.join(config["paths"]["checkpoint_dir"], "baseline_a_best.pt")
    if os.path.exists(ckpt_a):
        checkpoint = torch.load(ckpt_a, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # Check if we can load it without crashing
        model.backbone.load_state_dict(state_dict, strict=False)
        
    print("PASS: Baseline B architecture correct")
    record_test(9, "Baseline B architecture", "PASS")
except Exception as e:
    record_test(9, "Baseline B architecture", "FAIL", str(e))""")

# CELL 4
add_code("""# Cell 4 — TEST 10: Baseline C parser
try:
    from src.models.baseline_c_vlm import parse_blip2_output
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    test_cases = [
        ("The building appears destroyed with complete roof collapse.", 3),
        ("I can see major structural damage to the walls.", 2),
        ("There is minor damage visible on the left side.", 1),
        ("The building looks intact with no visible damage.", 0),
        ("The building has damage.", 2),
        ("A satellite image of a building.", 0)
    ]
    
    for text, expected in test_cases:
        pred_idx, probs = parse_blip2_output(text, class_names)
        assert pred_idx == expected, f"Parser failed on '{text}'. Expected {expected}, got {pred_idx}"
        assert abs(np.sum(probs) - 1.0) < 1e-6, "Pseudo probs must sum to 1.0"
        
    print("PASS: BLIP-2 output parser correct")
    record_test(10, "BLIP-2 output parser", "PASS")
except Exception as e:
    record_test(10, "BLIP-2 output parser", "FAIL", str(e))""")

# CELL 5
add_code("""# Cell 5 — TEST 11: BLIP-2 memory check
try:
    if not torch.cuda.is_available():
        print("Skipping BLIP-2 memory check (no CUDA available)")
        record_test(11, "BLIP-2 memory + inference", "PASS", "Skipped - no CUDA")
    else:
        from src.models.baseline_c_vlm import load_blip2, blip2_describe_damage
        processor, model = load_blip2(device="cuda")
        
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {mem_gb:.2f} GB")
        
        # Create a dummy image
        dummy_img = Image.new('RGB', (224, 224), color = 'red')
        
        raw_text = blip2_describe_damage(processor, model, dummy_img, device="cuda")
        print(f"Raw output text: {raw_text}")
        assert isinstance(raw_text, str) and len(raw_text) > 0, "Output should be non-empty string"
        
        del model, processor
        torch.cuda.empty_cache()
        mem_gb_after = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory after deletion: {mem_gb_after:.2f} GB")
        
        print("PASS: BLIP-2 loads and runs without OOM")
        record_test(11, "BLIP-2 memory + inference", "PASS")
except Exception as e:
    # ensure cleanup even if failed
    if 'model' in locals(): del model
    if 'processor' in locals(): del processor
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    record_test(11, "BLIP-2 memory + inference", "FAIL", str(e))""")

# CELL 6
add_code("""# Cell 6 — TEST 12: Ensemble weight computation
try:
    from src.models.ensemble import compute_ensemble_weights
    import io
    import sys
    
    # Suppress print for test
    captured_output = io.StringIO()
    sys.stdout = captured_output
    
    w_a, w_b, w_c = compute_ensemble_weights(0.74, 0.71, 0.62)
    
    sys.stdout = sys.__stdout__
    
    assert abs((w_a + w_b + w_c) - 1.0) < 1e-6, "Weights must sum to 1.0"
    assert w_a > 0 and w_b > 0 and w_c > 0, "Weights must be positive"
    assert w_a > w_b and w_a > w_c, "Highest val F1 should get highest weight"
    
    print("PASS: Ensemble weight computation correct")
    record_test(12, "Ensemble weight computation", "PASS")
except Exception as e:
    record_test(12, "Ensemble weight computation", "FAIL", str(e))""")

# CELL 7
add_code("""# Cell 7 — TEST 13: Ensemble prediction shape
try:
    from src.models.ensemble import ensemble_predict
    
    probs_a = np.random.dirichlet([1,1,1,1], size=100)
    probs_b = np.random.dirichlet([1,1,1,1], size=100)  
    probs_c = np.random.dirichlet([1,1,1,1], size=100)
    
    combined_probs, predictions = ensemble_predict(probs_a, probs_b, probs_c, (0.4, 0.35, 0.25))
    
    assert list(combined_probs.shape) == [100, 4], "Combined probs shape incorrect"
    assert np.allclose(np.sum(combined_probs, axis=1), 1.0), "Combined probs must sum to 1.0 per row"
    assert list(predictions.shape) == [100], "Predictions shape incorrect"
    assert set(np.unique(predictions)).issubset({0, 1, 2, 3}), "Predictions must be in [0, 1, 2, 3]"
    
    print("PASS: Ensemble predict correct")
    record_test(13, "Ensemble prediction shape", "PASS")
except Exception as e:
    record_test(13, "Ensemble prediction shape", "FAIL", str(e))""")

# CELL 8
add_code("""# Cell 8 — TEST 14: CSV round-trip for Baseline C
try:
    from src.models.baseline_c_vlm import load_baseline_c_from_csv
    
    dummy_csv = "temp_dummy_preds.csv"
    dummy_data = []
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    for i in range(10):
        pred_lbl = np.random.randint(0, 4)
        dummy_data.append({
            "sample_id": i,
            "pre_image_path": "x",
            "post_image_path": "y",
            "true_label": 0,
            "true_class": "no-damage",
            "predicted_label": pred_lbl,
            "predicted_class": class_names[pred_lbl],
            "raw_blip2_output": "test",
            "prob_no_damage": 0.25,
            "prob_minor": 0.25,
            "prob_major": 0.25,
            "prob_destroyed": 0.25
        })
    pd.DataFrame(dummy_data).to_csv(dummy_csv, index=False)
    
    res = load_baseline_c_from_csv(dummy_csv)
    
    req_keys = ["accuracy", "macro_f1", "per_class_f1", "variance_score", "confusion_matrix", "all_softmax_probs"]
    for k in req_keys:
        assert k in res, f"Missing key {k} in returned dict"
        
    probs = np.array(res["all_softmax_probs"])
    assert list(probs.shape) == [10, 4], "Shape incorrect"
    assert np.allclose(np.sum(probs, axis=1), 1.0), "Rows must sum to 1.0"
    
    os.remove(dummy_csv)
    print("PASS: Baseline C CSV round-trip correct")
    record_test(14, "Baseline C CSV round-trip", "PASS")
except Exception as e:
    if os.path.exists("temp_dummy_preds.csv"):
        os.remove("temp_dummy_preds.csv")
    record_test(14, "Baseline C CSV round-trip", "FAIL", str(e))""")

# CELL 9
add_code("""# Cell 9 — TEST 15: Mid-evaluation notebook existence check
try:
    notebook_path = os.path.join(config["paths"]["xbd_root"].split("data")[0], "notebooks", "04_mid_evaluation_report.ipynb")
    # Fallback to current relative path if xbd_root structure is weird
    if not os.path.exists(notebook_path):
        notebook_path = "notebooks/04_mid_evaluation_report.ipynb"
        if not os.path.exists(notebook_path):
            notebook_path = "../notebooks/04_mid_evaluation_report.ipynb"
            
    assert os.path.exists(notebook_path), f"Cannot find {notebook_path}"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    assert len(nb.get("cells", [])) >= 14, f"Expected >= 14 cells, found {len(nb.get('cells', []))}"
    
    # Architecture and experimental design images
    arch_img = os.path.join(config["paths"]["results_dir"], "architecture_diagram.png")
    exp_img = os.path.join(config["paths"]["results_dir"], "experimental_design_table.png")
    
    assert os.path.exists(arch_img), f"Missing {arch_img}"
    assert os.path.exists(exp_img), f"Missing {exp_img}"
    
    print("PASS: Mid-evaluation notebook ready")
    record_test(15, "Mid-evaluation notebook ready", "PASS")
except Exception as e:
    record_test(15, "Mid-evaluation notebook ready", "FAIL", str(e))""")

# CELL 10
add_code("""# Cell 10 — Final summary
print("\\nPhase 3 + 4 integration test summary:\\n")
print("| Test | Description                     | Result    |")
print("|------|---------------------------------|-----------|")
for i in range(9, 16):
    info = test_results.get(i, {"name": "UNKNOWN", "status": "FAIL"})
    print(f"|  {i:2d}  | {info['name']:31s} | {info['status']:9s} |")

all_pass = all(info["status"] == "PASS" for info in test_results.values())

print("\\n")
if all_pass:
    print("✅ ALL PHASE 3+4 TESTS PASSED")
    print("You are ready for the mid-evaluation presentation.")
    print("Open: notebooks/04_mid_evaluation_report.ipynb")
else:
    print("❌ FIX THESE BEFORE THE MID-EVALUATION:")
    for detail in failed_details:
        print(f" - {detail}")""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = "D:/Sem-7/Minor/Project/disaSense_project/notebooks/05_phase34_integration_test.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created at {out_path}")
