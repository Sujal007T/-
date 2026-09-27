import os
import yaml
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except ImportError:
    pass # Will fail gracefully later if not installed, but avoids crashing immediately

from src.preprocessing.patch_cropper import XBDPatchDataset

# PART 1: BLIP-2 loader
def load_blip2(device="cuda"):
    """
    Loads BLIP-2 (Salesforce/blip2-opt-2.7b) in float16.
    Returns (processor, model) tuple.
    """
    model_id = "Salesforce/blip2-opt-2.7b"
    print(f"Loading {model_id} in float16...")
    try:
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU memory used after loading BLIP-2: {mem_allocated:.2f} GB")
            
        return processor, model
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA OUT OF MEMORY ERROR: BLIP-2 requires ~8GB VRAM.")
            print("Please ensure no other models (Baselines A or B) are loaded in memory.")
        raise e

# PART 2: Single image inference
def blip2_describe_damage(processor, model, post_image: Image.Image, device="cuda") -> str:
    """
    Runs BLIP-2 on a single post-disaster building patch.
    Returns the raw generated text string.
    """
    prompt = ("Question: Describe the structural damage to this building. "
              "Is it: destroyed, major damage, minor damage, or no damage? "
              "Answer:")
    try:
        # Convert to RGB if not already
        if post_image.mode != "RGB":
            post_image = post_image.convert("RGB")
            
        inputs = processor(
            images=post_image, 
            text=prompt, 
            return_tensors="pt"
        ).to(device, torch.float16)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=50, 
                num_beams=4
            )
            
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    except Exception as e:
        print(f"Error during BLIP-2 generation: {e}")
        return "unknown"
    finally:
        torch.cuda.empty_cache()

# PART 3: Output parser
def parse_blip2_output(generated_text: str, class_names: list) -> tuple:
    """
    Converts BLIP-2 free-text output to a damage class label and 
    pseudo-probability distribution.
    """
    text = generated_text.lower()
    predicted_class_idx = -1
    
    # Step 1: Exact keyword match
    if any(k in text for k in ["destroyed", "collapse", "ruin"]):
        predicted_class_idx = 3
    elif any(k in text for k in ["major", "severe", "heavily"]):
        predicted_class_idx = 2
    elif any(k in text for k in ["minor", "partial", "slight"]):
        predicted_class_idx = 1
    elif any(k in text for k in ["no damage", "intact", "undamaged"]):
        predicted_class_idx = 0
        
    # Step 2: Partial match fallback
    if predicted_class_idx == -1:
        if "damage" in text:
            predicted_class_idx = 2 # conservative fallback
        else:
            predicted_class_idx = 0 # no damage keywords at all
            
    # Pseudo-probability construction
    probs = np.array([0.1, 0.1, 0.1, 0.1])
    probs[predicted_class_idx] = 0.7
    probs = probs / np.sum(probs)
    
    return predicted_class_idx, probs

# PART 4: Batch evaluation on xBD test set
def evaluate_baseline_c(xbd_root, processor, model, device,
                         disaster_types=None, results_dir=None,
                         save_predictions_csv=True):
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    # Use dummy transform to get PIL images instead of Tensors
    test_dataset = XBDPatchDataset(
        xbd_root, 
        split="test", 
        disaster_types=disaster_types,
        transform=lambda x: x 
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    csv_rows = []
    parse_failures = 0
    
    print(f"Starting Baseline C evaluation on {len(test_dataset)} samples...")
    for i in tqdm(range(len(test_dataset)), desc="Evaluating BLIP-2"):
        pre_patch, post_patch, label_tensor, idx = test_dataset[i]
        true_label = int(label_tensor.item()) if isinstance(label_tensor, torch.Tensor) else int(label_tensor)
        
        sample_meta = test_dataset.building_index[i]
        
        raw_text = blip2_describe_damage(processor, model, post_patch, device)
        if raw_text == "unknown":
            parse_failures += 1
            
        pred_label, pseudo_probs = parse_blip2_output(raw_text, class_names)
        
        all_preds.append(pred_label)
        all_labels.append(true_label)
        all_probs.append(pseudo_probs)
        
        csv_rows.append({
            "sample_id": i,
            "pre_image_path": sample_meta["pre_image_path"],
            "post_image_path": sample_meta["post_image_path"],
            "true_label": true_label,
            "true_class": class_names[true_label],
            "predicted_label": pred_label,
            "predicted_class": class_names[pred_label],
            "raw_blip2_output": raw_text,
            "prob_no_damage": pseudo_probs[0],
            "prob_minor": pseudo_probs[1],
            "prob_major": pseudo_probs[2],
            "prob_destroyed": pseudo_probs[3]
        })
        
        if (i + 1) % 50 == 0:
            acc = accuracy_score(all_labels, all_preds)
            print(f"Processed {i+1}/{len(test_dataset)} — running accuracy: {acc:.2%}")
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None, labels=[0,1,2,3])
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    
    per_class_f1 = {class_names[i]: float(per_class_f1_arr[i]) for i in range(4)}
    variance_score = float(np.std(per_class_f1_arr))
    failure_rate = parse_failures / len(test_dataset) if len(test_dataset) > 0 else 0
    
    csv_path = None
    if save_predictions_csv and results_dir:
        os.makedirs(os.path.join(results_dir, "baselines"), exist_ok=True)
        csv_path = os.path.join(results_dir, "baselines", "baseline_c_predictions.csv")
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False)
        
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "variance_score": variance_score,
        "confusion_matrix": cm.tolist(),
        "all_predictions": all_preds.tolist(),
        "all_labels": all_labels.tolist(),
        "all_softmax_probs": all_probs.tolist(),
        "parse_failure_rate": failure_rate,
        "predictions_csv_path": csv_path
    }

# PART 5: Load predictions from CSV
def load_baseline_c_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    all_labels = df["true_label"].values
    all_preds = df["predicted_label"].values
    all_probs = df[["prob_no_damage", "prob_minor", "prob_major", "prob_destroyed"]].values
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None, labels=[0,1,2,3])
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    
    per_class_f1 = {class_names[i]: float(per_class_f1_arr[i]) for i in range(4)}
    variance_score = float(np.std(per_class_f1_arr))
    
    parse_failures = (df["raw_blip2_output"] == "unknown").sum()
    failure_rate = parse_failures / len(df) if len(df) > 0 else 0
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "variance_score": variance_score,
        "confusion_matrix": cm.tolist(),
        "all_predictions": all_preds.tolist(),
        "all_labels": all_labels.tolist(),
        "all_softmax_probs": all_probs.tolist(),
        "parse_failure_rate": float(failure_rate),
        "predictions_csv_path": csv_path
    }

# PART 6: Parser self-test
def test_parser():
    class_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    test_cases = [
        ("The building appears destroyed with complete roof collapse.", 3),
        ("I can see major structural damage to the walls.", 2),
        ("There is minor damage visible on the left side.", 1),
        ("The building looks intact with no visible damage.", 0),
        ("The building has damage.", 2),
        ("A satellite image of a building.", 0)
    ]
    
    failures = 0
    for text, expected_idx in test_cases:
        pred_idx, probs = parse_blip2_output(text, class_names)
        status = "PASS" if pred_idx == expected_idx else "FAIL"
        print(f"[{status}] Expected {expected_idx}, Got {pred_idx} | Text: '{text}'")
        if pred_idx != expected_idx:
            failures += 1
            
    failure_rate = failures / len(test_cases)
    print(f"\nParse failure rate on test cases: {failure_rate:.1%}")
    if failure_rate > 0.5:
        print("WARNING: Parser may need adjustment")
        
# PART 7: Main block
if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    print("Testing output parser...")
    test_parser()
    
    print("\nLoading BLIP-2...")
    processor, model = load_blip2(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nRunning evaluation on xBD test set...")
    results = evaluate_baseline_c(
        xbd_root=config["paths"]["xbd_root"],
        processor=processor,
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        results_dir=config["paths"]["results_dir"],
        save_predictions_csv=True
    )
    
    print(f"\n=== BASELINE C (BLIP-2 Zero-Shot) RESULTS ===")
    print(f"Accuracy:      {results['accuracy']:.4f}")
    print(f"Macro-F1:      {results['macro_f1']:.4f}")
    print(f"Variance Score: {results['variance_score']:.4f}")
    print(f"Parse Failure Rate: {results['parse_failure_rate']:.1%}")
    print(f"Predictions saved: {results['predictions_csv_path']}")
    
    # Free BLIP-2 from GPU memory
    del model, processor
    torch.cuda.empty_cache()
    print("\nBLIP-2 released from GPU memory.")
    print("Run ensemble.py next — load Baseline C from CSV, not from model.")
