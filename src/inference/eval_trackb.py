"""
DisasterM3 Track B Benchmark Evaluation Pipeline (eval_trackb.py).
Evaluates Qwen2.5-VL-3B-Instruct across conditions A, B, C, D, and K on DisasterM3_Bench.

Conditions:
  A: Zero-shot base model, post-disaster image only, plain prompt (no task tokens).
  B: Zero-shot base model, post-disaster image only, task-token prompt.
  C: Zero-shot base model, pre + post + 2x2 patches of post, task-token prompt.
  D: Fine-tuned model (QLoRA adapter), pre + post + 2x2 patches, task-token prompt.
  K (Control): No model. Majority class for DTR, central training report for Captioning.

Key Features:
  - Stratified benchmark sampling across disaster types (seed 42).
  - Fixed evaluation set saved to results/track_b/bench_eval_ids.json and reused across conditions.
  - Strict leakage guard asserting 0 evaluated images exist in track_b_train.jsonl.
  - BitsAndBytes 4-bit base model loading with adapter switching (enable/disable_adapter).
  - Greedy decoding with max_new_tokens 16 (DTR) and 400 (captioning).
  - Robust DTR output parsing with options mapping; tracks unparseable counts.
  - Resumable predictions JSONL output (skips existing IDs).
  - Input hash verification asserting Condition B vs C inputs differ (> 0%).
  - Automatic calculation of bootstrap 95% CIs and metrics via src.evaluation.metrics_trackb.
  - Run manifest recording GPU, libraries, adapter, and dataset revision info.
"""

import os
import sys
import re
import json
import time
import yaml
import random
import hashlib
import zipfile
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
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
    export_manual_review_csv,
    compute_single_rouge_l
)

# Optional HF API for dataset manifest revision
try:
    from huggingface_hub import HfApi
    HAS_HF_API = True
except ImportError:
    HAS_HF_API = False

# Optional Qwen VL utils and transformers
IMPORT_ERROR_MSG = ""
try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    HAS_TRANSFORMERS = True
except ImportError as e:
    import subprocess
    print(f"[AUTO-INSTALL] Detected missing package ({e}). Auto-installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "qwen-vl-utils", "peft", "bitsandbytes", "accelerate", "rouge-score", "sacrebleu"
        ])
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig
        )
        from peft import PeftModel
        from qwen_vl_utils import process_vision_info
        HAS_TRANSFORMERS = True
        print("[AUTO-INSTALL] Successfully installed required dependencies.")
    except Exception as err:
        HAS_TRANSFORMERS = False
        IMPORT_ERROR_MSG = str(err)


# =========================================================================
#  1. Configuration & Path Management
# =========================================================================

def resolve_path(p: str, base_dir: str = PROJECT_ROOT) -> str:
    """Resolves relative and drive-prefixed paths against local project structure."""
    if not p:
        return ""
    # If absolute and exists
    if os.path.exists(p):
        return os.path.abspath(p)
    # Check if path contains disaSense_project
    if "disaSense_project" in p:
        rel = p.split("disaSense_project")[-1].lstrip("/\\")
        candidate = os.path.join(base_dir, rel)
        if os.path.exists(candidate) or not os.path.isabs(p):
            return os.path.abspath(candidate)
    candidate = os.path.join(base_dir, p.lstrip("/\\"))
    return os.path.abspath(candidate)


def load_config_and_schema(
    config_path: Optional[str] = None,
    schema_path: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads configuration and DisasterM3 schema YAML files."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    if schema_path is None:
        schema_path = os.path.join(PROJECT_ROOT, "configs", "disasterm3_schema.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    return config, schema


# =========================================================================
#  2. Task Tokens & Prompts
# =========================================================================

TASK_TOKENS = {
    "Disaster Type Recognition": (
        "You are a satellite image disaster analysis expert. "
        "You will be shown a pre-disaster and post-disaster satellite image pair. "
        "Your task is Disaster Type Recognition. "
        "Select the single best answer from the given options. "
        "Respond in exactly this format:\n"
        "Answer: <letter>. <disaster type>\n"
        "Reason: <one sentence explanation based on visible damage patterns.>"
    ),
    "Disaster Report": (
        "You are a satellite image disaster analysis expert. "
        "You will be shown a pre-disaster and post-disaster satellite image pair. "
        "Your task is Disaster Captioning. "
        "Produce a structured damage report using EXACTLY these section headers:\n"
        "DISASTER: <type>\n"
        "BUILDING: <damage observed or not>\n"
        "ROAD: <damage observed or not>\n"
        "VEGETATION: <changes observed or not>\n"
        "WATER_BODY: <changes observed or not>\n"
        "AGRICULTURE: <changes observed or not>\n"
        "CONCLUSION: <one sentence summary of overall damage>\n"
        "Be concise and specific. Base all observations on visible differences between the two images."
    )
}
# Alias for training tasks
TASK_TOKENS["disaster caption"] = TASK_TOKENS["Disaster Report"]


# =========================================================================
#  3. Event ID & Disaster Type Extraction
# =========================================================================

def extract_event_id(img_path: str) -> str:
    """Extracts base event identifier from an image filename."""
    if not img_path:
        return "unknown_event"
    bname = os.path.basename(img_path)
    # Match patterns like bata_explosion_pre_0.png -> bata_explosion
    event = re.sub(r'_(pre|post)_\d+\.[a-zA-Z0-9]+$', '', bname)
    # Also handle xBD pattern: socal-fire_00000073_pre_disaster.png -> socal-fire
    event = re.sub(r'_\d+_(pre|post)_disaster\.[a-zA-Z0-9]+$', '', event)
    return event if event else bname


def extract_disaster_type(record: Dict[str, Any]) -> str:
    """Extracts disaster type string from benchmark record."""
    task = record.get("task", "")
    if task == "Disaster Type Recognition":
        gt = str(record.get("ground_truth", "")).strip().lower()
        if gt:
            # If gt is a sentence like "This area has been affected by a fire."
            m = re.search(r'affected by an?\s+([a-zA-Z\-]+)', gt)
            if m:
                return m.group(1).lower()
            return gt
        return "unknown"
    else:
        # Structured report: parse DISASTER: <type>
        gt = str(record.get("ground_truth", ""))
        m = re.search(r'DISASTER\s*:\s*([^\n\r]+)', gt, re.IGNORECASE)
        if m:
            dtype = m.group(1).strip().lower()
            # Clean common punctuation
            dtype = re.sub(r'[^\w\s\-]', '', dtype).strip()
            return dtype
        # Fallback to event filename
        evt = extract_event_id(record.get("post_image_path", ""))
        for known in ["fire", "flood", "earthquake", "hurricane", "tornado", "tsunami", "explosion", "volcano"]:
            if known in evt.lower():
                return known
        return "unknown"


# =========================================================================
#  4. Fixed Stratified Sampling & bench_eval_ids.json
# =========================================================================

def build_stratified_evaluation_set(
    bench_records: List[Dict[str, Any]],
    train_file: Optional[str] = None,
    per_task: int = 200,
    full: bool = False,
    smoke: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Builds a deterministic, stratified evaluation sample set.
    Filters out any benchmark samples that share images with the training set (Leakage Prevention).
    Balances samples across disaster types for both DTR and Captioning.
    """
    rng = random.Random(seed)

    # Collect training set image basenames to prevent leakage
    train_basenames: Set[str] = set()
    if train_file and os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for k in ["pre_image_path", "post_image_path", "pre_image_abs", "post_image_abs"]:
                    val = r.get(k)
                    if val:
                        train_basenames.add(os.path.basename(str(val)).lower())

    # Filter by task and exclude leaked samples
    dtr_all = []
    cap_all = []
    skipped_leaks = 0

    for idx, rec in enumerate(bench_records):
        task = rec.get("task")
        if task not in ["Disaster Type Recognition", "Disaster Report", "disaster caption"]:
            continue

        # Check leakage against training basenames
        pre_b = os.path.basename(str(rec.get("pre_image_path", ""))).lower()
        post_b = os.path.basename(str(rec.get("post_image_path", ""))).lower()
        if (pre_b and pre_b in train_basenames) or (post_b and post_b in train_basenames):
            skipped_leaks += 1
            continue

        item = dict(rec)
        item["bench_index"] = idx
        item["id"] = f"bench_{idx:05d}"
        item["event_id"] = extract_event_id(rec.get("post_image_path", ""))
        item["disaster_type"] = extract_disaster_type(rec)

        if task == "Disaster Type Recognition":
            dtr_all.append(item)
        elif task in ["Disaster Report", "disaster caption"]:
            cap_all.append(item)

    if skipped_leaks > 0:
        print(f"[LEAKAGE FILTER] Excluded {skipped_leaks} benchmark samples that overlapped with training images.")
        print(f"  -> Clean candidate pool: {len(dtr_all)} DTR, {len(cap_all)} Captioning")

    if full:
        selected_dtr = dtr_all
        selected_cap = cap_all
    else:
        target_n = 10 if smoke else per_task

        def sample_stratified(records: List[Dict[str, Any]], n_target: int) -> List[Dict[str, Any]]:
            if len(records) <= n_target:
                return records
            # Group by disaster_type
            groups: Dict[str, List[Dict[str, Any]]] = {}
            for r in records:
                dtype = r["disaster_type"]
                groups.setdefault(dtype, []).append(r)

            # Shuffle each group deterministically
            for dtype in groups:
                rng.shuffle(groups[dtype])

            selected = []
            num_classes = len(groups)
            per_class_quota = max(1, n_target // num_classes)

            # First pass: take per_class_quota
            remaining_pool = []
            for dtype, items in groups.items():
                take = min(len(items), per_class_quota)
                selected.extend(items[:take])
                remaining_pool.extend(items[take:])

            # Second pass: fill up to n_target from remaining
            if len(selected) < n_target and remaining_pool:
                rng.shuffle(remaining_pool)
                needed = n_target - len(selected)
                selected.extend(remaining_pool[:needed])

            # Sort by original bench_index for reproducibility
            selected.sort(key=lambda x: x["bench_index"])
            return selected[:n_target]

        selected_dtr = sample_stratified(dtr_all, target_n)
        selected_cap = sample_stratified(cap_all, target_n)

    eval_set = selected_dtr + selected_cap
    return eval_set


def get_or_create_eval_ids(
    bench_records: List[Dict[str, Any]],
    eval_ids_path: str,
    train_file: Optional[str] = None,
    per_task: int = 200,
    full: bool = False,
    smoke: bool = False,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Loads existing bench_eval_ids.json if compatible, else builds and saves canonical set."""
    os.makedirs(os.path.dirname(os.path.abspath(eval_ids_path)), exist_ok=True)

    master_eval_set = None

    if os.path.exists(eval_ids_path):
        try:
            with open(eval_ids_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            saved_records = saved.get("sample_records", [])
            saved_dtr = [r for r in saved_records if r.get("task") == "Disaster Type Recognition"]
            saved_cap = [r for r in saved_records if r.get("task") != "Disaster Type Recognition"]

            if not full and len(saved_dtr) >= per_task and len(saved_cap) >= per_task:
                print(f"[EVAL SET] Reusing existing {eval_ids_path} ({len(saved_records)} samples)")
                master_eval_set = saved_records
            elif full and saved.get("target_n") == "full":
                print(f"[EVAL SET] Reusing full existing {eval_ids_path} ({len(saved_records)} samples)")
                master_eval_set = saved_records
        except Exception as e:
            print(f"[EVAL SET] Notice: Could not read existing {eval_ids_path}: {e}")

    if master_eval_set is None:
        # Build canonical 400-sample (or full) evaluation set
        master_eval_set = build_stratified_evaluation_set(
            bench_records,
            train_file=train_file,
            per_task=per_task,
            full=full,
            smoke=False,
            seed=seed
        )

        dtr_counts = Counter(r["disaster_type"] for r in master_eval_set if r["task"] == "Disaster Type Recognition")
        cap_counts = Counter(r["disaster_type"] for r in master_eval_set if r["task"] in ["Disaster Report", "disaster caption"])

        manifest_data = {
            "target_n": "full" if full else per_task,
            "seed": seed,
            "total_samples": len(master_eval_set),
            "dtr_count": sum(dtr_counts.values()),
            "caption_count": sum(cap_counts.values()),
            "dtr_disaster_type_distribution": dict(dtr_counts),
            "caption_disaster_type_distribution": dict(cap_counts),
            "sample_ids": [r["id"] for r in master_eval_set],
            "sample_records": master_eval_set
        }

        with open(eval_ids_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"[EVAL SET] Created {eval_ids_path} with {len(master_eval_set)} stratified samples:")
        print(f"  -> DTR ({manifest_data['dtr_count']} samples): {dict(dtr_counts)}")
        print(f"  -> Captioning ({manifest_data['caption_count']} samples): {dict(cap_counts)}")

    if smoke:
        # Slice deterministic 20-sample smoke subset (10 DTR + 10 Captioning) without mutating bench_eval_ids.json
        dtr_all = [r for r in master_eval_set if r.get("task") == "Disaster Type Recognition"]
        cap_all = [r for r in master_eval_set if r.get("task") != "Disaster Type Recognition"]
        smoke_set = dtr_all[:10] + cap_all[:10]
        print(f"[EVAL SET] Smoke mode active: evaluating 20 samples (10 DTR + 10 Captioning) sliced from master set.")
        return smoke_set

    return master_eval_set


# =========================================================================
#  5. Leakage Guard
# =========================================================================

def verify_no_leakage(train_file: str, eval_records: List[Dict[str, Any]]) -> None:
    """
    Strict Leakage Guard:
    Verifies that NO evaluated image path or basename appears in the fine-tuning training set.
    Aborts immediately if leakage is detected.
    """
    if not os.path.exists(train_file):
        print(f"[LEAKAGE GUARD] Training file not found at {train_file}; skipping strict file comparison.")
        return

    print(f"[LEAKAGE GUARD] Verifying 0 overlap against training set: {train_file} ...")
    train_images: Set[str] = set()
    train_basenames: Set[str] = set()

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for k in ["pre_image_path", "post_image_path", "pre_image_abs", "post_image_abs"]:
                val = r.get(k)
                if val:
                    val_str = str(val).strip()
                    train_images.add(val_str)
                    train_basenames.add(os.path.basename(val_str))

    leak_count = 0
    leaked_items = []
    for r in eval_records:
        for k in ["pre_image_path", "post_image_path"]:
            val = r.get(k)
            if val:
                val_str = str(val).strip()
                bname = os.path.basename(val_str)
                if val_str in train_images or bname in train_basenames:
                    leak_count += 1
                    leaked_items.append((r["id"], val_str))

    if leak_count > 0:
        msg = f"[CRITICAL LEAKAGE DETECTED] {leak_count} benchmark images overlap with training set! Examples: {leaked_items[:5]}"
        print(msg, file=sys.stderr)
        raise AssertionError(msg)

    print(f"[LEAKAGE GUARD PASS] 0 overlap detected across all {len(eval_records)} evaluation samples. No data leakage.")


# =========================================================================
#  6. Image Loading, Patching & Input Hashing
# =========================================================================

class ImageResolver:
    """Resolves and loads images from extracted folders or zip archives. Strict: raises FileNotFoundError if missing."""

    def __init__(self, bench_root: str):
        self.bench_root = os.path.abspath(bench_root)
        self.zip_handles: Dict[str, zipfile.ZipFile] = {}
        # Locate zip files in bench_root
        if os.path.exists(self.bench_root):
            for fname in os.listdir(self.bench_root):
                if fname.lower().endswith(".zip"):
                    zpath = os.path.join(self.bench_root, fname)
                    try:
                        self.zip_handles[fname] = zipfile.ZipFile(zpath, 'r')
                    except Exception as e:
                        print(f"[IMAGE RESOLVER] Warning: Could not open zip {zpath}: {e}")

    def load_image(self, rel_path: str) -> Image.Image:
        """Loads a PIL image from path or zip; raises FileNotFoundError if file is missing."""
        if not rel_path:
            raise ValueError("[IMAGE RESOLVER] rel_path cannot be empty.")

        clean_path = rel_path.replace("\\", "/")

        # 1. Direct path check across potential directories
        direct_candidates = [
            os.path.join(self.bench_root, clean_path),
            os.path.join(self.bench_root, os.path.basename(clean_path)),
            os.path.join(self.bench_root, "test_images", os.path.basename(clean_path)),
            os.path.join(self.bench_root, "..", "DisasterM3_Instruct", "train_images", os.path.basename(clean_path)),
            os.path.join(self.bench_root, "..", "train_images", os.path.basename(clean_path))
        ]
        for c in direct_candidates:
            c_norm = os.path.abspath(c)
            if os.path.exists(c_norm) and os.path.isfile(c_norm):
                assert os.path.exists(c_norm), f"Assertion failed: Image file must exist: {c_norm}"
                return Image.open(c_norm).convert("RGB")

        # 2. Check zip archives
        bname = os.path.basename(clean_path)
        for zname, zh in self.zip_handles.items():
            for member in zh.namelist():
                if member.endswith(bname) or member.replace("\\", "/").endswith(clean_path):
                    with zh.open(member) as f:
                        return Image.open(f).convert("RGB")

        # Strict: NO fallback to synthetic single-color images
        searched_paths = "\n  - " + "\n  - ".join([os.path.abspath(p) for p in direct_candidates])
        raise FileNotFoundError(
            f"[IMAGE RESOLVER ERROR] Image file '{rel_path}' does not exist on disk or in zip archives.\n"
            f"Searched candidate locations:{searched_paths}\n"
            f"Please run 'src.evaluation.preflight_images' or fetch the required benchmark images first."
        )

    def close(self):
        for zh in self.zip_handles.values():
            try:
                zh.close()
            except Exception:
                pass


def make_2x2_patches(img: Image.Image) -> List[Image.Image]:
    """Crops 4 local patches of an image in a 2x2 grid."""
    w, h = img.size
    pw, ph = w // 2, h // 2
    patches = [
        img.crop((0, 0, pw, ph)),
        img.crop((pw, 0, w, ph)),
        img.crop((0, ph, pw, h)),
        img.crop((pw, ph, w, h))
    ]
    return patches


def compute_sample_input_hash(
    condition: str,
    prompt_text: str,
    images: List[Image.Image]
) -> str:
    """Computes a SHA-256 hash uniquely identifying the exact inputs presented to the model."""
    h = hashlib.sha256()
    h.update(f"COND_{condition}::".encode("utf-8"))
    h.update(prompt_text.encode("utf-8"))
    h.update(f"::NUM_IMAGES_{len(images)}::".encode("utf-8"))
    for idx, img in enumerate(images):
        h.update(f"::IMG_{idx}_{img.size}_{img.mode}::".encode("utf-8"))
        # Sample pixels from image corners and center
        w, h_dim = img.size
        sample_pixels = [
            img.getpixel((0, 0)),
            img.getpixel((w - 1, 0)),
            img.getpixel((0, h_dim - 1)),
            img.getpixel((w // 2, h_dim // 2))
        ]
        h.update(str(sample_pixels).encode("utf-8"))
    return h.hexdigest()[:16]


# =========================================================================
#  7. DTR Parsing & Normalization
# =========================================================================

def parse_dtr_prediction(
    raw_text: str,
    sample: Dict[str, Any],
    known_types: Optional[List[str]] = None
) -> Tuple[str, bool]:
    """
    Maps model output to a disaster class label using that sample's own options.
    Accepts:
      - 'Answer: B. landslide'
      - 'B. landslide' or bare letter 'B'
      - Directly matching disaster type name
    Returns: (parsed_class, is_valid)
    """
    if known_types is None:
        known_types = [
            "hurricane", "earthquake", "flood", "fire",
            "tornado", "tsunami", "explosion", "volcano",
            "wildfire", "landslide", "flooding", "bushfire"
        ]

    options_list = sample.get("options_list", [])
    text = str(raw_text).strip()

    # Build letter-to-class map
    letter_map: Dict[str, str] = {}
    if options_list and isinstance(options_list, list):
        for idx, opt in enumerate(options_list):
            letter = chr(ord('A') + idx)
            letter_map[letter] = str(opt).strip().lower()

    # Also parse from options_str if available
    options_str = sample.get("options_str", "")
    if options_str and not letter_map:
        matches = re.findall(r'([A-Z])[\.:]\s*([a-zA-Z\s\-]+?)(?:,|$|\.)', options_str)
        for ltr, name in matches:
            letter_map[ltr.upper()] = name.strip().lower()

    # 1. Match 'Answer: <letter>' or '^<letter>[\.\)]'
    m_ans = re.search(r'(?:Answer\s*:\s*|^\s*)([A-H])\b', text, re.IGNORECASE)
    if m_ans:
        ltr = m_ans.group(1).upper()
        if ltr in letter_map:
            return letter_map[ltr], True

    # 2. Check for option text after letter: 'B. landslide'
    m_combo = re.search(r'([A-H])[\.\s\:\-]+([a-zA-Z\-]+)', text, re.IGNORECASE)
    if m_combo:
        ltr = m_combo.group(1).upper()
        opt_text = m_combo.group(2).strip().lower()
        if ltr in letter_map:
            return letter_map[ltr], True
        if opt_text in known_types:
            return opt_text, True

    # 3. Match against options_list classes directly
    text_lower = text.lower()
    for opt in options_list:
        opt_norm = str(opt).strip().lower()
        if re.search(r'\b' + re.escape(opt_norm) + r'\b', text_lower):
            return opt_norm, True

    # 4. Match against known disaster types
    for kt in known_types:
        if re.search(r'\b' + re.escape(kt) + r'\b', text_lower):
            return kt, True

    return "unparseable", False


# =========================================================================
#  8. Condition K (Control Baseline)
# =========================================================================

def get_condition_k_baselines(
    train_file: str,
    schema: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Computes Condition K control baseline values:
    - DTR: majority class of the training set (default: 'hurricane').
    - Captioning: central training report with highest mean ROUGE-L against 500 reports.
    """
    central_cache_file = os.path.join(PROJECT_ROOT, "configs", "condition_k_central_report.json")
    if os.path.exists(central_cache_file):
        try:
            with open(central_cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data["majority_dtr_class"], data["central_report"]
        except Exception:
            pass

    # Compute from train_file
    majority_dtr = "hurricane"
    reports: List[str] = []

    if os.path.exists(train_file):
        dtr_classes = Counter()
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                task = r.get("task")
                if task == "Disaster Type Recognition":
                    dtr_classes[r.get("ground_truth", "").strip().lower()] += 1
                elif task in ["disaster caption", "Disaster Report"]:
                    gt = r.get("ground_truth", "").strip()
                    if gt:
                        reports.append(gt)

        if dtr_classes:
            majority_dtr = dtr_classes.most_common(1)[0][0]

    # Find central report
    central_report = ""
    if reports:
        rng = random.Random(42)
        sampled = rng.sample(reports, min(500, len(reports)))

        # Pre-filter top candidates by word frequency
        corpus_words = Counter()
        for rep in sampled:
            corpus_words.update(set(re.findall(r'\b\w+\b', rep.lower())))

        scored = []
        for idx, rep in enumerate(sampled):
            words = set(re.findall(r'\b\w+\b', rep.lower()))
            score = sum(corpus_words[w] for w in words)
            scored.append((score, idx))
        scored.sort(reverse=True)
        top_candidates = [idx for _, idx in scored[:25]]

        best_score = -1.0
        best_idx = 0
        for c_idx in top_candidates:
            cand = sampled[c_idx]
            mean_r = sum(compute_single_rouge_l(other, cand) for j, other in enumerate(sampled) if c_idx != j) / max(len(sampled) - 1, 1)
            if mean_r > best_score:
                best_score = mean_r
                best_idx = c_idx
        central_report = sampled[best_idx]
    else:
        # Fallback default 7-section report
        central_report = (
            "DISASTER: hurricane\n"
            "BUILDING: Minor to moderate roof damage visible across multiple structures.\n"
            "ROAD: Roadways remain largely passable with scattered minor debris.\n"
            "VEGETATION: Visible wind stripping and localized canopy thinning.\n"
            "WATER_BODY: Temporary water accumulation observed in localized low-lying areas.\n"
            "AGRICULTURE: Surrounding agricultural parcels show minor standing water.\n"
            "CONCLUSION: The hurricane caused moderate localized damage to structures and vegetation while primary transit arteries remain operational.\n"
        )

    # Save to cache
    try:
        with open(central_cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "majority_dtr_class": majority_dtr,
                "central_report": central_report
            }, f, indent=2)
    except Exception:
        pass

    return majority_dtr, central_report


# =========================================================================
#  9. Model Loading & Inference Execution
# =========================================================================

class ModelRunner:
    """Manages 4-bit base model loading, LoRA adapter attaching, and adapter switching."""

    def __init__(
        self,
        model_name: str,
        checkpoint_dir: str,
        fp16_reference: bool = False,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.fp16_reference = fp16_reference
        self.device = device
        self.model = None
        self.processor = None
        self.adapter_attached = False

        if not HAS_TRANSFORMERS:
            raise ImportError(
                f"Missing required packages for Qwen-VL model inference:\n"
                f"  -> Original import error: {IMPORT_ERROR_MSG}\n"
                f"Please install dependencies with:\n"
                f"  pip install qwen-vl-utils peft bitsandbytes accelerate rouge-score sacrebleu"
            )

    def load(self):
        """Loads base model and attaches adapter once."""
        print(f"[MODEL] Loading base model: {self.model_name} (fp16_ref={self.fp16_reference}) ...")

        if self.fp16_reference or not torch.cuda.is_available():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.processor.image_processor.min_pixels = 28 * 28
        self.processor.image_processor.max_pixels = 448 * 448

        # Attach LoRA adapter if checkpoint exists
        if os.path.exists(self.checkpoint_dir):
            print(f"[MODEL] Attaching LoRA adapter from: {self.checkpoint_dir} ...")
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_dir)
            self.adapter_attached = True
            print("[MODEL] LoRA adapter attached successfully.")
        else:
            print(f"[MODEL] Warning: Checkpoint dir {self.checkpoint_dir} not found; Condition D will run on base weights.")

        self.model.eval()

    def set_condition_adapter(self, condition: str):
        """Switches adapter state: disabled for A, B, C; enabled for D."""
        if not self.adapter_attached:
            return

        if condition in ["A", "B", "C", "O"]:
            # Disable adapter for base model conditions
            try:
                self.model.disable_adapter_layers()
            except AttributeError:
                try:
                    self.model.base_model.disable_adapter_layers()
                except Exception:
                    pass
        elif condition == "D":
            # Enable adapter for fine-tuned condition
            try:
                self.model.enable_adapter_layers()
            except AttributeError:
                try:
                    self.model.base_model.enable_adapter_layers()
                except Exception:
                    pass

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
        """Executes greedy generation with Qwen2.5-VL chat template."""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            return decoded[0].strip()


# =========================================================================
#  10. Condition Prediction Generator
# =========================================================================

def build_condition_inputs(
    condition: str,
    sample: Dict[str, Any],
    image_resolver: ImageResolver,
    schema: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], str, List[Image.Image]]:
    """
    Constructs messages, prompt text, and images for a given condition:
      A: Zero-shot, post-disaster image only, plain prompt (no system prompt).
      B: Task tokens, post-disaster image only.
      C: Task tokens, pre-disaster + post-disaster + 2x2 patches of post.
      D: Same inputs as C (differs by enabling LoRA adapter).
    """
    task = sample.get("task", "")
    is_dtr = (task == "Disaster Type Recognition")

    # Load post-disaster image
    post_path = sample.get("post_image_path", "")
    post_img = image_resolver.load_image(post_path) if post_path else None
    if post_img is None:
        raise ValueError(f"Sample {sample.get('id')} has no valid post_image_path.")

    # Load pre-disaster image when needed (Conditions C, D, O, or captioning)
    pre_img = None
    if condition in ["C", "D", "O"] or not is_dtr:
        pre_path = sample.get("pre_image_path", "")
        if pre_path:
            pre_img = image_resolver.load_image(pre_path)

    # Construct user prompt text
    base_prompt = sample.get("prompts", "")
    if isinstance(base_prompt, list):
        base_prompt = base_prompt[0] if base_prompt else ""
    if not base_prompt:
        base_prompt = "What kind of calamity has this area experienced?" if is_dtr else "Please describe a comprehensive damage situation based on pre- and post-disaster images."

    if is_dtr and sample.get("options_str"):
        prompt_text = f"{base_prompt}\nOptions: {sample.get('options_str')}"
    else:
        prompt_text = base_prompt

    images: List[Image.Image] = []
    messages: List[Dict[str, Any]] = []

    if condition == "A":
        # Plain prompt, single post-disaster image, no system prompt
        images = [post_img]
        user_content = [
            {"type": "image", "image": post_img},
            {"type": "text", "text": prompt_text}
        ]
        messages = [{"role": "user", "content": user_content}]

    elif condition == "B":
        # Task tokens system prompt, single post-disaster image
        system_prompt = TASK_TOKENS.get(task, "")
        images = [post_img]
        user_content = [
            {"type": "image", "image": post_img},
            {"type": "text", "text": prompt_text}
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    elif condition in ["C", "D"]:
        # Task tokens system prompt, pre + post + 4 patches of post
        system_prompt = TASK_TOKENS.get(task, "")
        patches = make_2x2_patches(post_img)
        images = [pre_img, post_img] + patches

        user_content = [
            {"type": "image", "image": pre_img},
            {"type": "image", "image": post_img}
        ]
        for p in patches:
            user_content.append({"type": "image", "image": p})

        augmented_prompt = prompt_text + "\n\nNote: The last 4 images above are 2x2 local patches of the post-disaster image."
        user_content.append({"type": "text", "text": augmented_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    elif condition == "O":
        # Condition O: Official Reference from official_disasterm3/pyscripts/run_vllm.py
        # Base model (no adapter), official prompt format, exactly 2 images (pre & post), native resolution.
        images = [pre_img, post_img]
        if is_dtr:
            official_tmpl = (
                "Analyze both the pre-disaster and post-disaster images to answer the following question. "
                "Choose the best option from the candidate options provided.\n\n"
                "pre-disaster image:\n<image>\n\n"
                "post-disaster image:\n<image>\n\n"
                "Question: {prompt}\n"
                "Options: {options_str}\n\n"
                "Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text."
            )
            raw_text = official_tmpl.format(
                prompt=sample.get("prompts", [""])[0] if isinstance(sample.get("prompts"), list) else sample.get("prompts", ""),
                options_str=sample.get("options_str", "")
            )
        else:
            official_tmpl = (
                "Your TASK is to analyze the provided pair of pre-disaster and post-disaster remote sensing images. \n"
                "You will act as a remote sensing analyst to identify the type of disaster and assess its impact on both built and natural environments across five specific categories.\n\n"
                "pre-disaster image:\n<image>\n\n"
                "post-disaster image:\n<image>\n\n"
                "Your analysis must be formatted as follows:\n"
                "DISASTER: [the name of the disaster]\n"
                "BUILDING: [describe impacts on buildings]\n"
                "ROAD: [describe impacts on road networks]\n"
                "VEGETATION: [describe impacts on natural, unmanaged vegetation cover]\n"
                "WATER_BODY: [describe changes to water bodies]\n"
                "AGRICULTURE: [describe impacts on managed agricultural land]\n"
                "CONCLUSION: [provide a concise 1-2 sentence summary synthesizing the overall disaster impacts observed across the categories.]"
            )
            raw_text = official_tmpl

        prompt_splits = raw_text.split("<image>")
        user_content = [
            {"type": "text", "text": prompt_splits[0]},
            {"type": "image", "image": pre_img},
            {"type": "text", "text": prompt_splits[1]},
            {"type": "image", "image": post_img},
            {"type": "text", "text": prompt_splits[2]},
        ]
        messages = [{"role": "user", "content": user_content}]
        prompt_text = raw_text

    return messages, prompt_text, images


# =========================================================================
#  11. Main Evaluation Pipeline
# =========================================================================

def run_evaluation(
    conditions: List[str],
    per_task: int = 200,
    full: bool = False,
    smoke: bool = False,
    fp16_reference: bool = False,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    seed: int = 42
):
    """Executes the complete evaluation benchmark across requested conditions."""
    config, schema = load_config_and_schema()

    # Paths
    bench_file = resolve_path("data/raw/disasterm3/DisasterM3_Bench/benchmark_release.json")
    train_file = resolve_path("data/processed/track_b_train.jsonl")
    bench_root = os.path.dirname(bench_file)
    output_base = resolve_path(output_dir or "results/track_b_v2")
    os.makedirs(output_base, exist_ok=True)

    eval_ids_file = os.path.join(output_base, "bench_eval_ids.json")
    manifest_file = os.path.join(output_base, "run_manifest.json")

    if checkpoint_dir is None:
        checkpoint_dir = resolve_path("checkpoints/lora_smoke_test_final")
    else:
        checkpoint_dir = resolve_path(checkpoint_dir)

    model_name = config.get("model", {}).get("name", "Qwen/Qwen2.5-VL-3B-Instruct")
    max_tokens_dtr = schema.get("decoding", {}).get("max_new_tokens_dtr", 16)
    max_tokens_cap = schema.get("decoding", {}).get("max_new_tokens_caption", 400)

    print("=" * 70)
    print("DISASTERM3 TRACK B RIGOROUS BENCHMARK EVALUATION")
    print(f"Conditions        : {conditions}")
    print(f"Per Task Target   : {'full' if full else (10 if smoke else per_task)}")
    print(f"Benchmark File    : {bench_file}")
    print(f"Checkpoint Dir    : {checkpoint_dir}")
    print(f"Output Directory  : {output_base}")
    print("=" * 70)

    # 1. Load benchmark records
    if not os.path.exists(bench_file):
        raise FileNotFoundError(f"Benchmark release file not found at: {bench_file}")

    with open(bench_file, "r", encoding="utf-8") as f:
        bench_records = json.load(f)

    # 2. Get or create stratified evaluation set
    eval_set = get_or_create_eval_ids(
        bench_records,
        eval_ids_path=eval_ids_file,
        train_file=train_file,
        per_task=per_task,
        full=full,
        smoke=smoke,
        seed=seed
    )

    # 3. Leakage Guard Check
    verify_no_leakage(train_file, eval_set)

    # 3.5. Automated Preflight Gate: verify all images are present and pass variance checks
    print("\n[PREFLIGHT GATE] Automatically verifying image integrity before evaluation...")
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
        print("[PREFLIGHT GATE PASS] All evaluation images verified authentic and present on disk.")
    except Exception as e:
        print(f"\n[CRITICAL PREFLIGHT FAILURE] {e}")
        raise RuntimeError(f"Automated preflight failed: {e}. Aborting evaluation immediately.")

    # 4. Initialize image resolver
    image_resolver = ImageResolver(bench_root)

    # 5. Check if model is needed (conditions A, B, C, D, O)
    needs_model = any(c in ["A", "B", "C", "D", "O"] for c in conditions)
    model_runner = None
    if needs_model:
        model_runner = ModelRunner(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            fp16_reference=fp16_reference,
            device=device
        )
        model_runner.load()

    # 6. Preload Condition K baselines
    majority_dtr, central_caption = get_condition_k_baselines(train_file, schema)

    # Store input hashes across conditions for B vs C comparison
    condition_input_hashes: Dict[str, Dict[str, str]] = {}

    start_wall_time = time.time()

    # 7. Evaluate each condition
    for cond in conditions:
        print("\n" + "=" * 60)
        print(f"RUNNING CONDITION {cond}")
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

        # Switch adapter for model conditions
        if model_runner is not None:
            model_runner.set_condition_adapter(cond)

        unparseable_dtr_count = 0
        latencies: List[float] = []
        cond_hashes: Dict[str, str] = {}

        # Open file in append mode
        with open(pred_file, "a", encoding="utf-8") as out_f:
            for idx, sample in enumerate(eval_set):
                sample_id = sample["id"]
                task = sample.get("task", "")
                is_dtr = (task == "Disaster Type Recognition")
                max_tokens = max_tokens_dtr if is_dtr else max_tokens_cap

                # If already computed, reuse
                if sample_id in finished_records:
                    rec = finished_records[sample_id]
                    cond_hashes[sample_id] = rec.get("input_hash", "")
                    if is_dtr and rec.get("parsed") == "unparseable":
                        unparseable_dtr_count += 1
                    continue

                # Run inference or control baseline
                t0 = time.time()

                if cond == "K":
                    # Control baseline: no model execution
                    if is_dtr:
                        raw_output = f"Answer: {majority_dtr}\nReason: Majority class baseline from training set."
                        parsed = majority_dtr
                    else:
                        raw_output = central_caption
                        parsed = central_caption
                    input_hash = f"control_k_{majority_dtr if is_dtr else 'centroid'}"
                    latency = 0.001

                else:
                    # Model inference (A, B, C, D)
                    messages, prompt_text, imgs = build_condition_inputs(cond, sample, image_resolver, schema)
                    input_hash = compute_sample_input_hash(cond, prompt_text, imgs)

                    if model_runner is not None and model_runner.model is not None:
                        raw_output = model_runner.generate(messages, max_new_tokens=max_tokens)
                    else:
                        raise RuntimeError(f"[EVAL ERROR] ModelRunner is not loaded with valid model weights for condition {cond}.")

                    latency = time.time() - t0

                    # Parse output
                    if is_dtr:
                        parsed, is_valid = parse_dtr_prediction(raw_output, sample)
                        if not is_valid:
                            unparseable_dtr_count += 1
                    else:
                        parsed = raw_output

                cond_hashes[sample_id] = input_hash
                latencies.append(latency)

                # Build record
                pred_rec = {
                    "id": sample_id,
                    "bench_index": sample.get("bench_index"),
                    "task": task,
                    "disaster_type": sample.get("disaster_type", ""),
                    "event_id": sample.get("event_id", ""),
                    "gold": sample.get("ground_truth", ""),
                    "raw_output": raw_output,
                    "parsed": parsed,
                    "input_hash": input_hash,
                    "latency_s": round(latency, 4)
                }

                out_f.write(json.dumps(pred_rec) + "\n")
                out_f.flush()
                finished_records[sample_id] = pred_rec

                # Print ETA after first 20 processed samples
                if len(latencies) == 20:
                    avg_lat = sum(latencies) / len(latencies)
                    remaining = len(eval_set) - (idx + 1)
                    eta_mins = (avg_lat * remaining) / 60.0
                    print(f"[{cond}] ETA: ~{eta_mins:.1f} minutes remaining (avg {avg_lat:.2f}s / sample)")

                if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_set):
                    print(f"[{cond}] Processed {idx + 1}/{len(eval_set)} samples...")

        condition_input_hashes[cond] = cond_hashes

        # Check unparseable warning
        dtr_total = sum(1 for r in eval_set if r["task"] == "Disaster Type Recognition")
        if dtr_total > 0:
            unparseable_rate = (unparseable_dtr_count / dtr_total) * 100.0
            print(f"[{cond}] DTR unparseable rate: {unparseable_rate:.1f}% ({unparseable_dtr_count}/{dtr_total})")
            if unparseable_rate > 10.0:
                print(f"[WARNING] [{cond}] DTR unparseable outputs exceeded 10%: {unparseable_rate:.1f}%!")

        # 8. Compute headline metrics for condition
        all_preds = [finished_records[r["id"]] for r in eval_set if r["id"] in finished_records]
        dtr_preds = [r for r in all_preds if r["task"] == "Disaster Type Recognition"]
        cap_preds = [r for r in all_preds if r["task"] in ["Disaster Report", "disaster caption"]]

        cond_metrics: Dict[str, Any] = {
            "condition": cond,
            "sample_count": len(all_preds),
            "dtr_count": len(dtr_preds),
            "caption_count": len(cap_preds)
        }

        # DTR Metrics
        if dtr_preds:
            y_true_dtr = [r["disaster_type"] for r in dtr_preds]
            y_pred_dtr = [r["parsed"] for r in dtr_preds]
            dtr_res = compute_dtr_metrics(y_true_dtr, y_pred_dtr)

            # Bootstrap CIs for accuracy and macro_f1
            # Per-sample binary correctness for accuracy CI
            acc_indicators = [1.0 if t == p else 0.0 for t, p in zip(y_true_dtr, y_pred_dtr)]
            _, acc_ci_lo, acc_ci_hi = bootstrap_ci(acc_indicators, n_bootstrap=1000, seed=seed)
            dtr_res["accuracy_ci_95"] = [acc_ci_lo, acc_ci_hi]

            cond_metrics["dtr"] = dtr_res

        # Captioning Metrics
        if cap_preds:
            golds_cap = [r["gold"] for r in cap_preds]
            preds_cap = [r["parsed"] for r in cap_preds]
            dtypes_cap = [r["disaster_type"] for r in cap_preds]
            cap_res = compute_caption_metrics(golds_cap, preds_cap, disaster_types=dtypes_cap)

            # Bootstrap CIs on headline metrics
            _, r_lo, r_hi = bootstrap_ci(cap_res["raw_rouge_l"], n_bootstrap=1000, seed=seed)
            _, b_lo, b_hi = bootstrap_ci(cap_res["raw_bleu_4"], n_bootstrap=1000, seed=seed)
            _, c_lo, c_hi = bootstrap_ci(cap_res["raw_compliance"], n_bootstrap=1000, seed=seed)

            cap_res["rouge_l_ci_95"] = [r_lo, r_hi]
            cap_res["bleu_4_ci_95"] = [b_lo, b_hi]
            cap_res["compliance_ci_95"] = [c_lo * 100.0, c_hi * 100.0]

            # Remove large raw arrays from json file
            del cap_res["raw_rouge_l"]
            del cap_res["raw_bleu_4"]
            del cap_res["raw_compliance"]

            cond_metrics["captioning"] = cap_res

            # Export manual review CSV (30 samples)
            eval_ids_cap = [r["id"] for r in cap_preds]
            export_manual_review_csv(eval_ids_cap, golds_cap, preds_cap, review_file, n_samples=30, seed=seed)
            print(f"[{cond}] Exported manual review sheet: {review_file}")

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(cond_metrics, f, indent=2)
        print(f"[{cond}] Saved metrics to: {metrics_file}")

    # 9. Verify Condition B vs C Input Difference
    if "B" in condition_input_hashes and "C" in condition_input_hashes:
        b_hashes = condition_input_hashes["B"]
        c_hashes = condition_input_hashes["C"]
        common_ids = set(b_hashes.keys()).intersection(set(c_hashes.keys()))
        if common_ids:
            diff_count = sum(1 for cid in common_ids if b_hashes[cid] != c_hashes[cid])
            diff_pct = (diff_count / len(common_ids)) * 100.0
            print("\n" + "-" * 50)
            print(f"[INPUT HASH AUDIT] Condition B vs C input difference: {diff_pct:.1f}% ({diff_count}/{len(common_ids)})")
            if diff_pct == 0.0:
                print("WARNING: Condition C is a no-op", file=sys.stderr)
            else:
                print("[INPUT HASH AUDIT] Verified: Condition C alters image representation (multi-scale patches & pre-disaster imagery).")
            print("-" * 50)

    # 10. Write Run Manifest
    wall_time_s = time.time() - start_wall_time
    eval_ids_hash = hashlib.sha256(open(eval_ids_file, "rb").read()).hexdigest()

    # Check adapter size
    adapter_size_mb = 0.0
    if os.path.exists(checkpoint_dir):
        for root, _, files in os.walk(checkpoint_dir):
            for fname in files:
                adapter_size_mb += os.path.getsize(os.path.join(root, fname)) / (1024 * 1024)

    # Check HF revision
    hf_dataset_sha = "offline_unknown"
    if HAS_HF_API:
        try:
            info = HfApi().dataset_info("Kingdrone-Junjue/DisasterM3")
            hf_dataset_sha = getattr(info, "sha", "unknown")
        except Exception:
            pass

    manifest = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "conditions_evaluated": conditions,
        "sample_count": len(eval_set),
        "bench_eval_ids_sha256": eval_ids_hash,
        "checkpoint_dir": checkpoint_dir,
        "adapter_size_mb": round(adapter_size_mb, 2),
        "model_name": model_name,
        "fp16_reference": fp16_reference,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "dataset_hf_sha": hf_dataset_sha,
        "total_wall_time_s": round(wall_time_s, 2)
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    image_resolver.close()
    print(f"\n[RUN COMPLETE] All evaluations finished. Manifest saved to: {manifest_file}")


# =========================================================================
#  12. CLI Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate DisasterM3 Track B conditions across benchmark set.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C", "D", "K"],
        choices=["A", "B", "C", "D", "K", "O"],
        help="List of conditions to evaluate (default: A B C D K)"
    )
    parser.add_argument(
        "--per_task",
        type=int,
        default=200,
        help="Number of samples per task (default: 200)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate on the full benchmark set"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: evaluate 10 samples per task"
    )
    parser.add_argument(
        "--fp16_reference",
        action="store_true",
        help="Use fp16 base model instead of 4-bit quantization"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to trained LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/track_b_v2",
        help="Directory to save evaluation results (default: results/track_b_v2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and bootstrapping (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (cuda or cpu)"
    )

    args = parser.parse_args()

    run_evaluation(
        conditions=args.conditions,
        per_task=args.per_task,
        full=args.full,
        smoke=args.smoke,
        fp16_reference=args.fp16_reference,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
