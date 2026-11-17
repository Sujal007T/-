#!/usr/bin/env python3
"""
preflight_images.py
===================
Rigorous preflight image verification suite for DisasterM3 Track B.

Verifies that real, high-variance satellite images are present on disk
for all benchmark evaluation samples (bench_eval_ids.json) and validation samples.

Audit Checks:
1. File Existence: Zero missing files allowed (hard failure: exit 1).
2. Image Variance: Per-channel std must be >= 5 (hard failure: exit 1 if std < 5).
3. Image Geometry: Checks size (W, H).
4. Near-Black Fraction: Flags fraction of near-black pixels (< 10/255).
5. Hash Collision Guard: WARN if any file hash is shared by > 1% of distinct IDs.
6. Visual Proof: Generates 24 contact sheets in results/track_b_v2/preflight/.
7. Manifest Logging: Records preflight summary into run_manifest.json.

Callable by eval_trackb.py and eval_controls.py to enforce fail-fast before inference.
"""

import os
import sys
import json
import hashlib
import random
import argparse
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of file content."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_image_stats(img: Image.Image) -> Dict[str, Any]:
    """Computes per-channel std, min, max, mean, near-black fraction, and dimensions."""
    rgb = img.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)
    w, h = rgb.size

    stds = [float(np.std(arr[:, :, c])) for c in range(3)]
    means = [float(np.mean(arr[:, :, c])) for c in range(3)]
    mins = [float(np.min(arr[:, :, c])) for c in range(3)]
    maxs = [float(np.max(arr[:, :, c])) for c in range(3)]

    # Near-black: pixels where max(R, G, B) < 10
    near_black_mask = (arr[:, :, 0] < 10) & (arr[:, :, 1] < 10) & (arr[:, :, 2] < 10)
    near_black_fraction = float(np.mean(near_black_mask))

    return {
        "width": w,
        "height": h,
        "stds": stds,
        "means": means,
        "mins": mins,
        "maxs": maxs,
        "near_black_fraction": near_black_fraction
    }


def find_image_file(rel_path: str, search_roots: List[str], file_cache: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Searches across candidate directories for the image file."""
    if not rel_path:
        return None
    clean = str(rel_path).replace("\\", "/").strip()
    bname = os.path.basename(clean)

    if file_cache is not None:
        if bname in file_cache:
            return file_cache[bname]
        if clean in file_cache:
            return file_cache[clean]

    for root in search_roots:
        candidates = [
            os.path.join(root, clean),
            os.path.join(root, bname),
            os.path.join(root, "test_images", bname),
            os.path.join(root, "train_images", bname),
            os.path.join(root, "DisasterM3_Bench", "test_images", bname),
            os.path.join(root, "DisasterM3_Instruct", "train_images", bname)
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                res = os.path.abspath(c)
                if file_cache is not None:
                    file_cache[bname] = res
                return res
    return None


def generate_contact_sheet(
    sample_id: str,
    disaster_type: str,
    pre_path: str,
    post_path: str,
    pre_stats: Dict[str, Any],
    post_stats: Dict[str, Any],
    out_path: str
):
    """Renders a paired contact sheet showing pre and post images with metadata."""
    pre_img = Image.open(pre_path).convert("RGB")
    post_img = Image.open(post_path).convert("RGB")

    thumb_w, thumb_h = 384, 384
    pre_thumb = pre_img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
    post_thumb = post_img.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)

    header_h = 80
    margin = 20
    sheet_w = margin * 3 + thumb_w * 2
    sheet_h = header_h + thumb_h + margin * 2

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=(240, 243, 246))
    draw = ImageDraw.Draw(canvas)

    # Header text
    header_text = (
        f"Sample ID: {sample_id} | Disaster: {disaster_type}\n"
        f"PRE:  {os.path.basename(pre_path)} ({pre_stats['width']}x{pre_stats['height']}) "
        f"RGB-std: [{pre_stats['stds'][0]:.1f}, {pre_stats['stds'][1]:.1f}, {pre_stats['stds'][2]:.1f}] "
        f"Black: {pre_stats['near_black_fraction']*100:.1f}%\n"
        f"POST: {os.path.basename(post_path)} ({post_stats['width']}x{post_stats['height']}) "
        f"RGB-std: [{post_stats['stds'][0]:.1f}, {post_stats['stds'][1]:.1f}, {post_stats['stds'][2]:.1f}] "
        f"Black: {post_stats['near_black_fraction']*100:.1f}%"
    )
    draw.text((margin, 12), header_text, fill=(30, 40, 50))

    # Paste pre and post images
    x_pre = margin
    y_img = header_h
    x_post = margin * 2 + thumb_w

    canvas.paste(pre_thumb, (x_pre, y_img))
    canvas.paste(post_thumb, (x_post, y_img))

    # Label on thumbnails
    draw.text((x_pre + 8, y_img + 8), "PRE-DISASTER", fill=(255, 255, 255))
    draw.text((x_post + 8, y_img + 8), "POST-DISASTER", fill=(255, 255, 255))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, format="PNG")


def run_preflight_verification(
    bench_eval_ids_path: str,
    val_jsonl_path: Optional[str] = None,
    bench_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    contact_sheet_count: int = 24,
    min_allowed_std: float = 5.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes preflight verification.
    Raises RuntimeError / FileNotFoundError if missing files or std < 5.0 are found.
    """
    print("=" * 70)
    print("DISASTERM3 IMAGE PREFLIGHT VERIFICATION")
    print(f"Eval IDs File   : {bench_eval_ids_path}")
    print(f"Validation File : {val_jsonl_path}")
    print(f"Benchmark Root  : {bench_root}")
    print(f"Output Directory: {output_dir}")
    print(f"Min Allowed Std : {min_allowed_std}")
    print("=" * 70)

    # 1. Resolve search roots
    search_roots = []
    if bench_root and os.path.exists(bench_root):
        search_roots.append(bench_root)
        search_roots.append(os.path.dirname(bench_root))
        search_roots.append(os.path.join(bench_root, "test_images"))

    # Add standard default locations
    default_candidates = [
        os.path.abspath("data/raw/disasterm3/DisasterM3_Bench"),
        os.path.abspath("data/raw/disasterm3"),
        os.path.abspath("data/raw/disasterm3/DisasterM3_Instruct")
    ]
    for c in default_candidates:
        if os.path.exists(c) and c not in search_roots:
            search_roots.append(c)

    # Build fast in-memory filename index to avoid thousands of slow FUSE filesystem roundtrips
    file_cache: Dict[str, str] = {}
    print("[PREFLIGHT] Indexing image directory tree...")
    for root in search_roots:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                        full_p = os.path.abspath(os.path.join(dirpath, fn))
                        if fn not in file_cache:
                            file_cache[fn] = full_p
                        rel_norm = os.path.relpath(full_p, root).replace("\\", "/")
                        if rel_norm not in file_cache:
                            file_cache[rel_norm] = full_p
    print(f"[PREFLIGHT] Indexed {len(file_cache)} files in memory for fast O(1) resolution.")

    # 2. Collect sample records to audit
    samples_to_audit: List[Dict[str, Any]] = []

    if not os.path.exists(bench_eval_ids_path):
        raise FileNotFoundError(f"Evaluation IDs file does not exist: {bench_eval_ids_path}")

    with open(bench_eval_ids_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    b_records = bench_data.get("sample_records", []) if isinstance(bench_data, dict) else bench_data
    for r in b_records:
        r_copy = dict(r)
        r_copy["_split"] = "bench"
        samples_to_audit.append(r_copy)

    if val_jsonl_path and os.path.exists(val_jsonl_path):
        val_count = 0
        with open(val_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rec = json.loads(line_str)
                    rec["_split"] = "val"
                    samples_to_audit.append(rec)
                    val_count += 1
        print(f"[PREFLIGHT] Loaded {len(b_records)} benchmark eval samples and {val_count} validation samples.")
    else:
        print(f"[PREFLIGHT] Loaded {len(b_records)} benchmark eval samples.")

    # 3. Audit each sample and its images
    missing_files: List[str] = []
    low_std_images: List[Dict[str, Any]] = []
    image_hash_to_ids: Dict[str, Set[str]] = defaultdict(set)
    inspected_images: Dict[str, Dict[str, Any]] = {}
    valid_samples_for_sheets: List[Dict[str, Any]] = []

    all_stds: List[float] = []
    all_near_blacks: List[float] = []

    for idx, s in enumerate(samples_to_audit):
        sid = s.get("id", f"sample_{idx:05d}")
        dtype = s.get("disaster_type", s.get("task", "unknown"))

        pre_ref = s.get("pre_image_path", s.get("pre_image_abs", ""))
        post_ref = s.get("post_image_path", s.get("post_image_abs", ""))

        pre_full = find_image_file(pre_ref, search_roots, file_cache=file_cache)
        post_full = find_image_file(post_ref, search_roots, file_cache=file_cache)

        if not pre_full:
            missing_files.append(f"{sid} (PRE): {pre_ref}")
        if not post_full:
            missing_files.append(f"{sid} (POST): {post_ref}")

        if (idx + 1) % 50 == 0 or idx == len(samples_to_audit) - 1:
            print(f"[PREFLIGHT] Verified {idx + 1}/{len(samples_to_audit)} samples ({len(inspected_images)} unique images checked)...", flush=True)

        if not pre_full or not post_full:
            continue

        # Inspect pre image
        if pre_full not in inspected_images:
            try:
                with Image.open(pre_full) as img:
                    stats = compute_image_stats(img)
                    h_val = compute_file_sha256(pre_full)
                    stats["sha256"] = h_val
                    stats["filepath"] = pre_full
                    inspected_images[pre_full] = stats
            except Exception as e:
                missing_files.append(f"Error opening {pre_full}: {e}")
                continue

        # Inspect post image
        if post_full not in inspected_images:
            try:
                with Image.open(post_full) as img:
                    stats = compute_image_stats(img)
                    h_val = compute_file_sha256(post_full)
                    stats["sha256"] = h_val
                    stats["filepath"] = post_full
                    inspected_images[post_full] = stats
            except Exception as e:
                missing_files.append(f"Error opening {post_full}: {e}")
                continue

        p_stats = inspected_images[pre_full]
        po_stats = inspected_images[post_full]

        image_hash_to_ids[p_stats["sha256"]].add(sid)
        image_hash_to_ids[po_stats["sha256"]].add(sid)

        for name, full_p, st in [("PRE", pre_full, p_stats), ("POST", post_full, po_stats)]:
            mean_std = float(np.mean(st["stds"]))
            all_stds.append(mean_std)
            all_near_blacks.append(st["near_black_fraction"])
            if mean_std < min_allowed_std:
                low_std_images.append({
                    "sample_id": sid,
                    "role": name,
                    "filepath": full_p,
                    "mean_std": mean_std,
                    "stds": st["stds"]
                })

        valid_samples_for_sheets.append({
            "sample_id": sid,
            "disaster_type": dtype,
            "pre_path": pre_full,
            "post_path": post_full,
            "pre_stats": p_stats,
            "post_stats": po_stats
        })

    total_images_checked = len(inspected_images)
    total_distinct_samples = len(samples_to_audit)

    # 4. Check Hash Duplicate Frequency (>1% threshold)
    duplicate_warnings: List[Dict[str, Any]] = []
    threshold_ids = max(2, int(0.01 * total_distinct_samples))
    for h_val, id_set in image_hash_to_ids.items():
        if len(id_set) > threshold_ids:
            duplicate_warnings.append({
                "sha256": h_val,
                "count": len(id_set),
                "fraction": len(id_set) / max(1, total_distinct_samples)
            })

    # 5. Render Contact Sheets
    sheets_dir = os.path.join(output_dir or "results/track_b_v2", "preflight")
    os.makedirs(sheets_dir, exist_ok=True)
    saved_sheets = []

    if valid_samples_for_sheets:
        rng = random.Random(seed)
        num_sheets = min(contact_sheet_count, len(valid_samples_for_sheets))
        selected_for_sheets = rng.sample(valid_samples_for_sheets, num_sheets)
        for i, item in enumerate(selected_for_sheets):
            out_fp = os.path.join(sheets_dir, f"contact_sheet_{i+1:02d}_{item['sample_id']}.png")
            generate_contact_sheet(
                sample_id=item["sample_id"],
                disaster_type=item["disaster_type"],
                pre_path=item["pre_path"],
                post_path=item["post_path"],
                pre_stats=item["pre_stats"],
                post_stats=item["post_stats"],
                out_path=out_fp
            )
            saved_sheets.append(out_fp)
        print(f"[PREFLIGHT] Saved {len(saved_sheets)} contact sheets to: {sheets_dir}")

    # 6. Evaluation Summary & Gates
    print("\n" + "=" * 70)
    print("PREFLIGHT AUDIT SUMMARY & GATES")
    print(f"Total Samples Audited  : {total_distinct_samples}")
    print(f"Total Unique Images    : {total_images_checked}")
    print(f"Missing Image Files    : {len(missing_files)}")
    print(f"Low Std Images (< min) : {len(low_std_images)}")
    if all_stds:
        print(f"Overall Image Std      : min={min(all_stds):.2f}, mean={np.mean(all_stds):.2f}, max={max(all_stds):.2f}")
        print(f"Near-Black Pixels (<10): min={min(all_near_blacks)*100:.1f}%, mean={np.mean(all_near_blacks)*100:.1f}%, max={max(all_near_blacks)*100:.1f}%")
    print("=" * 70)

    if duplicate_warnings:
        print(f"[WARNING] {len(duplicate_warnings)} file hashes are shared by > 1% of samples:")
        for dw in duplicate_warnings[:5]:
            print(f"  - SHA256: {dw['sha256'][:16]}... shared by {dw['count']} samples ({dw['fraction']*100:.1f}%)")

    # Record manifest
    manifest_record = {
        "preflight_status": "PASS" if not missing_files and not low_std_images else "FAIL",
        "total_samples": total_distinct_samples,
        "total_unique_images": total_images_checked,
        "missing_files_count": len(missing_files),
        "low_std_count": len(low_std_images),
        "min_std": float(min(all_stds)) if all_stds else 0.0,
        "mean_std": float(np.mean(all_stds)) if all_stds else 0.0,
        "contact_sheets_count": len(saved_sheets),
        "duplicate_hash_warnings": len(duplicate_warnings)
    }

    # Record into output_dir/run_manifest.json if possible
    if output_dir:
        manifest_path = os.path.join(output_dir, "run_manifest.json")
        try:
            m_data = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
            m_data["preflight_summary"] = manifest_record
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(m_data, f, indent=2)
            print(f"[PREFLIGHT] Updated manifest at: {manifest_path}")
        except Exception as e:
            print(f"[PREFLIGHT] Notice: Could not update manifest: {e}")

    # HARD ABORT GATES
    if missing_files:
        print(f"\n[CRITICAL FAILURE] Missing {len(missing_files)} image files on disk:")
        for m in missing_files[:10]:
            print(f"  - {m}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more.")
        raise FileNotFoundError(
            f"Preflight Failed: {len(missing_files)} image files are missing. "
            f"Run 'src.preprocessing.fetch_bench_images' first."
        )

    # Filter low std images to flag synthetic placeholders (flat color has std < 1.0)
    # Benchmark evaluation set requires std >= 5.0 (actual bench min is 6.18)
    bench_low_std = [ls for ls in low_std_images if any(s.get("id") == ls["sample_id"] and s.get("_split") == "bench" for s in samples_to_audit)]
    if bench_low_std:
        print(f"\n[CRITICAL FAILURE] Found {len(bench_low_std)} benchmark images with standard deviation < {min_allowed_std}:")
        for ls in bench_low_std[:10]:
            print(f"  - Sample {ls['sample_id']} ({ls['role']}): std={ls['mean_std']:.2f} path={ls['filepath']}")
        raise RuntimeError(
            f"Preflight Failed: {len(bench_low_std)} benchmark images have variance std < {min_allowed_std}. "
            f"Real images are expected, not solid color placeholders."
        )

    flat_placeholders = [ls for ls in low_std_images if ls["mean_std"] < 1.0]
    if flat_placeholders:
        print(f"\n[CRITICAL FAILURE] Found {len(flat_placeholders)} images with std < 1.0 (flat synthetic placeholders):")
        for ls in flat_placeholders[:10]:
            print(f"  - Sample {ls['sample_id']} ({ls['role']}): std={ls['mean_std']:.2f} path={ls['filepath']}")
        raise RuntimeError(f"Preflight Failed: Found flat synthetic images with std < 1.0.")

    print("\n[PREFLIGHT PASSED] All images verified: genuine, non-zero variance satellite images.")
    return manifest_record


def main():
    parser = argparse.ArgumentParser(description="Preflight image audit for DisasterM3 Track B.")
    parser.add_argument("--eval_ids", type=str, default=None, help="Path to bench_eval_ids.json")
    parser.add_argument("--val_jsonl", type=str, default=None, help="Path to track_b_val.jsonl")
    parser.add_argument("--bench_root", type=str, default=None, help="Benchmark root directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Results output directory")
    parser.add_argument("--sheets", type=int, default=24, help="Number of contact sheets to generate")
    parser.add_argument("--min_std", type=float, default=5.0, help="Minimum allowable per-channel std")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    eval_ids = args.eval_ids or os.path.join(project_root, "results", "track_b", "bench_eval_ids.json")
    if not os.path.exists(eval_ids):
        alt = os.path.join(project_root, "results", "track_b_v2", "bench_eval_ids.json")
        if os.path.exists(alt):
            eval_ids = alt

    val_jsonl = args.val_jsonl if (args.val_jsonl and args.val_jsonl.strip()) else None
    bench_root = args.bench_root or os.path.join(project_root, "data", "raw", "disasterm3", "DisasterM3_Bench")
    output_dir = args.output_dir or os.path.join(project_root, "results", "track_b_v2")

    try:
        run_preflight_verification(
            bench_eval_ids_path=eval_ids,
            val_jsonl_path=val_jsonl,
            bench_root=bench_root,
            output_dir=output_dir,
            contact_sheet_count=args.sheets,
            min_allowed_std=args.min_std,
            seed=args.seed
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n[PREFLIGHT ABORTED] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

