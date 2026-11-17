#!/usr/bin/env python3
"""
fetch_bench_images.py
=====================
Downloads ONLY the benchmark image files referenced by bench_eval_ids.json
(or benchmark_release.json) from Kingdrone-Junjue/DisasterM3 on HuggingFace Hub.

Keeps repo-relative paths:
  DisasterM3_Bench/test_images/<image_file>.png
into paths.disasterm3_root (e.g. data/raw/disasterm3/).

Features:
- Resumable: skips already downloaded files with matching size.
- Multi-threaded: concurrent downloads with ThreadPoolExecutor.
- Validates: verifies all target files exist in repo before downloading.
- Prints file count, total size in MB, and any missing paths.
"""

import os
import sys
import json
import argparse
import time
from typing import Set, Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download


def resolve_path(p: str, project_root: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(project_root, p))


def load_target_image_paths(
    eval_ids_path: str,
    bench_file_path: str,
    project_root: str = None
) -> Set[str]:
    """Collects all image paths referenced by evaluation samples across all eval manifests."""
    target_paths: Set[str] = set()

    candidate_files = []
    if eval_ids_path:
        candidate_files.append(os.path.abspath(eval_ids_path))

    if project_root:
        for p in ["results/track_b/bench_eval_ids.json", "results/track_b_v2/bench_eval_ids.json"]:
            full_p = os.path.abspath(os.path.join(project_root, p))
            if full_p not in candidate_files:
                candidate_files.append(full_p)

    for c_path in candidate_files:
        if c_path and os.path.exists(c_path):
            print(f"[FETCH] Loading evaluation sample IDs from: {c_path}")
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                samples = data.get("sample_records", []) if isinstance(data, dict) else data
                for s in samples:
                    for k in ["pre_image_path", "post_image_path"]:
                        val = s.get(k)
                        if val:
                            clean = str(val).replace("\\", "/").strip()
                            if clean.startswith("DisasterM3_Bench/"):
                                clean = clean[len("DisasterM3_Bench/"):]
                            target_paths.add(clean)
            except Exception as e:
                print(f"[FETCH] Warning reading {c_path}: {e}")

    if target_paths:
        print(f"[FETCH] Total collected unique image paths: {len(target_paths)}")
        return target_paths

    # Fallback source: benchmark_release.json
    if os.path.exists(bench_file_path):
        print(f"[FETCH] Warning: No eval_ids files found. Scanning {bench_file_path}...")
        with open(bench_file_path, "r", encoding="utf-8") as f:
            bench_records = json.load(f)
        for r in bench_records:
            for k in ["pre_image_path", "post_image_path"]:
                val = r.get(k)
                if val:
                    clean = str(val).replace("\\", "/").strip()
                    if clean.startswith("DisasterM3_Bench/"):
                        clean = clean[len("DisasterM3_Bench/"):]
                    target_paths.add(clean)
        print(f"[FETCH] Found {len(target_paths)} unique image paths from benchmark release.")
        return target_paths

    raise FileNotFoundError(f"Neither eval_ids nor {bench_file_path} exists.")


def download_single_file(
    repo_id: str,
    filename: str,
    local_dir: str,
    expected_size: int
) -> Tuple[str, bool, int, str]:
    """
    Downloads one file using hf_hub_download.
    Returns: (filename, success, bytes_transferred, message)
    """
    target_local_path = os.path.join(local_dir, filename.replace("/", os.sep))
    if os.path.exists(target_local_path):
        actual_size = os.path.getsize(target_local_path)
        if expected_size <= 0 or actual_size == expected_size:
            return (filename, True, 0, "EXISTS")

    try:
        os.makedirs(os.path.dirname(target_local_path), exist_ok=True)
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        downloaded_size = os.path.getsize(target_local_path) if os.path.exists(target_local_path) else 0
        return (filename, True, downloaded_size, "DOWNLOADED")
    except Exception as e:
        return (filename, False, 0, str(e))


def fetch_benchmark_images(
    project_root: str,
    eval_ids_path: str = None,
    bench_file_path: str = None,
    dest_root: str = None,
    repo_id: str = "Kingdrone-Junjue/DisasterM3",
    max_workers: int = 8
) -> Dict[str, Any]:
    """Orchestrates targeted downloading of benchmark images."""
    config_path = os.path.join(project_root, "configs", "config.yaml")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if dest_root is None:
        raw_root = config.get("paths", {}).get("disasterm3_root", "data/raw/disasterm3")
        # In case config has drive path from colab, normalize
        if "/content/drive" in raw_root or ":\\" not in raw_root:
            dest_root = resolve_path("data/raw/disasterm3", project_root)
        else:
            dest_root = resolve_path(raw_root, project_root)

    if eval_ids_path is None:
        eval_ids_path = resolve_path("results/track_b/bench_eval_ids.json", project_root)
        if not os.path.exists(eval_ids_path):
            alt_path = resolve_path("results/track_b_v2/bench_eval_ids.json", project_root)
            if os.path.exists(alt_path):
                eval_ids_path = alt_path

    if bench_file_path is None:
        bench_file_path = resolve_path("data/raw/disasterm3/DisasterM3_Bench/benchmark_release.json", project_root)

    print("=" * 70)
    print("DISASTERM3 TARGETED BENCHMARK IMAGE FETCHER")
    print(f"Target Repo       : {repo_id}")
    print(f"Eval IDs File     : {eval_ids_path}")
    print(f"Destination Root  : {dest_root}")
    print(f"Concurrent Workers: {max_workers}")
    print("=" * 70)

    # 1. Collect target relative paths
    rel_paths = load_target_image_paths(eval_ids_path, bench_file_path, project_root=project_root)

    # 2. Query HuggingFace Repo Tree to get sizes and verify existence
    print("[FETCH] Querying HuggingFace dataset tree for 'DisasterM3_Bench/test_images'...")
    api = HfApi()
    try:
        repo_files = list(api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="DisasterM3_Bench/test_images"
        ))
        remote_file_map = {
            f.path.replace("\\", "/"): (f.size or 0)
            for f in repo_files if hasattr(f, "path")
        }
        print(f"[FETCH] Remote repo tree indexed: {len(remote_file_map)} files found under test_images.")
    except Exception as e:
        print(f"[FETCH] Warning: Failed to list repo tree ({e}). Proceeding with direct downloads...")
        remote_file_map = {}

    # Map each target path to repo filename
    targets_to_fetch: List[Tuple[str, int]] = []
    missing_in_repo: List[str] = []
    total_expected_bytes = 0

    for rp in sorted(rel_paths):
        repo_filename = f"DisasterM3_Bench/{rp}"
        if remote_file_map:
            if repo_filename in remote_file_map:
                sz = remote_file_map[repo_filename]
                targets_to_fetch.append((repo_filename, sz))
                total_expected_bytes += sz
            else:
                missing_in_repo.append(repo_filename)
        else:
            targets_to_fetch.append((repo_filename, 0))

    print(f"\n[SUMMARY] Target Images to Fetch : {len(targets_to_fetch)}")
    print(f"[SUMMARY] Total Expected Size    : {total_expected_bytes / (1024 * 1024):.2f} MB ({total_expected_bytes / (1024 * 1024 * 1024):.2f} GB)")
    if missing_in_repo:
        print(f"[WARNING] {len(missing_in_repo)} target paths were NOT found in Hugging Face repository:")
        for m in missing_in_repo[:10]:
            print(f"  - {m}")
        if len(missing_in_repo) > 10:
            print(f"  ... and {len(missing_in_repo) - 10} more.")

    # 3. Check existing files locally (Resumability check)
    already_existing = 0
    already_bytes = 0
    pending_tasks = []
    for repo_fn, sz in targets_to_fetch:
        local_fp = os.path.join(dest_root, repo_fn.replace("/", os.sep))
        if os.path.exists(local_fp) and os.path.getsize(local_fp) > 0:
            if sz <= 0 or os.path.getsize(local_fp) == sz:
                already_existing += 1
                already_bytes += os.path.getsize(local_fp)
                continue
        pending_tasks.append((repo_fn, sz))

    print(f"[CACHE] Already on disk and verified: {already_existing}/{len(targets_to_fetch)} files ({already_bytes / (1024 * 1024):.2f} MB)")
    print(f"[QUEUE] Pending download: {len(pending_tasks)} files ({(total_expected_bytes - already_bytes) / (1024 * 1024):.2f} MB)")

    if not pending_tasks:
        print("\n[ALL OK] All targeted benchmark images are already downloaded and verified on disk!")
        return {
            "total_targets": len(targets_to_fetch),
            "downloaded": 0,
            "already_present": already_existing,
            "missing_in_repo": len(missing_in_repo),
            "total_bytes": total_expected_bytes
        }

    # 4. Multithreaded Download
    print(f"\n[DOWNLOAD] Starting download with {max_workers} threads...")
    start_time = time.time()
    downloaded_count = 0
    bytes_downloaded = 0
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_file, repo_id, fn, dest_root, sz): fn
            for fn, sz in pending_tasks
        }
        with tqdm(total=len(pending_tasks), desc="Downloading images", unit="img") as pbar:
            for fut in as_completed(futures):
                fn, success, b_transferred, msg = fut.result()
                if success:
                    if msg == "DOWNLOADED":
                        downloaded_count += 1
                        bytes_downloaded += b_transferred
                else:
                    errors.append(f"{fn}: {msg}")
                pbar.update(1)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("FETCH COMPLETE SUMMARY")
    print(f"Total Targets        : {len(targets_to_fetch)}")
    print(f"Downloaded Now       : {downloaded_count} files ({bytes_downloaded / (1024 * 1024):.2f} MB)")
    print(f"Previously Existing  : {already_existing} files")
    print(f"Download Errors      : {len(errors)}")
    print(f"Elapsed Time         : {elapsed:.2f} s ({bytes_downloaded / (1024 * 1024 * max(elapsed, 0.001)):.2f} MB/s)")
    print("=" * 70)

    if errors:
        print(f"[ERRORS] {len(errors)} files failed to download:")
        for err in errors[:10]:
            print(f"  - {err}")
        raise RuntimeError(f"Failed to download {len(errors)} benchmark image files.")

    return {
        "total_targets": len(targets_to_fetch),
        "downloaded": downloaded_count,
        "already_present": already_existing,
        "missing_in_repo": len(missing_in_repo),
        "total_bytes": total_expected_bytes
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch targeted benchmark images from Hugging Face.")
    parser.add_argument("--eval_ids", type=str, default=None, help="Path to bench_eval_ids.json")
    parser.add_argument("--bench_file", type=str, default=None, help="Path to benchmark_release.json")
    parser.add_argument("--dest_root", type=str, default=None, help="Destination root directory")
    parser.add_argument("--repo_id", type=str, default="Kingdrone-Junjue/DisasterM3")
    parser.add_argument("--workers", type=int, default=8, help="Number of download worker threads")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fetch_benchmark_images(
        project_root=project_root,
        eval_ids_path=args.eval_ids,
        bench_file_path=args.bench_file,
        dest_root=args.dest_root,
        repo_id=args.repo_id,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
