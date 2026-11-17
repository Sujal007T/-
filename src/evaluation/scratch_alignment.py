import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.patch_cropper import XBDPatchDataset, crop_patch

def check():
    test_ds = XBDPatchDataset('data/raw/xbd/train', split='test')
    df_a = pd.read_csv('results/baselines/baseline_a_probs.csv')
    df_b = pd.read_csv('results/baselines/baseline_b_probs.csv')
    df_c = pd.read_csv('results/baselines/baseline_c_predictions.csv')
    
    print(f"Dataset length: {len(test_ds.building_index)}")
    print(f"df_a length: {len(df_a)}, df_b length: {len(df_b)}, df_c length: {len(df_c)}")
    
    # Check skipped patches in test_ds
    skipped = []
    for idx, b in enumerate(test_ds.building_index):
        pre_patch = crop_patch(b["pre_image_path"], b["bbox"], test_ds.alpha, test_ds.min_patch_size)
        post_patch = crop_patch(b["post_image_path"], b["bbox"], test_ds.alpha, test_ds.min_patch_size)
        if pre_patch is None or post_patch is None:
            skipped.append((idx, b["uid"], b["pre_image_path"], b["damage_label"], pre_patch is None, post_patch is None))
            
    print(f"Total test building patches where crop_patch returned None: {len(skipped)}")
    for s in skipped[:10]:
        print(f"  Idx {s[0]}: uid={s[1]}, label={s[3]}, pre_is_none={s[4]}, post_is_none={s[5]}")

    # Check differences across df_a, df_b, df_c labels
    labels_ds = np.array([b["damage_label"] for b in test_ds.building_index])
    labels_a = df_a["true_label"].values
    labels_b = df_b["true_label"].values
    labels_c = df_c["true_label"].values
    
    diff_ds_a = (labels_ds != labels_a)
    diff_ds_b = (labels_ds != labels_b)
    diff_ds_c = (labels_ds != labels_c)
    diff_ab = (labels_a != labels_b)
    diff_ac = (labels_a != labels_c)
    diff_bc = (labels_b != labels_c)
    
    print(f"\nLabel comparisons with ground truth in building_index:")
    print(f"  labels_ds != labels_a: {diff_ds_a.sum()}")
    print(f"  labels_ds != labels_b: {diff_ds_b.sum()}")
    print(f"  labels_ds != labels_c: {diff_ds_c.sum()}")
    print(f"  labels_a != labels_b: {diff_ab.sum()}")
    print(f"  labels_a != labels_c: {diff_ac.sum()}")
    print(f"  labels_b != labels_c: {diff_bc.sum()}")
    
    # Check if indices of diff_ds_a correspond to skipped
    skipped_indices = set(s[0] for s in skipped)
    print(f"\nAre diff_ds_a indices in skipped? {set(np.where(diff_ds_a)[0]).issubset(skipped_indices)}")
    print(f"Are diff_ds_b indices in skipped? {set(np.where(diff_ds_b)[0]).issubset(skipped_indices)}")
    print(f"Are diff_ds_c indices in skipped? {set(np.where(diff_ds_c)[0]).issubset(skipped_indices)}")
    
    overlap_skipped_diff = set(np.where(diff_ds_a)[0]).intersection(skipped_indices)
    print(f"Intersection of diff_ds_a and skipped: {len(overlap_skipped_diff)}")
    
if __name__ == "__main__":
    check()

