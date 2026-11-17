"""
Labeler Validation Script (validate_labeler.py).
Computes Cohen's kappa and agreement accuracy between auto-labels and human annotations
from results/track_b/labeler_validation.csv.

Rules:
  - If kappa < 0.6, the section is flagged as 'labeler unreliable' and excluded from headline claims.
"""

import os
import sys
import argparse
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import cohen_kappa_score, accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


def evaluate_labeler_validation(csv_path: str) -> Dict[str, Any]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Validation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = [
        "sample_id", "section", "gold_text", "pred_text_D",
        "auto_label_gold", "auto_label_pred",
        "human_label_gold", "human_label_pred"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in validation CSV: {col}")

    # Check if human columns have data (handles NaN or empty strings)
    has_human_gold = df["human_label_gold"].fillna("").astype(str).str.strip().ne("").any()
    has_human_pred = df["human_label_pred"].fillna("").astype(str).str.strip().ne("").any()

    sections = df["section"].unique().tolist()
    report = {
        "csv_path": csv_path,
        "total_rows": len(df),
        "total_samples": len(df["sample_id"].unique()),
        "human_labels_present": bool(has_human_gold or has_human_pred),
        "section_results": {}
    }

    if not (has_human_gold or has_human_pred):
        print(f"[VALIDATION NOTICE] {csv_path} has {len(df)} rows across {len(sections)} sections,")
        print("but 'human_label_gold' and 'human_label_pred' columns are currently blank.")
        print("Please review the rows and enter categorical labels, then re-run this script.")
        for sec in sections:
            sec_df = df[df["section"] == sec]
            report["section_results"][sec] = {
                "sample_count": len(sec_df),
                "gold_auto_distribution": sec_df["auto_label_gold"].value_counts().to_dict(),
                "pred_auto_distribution": sec_df["auto_label_pred"].value_counts().to_dict(),
                "status": "awaiting_human_annotation",
                "kappa_gold": None,
                "kappa_pred": None,
                "verdict": "UNVERIFIED"
            }
        return report

    print(f"\n{'='*70}\nSECTION LABELER VALIDATION RESULTS (Cohen's Kappa & Accuracy)\n{'='*70}")
    print(f"{'Section':<15} | {'Gold Acc':<9} | {'Gold Kappa':<11} | {'Pred Acc':<9} | {'Pred Kappa':<11} | {'Status':<20}")
    print("-" * 85)

    for sec in sections:
        sec_df = df[df["section"] == sec]
        res = {"sample_count": len(sec_df)}

        # Evaluate gold human agreement
        valid_gold = sec_df[sec_df["human_label_gold"].fillna("").astype(str).str.strip().ne("")]
        if len(valid_gold) > 0:
            g_auto = valid_gold["auto_label_gold"].tolist()
            g_human = valid_gold["human_label_gold"].tolist()
            acc_g = accuracy_score(g_human, g_auto)
            try:
                kap_g = cohen_kappa_score(g_human, g_auto)
            except Exception:
                kap_g = 0.0
            res["gold_acc"] = round(acc_g * 100.0, 2)
            res["gold_kappa"] = round(kap_g, 4)
        else:
            res["gold_acc"] = None
            res["gold_kappa"] = None

        # Evaluate pred human agreement
        valid_pred = sec_df[sec_df["human_label_pred"].fillna("").astype(str).str.strip().ne("")]
        if len(valid_pred) > 0:
            p_auto = valid_pred["auto_label_pred"].tolist()
            p_human = valid_pred["human_label_pred"].tolist()
            acc_p = accuracy_score(p_human, p_auto)
            try:
                kap_p = cohen_kappa_score(p_human, p_auto)
            except Exception:
                kap_p = 0.0
            res["pred_acc"] = round(acc_p * 100.0, 2)
            res["pred_kappa"] = round(kap_p, 4)
        else:
            res["pred_acc"] = None
            res["pred_kappa"] = None

        # Status check (kappa < 0.6 => unreliable)
        min_kappa = min(
            k for k in [res.get("gold_kappa"), res.get("pred_kappa")] if k is not None
        ) if any(k is not None for k in [res.get("gold_kappa"), res.get("pred_kappa")]) else 0.0

        if min_kappa < 0.6:
            verdict = "labeler unreliable (kappa < 0.6)"
        else:
            verdict = "reliable (kappa >= 0.6)"

        res["verdict"] = verdict
        report["section_results"][sec] = res

        g_acc_str = f"{res['gold_acc']:.1f}%" if res.get('gold_acc') is not None else "N/A"
        g_kap_str = f"{res['gold_kappa']:.3f}" if res.get('gold_kappa') is not None else "N/A"
        p_acc_str = f"{res['pred_acc']:.1f}%" if res.get('pred_acc') is not None else "N/A"
        p_kap_str = f"{res['pred_kappa']:.3f}" if res.get('pred_kappa') is not None else "N/A"

        print(f"{sec:<15} | {g_acc_str:<9} | {g_kap_str:<11} | {p_acc_str:<9} | {p_kap_str:<11} | {verdict:<20}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate section labeler with Cohen's Kappa.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to labeler_validation.csv"
    )
    args = parser.parse_args()

    csv_path = args.csv_path or os.path.join(PROJECT_ROOT, "results", "track_b", "labeler_validation.csv")
    evaluate_labeler_validation(csv_path)


if __name__ == "__main__":
    main()

