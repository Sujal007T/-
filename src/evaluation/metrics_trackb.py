"""
Track B Evaluation Metrics Module.
Implements rigorous statistical evaluation for:
1. Disaster Type Recognition (DTR, Classification):
   - Accuracy, Macro-F1, Per-class F1, Variance Score (std of per-class F1),
     Confusion Matrix, Per-Disaster-Type Accuracy, Majority-Class Baseline.
2. Disaster Captioning (Structured Report Generation):
   - (a) Schema Compliance: Strict section header verification, full compliance %,
         mean sections present, per-section presence rates.
   - (b) Generation Quality: ROUGE-L F1 overall & per-section, BLEU-4 (sacrebleu/NLTK),
         optional BERTScore.
   - (c) Manual review CSV export with seeded random sampling.
3. Statistical Testing:
   - 1000-iteration empirical bootstrap 95% Confidence Intervals.
   - Paired bootstrap test between conditions with difference CIs and verdicts.
"""

import os
import re
import csv
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Optional libraries
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE_SCORER = True
except ImportError:
    HAS_ROUGE_SCORER = False

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


# =========================================================================
#  1. Longest Common Subsequence & Pure-Python ROUGE-L Fallback
# =========================================================================

def _compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Computes the length of the Longest Common Subsequence between token lists using 1D DP."""
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1
    m, n = len(seq1), len(seq2)
    if m == 0:
        return 0
    dp = [0] * (m + 1)
    for y in seq2:
        prev = 0
        for i in range(m):
            temp = dp[i + 1]
            if seq1[i] == y:
                dp[i + 1] = prev + 1
            elif dp[i] > dp[i + 1]:
                dp[i + 1] = dp[i]
            prev = temp
    return dp[m]


def compute_single_rouge_l(ref: str, hyp: str) -> float:
    """Computes ROUGE-L F1 score between a reference and a hypothesis string."""
    if HAS_ROUGE_SCORER:
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            return scorer.score(ref, hyp)['rougeL'].fmeasure
        except Exception:
            pass

    # Pure Python robust fallback
    ref_tokens = re.findall(r'\b\w+\b', ref.lower())
    hyp_tokens = re.findall(r'\b\w+\b', hyp.lower())
    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs_len = _compute_lcs_length(ref_tokens, hyp_tokens)
    prec = lcs_len / len(hyp_tokens)
    rec = lcs_len / len(ref_tokens)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * (prec * rec) / (prec + rec)


def compute_single_bleu_4(ref: str, hyp: str) -> float:
    """Computes sentence-level BLEU-4 score (0.0 to 1.0)."""
    if HAS_SACREBLEU:
        try:
            res = sacrebleu.sentence_bleu(hyp, [ref])
            return res.score / 100.0
        except Exception:
            pass

    if HAS_NLTK:
        try:
            ref_tokens = [re.findall(r'\b\w+\b', ref.lower())]
            hyp_tokens = re.findall(r'\b\w+\b', hyp.lower())
            smooth = SmoothingFunction().method1
            return sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        except Exception:
            pass

    # Basic n-gram precision fallback
    ref_tokens = re.findall(r'\b\w+\b', ref.lower())
    hyp_tokens = re.findall(r'\b\w+\b', hyp.lower())
    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    common = set(hyp_tokens).intersection(set(ref_tokens))
    return len(common) / max(len(hyp_tokens), 1)


# =========================================================================
#  2. Disaster Type Recognition (DTR) Metrics
# =========================================================================

def compute_dtr_metrics(
    y_true: List[str],
    y_pred: List[str],
    all_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes all standard DTR classification metrics:
    - Accuracy
    - Macro-F1 (unweighted mean across classes)
    - Per-class F1
    - Variance Score (standard deviation across per-class F1)
    - Confusion Matrix
    - Per-disaster-type Accuracy
    - Majority-class baseline accuracy
    """
    if len(y_true) == 0:
        raise ValueError("Cannot compute DTR metrics on empty labels list.")

    # Normalize strings
    true_norm = [str(x).lower().strip() for x in y_true]
    pred_norm = [str(x).lower().strip() for x in y_pred]

    if all_classes is None:
        classes = sorted(list(set(true_norm + pred_norm)))
    else:
        classes = sorted(list(set([c.lower().strip() for c in all_classes] + true_norm + pred_norm)))

    acc = float(accuracy_score(true_norm, pred_norm))
    macro_f1 = float(f1_score(true_norm, pred_norm, labels=classes, average='macro', zero_division=0))
    per_class_f1_arr = f1_score(true_norm, pred_norm, labels=classes, average=None, zero_division=0)
    per_class_f1 = {cls_name: float(f1) for cls_name, f1 in zip(classes, per_class_f1_arr)}

    # Variance score: standard deviation of per-class F1
    variance_score = float(np.std(per_class_f1_arr))

    # Confusion Matrix
    cm = confusion_matrix(true_norm, pred_norm, labels=classes).tolist()

    # Per-disaster-type Accuracy
    per_type_acc = {}
    for cls_name in classes:
        indices = [i for i, t in enumerate(true_norm) if t == cls_name]
        if indices:
            cls_correct = sum(1 for i in indices if pred_norm[i] == cls_name)
            per_type_acc[cls_name] = float(cls_correct / len(indices))
        else:
            per_type_acc[cls_name] = 0.0

    # Majority class baseline
    counter = Counter(true_norm)
    majority_class, majority_count = counter.most_common(1)[0]
    majority_baseline_acc = float(majority_count / len(true_norm))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "variance_score": variance_score,
        "per_class_f1": per_class_f1,
        "per_disaster_type_accuracy": per_type_acc,
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_baseline_acc,
        "classes": classes,
        "confusion_matrix": cm,
        "sample_count": len(y_true)
    }


# =========================================================================
#  3. Structured Captioning / Report Metrics
# =========================================================================

DEFAULT_SECTIONS = [
    "DISASTER",
    "BUILDING",
    "ROAD",
    "VEGETATION",
    "WATER_BODY",
    "AGRICULTURE",
    "CONCLUSION"
]

def parse_report_sections(report_text: str, sections: Optional[List[str]] = None) -> Dict[str, str]:
    """Splits a structured damage report into section text by headers."""
    if sections is None:
        sections = DEFAULT_SECTIONS

    parsed = {sec: "" for sec in sections}
    text = str(report_text).strip()
    if not text:
        return parsed

    pattern = r'(?:^|\n)\s*(?:[*#_]*\s*)(' + '|'.join(sections) + r')\s*(?:[*#_]*)\s*[:：]\s*'
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            header = parts[i].upper().strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if header in parsed:
                parsed[header] = content
    return parsed


def compute_caption_metrics(
    golds: List[str],
    preds: List[str],
    required_sections: Optional[List[str]] = None,
    disaster_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes report generation metrics:
    - (a) Schema compliance rate (% of reports with all required sections present)
    - Mean sections present and per-section presence rates
    - (b) Overall ROUGE-L and BLEU-4
    - Per-section ROUGE-L
    - Per-disaster-type ROUGE-L (if disaster_types list provided)
    """
    if len(golds) == 0:
        raise ValueError("Cannot compute caption metrics on empty list.")
    if required_sections is None:
        required_sections = DEFAULT_SECTIONS

    n = len(golds)
    rouge_l_scores = []
    bleu_4_scores = []
    full_compliance_flags = []
    sections_present_counts = []
    section_presence = {sec: 0 for sec in required_sections}
    section_rouge_scores = {sec: [] for sec in required_sections}
    per_type_rouge = {}

    for i in range(n):
        gold = str(golds[i]).strip()
        pred = str(preds[i]).strip()

        # Overall text scores
        r_l = compute_single_rouge_l(gold, pred)
        b_4 = compute_single_bleu_4(gold, pred)
        rouge_l_scores.append(r_l)
        bleu_4_scores.append(b_4)

        # Schema compliance check
        parsed_pred = parse_report_sections(pred, required_sections)
        parsed_gold = parse_report_sections(gold, required_sections)

        present_count = 0
        for sec in required_sections:
            if parsed_pred[sec] and len(parsed_pred[sec].strip()) > 0:
                section_presence[sec] += 1
                present_count += 1
                if parsed_gold[sec]:
                    section_rouge_scores[sec].append(compute_single_rouge_l(parsed_gold[sec], parsed_pred[sec]))

        sections_present_counts.append(present_count)
        is_fully_compliant = (present_count == len(required_sections))
        full_compliance_flags.append(1.0 if is_fully_compliant else 0.0)

        # Track per-disaster-type if provided
        if disaster_types and i < len(disaster_types):
            d_type = str(disaster_types[i]).lower().strip()
            if d_type not in per_type_rouge:
                per_type_rouge[d_type] = []
            per_type_rouge[d_type].append(r_l)

    # Aggregations
    mean_rouge_l = float(np.mean(rouge_l_scores))
    mean_bleu_4 = float(np.mean(bleu_4_scores))
    full_compliance_pct = float(np.mean(full_compliance_flags) * 100.0)
    mean_sections = float(np.mean(sections_present_counts))
    per_section_pres_pct = {sec: float((cnt / n) * 100.0) for sec, cnt in section_presence.items()}
    per_section_rouge_mean = {
        sec: float(np.mean(scores)) if scores else 0.0
        for sec, scores in section_rouge_scores.items()
    }
    per_type_rouge_mean = {
        d_type: float(np.mean(scores)) if scores else 0.0
        for d_type, scores in per_type_rouge.items()
    }

    return {
        "rouge_l": mean_rouge_l,
        "bleu_4": mean_bleu_4,
        "full_compliance_pct": full_compliance_pct,
        "mean_sections_present": mean_sections,
        "total_required_sections": len(required_sections),
        "per_section_presence_pct": per_section_pres_pct,
        "per_section_rouge_l": per_section_rouge_mean,
        "per_disaster_type_rouge_l": per_type_rouge_mean,
        "sample_count": n,
        "raw_rouge_l": rouge_l_scores,
        "raw_bleu_4": bleu_4_scores,
        "raw_compliance": full_compliance_flags
    }


# =========================================================================
#  4. Bootstrap Confidence Intervals & Paired Hypothesis Testing
# =========================================================================

def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes the empirical bootstrap confidence interval for a metric list.
    Returns: (mean, ci_lower, ci_upper)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    arr = np.array(values, dtype=np.float64)
    point_mean = float(np.mean(arr))
    n = len(arr)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))

    return point_mean, ci_lower, ci_upper


def paired_bootstrap_test(
    scores_baseline: List[float],
    scores_target: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Runs a paired bootstrap test over matched sample IDs.
    Calculates diff = target[i] - baseline[i].
    Returns mean diff, 95% CI, empirical p-value, and verdict.
    """
    if len(scores_baseline) != len(scores_target):
        raise ValueError(f"Paired bootstrap requires equal lengths: {len(scores_baseline)} vs {len(scores_target)}")

    diffs = np.array(scores_target, dtype=np.float64) - np.array(scores_baseline, dtype=np.float64)
    point_diff = float(np.mean(diffs))
    n = len(diffs)

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resample = rng.choice(diffs, size=n, replace=True)
        boot_diffs[i] = np.mean(resample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100.0))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100.0))

    # Empirical p-value (proportion crossing zero in direction opposite to mean)
    if point_diff >= 0:
        p_val = 2.0 * float(np.mean(boot_diffs <= 0))
    else:
        p_val = 2.0 * float(np.mean(boot_diffs >= 0))
    p_val = min(max(p_val, 1.0 / n_bootstrap), 1.0)

    # Verdict: supported if CI excludes zero and improvement is positive
    is_supported = (ci_lower > 0.0) if point_diff > 0 else False
    verdict = "supported" if is_supported else ("statistically significant decrease" if ci_upper < 0.0 else "not supported")

    return {
        "mean_diff": point_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "verdict": verdict,
        "sample_count": n
    }


# =========================================================================
#  5. Manual Review CSV Exporter
# =========================================================================

def export_manual_review_csv(
    eval_ids: List[Any],
    golds: List[str],
    preds: List[str],
    output_path: str,
    n_samples: int = 30,
    seed: int = 42
) -> str:
    """
    Exports a seeded random sample of 30 generated reports for human blinded evaluation.
    Columns include blank rating columns (1-5 scales) and notes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n = min(len(eval_ids), len(golds), len(preds))
    indices = list(range(n))

    rng = random.Random(seed)
    sample_indices = rng.sample(indices, min(n_samples, n))

    fieldnames = [
        "review_id",
        "sample_id",
        "gold_report",
        "model_predicted_report",
        "factual_accuracy_1_5",
        "schema_compliance_1_5",
        "damage_severity_grading_1_5",
        "hallucination_penalty_1_5",
        "overall_quality_1_5",
        "reviewer_notes"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for review_idx, idx in enumerate(sample_indices, 1):
            writer.writerow({
                "review_id": review_idx,
                "sample_id": eval_ids[idx],
                "gold_report": golds[idx],
                "model_predicted_report": preds[idx],
                "factual_accuracy_1_5": "",
                "schema_compliance_1_5": "",
                "damage_severity_grading_1_5": "",
                "hallucination_penalty_1_5": "",
                "overall_quality_1_5": "",
                "reviewer_notes": ""
            })

    return output_path
