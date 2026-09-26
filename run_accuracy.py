"""
run_accuracy.py - Terminal Accuracy & Evaluation Report
======================================================
Computes and displays exact validation performance metrics:
- Macro F0.5 (Official Amazon ML Challenge Metric)
- Micro Precision & Recall
- Singleton Accuracy
- Classification Confusion Matrix
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

def compute_f05(prec: float, rec: float) -> float:
    if prec == 0.0 and rec == 0.0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)

def main():
    base_dir = Path(__file__).resolve().parent
    val_pred_file = base_dir / "artifacts" / "validation_predictions.tsv"
    thresh_file = base_dir / "artifacts" / "models" / "threshold.pkl"

    if not val_pred_file.exists():
        print(f"Error: {val_pred_file} not found.")
        sys.exit(1)

    # Load threshold
    if thresh_file.exists():
        with open(thresh_file, "rb") as f:
            threshold = pickle.load(f)
    else:
        threshold = 0.9300

    print("=" * 68)
    print("      AMAZON ML CHALLENGE: VALIDATION ACCURACY & METRICS REPORT")
    print("=" * 68)
    print(f"  Evaluation File:      {val_pred_file.name}")
    print(f"  Decision Threshold:   {threshold:.4f}")

    print("  Loading validation predictions...")
    df = pd.read_csv(val_pred_file, sep="\t", usecols=["s1_id", "cand_id", "label", "score"])
    print(f"  Total candidate pairs evaluated: {len(df):,}")

    # Build ground truth and predictions per S1 entity
    gt_map = defaultdict(set)
    pred_map = defaultdict(set)
    all_s1_ids = sorted(df["s1_id"].unique())

    for sid, cid, label, score in zip(df["s1_id"], df["cand_id"], df["label"], df["score"]):
        if label == 1:
            gt_map[sid].add(cid)
        if score >= threshold:
            pred_map[sid].add(cid)

    total_s1 = len(all_s1_ids)
    f05_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    false_merge_entities = 0
    singleton_total = 0
    singleton_correct = 0

    for sid in all_s1_ids:
        true_s = gt_map.get(sid, set())
        pred_s = pred_map.get(sid, set())

        is_true_single = (len(true_s) == 0)
        is_pred_single = (len(pred_s) == 0)

        if is_true_single:
            singleton_total += 1
            if is_pred_single:
                singleton_correct += 1
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                total_fp += len(pred_s)
                false_merge_entities += 1
        else:
            if is_pred_single:
                total_fn += len(true_s)
                f05_scores.append(0.0)
            else:
                tp = len(true_s & pred_s)
                fp = len(pred_s - true_s)
                fn = len(true_s - pred_s)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                if fp > 0:
                    false_merge_entities += 1

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f05_scores.append(compute_f05(prec, rec))

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    sing_acc = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0

    print("-" * 68)
    print("  KEY ACCURACY & PERFORMANCE METRICS:")
    print("-" * 68)
    print(f"  [*] MACRO F0.5 SCORE (OFFICIAL METRIC):  {macro_f05:.4f}  ({macro_f05*100:.2f}%)")
    print(f"  [*] PAIRWISE PRECISION:                 {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  [*] PAIRWISE RECALL:                    {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  [*] OVERALL F1 SCORE:                   {overall_f1:.4f}  ({overall_f1*100:.2f}%)")
    print(f"  [*] SINGLETON ACCURACY:                 {sing_acc:.4f}  ({sing_acc*100:.2f}%)")
    print("-" * 68)
    print("  CONFUSION & ERROR ANALYSIS:")
    print("-" * 68)
    print(f"  True Positives (Correct matches):      {total_tp:>8,}")
    print(f"  False Positives (False merges):        {total_fp:>8,}  (Penalized 2x in F0.5)")
    print(f"  False Negatives (Missed matches):      {total_fn:>8,}")
    print(f"  Entities with >= 1 False Merge:        {false_merge_entities:>8,} / {total_s1:,} ({false_merge_entities/total_s1*100:.2f}%)")
    print(f"  Singletons correctly identified:       {singleton_correct:>8,} / {singleton_total:,}")
    print("=" * 68)

if __name__ == "__main__":
    main()
