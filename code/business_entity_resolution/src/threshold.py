"""
threshold.py - Step 8: Threshold Optimization
==============================================
The model outputs P(match) in [0, 1] for each candidate pair.
We need a THRESHOLD: if P(match) >= threshold -> predict match.

Why not just use 0.5?
  Default 0.5 is arbitrary. Because F0.5 is precision-heavy, we often want
  a HIGHER threshold to avoid false merges (false positives).

  Example: with threshold=0.5, we might get F0.5=0.72
           with threshold=0.75, we get F0.5=0.81 (fewer false merges)

We try multiple thresholds and pick the one with best validation F0.5.

We also evaluate SEPARATE thresholds for S1-S2 and S1-S3 pairs,
because different sources may have different noise patterns.
"""

import numpy as np
import pandas as pd
import os

ARTIFACTS_DIR = "artifacts"
THRESHOLDS_TO_TRY = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
RANDOM_STATE = 42


def apply_threshold(scored_pairs: pd.DataFrame, threshold: float,
                    score_col: str = "score") -> dict:
    """
    Convert scored pairs to final predictions using threshold.
    
    For each S1 entity: collect all candidates with score >= threshold.
    If none meet the threshold, predict empty (singleton).
    
    Args:
        scored_pairs: DataFrame with s1_id, cand_id, score columns
        threshold: cutoff probability
        score_col: name of the score column
    
    Returns:
        { s1_id: set(predicted_match_ids) }
    """
    predictions = {}
    for s1_id, group in scored_pairs.groupby("s1_id"):
        above = group[group[score_col] >= threshold]["cand_id"].tolist()
        predictions[s1_id] = set(above)
    return predictions


def optimize_threshold(scored_pairs: pd.DataFrame,
                        ground_truth: dict,
                        all_s1_ids: list,
                        thresholds: list = None,
                        score_col: str = "score") -> dict:
    """
    Find the threshold that maximizes macro F0.5 on the validation set.
    
    Also evaluates per-source thresholds (S1-S2 vs S1-S3).
    
    Args:
        scored_pairs: DataFrame with s1_id, cand_id, score, and entity prefix info
        ground_truth: validation ground truth dict
        all_s1_ids: list of ALL S1 IDs (including those with no candidates)
        thresholds: list of thresholds to try
        score_col: score column name
    
    Returns:
        dict with 'best_threshold', 'best_f05', 'results_df', etc.
    """
    from evaluation import macro_f05, evaluate_predictions
    
    if thresholds is None:
        thresholds = THRESHOLDS_TO_TRY
    
    results = []
    
    print(f"  Testing {len(thresholds)} thresholds...")
    
    for thresh in thresholds:
        preds = apply_threshold(scored_pairs, thresh, score_col)
        
        # Ensure ALL S1 IDs have a prediction (even if empty)
        for s1_id in all_s1_ids:
            if s1_id not in preds:
                preds[s1_id] = set()
        
        metrics = evaluate_predictions(ground_truth, preds)
        
        results.append({
            "threshold": thresh,
            "macro_f05": metrics.get("macro_f05", 0.0),
            "micro_precision": metrics.get("micro_precision", 0.0),
            "micro_recall": metrics.get("micro_recall", 0.0),
            "false_merges": metrics.get("false_merge_entities", 0),
            "singleton_accuracy": metrics.get("singleton_accuracy", 0.0),
            "tp": metrics.get("true_positives", 0),
            "fp": metrics.get("false_positives", 0),
            "fn": metrics.get("false_negatives", 0),
        })
        
        print(f"    threshold={thresh:.2f}  "
              f"F0.5={results[-1]['macro_f05']:.4f}  "
              f"P={results[-1]['micro_precision']:.3f}  "
              f"R={results[-1]['micro_recall']:.3f}  "
              f"false_merges={results[-1]['false_merges']:,}")
    
    results_df = pd.DataFrame(results).sort_values("macro_f05", ascending=False)
    best_row   = results_df.iloc[0]
    best_thresh = best_row["threshold"]
    best_f05    = best_row["macro_f05"]
    
    print(f"\n  BEST threshold: {best_thresh:.2f}  (F0.5 = {best_f05:.4f})")
    
    # Save results
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(ARTIFACTS_DIR, "threshold_optimization.csv"), index=False)
    
    return {
        "best_threshold": best_thresh,
        "best_f05": best_f05,
        "results_df": results_df,
    }
