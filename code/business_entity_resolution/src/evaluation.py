"""
evaluation.py - Metrics and Evaluation
=======================================
Competition metric: Macro-averaged F0.5 score.

F0.5 = precision-heavy variant of F1:
  F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
  
With beta=0.5:
  F_0.5 = 1.25 * P * R / (0.25 * P + R)

This weights precision 2x more than recall because:
- Merging two different businesses (false positive) = VERY BAD
- Missing a link (false negative) = less bad

Macro-average = compute F0.5 per S1 entity, then average across ALL entities.
Singletons (no true matches): score 1.0 if predicted empty, 0.0 otherwise.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, Union


def f05_score(true_set: set, pred_set: set) -> float:
    """
    Compute F0.5 for a single S1 entity.
    
    Examples:
      true={S2-001, S3-002}, pred={S2-001, S3-002}  -> 1.0 (perfect)
      true={S2-001, S3-002}, pred={S2-001}           -> 0.83 (miss S3-002)
      true={S2-001, S3-002}, pred={S2-001,S2-999}    -> 0.71 (false merge S2-999)
      true={}(singleton),    pred={}                  -> 1.0 (correct singleton)
      true={}(singleton),    pred={S2-001}            -> 0.0 (wrong merge)
    """
    # Singleton case
    if not true_set:
        return 1.0 if not pred_set else 0.0
    
    # No prediction case
    if not pred_set:
        return 0.0
    
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_set)
    recall    = tp / len(true_set)
    
    beta2 = 0.25  # beta^2 for F0.5
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    
    return (1 + beta2) * precision * recall / denom


def macro_f05(ground_truth: dict, predictions: dict) -> float:
    """
    Compute macro-averaged F0.5 across all S1 entities in ground_truth.
    
    Args:
        ground_truth: { s1_id: set(true_match_ids) }
        predictions: { s1_id: set(predicted_match_ids) }
    """
    if not ground_truth:
        return 0.0
    
    total = 0.0
    for s1_id, true_set in ground_truth.items():
        true_s = set(true_set) if not isinstance(true_set, set) else true_set
        pred_s = set(predictions.get(s1_id, set()))
        total += f05_score(true_s, pred_s)
    
    return total / len(ground_truth)


def evaluate_predictions(ground_truth: dict, predictions: dict,
                          candidates: dict = None) -> dict:
    """
    Full evaluation: F0.5, precision, recall, singleton metrics.
    
    Returns a dict of metric_name -> value.
    """
    if not ground_truth:
        return {}
    
    f05_scores = []
    all_tp = all_fp = all_fn = 0
    singleton_correct = singleton_total = 0
    false_merge_count = 0
    
    for s1_id, true_set in ground_truth.items():
        true_s = set(true_set) if not isinstance(true_set, set) else true_set
        pred_s = set(predictions.get(s1_id, set()))
        
        f05_scores.append(f05_score(true_s, pred_s))
        
        tp = len(true_s & pred_s)
        fp = len(pred_s - true_s)
        fn = len(true_s - pred_s)
        
        all_tp += tp
        all_fp += fp
        all_fn += fn
        
        if not true_s:
            singleton_total += 1
            if not pred_s:
                singleton_correct += 1
        
        if fp > 0:
            false_merge_count += 1
    
    macro = np.mean(f05_scores)
    micro_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    micro_r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    
    singleton_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    
    metrics = {
        "macro_f05":       macro,
        "micro_precision": micro_p,
        "micro_recall":    micro_r,
        "true_positives":  all_tp,
        "false_positives": all_fp,
        "false_negatives": all_fn,
        "singleton_total":   singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_accuracy": singleton_acc,
        "false_merge_entities": false_merge_count,
        "total_entities": len(ground_truth),
    }
    
    if candidates is not None:
        # Candidate recall: what % of true matches are in candidates?
        total_true = 0
        total_found = 0
        for s1_id, true_s in ground_truth.items():
            if not true_s:
                continue
            cands = candidates.get(s1_id, set())
            total_true += len(true_s)
            total_found += len(set(true_s) & set(cands))
        metrics["candidate_recall"] = total_found / total_true if total_true > 0 else 1.0
    
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    """Print a formatted metrics report."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    f05 = metrics.get("macro_f05", 0)
    prec = metrics.get("micro_precision", 0)
    rec  = metrics.get("micro_recall", 0)
    cand_recall = metrics.get("candidate_recall", None)
    
    print(f"  Macro F0.5 (competition metric): {f05:.4f}")
    print(f"  Micro Precision:                 {prec:.4f}")
    print(f"  Micro Recall:                    {rec:.4f}")
    
    if cand_recall is not None:
        print(f"  Candidate Recall:                {cand_recall:.4f}  "
              f"(upper bound on recall)")
    
    print(f"\n  True Positives:  {metrics.get('true_positives',0):>8,}")
    print(f"  False Positives: {metrics.get('false_positives',0):>8,}  "
          f"(false merges)")
    print(f"  False Negatives: {metrics.get('false_negatives',0):>8,}  "
          f"(missed matches)")
    
    n_sing  = metrics.get("singleton_total", 0)
    n_sacc  = metrics.get("singleton_correct", 0)
    s_acc   = metrics.get("singleton_accuracy", 1.0)
    print(f"\n  Singletons total:   {n_sing:>6,}")
    print(f"  Singleton correct:  {n_sacc:>6,}")
    print(f"  Singleton accuracy: {s_acc:.4f}  "
          f"(fraction of no-match entities correctly left empty)")
    
    n_fm = metrics.get("false_merge_entities", 0)
    print(f"\n  Entities with >= 1 false merge: {n_fm:,}")
    
    print(f"{'='*60}")
