"""
Evaluation metric for Business Entity Resolution.
Computes macro-averaged F_0.5 per Source 1 entity, strictly following competition specification.
"""

from typing import Dict, List, Set, Union


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    Compute F_0.5 for a single Source 1 entity:
    - If true is empty (singleton):
        returns 1.0 if pred is empty, else 0.0
    - If true is non-empty:
        returns 0.0 if pred is empty
        otherwise F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not true_set:
        return 1.0 if not pred_set else 0.0
    
    if not pred_set:
        return 0.0
    
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0
    
    p = tp / len(pred_set)
    r = tp / len(true_set)
    
    denom = 0.25 * p + r
    if denom == 0.0:
        return 0.0
    
    return (1.25 * p * r) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Union[Set[str], List[str]]],
    predictions: Dict[str, Union[Set[str], List[str]]]
) -> float:
    """
    Compute macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    Every entity in ground_truth must be accounted for (defaulting to empty predictions if missing).
    """
    if not ground_truth:
        return 0.0
    
    total_f05 = 0.0
    for s1_id, true_matches in ground_truth.items():
        true_set = set(true_matches) if not isinstance(true_matches, set) else true_matches
        pred_matches = predictions.get(s1_id, set())
        pred_set = set(pred_matches) if not isinstance(pred_matches, set) else pred_matches
        
        f05 = compute_entity_f05(true_set, pred_set)
        total_f05 += f05
        
    return total_f05 / len(ground_truth)
