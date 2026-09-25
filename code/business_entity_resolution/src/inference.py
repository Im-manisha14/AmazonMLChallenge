"""
inference.py - Test Set Inference and Output Generation
========================================================
After training on the full training data, this module:
1. Loads the test data (S1, S2, S3 from the test split)
2. Generates candidates using the same blocking strategy
3. Scores candidates using the trained model
4. Applies the chosen threshold
5. Writes output/matching_results.tsv and output/candidate_pairs.tsv

Output format rules:
  - Tab-separated
  - One row per test S1 entity (even singletons)
  - matched_entity_ids: comma-separated S2/S3 IDs, or empty
  - No S1 IDs in matches, no duplicates, no missing S1 entities
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = "output"
ARTIFACTS_DIR = "artifacts"


def write_matching_results(predictions: dict, all_s1_ids: list,
                            output_path: str = "output/matching_results.tsv"):
    """
    Write matching_results.tsv.
    
    Rules:
    - Exactly one row per S1 entity
    - No duplicate S1 rows
    - No S1 IDs in matched_entity_ids
    - Only S2/S3 IDs
    - No duplicate matched IDs within a row
    - Empty for singletons
    - Tab-separated
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    seen_s1 = set()
    
    for s1_id in all_s1_ids:
        if s1_id in seen_s1:
            continue  # Skip duplicate S1 IDs (should not happen but be safe)
        seen_s1.add(s1_id)
        
        matched = predictions.get(s1_id, set())
        # Clean: remove S1 IDs, keep only S2/S3, deduplicate
        matched_clean = sorted(set(
            mid for mid in matched
            if mid.startswith("S2-") or mid.startswith("S3-")
        ))
        
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_clean) if matched_clean else "",
        })
    
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    
    n_non_empty = sum(1 for r in rows if r["matched_entity_ids"])
    n_empty     = len(rows) - n_non_empty
    print(f"    Written {len(rows):,} rows to {output_path}")
    print(f"    Non-empty (predicted matches): {n_non_empty:,}")
    print(f"    Empty (singletons predicted):  {n_empty:,}")
    
    return df


def write_candidate_pairs(candidates: dict, all_s1_ids: list,
                           output_path: str = "output/candidate_pairs.tsv"):
    """
    Write candidate_pairs.tsv.
    
    This is the candidate set that was fed into the ML model.
    Every ID in matching_results MUST also appear here.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    seen_s1 = set()
    
    for s1_id in all_s1_ids:
        if s1_id in seen_s1:
            continue
        seen_s1.add(s1_id)
        
        cands = candidates.get(s1_id, set())
        cands_clean = sorted(set(
            c for c in cands
            if c.startswith("S2-") or c.startswith("S3-")
        ))
        
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands_clean) if cands_clean else "",
        })
    
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    df.to_csv(output_path, sep="\t", index=False)
    
    n_with_cands = sum(1 for r in rows if r["candidate_entity_ids"])
    print(f"    Written {len(rows):,} rows to {output_path}")
    print(f"    S1 entities with candidates: {n_with_cands:,}")
    
    return df


def validate_submission_locally(matching_path: str, candidate_path: str,
                                 test_dir: str = "dataset/test"):
    """
    Run the official submission validator.
    Returns True if PASS, False if FAIL.
    """
    import subprocess
    validator_path = os.path.join("utils", "validate_submission.py")
    
    if not os.path.exists(validator_path):
        print("  WARNING: Validator not found at utils/validate_submission.py")
        return False
    
    print(f"\n  Running official submission validator...")
    result = subprocess.run(
        [sys.executable, validator_path,
         "--matching", matching_path,
         "--candidate", candidate_path,
         "--test-dir", test_dir],
        capture_output=True, text=True, encoding="utf-8"
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    passed = result.returncode == 0
    if passed:
        print("  VALIDATOR STATUS: PASS")
    else:
        print("  VALIDATOR STATUS: FAIL")
    return passed


def save_validation_predictions(scored_pairs: pd.DataFrame,
                                  ground_truth: dict,
                                  predictions: dict,
                                  threshold: float,
                                  output_path: str = "artifacts/validation_predictions.tsv"):
    """
    Save validation predictions with scores for debugging.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    for _, row in scored_pairs.iterrows():
        s1_id   = row["s1_id"]
        cand_id = row["cand_id"]
        score   = row.get("score", 0.0)
        true_matches = ground_truth.get(s1_id, set())
        true_label   = 1 if cand_id in true_matches else 0
        pred_label   = 1 if score >= threshold else 0
        rows.append({
            "source1_entity_id":  s1_id,
            "candidate_entity_id": cand_id,
            "true_label": true_label,
            "model_score": score,
            "predicted_label": pred_label,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"    Validation predictions saved to {output_path}")
    return df
