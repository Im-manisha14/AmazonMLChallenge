"""
error_analysis.py - Step 9: Error Analysis
===========================================
Identify patterns in prediction errors to improve the pipeline.

Types of errors:
  1. False Positives: We predicted a match, but they are different businesses
     -> These hurt precision (and more heavily F0.5)
     -> Common cause: similar names in same country (e.g. chain stores)
  
  2. False Negatives: We missed a true match
     -> These hurt recall
     -> Common cause: spelling variations, transliterations we didn't normalize
  
  3. Hard Negatives: Non-matching pairs with high model scores
     -> These are candidates for false positives
  
  4. Singleton False Positives: We predicted a match for a singleton entity
     -> These score 0.0 on that entity (very damaging)
"""

import pandas as pd
import numpy as np
import os

ARTIFACTS_DIR = "artifacts"


def run_error_analysis(val_s1_df: pd.DataFrame,
                       cand_df: pd.DataFrame,
                       scored_pairs: pd.DataFrame,
                       ground_truth: dict,
                       predictions: dict,
                       threshold: float,
                       max_examples: int = 10) -> pd.DataFrame:
    """
    Analyze prediction errors for validation set.
    
    Returns a DataFrame with error analysis, saved to artifacts/error_analysis.csv.
    
    Args:
        val_s1_df: Validation S1 DataFrame (preprocessed)
        cand_df: S2+S3 candidate DataFrame (preprocessed)
        scored_pairs: DataFrame with s1_id, cand_id, score
        ground_truth: { s1_id: set(true_match_ids) }
        predictions: { s1_id: set(predicted_ids) }
        threshold: decision threshold used
        max_examples: how many error examples to print
    """
    s1_lookup   = val_s1_df.set_index("entity_id").to_dict("index")
    cand_lookup = cand_df.set_index("entity_id").to_dict("index")
    
    # Merge scores with ground truth
    rows = []
    for _, row in scored_pairs.iterrows():
        s1_id   = row["s1_id"]
        cand_id = row["cand_id"]
        score   = row["score"]
        
        true_matches = ground_truth.get(s1_id, set())
        true_label   = 1 if cand_id in true_matches else 0
        pred_label   = 1 if score >= threshold else 0
        
        error_type = "correct"
        if true_label == 1 and pred_label == 0:
            error_type = "false_negative"
        elif true_label == 0 and pred_label == 1:
            is_singleton = (len(true_matches) == 0)
            error_type = "fp_on_singleton" if is_singleton else "false_positive"
        
        rows.append({
            "s1_id": s1_id,
            "cand_id": cand_id,
            "score": score,
            "true_label": true_label,
            "pred_label": pred_label,
            "error_type": error_type,
        })
    
    err_df = pd.DataFrame(rows)
    
    # Stats
    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS (threshold={threshold:.2f})")
    print(f"{'='*60}")
    
    type_counts = err_df["error_type"].value_counts()
    for etype, cnt in type_counts.items():
        print(f"  {etype:25s}: {cnt:>8,}")
    
    # Save
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    err_df.to_csv(os.path.join(ARTIFACTS_DIR, "error_analysis.csv"), index=False)
    print(f"\n  Error analysis saved to artifacts/error_analysis.csv")
    
    # Print false positive examples
    fps = err_df[err_df["error_type"] == "false_positive"].nlargest(max_examples, "score")
    if len(fps) > 0:
        print(f"\n  Top {min(max_examples, len(fps))} FALSE POSITIVES (predicted match, actually different):")
        for _, fp in fps.iterrows():
            s1 = s1_lookup.get(fp["s1_id"], {})
            cd = cand_lookup.get(fp["cand_id"], {})
            print(f"  ────────────────────────────────────────")
            print(f"  S1:   {s1.get('business_name','?')[:50]}")
            print(f"        {s1.get('business_address','?')[:50]}")
            print(f"        [{s1.get('country_normalized','?')}]")
            print(f"  CAND: {cd.get('business_name','?')[:50]}")
            print(f"        {cd.get('business_address','?')[:50]}")
            print(f"        [{cd.get('country_normalized','?')}]")
            print(f"  Score: {fp['score']:.3f} | ID: {fp['cand_id']}")
    
    # Print false negative examples
    fns = err_df[err_df["error_type"] == "false_negative"].nsmallest(max_examples, "score")
    if len(fns) > 0:
        print(f"\n  Top {min(max_examples, len(fns))} FALSE NEGATIVES (missed true matches):")
        for _, fn in fns.iterrows():
            s1 = s1_lookup.get(fn["s1_id"], {})
            cd = cand_lookup.get(fn["cand_id"], {})
            print(f"  ────────────────────────────────────────")
            print(f"  S1:   {s1.get('business_name','?')[:50]}")
            print(f"        {s1.get('business_address','?')[:50]}")
            print(f"        [{s1.get('country_normalized','?')}]")
            print(f"  CAND: {cd.get('business_name','?')[:50]}")
            print(f"        {cd.get('business_address','?')[:50]}")
            print(f"        [{cd.get('country_normalized','?')}]")
            print(f"  Score: {fn['score']:.3f} | ID: {fn['cand_id']}")
    
    return err_df
