"""
Validation split creation script.
Creates a reproducible, representative validation subset from train:
- 5,000 Source 1 entities (including ~5.6% singletons, stratified by country)
- All their true ground truth matches from S2 and S3
- An additional 100,000 distractor records from S2 and S3 to test blocking selectivity and recall.
"""

import os
import random
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))

VAL_DIR = "dataset/validation_split"


def create_validation_split(num_s1: int = 5000, num_distractors: int = 100000, seed: int = 42):
    random.seed(seed)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    print(f"Creating validation split with {num_s1} S1 entities and {num_distractors} distractors...")
    
    # 1. Read balanced sample of US and India
    print("Loading S1 records...")
    # Read 3000 US and 2000 India from train_source1
    us_records = []
    india_records = []
    
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        header_line = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                country = parts[3]
                if country == "US" and len(us_records) < 3000:
                    us_records.append(parts)
                elif country == "India" and len(india_records) < 2000:
                    india_records.append(parts)
            if len(us_records) >= 3000 and len(india_records) >= 2000:
                break
                
    all_s1 = us_records + india_records
    val_s1 = pd.DataFrame(all_s1, columns=header_line)
    val_s1_ids = set(val_s1["entity_id"])
    print(f"Sampled {len(val_s1)} S1 entities. Country distribution:")
    print(val_s1["country"].value_counts())
    
    # 2. Extract ground truth for these S1 entities
    print("Extracting ground truth for sampled S1 entities...")
    val_gt = {}
    target_ids = set()
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2 and parts[0] in val_s1_ids:
                matches = parts[1].split(",") if parts[1] else []
                val_gt[parts[0]] = matches
                target_ids.update(matches)
                
    # Fill in any S1 that had no row or was missing
    for s1_id in val_s1_ids:
        if s1_id not in val_gt:
            val_gt[s1_id] = []
            
    num_singletons = sum(1 for m in val_gt.values() if not m)
    total_true_matches = sum(len(m) for m in val_gt.values())
    print(f"Ground truth extracted: {len(val_gt)} entities, {num_singletons} singletons ({num_singletons/len(val_gt)*100:.2f}%), {total_true_matches} total true matches ({total_true_matches/(len(val_gt)-num_singletons):.2f} avg matches/non-singleton).")
    
    # Save validation ground truth
    gt_df = pd.DataFrame([
        {"source1_entity_id": k, "matched_entity_ids": ",".join(v)}
        for k, v in val_gt.items()
    ])
    gt_df.to_csv(os.path.join(VAL_DIR, "val_ground_truth.tsv"), sep="\t", index=False)
    val_s1.to_csv(os.path.join(VAL_DIR, "val_source1.tsv"), sep="\t", index=False)
    
    # 3. Collect S2 and S3: all target IDs + random distractors
    s2_targets = {tid for tid in target_ids if tid.startswith("S2-")}
    s3_targets = {tid for tid in target_ids if tid.startswith("S3-")}
    print(f"Target IDs: {len(s2_targets)} in S2, {len(s3_targets)} in S3.")
    
    print("Scanning S2 for targets and distractors...")
    s2_matched = []
    s2_distractors = []
    num_distractor_per_source = num_distractors // 2
    
    with open("dataset/train/train_source2.tsv", "r", encoding="utf-8") as f:
        h = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in s2_targets:
                s2_matched.append(parts)
            elif len(s2_distractors) < num_distractor_per_source:
                s2_distractors.append(parts)
            if len(s2_matched) == len(s2_targets) and len(s2_distractors) >= num_distractor_per_source:
                break
                
    print("Scanning S3 for targets and distractors...")
    s3_matched = []
    s3_distractors = []
    with open("dataset/train/train_source3.tsv", "r", encoding="utf-8") as f:
        h = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            eid = parts[0]
            if eid in s3_targets:
                s3_matched.append(parts)
            elif len(s3_distractors) < num_distractor_per_source:
                s3_distractors.append(parts)
            if len(s3_matched) == len(s3_targets) and len(s3_distractors) >= num_distractor_per_source:
                break
                
    s2_all = s2_matched + s2_distractors
    s3_all = s3_matched + s3_distractors
    print(f"Total S2 records for validation: {len(s2_all)} (Targets: {len(s2_matched)}/{len(s2_targets)})")
    print(f"Total S3 records for validation: {len(s3_all)} (Targets: {len(s3_matched)}/{len(s3_targets)})")
    
    val_s2 = pd.DataFrame(s2_all, columns=h)
    val_s3 = pd.DataFrame(s3_all, columns=h)
    val_s2.to_csv(os.path.join(VAL_DIR, "val_source2.tsv"), sep="\t", index=False)
    val_s3.to_csv(os.path.join(VAL_DIR, "val_source3.tsv"), sep="\t", index=False)
    print("Validation split successfully created in:", VAL_DIR)


if __name__ == "__main__":
    create_validation_split(num_s1=5000, num_distractors=100000, seed=42)
