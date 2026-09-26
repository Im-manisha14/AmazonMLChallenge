"""
create_validation_benchmark.py - Phase 2 Disjoint Grouped Validation Benchmark
=============================================================================
Creates:
- Validation A: 10,000 S1 entities (stratified by US/India)
- Validation B: 3,000 independent S1 entities (stratified by US/India)
Disjoint from each other and training.
"""

import sys
import time
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path("c:/ml challenge/student_resource")
TRAIN_DIR = BASE_DIR / "dataset/train"
BENCH_DIR = BASE_DIR / "benchmark"

print("=" * 70)
print("  PHASE 2: BUILDING DISJOINT GROUPED VALIDATION BENCHMARKS")
print("=" * 70)

t0 = time.time()

# 1. Load train S1
print("[1] Loading train_source1.tsv...")
s1_df = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
print(f"  Loaded {len(s1_df):,} Source 1 entities.")
print(f"  Country distribution:\n{s1_df['country'].value_counts()}")

# 2. Load Ground Truth
print("\n[2] Loading train_ground_truth.tsv...")
gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt_map = {}
for sid, matches in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
    m_clean = str(matches).strip()
    if m_clean:
        gt_map[sid] = set(m_clean.split(","))
    else:
        gt_map[sid] = set()

# 3. Stratified Sample for Val A (10,000) and Val B (3,000)
print("\n[3] Splitting into Validation A (10k) and Validation B (3k)...")
us_s1 = s1_df[s1_df["country"].str.lower().str.strip() == "us"]["entity_id"].tolist()
in_s1 = s1_df[s1_df["country"].str.lower().str.strip() == "india"]["entity_id"].tolist()

random.shuffle(us_s1)
random.shuffle(in_s1)

# Proportions: ~60% US, ~40% India
val_a_us = us_s1[:6_000]
val_a_in = in_s1[:4_000]
val_a_ids = set(val_a_us + val_a_in)

val_b_us = us_s1[6_000:7_800]
val_b_in = in_s1[4_000:5_200]
val_b_ids = set(val_b_us + val_b_in)

# Verify disjoint
assert len(val_a_ids & val_b_ids) == 0, "Error: Val A and Val B overlap!"
assert len(val_a_ids) == 10_000, f"Expected 10,000 in Val A, got {len(val_a_ids)}"
assert len(val_b_ids) == 3_000, f"Expected 3,000 in Val B, got {len(val_b_ids)}"

val_a_df = s1_df[s1_df["entity_id"].isin(val_a_ids)].reset_index(drop=True)
val_b_df = s1_df[s1_df["entity_id"].isin(val_b_ids)].reset_index(drop=True)

val_a_gt_df = gt_df[gt_df["source1_entity_id"].isin(val_a_ids)].reset_index(drop=True)
val_b_gt_df = gt_df[gt_df["source1_entity_id"].isin(val_b_ids)].reset_index(drop=True)

# Save TSVs
val_a_df.to_csv(BENCH_DIR / "val_a_s1.tsv", sep="\t", index=False)
val_b_df.to_csv(BENCH_DIR / "val_b_s1.tsv", sep="\t", index=False)
val_a_gt_df.to_csv(BENCH_DIR / "val_a_gt.tsv", sep="\t", index=False)
val_b_gt_df.to_csv(BENCH_DIR / "val_b_gt.tsv", sep="\t", index=False)

def report_stats(name, df, gt_subset_df):
    total = len(df)
    singles = sum(1 for m in gt_subset_df["matched_entity_ids"] if not str(m).strip())
    matches = sum(len(str(m).strip().split(",")) for m in gt_subset_df["matched_entity_ids"] if str(m).strip())
    us_cnt = sum(df["country"].str.lower().str.strip() == "us")
    in_cnt = sum(df["country"].str.lower().str.strip() == "india")
    print(f"\n  --- {name} Statistics ---")
    print(f"  Total Entities:     {total:,}")
    print(f"  US / India:         {us_cnt:,} ({us_cnt/total*100:.1f}%) / {in_cnt:,} ({in_cnt/total*100:.1f}%)")
    print(f"  Total True Matches: {matches:,} (Avg {matches/total:.2f} per S1)")
    print(f"  Singletons (0 match): {singles:,} ({singles/total*100:.2f}%)")
    print(f"  Non-singletons:     {total - singles:,} ({(total - singles)/total*100:.2f}%)")

report_stats("Validation A (Primary Benchmark)", val_a_df, val_a_gt_df)
report_stats("Validation B (Confirmation Holdout)", val_b_df, val_b_gt_df)

print("\n" + "=" * 70)
print(f"  BENCHMARK SPLITS CREATED IN {time.time()-t0:.1f}s")
print("=" * 70)
