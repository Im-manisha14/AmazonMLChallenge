"""
data_loader.py - Step 1: Data Loading and Validation
=====================================================
Loads all TSV files with strict tab separation and validates data quality.

Why TSV?
  Business addresses contain commas (e.g. "123 Main St, Springfield, IL").
  If we used CSV, pandas would split addresses into multiple columns.
  TSV (tab-separated) avoids this problem entirely.
"""

import os
import sys
import pandas as pd
import numpy as np

# Paths (relative to student_resource/)
TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"

TRAIN_S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_S1_PATH  = os.path.join(TEST_DIR,  "test_source1.tsv")
TEST_S2_PATH  = os.path.join(TEST_DIR,  "test_source2.tsv")
TEST_S3_PATH  = os.path.join(TEST_DIR,  "test_source3.tsv")

REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]


def _load_tsv(path: str, expected_prefix: str = None) -> pd.DataFrame:
    """
    Load a TSV file with UTF-8 encoding. Validates columns and optionally
    checks that entity_id values start with the expected prefix (S1-, S2-, S3-).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")

    df.replace("", np.nan, inplace=True)

    if expected_prefix:
        valid = df["entity_id"].dropna()
        bad = valid[~valid.str.startswith(expected_prefix)]
        if len(bad) > 0:
            print(f"  WARNING: {len(bad)} entity_ids in {os.path.basename(path)} "
                  f"do not start with '{expected_prefix}'. Sample: {bad.head(3).tolist()}")

    return df


def load_source1(split: str = "train") -> pd.DataFrame:
    """Load Source 1 (reference source) for a given split."""
    path = TRAIN_S1_PATH if split == "train" else TEST_S1_PATH
    return _load_tsv(path, expected_prefix="S1-")


def load_source2(split: str = "train") -> pd.DataFrame:
    """Load Source 2 for a given split."""
    path = TRAIN_S2_PATH if split == "train" else TEST_S2_PATH
    return _load_tsv(path, expected_prefix="S2-")


def load_source3(split: str = "train") -> pd.DataFrame:
    """Load Source 3 for a given split."""
    path = TRAIN_S3_PATH if split == "train" else TEST_S3_PATH
    return _load_tsv(path, expected_prefix="S3-")


def load_ground_truth(split: str = "train") -> pd.DataFrame:
    """Load the ground truth file (train only)."""
    if split != "train":
        raise ValueError("Ground truth is only available for the train split.")
    if not os.path.exists(TRAIN_GT_PATH):
        raise FileNotFoundError(f"Ground truth file not found: {TRAIN_GT_PATH}")
    df = pd.read_csv(TRAIN_GT_PATH, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """
    Convert ground truth DataFrame into a dict:
      { "S1-XXXXX": set(["S2-YYY", "S3-ZZZ"]) }
    Singletons have empty sets.
    """
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        if pd.isna(raw) or str(raw).strip() == "":
            gt_dict[s1_id] = set()
        else:
            gt_dict[s1_id] = set(str(raw).split(","))
    return gt_dict


def print_source_stats(df: pd.DataFrame, name: str):
    """Print a detailed summary of a source DataFrame."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    missing = df.isnull().sum()
    print(f"\n  Missing values:")
    for col, cnt in missing.items():
        pct = 100.0 * cnt / len(df)
        print(f"    {col:30s}: {cnt:>8,}  ({pct:.2f}%)")

    n_unique = df["entity_id"].nunique()
    n_total  = len(df)
    dup_msg = "OK" if n_unique == n_total else f"WARNING: {n_total - n_unique:,} DUPLICATES!"
    print(f"\n  Unique entity_ids: {n_unique:,} / {n_total:,}  ({dup_msg})")

    if "country" in df.columns:
        print(f"\n  Country distribution:")
        for country, cnt in df["country"].value_counts(dropna=False).items():
            pct = 100.0 * cnt / len(df)
            print(f"    {str(country):20s}: {cnt:>8,}  ({pct:.2f}%)")

    dup_names = df["business_name"].dropna().duplicated().sum()
    dup_addr  = df["business_address"].dropna().duplicated().sum()
    print(f"\n  Duplicate business_name values: {dup_names:,}")
    print(f"  Duplicate business_address values: {dup_addr:,}")

    print(f"\n  Sample records (first 3):")
    for _, row in df.head(3).iterrows():
        print(f"    ID={str(row['entity_id'])[:20]!r}  name={str(row['business_name'])[:40]!r}  "
              f"country={str(row['country'])!r}")


def validate_ground_truth(gt_df: pd.DataFrame, s1_df: pd.DataFrame,
                          s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Sanity-check the ground truth against source files."""
    print(f"\n{'='*60}")
    print(f"  GROUND TRUTH VALIDATION")
    print(f"{'='*60}")

    s1_ids = set(s1_df["entity_id"].dropna())
    s2_ids = set(s2_df["entity_id"].dropna())
    s3_ids = set(s3_df["entity_id"].dropna())
    gt_s1_ids = set(gt_df["source1_entity_id"].dropna())

    missing_gt = s1_ids - gt_s1_ids
    extra_gt   = gt_s1_ids - s1_ids
    print(f"  S1 entities:          {len(s1_ids):,}")
    print(f"  GT rows:              {len(gt_s1_ids):,}")
    print(f"  S1 missing from GT:   {len(missing_gt):,}")
    print(f"  GT rows not in S1:    {len(extra_gt):,}")

    singletons = 0
    total_matches = 0
    for _, row in gt_df.iterrows():
        raw = str(row.get("matched_entity_ids", "")).strip()
        if raw == "" or raw == "nan":
            singletons += 1
        else:
            total_matches += len(raw.split(","))

    non_singletons = len(gt_df) - singletons
    print(f"\n  Singletons (no match):     {singletons:,}  ({100.0*singletons/len(gt_df):.2f}%)")
    print(f"  Non-singletons:            {non_singletons:,}  ({100.0*non_singletons/len(gt_df):.2f}%)")
    print(f"  Total true match links:    {total_matches:,}")
    if non_singletons > 0:
        print(f"  Avg matches/non-singleton: {total_matches/non_singletons:.2f}")


def inspect_all(split: str = "train"):
    """Load and print statistics for all files in a split."""
    print(f"\n[DATA INSPECTION] Split = {split.upper()}")
    s1 = load_source1(split)
    s2 = load_source2(split)
    s3 = load_source3(split)
    print_source_stats(s1, f"Source 1 ({split})")
    print_source_stats(s2, f"Source 2 ({split})")
    print_source_stats(s3, f"Source 3 ({split})")
    if split == "train":
        gt = load_ground_truth(split)
        validate_ground_truth(gt, s1, s2, s3)
    return s1, s2, s3


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    inspect_all(split)
