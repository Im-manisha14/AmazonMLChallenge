"""
model.py - Step 6/7: Matching Model + Training Labels
======================================================
We frame entity resolution as a BINARY CLASSIFICATION problem:
  - Input: feature vector for a candidate pair (S1, S2/S3)
  - Output: P(match) = probability that they refer to the same real business

Why LightGBM?
  - Fast on tabular/feature data
  - Handles class imbalance well
  - License: MIT (fully compliant with competition requirements)
  - ~zero parameters in the traditional DL sense (tree ensemble)
  - Works well even with thousands of training examples

Training Labels:
  - Positive pairs: (S1_id, S2/S3_id) pairs from ground truth
  - Negative pairs: (S1_id, S2/S3_id) candidate pairs NOT in ground truth
    - Easy negatives: random non-matching candidates
    - Hard negatives: candidates with HIGH similarity but WRONG answer
      (These are the dangerous cases that cause false merges!)
"""

import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


ARTIFACTS_DIR    = "artifacts"
MODELS_DIR       = os.path.join(ARTIFACTS_DIR, "models")
MODEL_PATH       = os.path.join(MODELS_DIR, "lgbm_matcher.pkl")
HARD_NEG_RATIO   = 3   # For each positive pair, sample 3 hard negatives
EASY_NEG_RATIO   = 2   # ... plus 2 easy negatives
RANDOM_STATE     = 42


# ─── Label Creation ──────────────────────────────────────────────────────────

def create_training_labels(feat_df: pd.DataFrame, ground_truth: dict,
                            candidates: dict) -> pd.DataFrame:
    """
    Add a 'label' column to the feature DataFrame:
      1 = true match (in ground truth)
      0 = not a match (in candidates but not in ground truth)
    
    Args:
        feat_df: feature matrix with s1_id and cand_id columns
        ground_truth: { s1_id: set(true_match_ids) }
        candidates: { s1_id: set(candidate_ids) } (for reference)
    
    Returns:
        Feature DataFrame with 'label' column added.
    """
    labels = []
    for _, row in feat_df.iterrows():
        s1_id   = row["s1_id"]
        cand_id = row["cand_id"]
        true_matches = ground_truth.get(s1_id, set())
        labels.append(1 if cand_id in true_matches else 0)
    
    feat_df = feat_df.copy()
    feat_df["label"] = labels
    
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_rate = 100.0 * n_pos / len(labels) if labels else 0.0
    print(f"    Labels: {n_pos:,} positive ({pos_rate:.2f}%)  {n_neg:,} negative")
    return feat_df


def sample_for_training(feat_df: pd.DataFrame,
                        hard_neg_ratio: int = HARD_NEG_RATIO,
                        easy_neg_ratio: int = EASY_NEG_RATIO,
                        random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Create a balanced training set with hard negatives.
    
    Hard negatives = negative pairs with HIGH similarity scores.
    These are the most important examples because:
    - They look like matches but aren't (the model needs to learn to reject them)
    - F0.5 heavily penalizes false positives, so getting these right is crucial
    
    Strategy:
    1. Keep all positive pairs
    2. Sort negatives by 'combined_sim' (descending)
    3. Take top-N as hard negatives
    4. Take random sample as easy negatives
    """
    pos = feat_df[feat_df["label"] == 1]
    neg = feat_df[feat_df["label"] == 0]
    
    n_pos = len(pos)
    n_hard = min(n_pos * hard_neg_ratio, len(neg))
    n_easy = min(n_pos * easy_neg_ratio, len(neg) - n_hard)
    
    print(f"    Sampling: {n_pos:,} positives + {n_hard:,} hard negatives + {n_easy:,} easy negatives")
    
    # Hard negatives: highest combined similarity that is actually non-matching
    if "combined_sim" in neg.columns:
        neg_sorted = neg.sort_values("combined_sim", ascending=False)
    else:
        neg_sorted = neg
    
    hard_negs = neg_sorted.head(n_hard)
    remaining = neg_sorted.iloc[n_hard:]
    easy_negs = remaining.sample(min(n_easy, len(remaining)), random_state=random_state)
    
    sampled = pd.concat([pos, hard_negs, easy_negs], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state)  # Shuffle
    
    print(f"    Final training set: {len(sampled):,} pairs "
          f"({len(pos)/len(sampled)*100:.1f}% positive)")
    return sampled


# ─── Model Training ──────────────────────────────────────────────────────────

def get_feature_cols(feat_df: pd.DataFrame) -> list:
    """Return feature column names (exclude ID and label columns)."""
    exclude = {"s1_id", "cand_id", "label"}
    return [c for c in feat_df.columns if c not in exclude]


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                feature_cols: list = None) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier on the pairwise feature matrix.
    
    LightGBM (Gradient Boosted Decision Trees):
    - Builds an ensemble of decision trees, each correcting the errors of the previous
    - Very fast, memory-efficient, excellent on tabular data
    - MIT license: fully compliant with competition
    - No "parameters" in the neural net sense; tree models are unlimited in theory but
      constrained by n_estimators and max_depth
    
    Args:
        train_df: feature DataFrame with 'label' column
        val_df: optional validation DataFrame for early stopping
        feature_cols: list of feature column names to use
    
    Returns:
        Trained LightGBM classifier
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if feature_cols is None:
        feature_cols = get_feature_cols(train_df)
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    
    # Class weight to handle imbalance: positives are rare
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    print(f"    Training LightGBM on {len(X_train):,} pairs "
          f"(scale_pos_weight={scale_pos_weight:.2f})...")
    print(f"    Features used: {len(feature_cols)}")
    
    # LightGBM hyperparameters (tuned for F0.5 / precision-heavy objective)
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,      # L1 regularization (encourages sparsity)
        reg_lambda=0.1,     # L2 regularization
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,           # Use all CPU cores
        verbose=-1,          # Suppress training output
    )
    
    if val_df is not None and len(val_df) > 0:
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["label"].values
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=50)
            ],
        )
    else:
        model.fit(X_train, y_train)
    
    print(f"    Model trained. Best iteration: {model.best_iteration_}")
    
    # Feature importances (useful for understanding what drives matching)
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print(f"\n    Top 10 most important features:")
        for fname, imp in importances[:10]:
            print(f"      {fname:35s}: {imp:.0f}")
    
    return model


def predict_scores(model: lgb.LGBMClassifier, feat_df: pd.DataFrame,
                   feature_cols: list) -> np.ndarray:
    """
    Predict match probability for each pair.
    
    Returns: array of shape [n_pairs] with P(match) values in [0, 1].
    """
    X = feat_df[feature_cols].values.astype(np.float32)
    # predict_proba returns [[P(0), P(1)], ...]
    return model.predict_proba(X)[:, 1]


def save_model(model, feature_cols: list, path: str = MODEL_PATH):
    """Save model and feature column names to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)
    print(f"    Model saved to {path}")


def load_model(path: str = MODEL_PATH):
    """Load model from disk. Returns (model, feature_cols)."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]
