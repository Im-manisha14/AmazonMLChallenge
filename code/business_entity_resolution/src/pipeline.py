"""
Amazon ML Challenge 2026: Business Entity Resolution
=====================================================
High-performance, memory-efficient, fully-vectorised ML pipeline.
Supports open-set countries (US, India, France, etc.) and streaming inference.
"""

import os
import sys
import gc
import re
import time
import pickle
import random
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein, JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(".")
TRAIN_DIR = BASE_DIR / "dataset/train"
TEST_DIR = BASE_DIR / "dataset/test"
OUT_DIR = BASE_DIR / "output"
ART_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ART_DIR / "models"
MODEL_PATH = MODEL_DIR / "lgbm_model.pkl"
THRESH_PATH = MODEL_DIR / "threshold.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

VAL_S1_COUNT = 5_000
TRAIN_S1_COUNT = 15_000
VAL_DISTRACTOR_COUNT = 30_000
TRAIN_DISTRACTOR_COUNT = 25_000
TFIDF_K = 25

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def elapsed_str(t0):
    s = time.time() - t0
    return f"{int(s // 60)}m {int(s % 60):02d}s"

# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & INSPECTION
# ══════════════════════════════════════════════════════════════════════════
def load_data(split="train"):
    d = TRAIN_DIR if split == "train" else TEST_DIR
    log(f"  Loading {split} datasets from {d}...")
    s1 = pd.read_csv(d / f"{split}_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(d / f"{split}_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(d / f"{split}_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = None
    if split == "train":
        gt = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    log(f"  Loaded {split}: S1={len(s1):,}, S2={len(s2):,}, S3={len(s3):,}")
    return s1, s2, s3, gt

def parse_ground_truth(gt_df):
    mapping = {}
    s1_ids = gt_df["source1_entity_id"].values
    matched = gt_df["matched_entity_ids"].values
    for s1_id, m in zip(s1_ids, matched):
        m_str = str(m).strip()
        mapping[s1_id] = set(m_str.split(",")) if m_str else set()
    return mapping

def generate_data_report(s1, s2, s3, gt, gt_map):
    report_path = ART_DIR / "data_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AMAZON ML CHALLENGE: DATA INSPECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for name, df in [("Source 1 (Train)", s1), ("Source 2 (Train)", s2), ("Source 3 (Train)", s3)]:
            f.write(f"--- {name} ---\n")
            f.write(f"Rows: {len(df):,}, Columns: {len(df.columns)} ({list(df.columns)})\n")
            f.write(f"Duplicate entity IDs: {df['entity_id'].duplicated().sum()}\n")
            f.write(f"Country Distribution:\n{df['country'].value_counts().to_string()}\n")
            f.write(f"Unique Business Names: {df['business_name'].nunique():,}\n")
            f.write(f"Unique Addresses: {df['business_address'].nunique():,}\n\n")
            
        singletons = sum(1 for v in gt_map.values() if not v)
        f.write("--- Ground Truth Statistics ---\n")
        f.write(f"Total Source 1 entities: {len(gt_map):,}\n")
        f.write(f"Singletons (0 matches): {singletons:,} ({100.0 * singletons / len(gt_map):.2f}%)\n")
        f.write(f"Non-singletons: {len(gt_map) - singletons:,} ({100.0 * (len(gt_map) - singletons) / len(gt_map):.2f}%)\n")
    log(f"  Data report written to {report_path}")

# ══════════════════════════════════════════════════════════════════════════
# 2. ULTRA-FAST VECTORIZED PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════
def clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r'&', ' and ', regex=False)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nn"] = clean_series(df["business_name"])
    df["na"] = clean_series(df["business_address"])
    df["nc"] = df["country"].fillna("").astype(str).str.lower().str.strip()
    
    # Fast token and prefix extractions
    df["n_tok0"] = df["nn"].str.split().str[0].fillna("")
    df["n_pref3"] = df["nn"].str[:3].fillna("")
    df["n_pref4"] = df["nn"].str[:4].fillna("")
    df["postal"] = df["na"].str.extract(r'\b(\d{5,6})\b', expand=False).fillna("")
    return df

# ══════════════════════════════════════════════════════════════════════════
# 3. MULTI-STRATEGY BLOCKING
# ══════════════════════════════════════════════════════════════════════════
def build_blocking_indexes(df: pd.DataFrame):
    idx = {
        "tok0": defaultdict(list),
        "pref3": defaultdict(list),
        "pref4": defaultdict(list),
        "postal": defaultdict(list),
        "tok01": defaultdict(list)
    }
    eids = df["entity_id"].values
    ncs = df["nc"].values
    nns = df["nn"].values
    postals = df["postal"].values
    tok0s = df["n_tok0"].values
    pref3s = df["n_pref3"].values
    pref4s = df["n_pref4"].values
    
    for i in range(len(eids)):
        eid = eids[i]
        c = ncs[i]
        t0 = tok0s[i]
        p3 = pref3s[i]
        p4 = pref4s[i]
        post = postals[i]
        name = nns[i]
        
        if t0:
            idx["tok0"][f"{c}|{t0}"].append(eid)
        if p3:
            idx["pref3"][f"{c}|{p3}"].append(eid)
        if p4:
            idx["pref4"][f"{c}|{p4}"].append(eid)
        if post:
            idx["postal"][f"{c}|{post}"].append(eid)
            
        toks = name.split(maxsplit=2)
        if len(toks) >= 2:
            idx["tok01"][f"{c}|{toks[0]}|{toks[1]}"].append(eid)
            
    return idx

def get_deterministic_candidates(s1_df: pd.DataFrame, idx2, idx3, max_per_bucket=100):
    eids = s1_df["entity_id"].values
    ncs = s1_df["nc"].values
    nns = s1_df["nn"].values
    postals = s1_df["postal"].values
    tok0s = s1_df["n_tok0"].values
    pref3s = s1_df["n_pref3"].values
    pref4s = s1_df["n_pref4"].values
    
    candidates = defaultdict(set)
    for i in range(len(eids)):
        sid = eids[i]
        c = ncs[i]
        t0 = tok0s[i]
        p3 = pref3s[i]
        p4 = pref4s[i]
        post = postals[i]
        name = nns[i]
        
        cands = set()
        
        for idx in (idx2, idx3):
            # Strategy 1: Exact first name token + country
            if t0:
                k = f"{c}|{t0}"
                c_list = idx["tok0"].get(k, [])
                if len(c_list) <= max_per_bucket:
                    cands.update(c_list)
                    
            # Strategy 2: First two name tokens + country
            toks = name.split(maxsplit=2)
            if len(toks) >= 2:
                k2 = f"{c}|{toks[0]}|{toks[1]}"
                c_list = idx["tok01"].get(k2, [])
                if len(c_list) <= max_per_bucket:
                    cands.update(c_list)
                    
            # Strategy 3: Prefix 4
            if p4:
                k4 = f"{c}|{p4}"
                c_list = idx["pref4"].get(k4, [])
                if len(c_list) <= max_per_bucket:
                    cands.update(c_list)
                    
            # Strategy 4: Postal code + country
            if post:
                kp = f"{c}|{post}"
                c_list = idx["postal"].get(kp, [])
                if len(c_list) <= max_per_bucket:
                    cands.update(c_list)
                    
        candidates[sid] = cands
    return candidates

def fit_tfidf_vectorizers(s2_df, s3_df):
    cand_df = pd.concat([s2_df, s3_df], ignore_index=True)
    vec_name = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=40_000,
        sublinear_tf=True
    )
    vec_addr = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_features=40_000,
        sublinear_tf=True
    )
    
    log("    Fitting TF-IDF Name (word 1-2)...")
    vec_name.fit(cand_df["nn"])
    log("    Fitting TF-IDF Address (word 1-2)...")
    vec_addr.fit(cand_df["na"])
    
    mat_n2 = vec_name.transform(s2_df["nn"])
    mat_n3 = vec_name.transform(s3_df["nn"])
    mat_a2 = vec_addr.transform(s2_df["na"])
    mat_a3 = vec_addr.transform(s3_df["na"])
    
    return vec_name, vec_addr, mat_n2, mat_n3, mat_a2, mat_a3

def tfidf_sparse_retrieve(s1_df, vec_name, vec_addr, mat_n2, mat_n3, mat_a2, mat_a3,
                          s2_ids, s3_ids, K=15, batch_size=300):
    log(f"    TF-IDF sparse candidate retrieval (K={K}) for {len(s1_df):,} S1...")
    tfidf_cands = defaultdict(set)
    
    N = len(s1_df)
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch = s1_df.iloc[i:end]
        batch_ids = batch["entity_id"].values
        
        bn = vec_name.transform(batch["nn"])
        ba = vec_addr.transform(batch["na"])
        
        # Name similarity S2
        sim_n2 = bn.dot(mat_n2.T).tocsr()
        for r in range(len(batch_ids)):
            row = sim_n2.getrow(r)
            if row.nnz > 0:
                top_idx = row.indices[np.argpartition(row.data, -min(K, row.nnz))[-min(K, row.nnz):]]
                tfidf_cands[batch_ids[r]].update(s2_ids[top_idx])
        del sim_n2
        
        # Name similarity S3
        sim_n3 = bn.dot(mat_n3.T).tocsr()
        for r in range(len(batch_ids)):
            row = sim_n3.getrow(r)
            if row.nnz > 0:
                top_idx = row.indices[np.argpartition(row.data, -min(K, row.nnz))[-min(K, row.nnz):]]
                tfidf_cands[batch_ids[r]].update(s3_ids[top_idx])
        del sim_n3
        
        # Address similarity S2
        sim_a2 = ba.dot(mat_a2.T).tocsr()
        for r in range(len(batch_ids)):
            row = sim_a2.getrow(r)
            if row.nnz > 0:
                top_idx = row.indices[np.argpartition(row.data, -min(K, row.nnz))[-min(K, row.nnz):]]
                tfidf_cands[batch_ids[r]].update(s2_ids[top_idx])
        del sim_a2
        
        # Address similarity S3
        sim_a3 = ba.dot(mat_a3.T).tocsr()
        for r in range(len(batch_ids)):
            row = sim_a3.getrow(r)
            if row.nnz > 0:
                top_idx = row.indices[np.argpartition(row.data, -min(K, row.nnz))[-min(K, row.nnz):]]
                tfidf_cands[batch_ids[r]].update(s3_ids[top_idx])
        del sim_a3
        del bn, ba
        
        if end % 25_000 == 0 or end == N:
            log(f"      TF-IDF processed {end:,}/{N:,} S1 entities")
            gc.collect()
            
    return tfidf_cands

def merge_candidate_sets(det_dict, tfidf_dict, all_s1_ids):
    merged = {}
    for sid in all_s1_ids:
        cands = set(det_dict.get(sid, set()))
        if sid in tfidf_dict:
            cands.update(tfidf_dict[sid])
        merged[sid] = cands
    return merged

def compute_candidate_recall(candidate_dict, ground_truth_dict):
    total_true = 0
    recalled = 0
    for sid, true_matches in ground_truth_dict.items():
        if not true_matches:
            continue
        cands = candidate_dict.get(sid, set())
        total_true += len(true_matches)
        recalled += len(true_matches & cands)
    return recalled / total_true if total_true > 0 else 1.0

# ══════════════════════════════════════════════════════════════════════════
# 4. PAIRWISE FEATURE EXTRACTION (HIGH PERFORMANCE)
# ══════════════════════════════════════════════════════════════════════════
def get_char_ngrams(s, n=3):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    u = len(set1 | set2)
    return len(set1 & set2) / u if u > 0 else 0.0

def compute_pairwise_features(s1_df, cand_df, candidate_dict):
    s1_dict = s1_df.set_index("entity_id").to_dict(orient="index")
    cand_dict = cand_df.set_index("entity_id").to_dict(orient="index")
    
    pairs = []
    for sid, cands in candidate_dict.items():
        for cid in cands:
            if cid in cand_dict and sid in s1_dict:
                pairs.append((sid, cid))
                
    N = len(pairs)
    log(f"    Computing features for {N:,} candidate pairs...")
    if N == 0:
        return pd.DataFrame()
        
    s1_names = [s1_dict[sid]["nn"] for sid, _ in pairs]
    s1_addrs = [s1_dict[sid]["na"] for sid, _ in pairs]
    s1_cntrs = [s1_dict[sid]["nc"] for sid, _ in pairs]
    s1_posts = [s1_dict[sid]["postal"] for sid, _ in pairs]
    
    c_names = [cand_dict[cid]["nn"] for _, cid in pairs]
    c_addrs = [cand_dict[cid]["na"] for _, cid in pairs]
    c_cntrs = [cand_dict[cid]["nc"] for _, cid in pairs]
    c_posts = [cand_dict[cid]["postal"] for _, cid in pairs]
    
    n_exact = np.zeros(N, dtype=np.int8)
    n_lev = np.zeros(N, dtype=np.float32)
    n_jw = np.zeros(N, dtype=np.float32)
    n_jac = np.zeros(N, dtype=np.float32)
    n_cgram = np.zeros(N, dtype=np.float32)
    len_n_ratio = np.zeros(N, dtype=np.float32)
    
    a_exact = np.zeros(N, dtype=np.int8)
    a_lev = np.zeros(N, dtype=np.float32)
    a_jw = np.zeros(N, dtype=np.float32)
    a_jac = np.zeros(N, dtype=np.float32)
    a_cgram = np.zeros(N, dtype=np.float32)
    len_a_ratio = np.zeros(N, dtype=np.float32)
    
    c_exact = np.zeros(N, dtype=np.int8)
    post_exact = np.zeros(N, dtype=np.int8)
    num_jac = np.zeros(N, dtype=np.float32)
    is_s2 = np.zeros(N, dtype=np.int8)
    
    for i in range(N):
        n1, n2 = s1_names[i], c_names[i]
        a1, a2 = s1_addrs[i], c_addrs[i]
        c1, c2 = s1_cntrs[i], c_cntrs[i]
        p1, p2 = s1_posts[i], c_posts[i]
        cid = pairs[i][1]
        
        # Name exact & similarity
        if n1 and n1 == n2:
            n_exact[i] = 1
            n_lev[i] = 1.0
            n_jw[i] = 1.0
            n_jac[i] = 1.0
            n_cgram[i] = 1.0
            len_n_ratio[i] = 1.0
        else:
            if HAS_RAPIDFUZZ:
                n_lev[i] = Levenshtein.normalized_similarity(n1, n2)
                n_jw[i] = JaroWinkler.similarity(n1, n2)
            t1, t2 = set(n1.split()), set(n2.split())
            n_jac[i] = jaccard_similarity(t1, t2)
            g1, g2 = get_char_ngrams(n1, 3), get_char_ngrams(n2, 3)
            n_cgram[i] = jaccard_similarity(g1, g2)
            l1, l2 = len(n1), len(n2)
            len_n_ratio[i] = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
            
        # Address exact & similarity
        if a1 and a1 == a2:
            a_exact[i] = 1
            a_lev[i] = 1.0
            a_jw[i] = 1.0
            a_jac[i] = 1.0
            a_cgram[i] = 1.0
            len_a_ratio[i] = 1.0
        else:
            if HAS_RAPIDFUZZ:
                a_lev[i] = Levenshtein.normalized_similarity(a1, a2)
                a_jw[i] = JaroWinkler.similarity(a1, a2)
            at1, at2 = set(a1.split()), set(a2.split())
            a_jac[i] = jaccard_similarity(at1, at2)
            ag1, ag2 = get_char_ngrams(a1, 3), get_char_ngrams(a2, 3)
            a_cgram[i] = jaccard_similarity(ag1, ag2)
            al1, al2 = len(a1), len(a2)
            len_a_ratio[i] = min(al1, al2) / max(al1, al2) if max(al1, al2) > 0 else 1.0
            
        # Country & Postal & Numerics
        c_exact[i] = 1 if (c1 and c1 == c2) else 0
        post_exact[i] = 1 if (p1 and p1 == p2) else 0
        
        num1 = set(re.findall(r'\d+', a1))
        num2 = set(re.findall(r'\d+', a2))
        num_jac[i] = jaccard_similarity(num1, num2)
        
        is_s2[i] = 1 if cid.startswith("S2-") else 0
        
    feat_df = pd.DataFrame({
        "s1_id": [p[0] for p in pairs],
        "cand_id": [p[1] for p in pairs],
        "n_exact": n_exact,
        "n_lev": n_lev,
        "n_jw": n_jw,
        "n_jac": n_jac,
        "n_cgram": n_cgram,
        "len_n_ratio": len_n_ratio,
        "a_exact": a_exact,
        "a_lev": a_lev,
        "a_jw": a_jw,
        "a_jac": a_jac,
        "a_cgram": a_cgram,
        "len_a_ratio": len_a_ratio,
        "c_exact": c_exact,
        "post_exact": post_exact,
        "num_jac": num_jac,
        "is_s2": is_s2,
    })
    
    feat_df["hi_conf"] = ((feat_df["n_exact"] == 1) & (feat_df["a_exact"] == 1) & (feat_df["c_exact"] == 1)).astype(np.int8)
    feat_df["high_sim_all"] = ((feat_df["n_lev"] > 0.85) & (feat_df["a_lev"] > 0.80) & (feat_df["c_exact"] == 1)).astype(np.int8)
    
    return feat_df

# ══════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING & HARD NEGATIVE MINING
# ══════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    "n_exact", "n_lev", "n_jw", "n_jac", "n_cgram", "len_n_ratio",
    "a_exact", "a_lev", "a_jw", "a_jac", "a_cgram", "len_a_ratio",
    "c_exact", "post_exact", "num_jac", "is_s2", "hi_conf", "high_sim_all"
]

def add_labels_to_features(feat_df, ground_truth_dict):
    labels = np.zeros(len(feat_df), dtype=np.int8)
    s1_ids = feat_df["s1_id"].values
    cand_ids = feat_df["cand_id"].values
    for i in range(len(s1_ids)):
        true_set = ground_truth_dict.get(s1_ids[i], set())
        if cand_ids[i] in true_set:
            labels[i] = 1
    feat_df["label"] = labels
    return feat_df

def sample_training_pairs(feat_df, hard_neg_ratio=3, easy_neg_ratio=1):
    positives = feat_df[feat_df["label"] == 1]
    negatives = feat_df[feat_df["label"] == 0]
    
    if len(positives) == 0:
        return feat_df
        
    n_pos = len(positives)
    hard_score = negatives["n_lev"] * 0.6 + negatives["a_lev"] * 0.4
    hard_negs = negatives.iloc[np.argsort(hard_score)[::-1][:n_pos * hard_neg_ratio]]
    
    remaining_negs = negatives.drop(hard_negs.index)
    easy_negs = remaining_negs.sample(min(len(remaining_negs), n_pos * easy_neg_ratio), random_state=SEED) if len(remaining_negs) > 0 else remaining_negs
    
    sampled = pd.concat([positives, hard_negs, easy_negs]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    log(f"    Sampled training dataset: Positives={len(positives):,}, Hard Negs={len(hard_negs):,}, Easy Negs={len(easy_negs):,}")
    return sampled

def train_lightgbm_model(train_df, val_df=None):
    X_tr = train_df[FEATURE_COLS].values.astype(np.float32)
    y_tr = train_df["label"].values.astype(np.int32)
    
    try:
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        )
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[FEATURE_COLS].values.astype(np.float32)
            y_val = val_df["label"].values.astype(np.int32)
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])
        else:
            clf.fit(X_tr, y_tr)
        return clf
    except Exception as e:
        log(f"    Fallback to HistGradientBoostingClassifier due to: {e}")
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            max_depth=6,
            random_state=SEED
        )
        clf.fit(X_tr, y_tr)
        return clf

def predict_match_probabilities(model, X_df):
    X = X_df[FEATURE_COLS].values.astype(np.float32)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)

# ══════════════════════════════════════════════════════════════════════════
# 6. EVALUATION & THRESHOLD OPTIMIZATION (MACRO F0.5)
# ══════════════════════════════════════════════════════════════════════════
def calculate_entity_f05(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate_predictions(ground_truth_dict, predictions_dict, all_s1_ids):
    f05_scores = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    false_merges = 0
    singleton_correct = 0
    singleton_total = 0
    
    for sid in all_s1_ids:
        true_set = ground_truth_dict.get(sid, set())
        pred_set = predictions_dict.get(sid, set())
        
        is_true_singleton = (len(true_set) == 0)
        is_pred_singleton = (len(pred_set) == 0)
        
        if is_true_singleton:
            singleton_total += 1
            if is_pred_singleton:
                singleton_correct += 1
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
                total_fp += len(pred_set)
                false_merges += 1
        else:
            if is_pred_singleton:
                total_fn += len(true_set)
                f05_scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                if fp > 0:
                    false_merges += 1
                    
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f05 = calculate_entity_f05(prec, rec)
                f05_scores.append(f05)
                
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    sing_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    
    return {
        "macro_f05": macro_f05,
        "precision": overall_prec,
        "recall": overall_rec,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "false_merges": false_merges,
        "singleton_accuracy": sing_acc
    }

def find_best_threshold(val_feat_df, ground_truth_dict, val_s1_ids):
    log("    Optimizing threshold for Macro F0.5...")
    scores = val_feat_df["score"].values
    s1_ids = val_feat_df["s1_id"].values
    cand_ids = val_feat_df["cand_id"].values
    
    best_thr = 0.50
    best_f05 = -1.0
    
    candidate_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for thr in candidate_thresholds:
        preds = defaultdict(set)
        for i in range(len(scores)):
            if scores[i] >= thr:
                preds[s1_ids[i]].add(cand_ids[i])
                
        metrics = evaluate_predictions(ground_truth_dict, preds, val_s1_ids)
        log(f"      Threshold {thr:.2f} -> Macro F0.5: {metrics['macro_f05']:.4f} (Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, False Merges: {metrics['false_merges']})")
        
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_thr = thr
            
    fine_range = np.linspace(max(0.1, best_thr - 0.08), min(0.98, best_thr + 0.08), 17)
    for thr in fine_range:
        preds = defaultdict(set)
        for i in range(len(scores)):
            if scores[i] >= thr:
                preds[s1_ids[i]].add(cand_ids[i])
        metrics = evaluate_predictions(ground_truth_dict, preds, val_s1_ids)
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_thr = thr
            
    log(f"    Optimal Threshold: {best_thr:.4f} (Peak Validation Macro F0.5: {best_f05:.4f})")
    return best_thr, best_f05

# ══════════════════════════════════════════════════════════════════════════
# 7. SUBMISSION WRITING & VALIDATION
# ══════════════════════════════════════════════════════════════════════════
def write_submission_files(predictions_dict, candidate_dict, all_s1_ids):
    match_path = OUT_DIR / "matching_results.tsv"
    cand_path = OUT_DIR / "candidate_pairs.tsv"
    
    log(f"  Writing matching results to {match_path}...")
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = sorted(list(predictions_dict.get(sid, set())))
            m_str = ",".join(matches)
            f.write(f"{sid}\t{m_str}\n")
            
    log(f"  Writing candidate pairs to {cand_path}...")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = sorted(list(candidate_dict.get(sid, set())))
            c_str = ",".join(cands)
            f.write(f"{sid}\t{c_str}\n")

def run_official_validator():
    cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(OUT_DIR / "matching_results.tsv"),
        "--candidate", str(OUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(TEST_DIR)
    ]
    log(f"  Executing validator: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    log(res.stdout)
    if res.stderr:
        log("Validator Stderr:", res.stderr)
    return res.returncode == 0 and "PASS" in res.stdout

# ══════════════════════════════════════════════════════════════════════════
# 8. MAIN EXECUTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def run_pipeline(mode="all"):
    t0 = time.time()
    log("\n" + "=" * 70)
    log(f"  AMAZON ML CHALLENGE: BUSINESS ENTITY RESOLUTION PIPELINE (mode={mode})")
    log("=" * 70)
    
    # Step 1: Load Data
    log(f"\n[1] Loading data... [{elapsed_str(t0)}]")
    s1, s2, s3, gt_df = load_data("train")
    gt_map = parse_ground_truth(gt_df)
    
    # Step 2: Data Inspection
    log(f"\n[2] Inspecting data... [{elapsed_str(t0)}]")
    generate_data_report(s1, s2, s3, gt_df, gt_map)
    
    # Step 3: Validation & Training Splits
    log(f"\n[3] Creating validation & training splits... [{elapsed_str(t0)}]")
    all_s1_ids = s1["entity_id"].tolist()
    rng = random.Random(SEED)
    rng.shuffle(all_s1_ids)
    
    val_s1_id_set = set(all_s1_ids[:VAL_S1_COUNT])
    train_s1_id_set = set(all_s1_ids[VAL_S1_COUNT:VAL_S1_COUNT + TRAIN_S1_COUNT])
    
    val_s1 = s1[s1["entity_id"].isin(val_s1_id_set)].reset_index(drop=True)
    val_gt = {k: v for k, v in gt_map.items() if k in val_s1_id_set}
    val_true_ids = set()
    for v in val_gt.values():
        val_true_ids.update(v)
        
    val_s2_true = s2[s2["entity_id"].isin(val_true_ids)]
    val_s2_dist = s2[~s2["entity_id"].isin(val_true_ids)].sample(min(VAL_DISTRACTOR_COUNT, len(s2)), random_state=SEED)
    val_s2 = pd.concat([val_s2_true, val_s2_dist], ignore_index=True)
    
    val_s3_true = s3[s3["entity_id"].isin(val_true_ids)]
    val_s3_dist = s3[~s3["entity_id"].isin(val_true_ids)].sample(min(VAL_DISTRACTOR_COUNT, len(s3)), random_state=SEED)
    val_s3 = pd.concat([val_s3_true, val_s3_dist], ignore_index=True)
    
    train_s1 = s1[s1["entity_id"].isin(train_s1_id_set)].reset_index(drop=True)
    train_gt = {k: v for k, v in gt_map.items() if k in train_s1_id_set}
    train_true_ids = set()
    for v in train_gt.values():
        train_true_ids.update(v)
        
    train_s2_true = s2[s2["entity_id"].isin(train_true_ids)]
    train_s2_dist = s2[~s2["entity_id"].isin(train_true_ids)].sample(min(TRAIN_DISTRACTOR_COUNT, len(s2)), random_state=SEED)
    train_s2 = pd.concat([train_s2_true, train_s2_dist], ignore_index=True)
    
    train_s3_true = s3[s3["entity_id"].isin(train_true_ids)]
    train_s3_dist = s3[~s3["entity_id"].isin(train_true_ids)].sample(min(TRAIN_DISTRACTOR_COUNT, len(s3)), random_state=SEED)
    train_s3 = pd.concat([train_s3_true, train_s3_dist], ignore_index=True)
    
    # Release raw 10M rows from RAM
    del s1, s2, s3, gt_df
    gc.collect()
    
    log(f"  Validation Set: {len(val_s1):,} S1 entities, {len(val_s2):,} S2 candidates, {len(val_s3):,} S3 candidates")
    log(f"  Training Set:   {len(train_s1):,} S1 entities, {len(train_s2):,} S2 candidates, {len(train_s3):,} S3 candidates")
    
    # Step 4: Text Preprocessing
    log(f"\n[4] Preprocessing text (vectorized)... [{elapsed_str(t0)}]")
    val_s1 = preprocess_dataframe(val_s1)
    val_s2 = preprocess_dataframe(val_s2)
    val_s3 = preprocess_dataframe(val_s3)
    train_s1 = preprocess_dataframe(train_s1)
    train_s2 = preprocess_dataframe(train_s2)
    train_s3 = preprocess_dataframe(train_s3)
    log(f"  Preprocessing complete [{elapsed_str(t0)}]")
    
    # Step 5: Candidate Generation
    log(f"\n[5] Multi-strategy blocking & candidate generation... [{elapsed_str(t0)}]")
    log("  Building validation blocking indexes...")
    v_idx2 = build_blocking_indexes(val_s2)
    v_idx3 = build_blocking_indexes(val_s3)
    val_det_cands = get_deterministic_candidates(val_s1, v_idx2, v_idx3)
    del v_idx2, v_idx3
    
    log("  Building TF-IDF indexes for validation...")
    v_vn, v_va, vm_n2, vm_n3, vm_a2, vm_a3 = fit_tfidf_vectorizers(val_s2, val_s3)
    val_tfidf_cands = tfidf_sparse_retrieve(
        val_s1, v_vn, v_va, vm_n2, vm_n3, vm_a2, vm_a3,
        val_s2["entity_id"].values, val_s3["entity_id"].values, K=TFIDF_K
    )
    val_candidates = merge_candidate_sets(val_det_cands, val_tfidf_cands, val_s1["entity_id"].tolist())
    val_cand_rec = compute_candidate_recall(val_candidates, val_gt)
    avg_val_cands = np.mean([len(v) for v in val_candidates.values()])
    log(f"  Validation Candidate Recall: {val_cand_rec:.4f} (Avg candidates per S1: {avg_val_cands:.1f})")
    
    # Training Candidates
    log("  Building training blocking indexes...")
    t_idx2 = build_blocking_indexes(train_s2)
    t_idx3 = build_blocking_indexes(train_s3)
    train_det_cands = get_deterministic_candidates(train_s1, t_idx2, t_idx3)
    del t_idx2, t_idx3
    
    log("  Building TF-IDF indexes for training...")
    t_vn, t_va, tm_n2, tm_n3, tm_a2, tm_a3 = fit_tfidf_vectorizers(train_s2, train_s3)
    train_tfidf_cands = tfidf_sparse_retrieve(
        train_s1, t_vn, t_va, tm_n2, tm_n3, tm_a2, tm_a3,
        train_s2["entity_id"].values, train_s3["entity_id"].values, K=TFIDF_K
    )
    train_candidates = merge_candidate_sets(train_det_cands, train_tfidf_cands, train_s1["entity_id"].tolist())
    
    # Step 6: Feature Extraction
    log(f"\n[6] Extracting pairwise features... [{elapsed_str(t0)}]")
    val_cand_df = pd.concat([val_s2, val_s3], ignore_index=True)
    val_feat_df = compute_pairwise_features(val_s1, val_cand_df, val_candidates)
    val_feat_df = add_labels_to_features(val_feat_df, val_gt)
    
    train_cand_df = pd.concat([train_s2, train_s3], ignore_index=True)
    train_feat_df = compute_pairwise_features(train_s1, train_cand_df, train_candidates)
    train_feat_df = add_labels_to_features(train_feat_df, train_gt)
    train_feat_sampled = sample_training_pairs(train_feat_df)
    
    # Free up memory
    del val_cand_df, train_cand_df, train_feat_df
    gc.collect()
    
    # Step 7: Train Model
    log(f"\n[7] Training LightGBM matcher... [{elapsed_str(t0)}]")
    model = train_lightgbm_model(train_feat_sampled, val_feat_df)
    log(f"  Model training complete [{elapsed_str(t0)}]")
    
    # Step 8: Validation & Threshold Search
    log(f"\n[8] Evaluating validation & optimizing threshold... [{elapsed_str(t0)}]")
    val_feat_df["score"] = predict_match_probabilities(model, val_feat_df)
    best_thr, best_val_f05 = find_best_threshold(val_feat_df, val_gt, val_s1["entity_id"].tolist())
    
    val_preds = defaultdict(set)
    for sid, cid, sc in zip(val_feat_df["s1_id"], val_feat_df["cand_id"], val_feat_df["score"]):
        if sc >= best_thr:
            val_preds[sid].add(cid)
            
    val_metrics = evaluate_predictions(val_gt, val_preds, val_s1["entity_id"].tolist())
    
    log("\n  " + "=" * 40)
    log("  OFFICIAL VALIDATION METRICS")
    log("  " + "=" * 40)
    log(f"  Macro F0.5:          {val_metrics['macro_f05']:.4f}")
    log(f"  Precision:           {val_metrics['precision']:.4f}")
    log(f"  Recall:              {val_metrics['recall']:.4f}")
    log(f"  Candidate Recall:    {val_cand_rec:.4f}")
    log(f"  False Merges:        {val_metrics['false_merges']:,}")
    log(f"  Singleton Accuracy:  {val_metrics['singleton_accuracy']:.4f}")
    log(f"  TP / FP / FN:        {val_metrics['tp']} / {val_metrics['fp']} / {val_metrics['fn']}")
    log("  " + "=" * 40)
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(THRESH_PATH, "wb") as f:
        pickle.dump(best_thr, f)
        
    val_feat_df.to_csv(ART_DIR / "validation_predictions.tsv", sep="\t", index=False)
    del val_feat_df
    gc.collect()
    
    # Step 9: Test Inference (Country Partitioned & Streaming)
    log(f"\n[9] Running inference on test data (country partitioned)... [{elapsed_str(t0)}]")
    test_s1, test_s2, test_s3, _ = load_data("test")
    test_s1 = preprocess_dataframe(test_s1)
    test_s2 = preprocess_dataframe(test_s2)
    test_s3 = preprocess_dataframe(test_s3)
    
    all_test_s1_ids = test_s1["entity_id"].tolist()
    countries = test_s1["nc"].unique()
    log(f"  Found {len(countries)} unique countries in test data: {list(countries)}")
    
    test_preds = defaultdict(set)
    test_candidates = defaultdict(set)
    
    for c in countries:
        c_s1 = test_s1[test_s1["nc"] == c].reset_index(drop=True)
        c_s2 = test_s2[test_s2["nc"] == c].reset_index(drop=True)
        c_s3 = test_s3[test_s3["nc"] == c].reset_index(drop=True)
        
        log(f"\n  Processing country '{c}': S1={len(c_s1):,}, S2={len(c_s2):,}, S3={len(c_s3):,}")
        if len(c_s2) == 0 and len(c_s3) == 0:
            log(f"    No S2/S3 records for country '{c}', all S1 are singletons.")
            continue
            
        c_idx2 = build_blocking_indexes(c_s2)
        c_idx3 = build_blocking_indexes(c_s3)
        c_det_cands = get_deterministic_candidates(c_s1, c_idx2, c_idx3)
        del c_idx2, c_idx3
        
        c_vn, c_va, cm_n2, cm_n3, cm_a2, cm_a3 = fit_tfidf_vectorizers(c_s2, c_s3)
        c_tfidf_cands = tfidf_sparse_retrieve(
            c_s1, c_vn, c_va, cm_n2, cm_n3, cm_a2, cm_a3,
            c_s2["entity_id"].values, c_s3["entity_id"].values, K=15, batch_size=300
        )
        c_merged = merge_candidate_sets(c_det_cands, c_tfidf_cands, c_s1["entity_id"].tolist())
        del c_det_cands, c_tfidf_cands, c_vn, c_va, cm_n2, cm_n3, cm_a2, cm_a3
        gc.collect()
        
        c_cand_df = pd.concat([c_s2, c_s3], ignore_index=True)
        
        # Batch inference for this country
        c_s1_len = len(c_s1)
        sub_batch = 25_000
        for b_st in range(0, c_s1_len, sub_batch):
            b_en = min(b_st + sub_batch, c_s1_len)
            b_s1 = c_s1.iloc[b_st:b_en]
            b_ids = set(b_s1["entity_id"])
            b_cands = {k: c_merged[k] for k in b_ids if k in c_merged}
            
            for k, v in b_cands.items():
                test_candidates[k].update(v)
                
            b_feat = compute_pairwise_features(b_s1, c_cand_df, b_cands)
            if len(b_feat) > 0:
                scores = predict_match_probabilities(model, b_feat)
                for sid, cid, sc in zip(b_feat["s1_id"], b_feat["cand_id"], scores):
                    if sc >= best_thr:
                        test_preds[sid].add(cid)
            del b_feat
            gc.collect()
            log(f"    Scored country '{c}' batch {b_en:,}/{c_s1_len:,} S1 entities")
            
        del c_s1, c_s2, c_s3, c_cand_df, c_merged
        gc.collect()
    
    # Step 10: Write Submissions & Validate
    log(f"\n[10] Writing final TSVs... [{elapsed_str(t0)}]")
    write_submission_files(test_preds, test_candidates, all_test_s1_ids)
    
    log(f"\n[11] Running official submission validator... [{elapsed_str(t0)}]")
    is_valid = run_official_validator()
    
    matched_count = sum(1 for p in test_preds.values() if p)
    log("\n" + "=" * 70)
    log("  FINAL PIPELINE SUMMARY")
    log("=" * 70)
    log(f"  TRAINING DATA:")
    log(f"    Source 1: 2,206,821 | Source 2: 5,034,616 | Source 3: 5,285,603")
    log(f"  CANDIDATE GENERATION:")
    log(f"    Validation Recall: {val_cand_rec:.4f} | Avg per S1: {avg_val_cands:.1f}")
    log(f"  VALIDATION METRICS:")
    log(f"    Macro F0.5:         {val_metrics['macro_f05']:.4f}")
    log(f"    Precision:          {val_metrics['precision']:.4f}")
    log(f"    Recall:             {val_metrics['recall']:.4f}")
    log(f"    False Merges:       {val_metrics['false_merges']:,}")
    log(f"    Singleton Accuracy: {val_metrics['singleton_accuracy']:.4f}")
    log(f"  FINAL TEST RESULTS:")
    log(f"    Optimal Threshold:  {best_thr:.4f}")
    log(f"    Test S1 Count:      {len(all_test_s1_ids):,}")
    log(f"    Predicted Matches:  {matched_count:,}")
    log(f"    Predicted Singletons: {len(all_test_s1_ids) - matched_count:,}")
    log(f"  OFFICIAL VALIDATOR:   {'PASS' if is_valid else 'FAIL'}")
    log(f"  Total Runtime:        {elapsed_str(t0)}")
    log("=" * 70)
    return is_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["train", "validate", "test", "all"])
    args = parser.parse_args()
    success = run_pipeline(args.mode)
    sys.exit(0 if success else 1)
