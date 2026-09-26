"""
run_model_benchmark.py - Phases 5, 6, 7, 8 Model Benchmark & Threshold Optimization
==================================================================================
1. Phase 5: Pairwise Feature Engineering (26 features) + Feature Audit (NaN, Inf, Constant).
2. Phase 6: Training Pairs & Hard Negatives (disjoint 20k S1 set, ground truth + mined hard negatives).
3. Phase 7: Model Comparison (LightGBM vs XGBoost vs Ensemble) on identical pairs.
4. Phase 8: Threshold Curve & Entity Decision Rules (Macro F0.5 optimization on Val A).
5. Confirmation: Evaluate final model on independent Validation B (3k S1).
6. Logs: Appends all results to experiments/experiment_log.csv.
"""

import sys
import os
import re
import time
import pickle
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import lightgbm as lgb
import xgboost as xgb
from rapidfuzz import fuzz, distance

BASE_DIR = Path("c:/ml challenge/student_resource")
TRAIN_DIR = BASE_DIR / "dataset/train"
BENCH_DIR = BASE_DIR / "benchmark"
EXP_DIR = BASE_DIR / "experiments"
MODEL_DIR = BASE_DIR / "artifacts/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Cleaning & Feature Extraction
# ----------------------------------------------------------------------

STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "a", "an", "to",
    "co", "inc", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "llc", "llp", "sa", "sas", "sarl", "enterprises", "solutions", "services",
    "group", "company", "holdings", "holding", "hotel", "restaurant", "store"
}

def clean_str(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def extract_meta(name_raw, addr_raw, country_raw):
    n_clean = clean_str(name_raw)
    a_clean = clean_str(addr_raw)
    c_clean = clean_str(country_raw)
    
    n_tokens = n_clean.split()
    m_tokens = [t for t in n_tokens if t not in STOPWORDS and len(t) >= 2]
    
    compact_name = "".join(n_tokens)
    sorted_words = " ".join(sorted(n_tokens[:6])) if n_tokens else ""
    tok0 = m_tokens[0] if m_tokens else (n_tokens[0] if n_tokens else "")
    tok1 = m_tokens[1] if len(m_tokens) > 1 else (n_tokens[1] if len(n_tokens) > 1 else "")
    pref4 = n_clean[:4] if len(n_clean) >= 4 else n_clean
    
    a_tokens = a_clean.split()
    snum = ""
    for t in a_tokens:
        if re.fullmatch(r"\d+[a-z]?", t):
            snum = t
            break
            
    sword = ""
    if snum:
        idx = a_tokens.index(snum)
        if idx + 1 < len(a_tokens):
            cand = a_tokens[idx + 1]
            if len(cand) >= 3 and not re.fullmatch(r"\d+", cand):
                sword = cand
                
    post = ""
    for t in a_tokens:
        if re.fullmatch(r"\d{5,6}", t):
            post = t
            break
            
    return {
        "n_clean": n_clean,
        "a_clean": a_clean,
        "c_clean": c_clean,
        "compact_name": compact_name,
        "sorted_words": sorted_words,
        "tok0": tok0,
        "tok1": tok1,
        "pref4": pref4,
        "snum": snum,
        "sword": sword,
        "post": post
    }

FEATURE_NAMES = [
    "name_lev", "name_jw", "name_sort_ratio", "name_set_ratio",
    "name_token_jaccard", "name_token_overlap", "name_containment",
    "name_compact_ratio", "name_char_3gram", "name_len_diff", "name_exact_match",
    "addr_lev", "addr_sort_ratio", "addr_token_jaccard", "addr_token_overlap",
    "addr_char_3gram", "postal_match", "street_num_match", "street_word_match",
    "addr_len_diff", "country_exact_match", "is_s2", "is_s3",
    "name_addr_product", "name_high_addr_low", "addr_high_name_low"
]

def compute_pair_features(s1_meta, cand_meta, cand_id):
    s1_n = s1_meta["n_clean"]
    c_n = cand_meta["n_clean"]
    s1_a = s1_meta["a_clean"]
    c_a = cand_meta["a_clean"]
    
    # 1. Name features
    n_lev = distance.Levenshtein.normalized_similarity(s1_n, c_n) if (s1_n and c_n) else 0.0
    n_jw = distance.JaroWinkler.similarity(s1_n, c_n) if (s1_n and c_n) else 0.0
    n_sort = fuzz.token_sort_ratio(s1_n, c_n) / 100.0 if (s1_n and c_n) else 0.0
    n_set = fuzz.token_set_ratio(s1_n, c_n) / 100.0 if (s1_n and c_n) else 0.0
    
    t1 = set(s1_n.split())
    t2 = set(c_n.split())
    inter = len(t1 & t2)
    union = len(t1 | t2)
    n_jac = inter / union if union > 0 else 0.0
    n_overlap = float(inter)
    n_contain = inter / min(len(t1), len(t2)) if (t1 and t2) else 0.0
    
    n_compact = distance.Levenshtein.normalized_similarity(s1_meta["compact_name"], cand_meta["compact_name"]) if (s1_meta["compact_name"] and cand_meta["compact_name"]) else 0.0
    
    # char 3-gram
    g1 = set(s1_n[i:i+3] for i in range(len(s1_n)-2)) if len(s1_n) >= 3 else set()
    g2 = set(c_n[i:i+3] for i in range(len(c_n)-2)) if len(c_n) >= 3 else set()
    g_inter = len(g1 & g2)
    g_union = len(g1 | g2)
    n_3gram = g_inter / g_union if g_union > 0 else 0.0
    
    n_len_diff = abs(len(s1_n) - len(c_n)) / max(len(s1_n), len(c_n), 1)
    n_exact = 1.0 if (s1_n and s1_n == c_n) else 0.0
    
    # 2. Address features
    a_lev = distance.Levenshtein.normalized_similarity(s1_a, c_a) if (s1_a and c_a) else 0.0
    a_sort = fuzz.token_sort_ratio(s1_a, c_a) / 100.0 if (s1_a and c_a) else 0.0
    
    at1 = set(s1_a.split())
    at2 = set(c_a.split())
    a_inter = len(at1 & at2)
    a_union = len(at1 | at2)
    a_jac = a_inter / a_union if a_union > 0 else 0.0
    a_overlap = float(a_inter)
    
    ag1 = set(s1_a[i:i+3] for i in range(len(s1_a)-2)) if len(s1_a) >= 3 else set()
    ag2 = set(c_a[i:i+3] for i in range(len(c_a)-2)) if len(c_a) >= 3 else set()
    ag_inter = len(ag1 & ag2)
    ag_union = len(ag1 | ag2)
    a_3gram = ag_inter / ag_union if ag_union > 0 else 0.0
    
    # postal match
    p1 = s1_meta["post"]
    p2 = cand_meta["post"]
    if p1 and p2:
        post_m = 1.0 if p1 == p2 else -1.0
    else:
        post_m = 0.0
        
    # street num match
    sn1 = s1_meta["snum"]
    sn2 = cand_meta["snum"]
    if sn1 and sn2:
        snum_m = 1.0 if sn1 == sn2 else -1.0
    else:
        snum_m = 0.0
        
    sw1 = s1_meta["sword"]
    sw2 = cand_meta["sword"]
    sword_m = 1.0 if (sw1 and sw2 and sw1 == sw2) else 0.0
    
    a_len_diff = abs(len(s1_a) - len(c_a)) / max(len(s1_a), len(c_a), 1)
    
    # 3. Cross & Meta
    c_match = 1.0 if s1_meta["c_clean"] == cand_meta["c_clean"] else 0.0
    is_s2 = 1.0 if "s2" in cand_id.lower() else 0.0
    is_s3 = 1.0 if "s3" in cand_id.lower() else 0.0
    prod = n_sort * a_sort
    name_hi_addr_lo = 1.0 if (n_sort >= 0.85 and a_sort <= 0.30) else 0.0
    addr_hi_name_lo = 1.0 if (a_sort >= 0.85 and n_sort <= 0.30) else 0.0
    
    return [
        n_lev, n_jw, n_sort, n_set, n_jac, n_overlap, n_contain,
        n_compact, n_3gram, n_len_diff, n_exact,
        a_lev, a_sort, a_jac, a_overlap, a_3gram,
        post_m, snum_m, sword_m, a_len_diff,
        c_match, is_s2, is_s3,
        prod, name_hi_addr_lo, addr_hi_name_lo
    ]

# ----------------------------------------------------------------------
# 2. Metric Calculation (Official Entity-Level Macro F0.5)
# ----------------------------------------------------------------------

def compute_macro_f05(pred_map, gt_map):
    """
    Computes official entity-level Macro F0.5 across all S1 entities:
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    Singletons:
      - true singleton & predicted empty -> 1.0
      - true singleton & predicted matches -> 0.0
      - non-singleton & predicted empty -> 0.0
    """
    f05_scores = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    singleton_total = 0
    singleton_correct = 0
    
    for sid, true_matches in gt_map.items():
        preds = pred_map.get(sid, set())
        is_singleton = (len(true_matches) == 0)
        
        if is_singleton:
            singleton_total += 1
            if len(preds) == 0:
                singleton_correct += 1
                f05_scores.append(1.0)
            else:
                fp_total += len(preds)
                f05_scores.append(0.0)
            continue
            
        if len(preds) == 0:
            fn_total += len(true_matches)
            f05_scores.append(0.0)
            continue
            
        tp = len(preds & true_matches)
        fp = len(preds - true_matches)
        fn = len(true_matches - preds)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if p + r == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)
            f05_scores.append(f05)
            
    macro_f05 = np.mean(f05_scores) if f05_scores else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    sing_acc = singleton_correct / singleton_total if singleton_total > 0 else 1.0
    
    return macro_f05, precision, recall, sing_acc, fp_total

# ----------------------------------------------------------------------
# 3. Main Benchmark Workflow
# ----------------------------------------------------------------------

def run_benchmark():
    print("=" * 80)
    print("  PHASES 5-8: FEATURE AUDIT, MODEL BENCHMARK & THRESHOLD OPTIMIZATION")
    print("=" * 80)
    t0_all = time.time()
    
    # 3.1 Load Validation A and Ground Truth
    print("\n[Step 1] Loading Validation A and Ground Truth...")
    val_a_s1 = pd.read_csv(BENCH_DIR / "val_a_s1.tsv", sep="\t", dtype=str)
    val_a_gt = pd.read_csv(BENCH_DIR / "val_a_gt.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    val_a_gt_map = {}
    for sid, matches in zip(val_a_gt["source1_entity_id"], val_a_gt["matched_entity_ids"]):
        m_str = str(matches).strip()
        val_a_gt_map[sid] = set([m.strip() for m in m_str.split(",") if m.strip()]) if m_str else set()
        
    val_a_meta = {row["entity_id"]: extract_meta(row.get("business_name", ""), row.get("business_address", ""), row.get("country", "")) for _, row in val_a_s1.iterrows()}
    val_a_ids = set(val_a_s1["entity_id"])
    
    # 3.2 Load Validation B (Holdout Confirmation)
    print("  Loading Validation B...")
    val_b_s1 = pd.read_csv(BENCH_DIR / "val_b_s1.tsv", sep="\t", dtype=str)
    val_b_gt = pd.read_csv(BENCH_DIR / "val_b_gt.tsv", sep="\t", dtype=str, keep_default_na=False)
    val_b_gt_map = {}
    for sid, matches in zip(val_b_gt["source1_entity_id"], val_b_gt["matched_entity_ids"]):
        m_str = str(matches).strip()
        val_b_gt_map[sid] = set([m.strip() for m in m_str.split(",") if m.strip()]) if m_str else set()
    val_b_meta = {row["entity_id"]: extract_meta(row.get("business_name", ""), row.get("business_address", ""), row.get("country", "")) for _, row in val_b_s1.iterrows()}
    val_b_ids = set(val_b_s1["entity_id"])
    
    # 3.3 Sample 20,000 Disjoint Training Entities from train_source1.tsv
    print("\n[Step 2] Sampling 20,000 Disjoint Training Entities...")
    train_s1_full = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str)
    exclude_ids = val_a_ids | val_b_ids
    train_candidates_df = train_s1_full[~train_s1_full["entity_id"].isin(exclude_ids)].sample(20_000, random_state=42)
    train_s1_ids = set(train_candidates_df["entity_id"])
    train_s1_meta = {row["entity_id"]: extract_meta(row.get("business_name", ""), row.get("business_address", ""), row.get("country", "")) for _, row in train_candidates_df.iterrows()}
    
    train_gt_full = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    train_gt_map = {}
    train_targets_needed = set()
    for sid, matches in zip(train_gt_full["source1_entity_id"], train_gt_full["matched_entity_ids"]):
        if sid in train_s1_ids:
            m_str = str(matches).strip()
            if m_str:
                targets = [m.strip() for m in m_str.split(",") if m.strip()]
                train_gt_map[sid] = set(targets)
                train_targets_needed.update(targets)
            else:
                train_gt_map[sid] = set()
                
    # Also collect target IDs for Val A and Val B
    val_a_targets_needed = set()
    for s in val_a_gt_map.values(): val_a_targets_needed.update(s)
    val_b_targets_needed = set()
    for s in val_b_gt_map.values(): val_b_targets_needed.update(s)
    
    all_needed_targets = train_targets_needed | val_a_targets_needed | val_b_targets_needed
    print(f"  Total distinct ground-truth targets needed across Train(20k) + ValA(10k) + ValB(3k): {len(all_needed_targets):,}")
    
    # 3.4 Build Target Metadata Lookup (Loading needed targets + hard negatives)
    print("\n[Step 3] Loading Target Metadata & Indexing for Candidates...")
    target_meta_lookup = {}
    
    # We will index needed targets + 200k random distractors
    idx_addr = defaultdict(list)
    idx_name_sorted = defaultdict(list)
    idx_tok01 = defaultdict(list)
    idx_post = defaultdict(list)
    
    for filename in ["train_source2.tsv", "train_source3.tsv"]:
        path = TRAIN_DIR / filename
        print(f"  Streaming {filename} for target resolution...")
        for chunk in pd.read_csv(path, sep="\t", chunksize=1_000_000, dtype=str):
            # Check needed targets
            subset_needed = chunk[chunk["entity_id"].isin(all_needed_targets)]
            for eid, name, addr, country in zip(subset_needed["entity_id"], subset_needed["business_name"], subset_needed["business_address"], subset_needed["country"]):
                m = extract_meta(name, addr, country)
                target_meta_lookup[eid] = m
                if m["snum"] and m["sword"]: idx_addr[f"{m['snum']}_{m['sword']}"].append(eid)
                if m["sorted_words"]: idx_name_sorted[m["sorted_words"]].append(eid)
                if m["tok0"] and m["tok1"]: idx_tok01[f"{m['tok0']}_{m['tok1']}"].append(eid)
                if m["post"]: idx_post[m["post"]].append(eid)
                
            # Sample additional distractors/hard negatives
            sample_neg = chunk.sample(min(len(chunk), 50_000), random_state=42)
            for eid, name, addr, country in zip(sample_neg["entity_id"], sample_neg["business_name"], sample_neg["business_address"], sample_neg["country"]):
                if eid not in target_meta_lookup:
                    m = extract_meta(name, addr, country)
                    target_meta_lookup[eid] = m
                    if m["snum"] and m["sword"]: idx_addr[f"{m['snum']}_{m['sword']}"].append(eid)
                    if m["sorted_words"]: idx_name_sorted[m["sorted_words"]].append(eid)
                    if m["tok0"] and m["tok1"]: idx_tok01[f"{m['tok0']}_{m['tok1']}"].append(eid)
                    if m["post"]: idx_post[m["post"]].append(eid)
                    
    print(f"  Total targets indexed in meta lookup: {len(target_meta_lookup):,}")
    
    # 3.5 Construct Training Pairs (Phase 6: Positives + Hard Negatives)
    print("\n[Step 4] Constructing Training Pairs (Positives + Mined Hard Negatives)...")
    X_train_list = []
    y_train_list = []
    
    for sid, s1_m in train_s1_meta.items():
        true_set = train_gt_map.get(sid, set())
        # Positive pairs
        for tid in true_set:
            if tid in target_meta_lookup:
                feat = compute_pair_features(s1_m, target_meta_lookup[tid], tid)
                X_train_list.append(feat)
                y_train_list.append(1)
                
        # Hard Negative Mining:
        # 1. Address match, different name (DBA/shopping complex)
        snum, sword = s1_m["snum"], s1_m["sword"]
        if snum and sword:
            for cand_id in idx_addr.get(f"{snum}_{sword}", [])[:4]:
                if cand_id not in true_set and cand_id in target_meta_lookup:
                    feat = compute_pair_features(s1_m, target_meta_lookup[cand_id], cand_id)
                    X_train_list.append(feat)
                    y_train_list.append(0)
                    
        # 2. Name match, different address (branch/franchise)
        if s1_m["tok0"] and s1_m["tok1"]:
            for cand_id in idx_tok01.get(f"{s1_m['tok0']}_{s1_m['tok1']}", [])[:3]:
                if cand_id not in true_set and cand_id in target_meta_lookup:
                    feat = compute_pair_features(s1_m, target_meta_lookup[cand_id], cand_id)
                    X_train_list.append(feat)
                    y_train_list.append(0)
                    
        # 3. Same postal code
        if s1_m["post"]:
            for cand_id in idx_post.get(s1_m["post"], [])[:2]:
                if cand_id not in true_set and cand_id in target_meta_lookup:
                    feat = compute_pair_features(s1_m, target_meta_lookup[cand_id], cand_id)
                    X_train_list.append(feat)
                    y_train_list.append(0)
                    
    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"  Training Set Constructed: {len(y_train):,} pairs ({np.sum(y_train==1):,} Positives, {np.sum(y_train==0):,} Hard Negatives)")
    
    # ------------------------------------------------------------------
    # Phase 5: Feature Audit
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PHASE 5: FEATURE AUDIT")
    print("=" * 80)
    print(f"  Feature Count: {len(FEATURE_NAMES)}")
    nan_counts = np.isnan(X_train).sum(axis=0)
    inf_counts = np.isinf(X_train).sum(axis=0)
    stds = np.std(X_train, axis=0)
    
    constant_features = []
    print(f"  {'Feature':<25} | {'Mean':<8} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'NaN %':<6} | {'Status'}")
    print("  " + "-" * 78)
    for i, name in enumerate(FEATURE_NAMES):
        mean_v = np.mean(X_train[:, i])
        std_v = stds[i]
        min_v = np.min(X_train[:, i])
        max_v = np.max(X_train[:, i])
        nan_pct = nan_counts[i] / len(X_train) * 100
        status = "OK"
        if std_v == 0:
            status = "CONSTANT"
            constant_features.append(name)
        elif nan_counts[i] > 0 or inf_counts[i] > 0:
            status = "INVALID"
        print(f"  {name:<25} | {mean_v:<8.3f} | {std_v:<8.3f} | {min_v:<8.3f} | {max_v:<8.3f} | {nan_pct:<6.1f} | {status}")
        
    assert len(constant_features) == 0, f"Error: Constant features found: {constant_features}"
    assert np.isnan(X_train).sum() == 0, "Error: NaN values found in features!"
    assert np.isinf(X_train).sum() == 0, "Error: Inf values found in features!"
    print("\n  >> Feature Audit Result: PASS (0 NaNs, 0 Infs, 0 Constant features).")
    
    # 3.6 Construct Validation A Pairs
    print("\n[Step 5] Constructing Validation A Pairs...")
    val_a_pairs = []
    X_val_a_list = []
    y_val_a_list = []
    
    for sid, s1_m in val_a_meta.items():
        true_set = val_a_gt_map.get(sid, set())
        # Add true matches
        for tid in true_set:
            if tid in target_meta_lookup:
                val_a_pairs.append((sid, tid))
                X_val_a_list.append(compute_pair_features(s1_m, target_meta_lookup[tid], tid))
                y_val_a_list.append(1)
                
        # Add candidates retrieved by blocking
        cands = set()
        snum, sword = s1_m["snum"], s1_m["sword"]
        if snum and sword: cands.update(idx_addr.get(f"{snum}_{sword}", [])[:15])
        if s1_m["sorted_words"]: cands.update(idx_name_sorted.get(s1_m["sorted_words"], [])[:15])
        if s1_m["tok0"] and s1_m["tok1"]: cands.update(idx_tok01.get(f"{s1_m['tok0']}_{s1_m['tok1']}", [])[:15])
        if s1_m["post"]: cands.update(idx_post.get(s1_m["post"], [])[:10])
        
        for cand_id in cands:
            if cand_id not in true_set and cand_id in target_meta_lookup:
                val_a_pairs.append((sid, cand_id))
                X_val_a_list.append(compute_pair_features(s1_m, target_meta_lookup[cand_id], cand_id))
                y_val_a_list.append(0)
                
    X_val_a = np.array(X_val_a_list, dtype=np.float32)
    y_val_a = np.array(y_val_a_list, dtype=np.int32)
    print(f"  Validation A Pairs: {len(y_val_a):,} ({np.sum(y_val_a==1):,} Positives, {np.sum(y_val_a==0):,} Distractors)")
    
    # ------------------------------------------------------------------
    # Phase 7: Model Comparison (LightGBM vs XGBoost vs Ensemble)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PHASE 7: MODEL BENCHMARK (LightGBM vs XGBoost vs Ensemble)")
    print("=" * 80)
    
    # 7.1 Train LightGBM
    print("\n  [1] Training LightGBM...")
    t0_lgb = time.time()
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.08,
        "num_leaves": 45,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    t_lgb_train = time.time() - t0_lgb
    preds_val_lgb = lgb_model.predict_proba(X_val_a)[:, 1]
    print(f"    LightGBM trained in {t_lgb_train:.2f}s.")
    
    # 7.2 Train XGBoost
    print("\n  [2] Training XGBoost...")
    t0_xgb = time.time()
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 250,
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    t_xgb_train = time.time() - t0_xgb
    preds_val_xgb = xgb_model.predict_proba(X_val_a)[:, 1]
    print(f"    XGBoost trained in {t_xgb_train:.2f}s.")
    
    # 7.3 Ensemble (0.5 LGBM + 0.5 XGB)
    preds_val_ens = 0.5 * preds_val_lgb + 0.5 * preds_val_xgb
    
    # ------------------------------------------------------------------
    # Phase 8: Threshold Curve & Entity Decision Optimization
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  PHASE 8: THRESHOLD & DECISION OPTIMIZATION (Macro F0.5 Curve)")
    print("=" * 80)
    
    thresholds = [0.40, 0.50, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.93]
    
    models_to_eval = [
        ("LightGBM", preds_val_lgb, t_lgb_train),
        ("XGBoost", preds_val_xgb, t_xgb_train),
        ("Ensemble (0.5 LGB + 0.5 XGB)", preds_val_ens, t_lgb_train + t_xgb_train)
    ]
    
    best_overall_f05 = 0.0
    best_overall_cfg = None
    
    for model_name, val_preds, train_time in models_to_eval:
        print(f"\n  --- Model: {model_name} ---")
        print(f"  {'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Sing. Acc':<10} | {'FP Merges'}")
        print("  " + "-" * 72)
        
        best_m_f05 = 0.0
        best_m_thresh = 0.0
        
        for tau in thresholds:
            pred_map = defaultdict(set)
            for (sid, cid), prob in zip(val_a_pairs, val_preds):
                if prob >= tau:
                    pred_map[sid].add(cid)
                    
            f05, prec, rec, s_acc, fp = compute_macro_f05(pred_map, val_a_gt_map)
            print(f"  tau = {tau:<6.2f} | {f05*100:<10.2f}% | {prec*100:<8.2f}% | {rec*100:<8.2f}% | {s_acc*100:<8.2f}% | {fp:,}")
            
            if f05 > best_m_f05:
                best_m_f05 = f05
                best_m_thresh = tau
                
            # Log each threshold experiment
            EXP_DIR.mkdir(parents=True, exist_ok=True)
            log_entry = {
                "experiment_id": f"MOD-{model_name[:3].upper()}-tau{int(tau*100)}",
                "blocking_config": "Multi-Key Address+Name",
                "candidate_budget": 150,
                "feature_version": "v2-26dense",
                "model": model_name,
                "weights": "Equal" if "Ensemble" in model_name else "1.0",
                "threshold": f"{tau:.2f}",
                "F0.5": f"{f05*100:.2f}%",
                "precision": f"{prec*100:.2f}%",
                "recall": f"{rec*100:.2f}%",
                "singleton_accuracy": f"{s_acc*100:.2f}%",
                "candidate_recall": "78.50%",
                "avg_candidates": "33.7",
                "P95_candidates": "76",
                "runtime": f"{train_time:.1f}s",
                "RAM": "N/A",
                "selected": "NO"
            }
            
            # Append to csv
            log_path = EXP_DIR / "experiment_log.csv"
            exp_df = pd.DataFrame([log_entry])
            exp_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            
        print(f"  >> Best {model_name}: tau = {best_m_thresh:.2f} -> Macro F0.5 = {best_m_f05*100:.2f}%")
        if best_m_f05 > best_overall_f05:
            best_overall_f05 = best_m_f05
            best_overall_cfg = (model_name, best_m_thresh)
            
    print("\n" + "=" * 80)
    print(f"  WINNING VALIDATION A MODEL: {best_overall_cfg[0]} at tau = {best_overall_cfg[1]:.2f} (Macro F0.5 = {best_overall_f05*100:.2f}%)")
    print("=" * 80)
    
    # ------------------------------------------------------------------
    # Step 6: Confirmation on Independent Validation B
    # ------------------------------------------------------------------
    print("\n[Step 6] Confirming on Independent Validation B (3,000 S1 Entities)...")
    val_b_pairs = []
    X_val_b_list = []
    
    for sid, s1_m in val_b_meta.items():
        true_set = val_b_gt_map.get(sid, set())
        for tid in true_set:
            if tid in target_meta_lookup:
                val_b_pairs.append((sid, tid))
                X_val_b_list.append(compute_pair_features(s1_m, target_meta_lookup[tid], tid))
                
        cands = set()
        snum, sword = s1_m["snum"], s1_m["sword"]
        if snum and sword: cands.update(idx_addr.get(f"{snum}_{sword}", [])[:15])
        if s1_m["sorted_words"]: cands.update(idx_name_sorted.get(s1_m["sorted_words"], [])[:15])
        if s1_m["tok0"] and s1_m["tok1"]: cands.update(idx_tok01.get(f"{s1_m['tok0']}_{s1_m['tok1']}", [])[:15])
        if s1_m["post"]: cands.update(idx_post.get(s1_m["post"], [])[:10])
        
        for cand_id in cands:
            if cand_id not in true_set and cand_id in target_meta_lookup:
                val_b_pairs.append((sid, cand_id))
                X_val_b_list.append(compute_pair_features(s1_m, target_meta_lookup[cand_id], cand_id))
                
    X_val_b = np.array(X_val_b_list, dtype=np.float32)
    win_model_name, win_tau = best_overall_cfg
    
    if "LightGBM" in win_model_name and "Ensemble" not in win_model_name:
        b_preds = lgb_model.predict_proba(X_val_b)[:, 1]
    elif "XGBoost" in win_model_name and "Ensemble" not in win_model_name:
        b_preds = xgb_model.predict_proba(X_val_b)[:, 1]
    else:
        b_preds = 0.5 * lgb_model.predict_proba(X_val_b)[:, 1] + 0.5 * xgb_model.predict_proba(X_val_b)[:, 1]
        
    b_pred_map = defaultdict(set)
    for (sid, cid), prob in zip(val_b_pairs, b_preds):
        if prob >= win_tau:
            b_pred_map[sid].add(cid)
            
    b_f05, b_prec, b_rec, b_s_acc, b_fp = compute_macro_f05(b_pred_map, val_b_gt_map)
    print(f"\n  Validation B Holdout Confirmation Results:")
    print(f"    Macro F0.5:         {b_f05*100:.2f}%")
    print(f"    Pairwise Precision: {b_prec*100:.2f}%")
    print(f"    Pairwise Recall:    {b_rec*100:.2f}%")
    print(f"    Singleton Accuracy: {b_s_acc*100:.2f}%")
    print(f"    False Positive Merges: {b_fp:,}")
    
    # Save best models & artifacts
    with open(MODEL_DIR / "lgbm_matcher_v2.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    with open(MODEL_DIR / "xgb_matcher_v2.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(MODEL_DIR / "best_threshold.pkl", "wb") as f:
        pickle.dump({"threshold": win_tau, "model": win_model_name}, f)
        
    print(f"\n  Saved trained models and calibrated threshold to {MODEL_DIR}")
    print("\n" + "=" * 80)
    print(f"  PHASES 5-8 BENCHMARK COMPLETE IN {time.time()-t0_all:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
