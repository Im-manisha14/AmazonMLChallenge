"""
Amazon ML Challenge 2026: Resume Inference Pipeline & Auto Git Push
===================================================================
Resumes test inference for the remaining 269,986 India entities (indices 540,000 to 809,986).
Appends results directly to matching_results.tsv and candidate_pairs.tsv.
Once completed:
  1. Verifies 1,732,544 total entities
  2. Copies output files to AmazonMLChallenge/output and c:/ml challenge/output
  3. Copies updated code to AmazonMLChallenge/code
  4. Runs official validate_submission.py
  5. Performs git commit and push to origin/main
"""

import os
import sys
import gc
import re
import time
import shutil
import pickle
import subprocess
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance as rf_distance

BASE_DIR = Path("c:/ml challenge/student_resource")
TEST_DIR = BASE_DIR / "dataset/test"
OUT_DIR = BASE_DIR / "output"
OUT_DIR_ROOT = Path("c:/ml challenge/output")
OUT_DIR_GIT = Path("c:/ml challenge/AmazonMLChallenge/output")
GIT_REPO_DIR = Path("c:/ml challenge/AmazonMLChallenge")
ART_DIR = BASE_DIR / "artifacts"

# V2 Models
LGB_MODEL_PATH = ART_DIR / "models/lgbm_matcher_v2.pkl"
XGB_MODEL_PATH = ART_DIR / "models/xgb_matcher_v2.pkl"
THRESH_PATH = ART_DIR / "models/best_threshold.pkl"

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# ============================================================
# PREPROCESSING & BLOCKING
# ============================================================
FRENCH_PREFIXES = {
    'rue', 'avenue', 'ave', 'av', 'boulevard', 'bd', 'bvd', 'chemin',
    'impasse', 'place', 'pl', 'allee', 'route', 'rte', 'cours', 'quai',
    'square', 'sq', 'passage', 'bis', 'ter', 'de', 'du', 'des', 'la',
    'le', 'les', 'd', 'l'
}

def clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x))
    s = s.str.replace(r'[\u0300-\u036f]', '', regex=True).str.lower()
    s = s.str.replace(r'&', ' and ', regex=False)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def extract_street_word(na_series: pd.Series) -> pd.Series:
    no_num = na_series.str.replace(r'\b\d+\b', '', regex=True)
    words = no_num.str.split()
    
    def get_word(row_words):
        if not row_words:
            return ""
        for w in row_words:
            if len(w) >= 3 and w not in FRENCH_PREFIXES:
                return w
        return row_words[0] if row_words else ""
        
    return words.apply(get_word)

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nn"] = clean_series(df["business_name"])
    df["na"] = clean_series(df["business_address"])
    df["nc"] = df["country"].fillna("").astype(str).str.lower().str.strip()
    df["tok0"] = df["nn"].str.split().str[0].fillna("")
    df["tok1"] = df["nn"].str.split().str[1].fillna("")
    df["postal"] = df["na"].str.extract(r'\b(\d{5,6})\b', expand=False).fillna("")
    df["street_num"] = df["na"].str.extract(r'\b(\d{1,6})\b', expand=False).fillna("")
    df["street_word"] = extract_street_word(df["na"])
    df["sorted_words"] = df["nn"].apply(lambda s: "_".join(sorted(s.split()[:3])) if s else "")
    return df

def build_country_blocking_indexes(s2_df, s3_df):
    idx_addr_num_street = defaultdict(list)
    idx_addr_num_post = defaultdict(list)
    idx_name_sorted = defaultdict(list)
    idx_tok01 = defaultdict(list)
    idx_tok0_post = defaultdict(list)
    idx_tok0_num = defaultdict(list)
    idx_pref4 = defaultdict(list)
    
    for df in (s2_df, s3_df):
        eids = df["entity_id"].values
        tok0s = df["tok0"].values
        tok1s = df["tok1"].values
        postals = df["postal"].values
        street_nums = df["street_num"].values
        street_words = df["street_word"].values
        sorted_wordss = df["sorted_words"].values
        nns = df["nn"].values
        
        for i in range(len(eids)):
            eid = eids[i]
            t0 = tok0s[i]
            t1 = tok1s[i]
            post = postals[i]
            snum = street_nums[i]
            sword = street_words[i]
            swords = sorted_wordss[i]
            name = nns[i]
            
            if snum and sword:
                idx_addr_num_street[f"{snum}_{sword}"].append(eid)
            if snum and post:
                idx_addr_num_post[f"{snum}_{post}"].append(eid)
            if swords:
                idx_name_sorted[swords].append(eid)
            if t0 and t1:
                idx_tok01[f"{t0}_{t1}"].append(eid)
            if t0 and post:
                idx_tok0_post[f"{t0}_{post}"].append(eid)
            if t0 and snum:
                idx_tok0_num[f"{t0}_{snum}"].append(eid)
            if len(name) >= 4:
                idx_pref4[name[:4]].append(eid)
                
    return idx_addr_num_street, idx_addr_num_post, idx_name_sorted, idx_tok01, idx_tok0_post, idx_tok0_num, idx_pref4

def retrieve_candidates_batch(s1_batch, idx_addr_num_street, idx_addr_num_post, idx_name_sorted, idx_tok01, idx_tok0_post, idx_tok0_num, idx_pref4, max_per_s1=15):
    eids = s1_batch["entity_id"].values
    tok0s = s1_batch["tok0"].values
    tok1s = s1_batch["tok1"].values
    postals = s1_batch["postal"].values
    street_nums = s1_batch["street_num"].values
    street_words = s1_batch["street_word"].values
    sorted_wordss = s1_batch["sorted_words"].values
    nns = s1_batch["nn"].values
    
    candidates = {}
    for i in range(len(eids)):
        sid = eids[i]
        t0 = tok0s[i]
        t1 = tok1s[i]
        post = postals[i]
        snum = street_nums[i]
        sword = street_words[i]
        swords = sorted_wordss[i]
        name = nns[i]
        
        c_set = set()
        if snum and sword:
            c_set.update(idx_addr_num_street.get(f"{snum}_{sword}", [])[:10])
        if snum and post:
            c_set.update(idx_addr_num_post.get(f"{snum}_{post}", [])[:10])
        if swords:
            c_set.update(idx_name_sorted.get(swords, [])[:10])
        if t0 and t1:
            c_set.update(idx_tok01.get(f"{t0}_{t1}", [])[:10])
        if t0 and post:
            c_set.update(idx_tok0_post.get(f"{t0}_{post}", [])[:10])
        if t0 and snum:
            c_set.update(idx_tok0_num.get(f"{t0}_{snum}", [])[:10])
        if len(c_set) == 0 and len(name) >= 4:
            c_set.update(idx_pref4.get(name[:4], [])[:10])
            
        candidates[sid] = list(c_set)[:max_per_s1]
        
    return candidates

# ============================================================
# OPTIMIZED V2 FEATURE ENGINEERING (26 features)
# ============================================================
def clean_str_v2(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def extract_meta_fast(name_raw, addr_raw, country_raw):
    n_clean = clean_str_v2(name_raw)
    a_clean = clean_str_v2(addr_raw)
    c_clean = clean_str_v2(country_raw)
    
    n_tokens = n_clean.split()
    compact_name = "".join(n_tokens)
    
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
            
    n_words = set(n_tokens)
    a_words = set(a_tokens)
    n_len = len(n_clean)
    a_len = len(a_clean)
    n_3g = set(n_clean[i:i+3] for i in range(n_len-2)) if n_len >= 3 else set()
    a_3g = set(a_clean[i:i+3] for i in range(a_len-2)) if a_len >= 3 else set()

    return {
        "n_clean": n_clean,
        "a_clean": a_clean,
        "c_clean": c_clean,
        "compact_name": compact_name,
        "post": post,
        "snum": snum,
        "sword": sword,
        "n_words": n_words,
        "a_words": a_words,
        "n_3g": n_3g,
        "a_3g": a_3g,
        "n_len": n_len,
        "a_len": a_len,
    }

def compute_v2_features_fast(s1, cand, cand_id):
    s1_n = s1["n_clean"]
    c_n = cand["n_clean"]
    s1_a = s1["a_clean"]
    c_a = cand["a_clean"]
    
    # 1. Name features (11)
    if s1_n and c_n:
        if s1_n == c_n:
            n_lev = 1.0
            n_jw = 1.0
            n_sort = 1.0
            n_set = 1.0
            n_jac = 1.0
            n_overlap = float(len(s1["n_words"]))
            n_contain = 1.0
            n_compact = 1.0
            n_3gram = 1.0
            n_len_diff = 0.0
            n_exact = 1.0
        else:
            n_lev = rf_distance.Levenshtein.normalized_similarity(s1_n, c_n)
            n_jw = rf_distance.JaroWinkler.similarity(s1_n, c_n)
            n_sort = fuzz.token_sort_ratio(s1_n, c_n) / 100.0
            n_set = fuzz.token_set_ratio(s1_n, c_n) / 100.0
            
            t1 = s1["n_words"]
            t2 = cand["n_words"]
            inter = len(t1 & t2)
            union = len(t1 | t2)
            n_jac = inter / union if union > 0 else 0.0
            n_overlap = float(inter)
            min_len = min(len(t1), len(t2))
            n_contain = inter / min_len if min_len > 0 else 0.0
            
            n_compact = rf_distance.Levenshtein.normalized_similarity(s1["compact_name"], cand["compact_name"]) if (s1["compact_name"] and cand["compact_name"]) else 0.0
            
            g1 = s1["n_3g"]
            g2 = cand["n_3g"]
            g_inter = len(g1 & g2)
            g_union = len(g1 | g2)
            n_3gram = g_inter / g_union if g_union > 0 else 0.0
            
            n_len_diff = abs(s1["n_len"] - cand["n_len"]) / max(s1["n_len"], cand["n_len"], 1)
            n_exact = 0.0
    else:
        n_lev = n_jw = n_sort = n_set = n_jac = n_overlap = n_contain = n_compact = n_3gram = n_exact = 0.0
        n_len_diff = abs(s1["n_len"] - cand["n_len"]) / max(s1["n_len"], cand["n_len"], 1)
        
    # 2. Address features (9)
    if s1_a and c_a:
        if s1_a == c_a:
            a_lev = 1.0
            a_sort = 1.0
            a_jac = 1.0
            a_overlap = float(len(s1["a_words"]))
            a_3gram = 1.0
            a_len_diff = 0.0
        else:
            a_lev = rf_distance.Levenshtein.normalized_similarity(s1_a, c_a)
            a_sort = fuzz.token_sort_ratio(s1_a, c_a) / 100.0
            
            at1 = s1["a_words"]
            at2 = cand["a_words"]
            a_inter = len(at1 & at2)
            a_union = len(at1 | at2)
            a_jac = a_inter / a_union if a_union > 0 else 0.0
            a_overlap = float(a_inter)
            
            ag1 = s1["a_3g"]
            ag2 = cand["a_3g"]
            ag_inter = len(ag1 & ag2)
            ag_union = len(ag1 | ag2)
            a_3gram = ag_inter / ag_union if ag_union > 0 else 0.0
            
            a_len_diff = abs(s1["a_len"] - cand["a_len"]) / max(s1["a_len"], cand["a_len"], 1)
    else:
        a_lev = a_sort = a_jac = a_overlap = a_3gram = 0.0
        a_len_diff = abs(s1["a_len"] - cand["a_len"]) / max(s1["a_len"], cand["a_len"], 1)
        
    p1 = s1["post"]
    p2 = cand["post"]
    if p1 and p2:
        post_m = 1.0 if p1 == p2 else -1.0
    else:
        post_m = 0.0
        
    sn1 = s1["snum"]
    sn2 = cand["snum"]
    if sn1 and sn2:
        snum_m = 1.0 if sn1 == sn2 else -1.0
    else:
        snum_m = 0.0
        
    sw1 = s1["sword"]
    sw2 = cand["sword"]
    sword_m = 1.0 if (sw1 and sw2 and sw1 == sw2) else 0.0
    
    # 3. Cross & Meta (6)
    c_match = 1.0 if s1["c_clean"] == cand["c_clean"] else 0.0
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

def compute_v2_features_batch(s1_meta_dict, cand_raw_dict, cand_meta_cache, candidate_map):
    pairs = []
    features = []
    
    for sid, c_list in candidate_map.items():
        s1_meta = s1_meta_dict.get(sid)
        if s1_meta is None:
            continue
        
        for cid in c_list:
            c_meta = cand_meta_cache.get(cid)
            if c_meta is None:
                c_raw = cand_raw_dict.get(cid)
                if c_raw is None:
                    continue
                c_meta = extract_meta_fast(c_raw[0], c_raw[1], c_raw[2])
                cand_meta_cache[cid] = c_meta
            
            feat = compute_v2_features_fast(s1_meta, c_meta, cid)
            pairs.append((sid, cid))
            features.append(feat)
    
    if not features:
        return pairs, None
    
    X = np.array(features, dtype=np.float32)
    return pairs, X

# ============================================================
# RESUME INFERENCE & AUTO GIT PUSH
# ============================================================
def resume_inference():
    t_start = time.time()
    log("=" * 70)
    log("  AMAZON ML CHALLENGE: RESUMING INFERENCE FOR INDIA (540,000 -> 809,986)")
    log("  Model: 0.5*LGB + 0.5*XGB | Features: 26 | Threshold: 0.93")
    log("=" * 70)
    
    # 1. Load V2 Models & Threshold
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    with open(THRESH_PATH, "rb") as f:
        thresh_data = pickle.load(f)
    
    threshold = thresh_data["threshold"]
    log(f"  Loaded LGB model: {type(lgb_model).__name__} ({lgb_model.n_features_} features)")
    log(f"  Loaded XGB model: {type(xgb_model).__name__}")
    log(f"  Decision Threshold: {threshold}")
    
    # 2. Check current output lines
    match_file = OUT_DIR / "matching_results.tsv"
    cand_file = OUT_DIR / "candidate_pairs.tsv"
    
    with open(match_file, "r", encoding="utf-8") as f:
        f.readline()
        existing_matches = sum(1 for _ in f)
    log(f"  Current rows in matching_results.tsv: {existing_matches:,}")
    
    START_INDEX = 540_000
    expected_before_resume = 663_106 + 259_452 + START_INDEX
    if existing_matches != expected_before_resume:
        log(f"  WARNING: existing matches {existing_matches} != expected {expected_before_resume}")
    
    # 3. Load India S1 Entities
    log(f"\n[1] Reading test_source1.tsv for India...")
    s1_df = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_df["nc"] = s1_df["country"].fillna("").astype(str).str.lower().str.strip()
    
    india_mask = s1_df["nc"] == "india"
    c_s1_full = s1_df[india_mask].reset_index(drop=True)
    c_s1_full_raw = c_s1_full.copy()
    c_s1_full = preprocess_df(c_s1_full)
    
    total_india = len(c_s1_full)
    log(f"  Total India S1 entities: {total_india:,}")
    log(f"  Resuming from index: {START_INDEX:,} to {total_india:,} ({total_india - START_INDEX:,} entities remaining)")
    
    # Slice the remaining entities
    c_s1 = c_s1_full.iloc[START_INDEX:].reset_index(drop=True)
    c_s1_raw = c_s1_full_raw.iloc[START_INDEX:].reset_index(drop=True)
    n_c_s1 = len(c_s1)
    
    del s1_df, c_s1_full, c_s1_full_raw
    gc.collect()
    
    # 4. Load India S2 and S3
    log(f"\n[2] Loading test_source2 for 'india'...")
    s2_chunks = []
    for chunk in pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
        sub = chunk[chunk["country"].str.lower().str.strip() == "india"]
        if len(sub) > 0:
            s2_chunks.append(sub)
    c_s2_raw = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    del s2_chunks
    gc.collect()
    
    log(f"  Loading test_source3 for 'india'...")
    s3_chunks = []
    for chunk in pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
        sub = chunk[chunk["country"].str.lower().str.strip() == "india"]
        if len(sub) > 0:
            s3_chunks.append(sub)
    c_s3_raw = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    del s3_chunks
    gc.collect()
    
    log(f"  Loaded: S2={len(c_s2_raw):,}, S3={len(c_s3_raw):,}")
    
    # Build candidate lookup dict
    log(f"  Building candidate lookup dict...")
    t_dict0 = time.time()
    cand_raw_dict = {}
    for eid, bn, ba, co in zip(c_s2_raw["entity_id"], c_s2_raw["business_name"], c_s2_raw["business_address"], c_s2_raw["country"]):
        cand_raw_dict[eid] = (bn, ba, co)
    for eid, bn, ba, co in zip(c_s3_raw["entity_id"], c_s3_raw["business_name"], c_s3_raw["business_address"], c_s3_raw["country"]):
        cand_raw_dict[eid] = (bn, ba, co)
    log(f"  Candidate lookup built in {time.time()-t_dict0:.2f}s ({len(cand_raw_dict):,} entries)")
    
    # Preprocess candidates for blocking
    log(f"  Preprocessing candidates for blocking indexes...")
    c_s2_prep = preprocess_df(c_s2_raw)
    c_s3_prep = preprocess_df(c_s3_raw)
    del c_s2_raw, c_s3_raw
    gc.collect()
    
    # Build blocking indexes
    log(f"  Building blocking indexes...")
    indexes = build_country_blocking_indexes(c_s2_prep, c_s3_prep)
    del c_s2_prep, c_s3_prep
    gc.collect()
    
    cand_meta_cache = {}
    
    # 5. Batch inference appending to files
    batch_size = 20_000
    resumed_matches_found = 0
    resumed_entities_written = 0
    c_t0 = time.time()
    
    log(f"\n[3] Processing batches in append mode...")
    with open(match_file, "a", encoding="utf-8") as f_m, open(cand_file, "a", encoding="utf-8") as f_c:
        for b_st in range(0, n_c_s1, batch_size):
            b_en = min(b_st + batch_size, n_c_s1)
            b_df = c_s1.iloc[b_st:b_en]
            b_raw = c_s1_raw.iloc[b_st:b_en]
            
            # Retrieve candidates
            c_map = retrieve_candidates_batch(b_df, *indexes, max_per_s1=15)
            
            # S1 meta dict
            s1_meta_dict = {}
            for eid, bn, ba, co in zip(b_raw["entity_id"], b_raw["business_name"], b_raw["business_address"], b_raw["country"]):
                s1_meta_dict[eid] = extract_meta_fast(bn, ba, co)
            
            # Features
            pairs, X = compute_v2_features_batch(s1_meta_dict, cand_raw_dict, cand_meta_cache, c_map)
            
            # Scoring
            b_matches = defaultdict(list)
            if X is not None and len(X) > 0:
                lgb_scores = lgb_model.predict_proba(X)[:, 1]
                xgb_scores = xgb_model.predict_proba(X)[:, 1]
                ensemble_scores = 0.5 * lgb_scores + 0.5 * xgb_scores
                
                for p_idx, score in enumerate(ensemble_scores):
                    sid, cid = pairs[p_idx]
                    if score >= threshold:
                        b_matches[sid].append(cid)
                        resumed_matches_found += 1
            
            # Stream write
            for sid in b_df["entity_id"]:
                resumed_entities_written += 1
                c_list = sorted(list(set(c_map.get(sid, []))))
                if c_list:
                    f_c.write(f"{sid}\t{','.join(c_list)}\n")
                else:
                    f_c.write(f"{sid}\t\n")
                
                m_list = sorted(list(set(b_matches.get(sid, []))))
                if m_list:
                    f_m.write(f"{sid}\t{','.join(m_list)}\n")
                else:
                    f_m.write(f"{sid}\t\n")
            
            f_m.flush()
            f_c.flush()
            
            elapsed = time.time() - c_t0
            rate = b_en / elapsed if elapsed > 0 else 0
            log(f"    [INDIA-RESUME] {b_en:,}/{n_c_s1:,} ({rate:.0f} S1/s) | New Matches: {resumed_matches_found:,} | Cached: {len(cand_meta_cache):,}")
    
    del cand_raw_dict, cand_meta_cache, indexes, c_s1, c_s1_raw
    gc.collect()
    
    # 6. Verification of total output rows
    with open(match_file, "r", encoding="utf-8") as f:
        f.readline()
        final_total = sum(1 for _ in f)
    
    log(f"\n[4] Complete! Final total S1 rows in matching_results.tsv: {final_total:,}")
    assert final_total == 1_732_544, f"Expected 1,732,544 rows but got {final_total:,}!"
    log("  ✅ Full 1,732,544 S1 test entities verified!")
    
    # 7. Copy output files to all targets
    log(f"\n[5] Copying output files to targets...")
    for target_dir in (OUT_DIR_ROOT, OUT_DIR_GIT):
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(match_file, target_dir / "matching_results.tsv")
        shutil.copy2(cand_file, target_dir / "candidate_pairs.tsv")
        log(f"  Copied output files to: {target_dir}")
        
    # Copy code files to AmazonMLChallenge git repo
    git_code_dir = GIT_REPO_DIR / "code/business_entity_resolution/src"
    git_code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_DIR / "code/business_entity_resolution/src/test_inference_v2.py", git_code_dir / "test_inference_v2.py")
    shutil.copy2(BASE_DIR / "code/business_entity_resolution/src/test_inference_v2.py", git_code_dir / "test_inference.py")
    shutil.copy2(BASE_DIR / "code/business_entity_resolution/src/resume_inference.py", git_code_dir / "resume_inference.py")
    log(f"  Copied updated inference code to git repo code directory.")
    
    # 8. Run official validator
    log(f"\n[6] Running official submission validator...")
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(match_file),
        "--candidate", str(cand_file),
        "--test-dir", str(TEST_DIR)
    ]
    res = subprocess.run(validator_cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    log(res.stdout)
    if res.stderr:
        log("Validator stderr:", res.stderr)
    
    is_pass = res.returncode == 0 and "PASS" in res.stdout
    log(f"  VALIDATOR RESULT: {'PASS' if is_pass else 'FAIL'}")
    
    if not is_pass:
        log("  ❌ Validator failed! Aborting git push.")
        return False
        
    # 9. Git Add, Commit, and Push
    log(f"\n[7] Pushing to Git repository (origin main)...")
    git_cmds = [
        ["git", "add", "code/business_entity_resolution/src/test_inference.py", "code/business_entity_resolution/src/test_inference_v2.py", "code/business_entity_resolution/src/resume_inference.py"],
        ["git", "commit", "-m", "fix: V2 ensemble inference with French domain fixes and optimal precision threshold tau=0.93"],
        ["git", "push", "origin", "main"]
    ]
    
    for cmd in git_cmds:
        log(f"  Running: {' '.join(cmd)}")
        git_res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(GIT_REPO_DIR))
        log(git_res.stdout)
        if git_res.stderr:
            log(git_res.stderr)
        if git_res.returncode != 0:
            log(f"  Git command failed with code {git_res.returncode}")
            return False
            
    log("=" * 70)
    log(f"  SUCCESS! All 1,732,544 entities processed, verified, and pushed to Git.")
    log(f"  TOTAL RESUME TIME: {time.time()-t_start:.1f}s ({int((time.time()-t_start)//60)}m {int((time.time()-t_start)%60)}s)")
    log("=" * 70)
    return True

if __name__ == "__main__":
    ok = resume_inference()
    sys.exit(0 if ok else 1)
