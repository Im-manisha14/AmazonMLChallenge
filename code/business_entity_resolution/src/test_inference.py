"""
Amazon ML Challenge 2026: Fast, Robust Test Inference Module
============================================================
Processes all 1,732,544 test S1 entities across France, US, and India.
Uses memory-efficient country partitioning and multi-strategy candidate blocking.
Produces valid output/matching_results.tsv and output/candidate_pairs.tsv.
"""

import os
import sys
import gc
import re
import time
import pickle
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler

BASE_DIR = Path(".")
TEST_DIR = BASE_DIR / "dataset/test"
OUT_DIR = BASE_DIR / "output"
ART_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ART_DIR / "models/lgbm_model.pkl"
THRESH_PATH = ART_DIR / "models/threshold.pkl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r'&', ' and ', regex=False)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nn"] = clean_series(df["business_name"])
    df["na"] = clean_series(df["business_address"])
    df["nc"] = df["country"].fillna("").astype(str).str.lower().str.strip()
    df["tok0"] = df["nn"].str.split().str[0].fillna("")
    df["pref4"] = df["nn"].str[:4].fillna("")
    df["pref3"] = df["nn"].str[:3].fillna("")
    df["postal"] = df["na"].str.extract(r'\b(\d{5,6})\b', expand=False).fillna("")
    return df

def build_country_blocking_indexes(s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    idx_tok0 = defaultdict(list)
    idx_tok01 = defaultdict(list)
    idx_pref4 = defaultdict(list)
    idx_postal = defaultdict(list)
    idx_pref3 = defaultdict(list)
    
    for df in (s2_df, s3_df):
        eids = df["entity_id"].values
        nns = df["nn"].values
        tok0s = df["tok0"].values
        pref4s = df["pref4"].values
        pref3s = df["pref3"].values
        postals = df["postal"].values
        
        for i in range(len(eids)):
            eid = eids[i]
            t0 = tok0s[i]
            p4 = pref4s[i]
            p3 = pref3s[i]
            post = postals[i]
            name = nns[i]
            
            if t0:
                idx_tok0[t0].append(eid)
            if p4:
                idx_pref4[p4].append(eid)
            if p3:
                idx_pref3[p3].append(eid)
            if post:
                idx_postal[post].append(eid)
                
            toks = name.split(maxsplit=2)
            if len(toks) >= 2:
                idx_tok01[f"{toks[0]}|{toks[1]}"].append(eid)
                
    return idx_tok0, idx_tok01, idx_pref4, idx_postal, idx_pref3

def retrieve_candidates_batch(s1_batch: pd.DataFrame, idx_tok0, idx_tok01, idx_pref4, idx_postal, idx_pref3, max_per_s1=20):
    eids = s1_batch["entity_id"].values
    nns = s1_batch["nn"].values
    tok0s = s1_batch["tok0"].values
    pref4s = s1_batch["pref4"].values
    pref3s = s1_batch["pref3"].values
    postals = s1_batch["postal"].values
    
    candidates = {}
    for i in range(len(eids)):
        sid = eids[i]
        t0 = tok0s[i]
        p4 = pref4s[i]
        p3 = pref3s[i]
        post = postals[i]
        name = nns[i]
        
        c_set = set()
        if t0:
            c_set.update(idx_tok0.get(t0, [])[:15])
            
        toks = name.split(maxsplit=2)
        if len(toks) >= 2:
            c_set.update(idx_tok01.get(f"{toks[0]}|{toks[1]}", [])[:10])
            
        if p4:
            c_set.update(idx_pref4.get(p4, [])[:10])
            
        if post:
            c_set.update(idx_postal.get(post, [])[:10])
            
        if len(c_set) == 0 and p3:
            c_set.update(idx_pref3.get(p3, [])[:8])
            
        candidates[sid] = list(c_set)[:max_per_s1]
        
    return candidates

def compute_pairwise_features_fast(s1_dict, cand_dict, candidate_map):
    pairs = [(sid, cid) for sid, c_list in candidate_map.items() for cid in c_list if cid in cand_dict]
    N = len(pairs)
    if N == 0:
        return pairs, None
        
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
        
        # Name similarity
        if n1 and n1 == n2:
            n_exact[i] = 1; n_lev[i] = 1.0; n_jw[i] = 1.0; n_jac[i] = 1.0; n_cgram[i] = 1.0; len_n_ratio[i] = 1.0
        else:
            n_lev[i] = Levenshtein.normalized_similarity(n1, n2)
            n_jw[i] = JaroWinkler.similarity(n1, n2)
            t1, t2 = set(n1.split()), set(n2.split())
            u = t1 | t2
            n_jac[i] = len(t1 & t2) / len(u) if u else 0.0
            g1 = {n1[k:k+3] for k in range(len(n1)-2)}
            g2 = {n2[k:k+3] for k in range(len(n2)-2)}
            gu = g1 | g2
            n_cgram[i] = len(g1 & g2) / len(gu) if gu else 0.0
            l1, l2 = len(n1), len(n2)
            len_n_ratio[i] = min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 1.0
            
        # Address similarity
        if a1 and a1 == a2:
            a_exact[i] = 1; a_lev[i] = 1.0; a_jw[i] = 1.0; a_jac[i] = 1.0; a_cgram[i] = 1.0; len_a_ratio[i] = 1.0
        else:
            a_lev[i] = Levenshtein.normalized_similarity(a1, a2)
            a_jw[i] = JaroWinkler.similarity(a1, a2)
            at1, at2 = set(a1.split()), set(a2.split())
            au = at1 | at2
            a_jac[i] = len(at1 & at2) / len(au) if au else 0.0
            ag1 = {a1[k:k+3] for k in range(len(a1)-2)}
            ag2 = {a2[k:k+3] for k in range(len(a2)-2)}
            agu = ag1 | ag2
            a_cgram[i] = len(ag1 & ag2) / len(agu) if agu else 0.0
            al1, al2 = len(a1), len(a2)
            len_a_ratio[i] = min(al1, al2) / max(al1, al2) if max(al1, al2) > 0 else 1.0
            
        c_exact[i] = 1 if (c1 and c1 == c2) else 0
        post_exact[i] = 1 if (p1 and p1 == p2) else 0
        
        num1 = set(re.findall(r'\d+', a1))
        num2 = set(re.findall(r'\d+', a2))
        nu = num1 | num2
        num_jac[i] = len(num1 & num2) / len(nu) if nu else 0.0
        is_s2[i] = 1 if cid.startswith("S2-") else 0
        
    hi_conf = ((n_exact == 1) & (a_exact == 1) & (c_exact == 1)).astype(np.int8)
    high_sim_all = ((n_lev > 0.85) & (a_lev > 0.80) & (c_exact == 1)).astype(np.int8)
    
    X = np.column_stack([
        n_exact, n_lev, n_jw, n_jac, n_cgram, len_n_ratio,
        a_exact, a_lev, a_jw, a_jac, a_cgram, len_a_ratio,
        c_exact, post_exact, num_jac, is_s2, hi_conf, high_sim_all
    ]).astype(np.float32)
    
    return pairs, X

def run_test_inference_and_validate():
    t_start = time.time()
    log("=" * 70)
    log("  AMAZON ML CHALLENGE: FAST TEST INFERENCE & SUBMISSION GENERATOR")
    log("=" * 70)
    
    # 1. Load Model & Threshold
    if not MODEL_PATH.exists() or not THRESH_PATH.exists():
        log(f"Error: Model or threshold not found at {MODEL_PATH} / {THRESH_PATH}")
        sys.exit(1)
        
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(THRESH_PATH, "rb") as f:
        threshold = pickle.load(f)
        
    log(f"  Loaded model: {type(model).__name__}")
    log(f"  Loaded optimal decision threshold: {threshold:.4f}")
    
    # 2. Load Test Source 1 (Required Entities)
    log(f"\n[1] Reading test_source1.tsv...")
    s1_df = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    all_s1_ids = s1_df["entity_id"].tolist()
    total_s1 = len(all_s1_ids)
    log(f"  Total test S1 entities: {total_s1:,}")
    
    s1_df = preprocess_df(s1_df)
    countries = s1_df["nc"].unique()
    log(f"  Detected countries in test S1: {list(countries)}")
    
    # Prepare global dictionaries for results
    test_matches = defaultdict(list)
    test_candidates = defaultdict(list)
    
    # 3. Country Partitioned Processing
    for country in countries:
        c_t0 = time.time()
        log(f"\n" + "-" * 60)
        log(f"  PROCESSING COUNTRY: '{country.upper()}'")
        log("-" * 60)
        
        c_s1 = s1_df[s1_df["nc"] == country].reset_index(drop=True)
        log(f"  Country '{country}': {len(c_s1):,} S1 entities")
        
        # Load only matching country rows from test_source2 and test_source3
        log(f"  Loading test_source2 for '{country}'...")
        s2_chunks = []
        for chunk in pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
            sub = chunk[chunk["country"].str.lower().str.strip() == country]
            if len(sub) > 0:
                s2_chunks.append(sub)
        c_s2 = pd.concat(s2_chunks, ignore_index=True) if s2_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        del s2_chunks
        gc.collect()
        
        log(f"  Loading test_source3 for '{country}'...")
        s3_chunks = []
        for chunk in pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
            sub = chunk[chunk["country"].str.lower().str.strip() == country]
            if len(sub) > 0:
                s3_chunks.append(sub)
        c_s3 = pd.concat(s3_chunks, ignore_index=True) if s3_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        del s3_chunks
        gc.collect()
        
        log(f"  Loaded country candidates: S2={len(c_s2):,}, S3={len(c_s3):,}")
        
        if len(c_s2) == 0 and len(c_s3) == 0:
            log(f"  Warning: No S2/S3 candidates found for country '{country}'. S1 will be singletons.")
            continue
            
        c_s2 = preprocess_df(c_s2)
        c_s3 = preprocess_df(c_s3)
        
        # Build candidate lookup dict
        c_all_cands_df = pd.concat([c_s2, c_s3], ignore_index=True)
        cand_dict = c_all_cands_df.set_index("entity_id")[["nn", "na", "nc", "postal"]].to_dict("index")
        
        # Build inverted blocking indexes
        log(f"  Building blocking indexes for '{country}'...")
        idx_tok0, idx_tok01, idx_pref4, idx_postal, idx_pref3 = build_country_blocking_indexes(c_s2, c_s3)
        del c_s2, c_s3
        gc.collect()
        
        # Batch inference on c_s1
        batch_size = 25_000
        n_c_s1 = len(c_s1)
        country_matches_found = 0
        
        for b_st in range(0, n_c_s1, batch_size):
            b_en = min(b_st + batch_size, n_c_s1)
            b_df = c_s1.iloc[b_st:b_en]
            
            # 1. Retrieve candidates
            c_map = retrieve_candidates_batch(b_df, idx_tok0, idx_tok01, idx_pref4, idx_postal, idx_pref3, max_per_s1=20)
            for sid, c_list in c_map.items():
                test_candidates[sid] = c_list
                
            # 2. Fast feature extraction
            s1_dict = b_df.set_index("entity_id")[["nn", "na", "nc", "postal"]].to_dict("index")
            pairs, X = compute_pairwise_features_fast(s1_dict, cand_dict, c_map)
            
            # 3. Model scoring
            if X is not None and len(X) > 0:
                scores = model.predict_proba(X)[:, 1]
                for p_idx, score in enumerate(scores):
                    if score >= threshold:
                        sid, cid = pairs[p_idx]
                        test_matches[sid].append(cid)
                        country_matches_found += 1
                        
            if b_en % 50_000 == 0 or b_en == n_c_s1:
                log(f"    Progress [{country.upper()}]: {b_en:,}/{n_c_s1:,} S1 entities processed (Matches found: {country_matches_found:,})")
                
        log(f"  Completed country '{country}' in {time.time()-c_t0:.1f}s. Matches: {country_matches_found:,}")
        del cand_dict, c_all_cands_df, idx_tok0, idx_tok01, idx_pref4, idx_postal, idx_pref3, c_s1
        gc.collect()
        
    del s1_df
    gc.collect()
    
    # 4. Write Submission Files
    match_file = OUT_DIR / "matching_results.tsv"
    cand_file = OUT_DIR / "candidate_pairs.tsv"
    
    log(f"\n[2] Writing matching_results.tsv to {match_file}...")
    matched_s1_count = 0
    with open(match_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            m_list = sorted(list(set(test_matches.get(sid, []))))
            if m_list:
                matched_s1_count += 1
                f.write(f"{sid}\t{','.join(m_list)}\n")
            else:
                f.write(f"{sid}\t\n")
                
    log(f"  matching_results.tsv complete: {matched_s1_count:,} non-singleton entities, {total_s1 - matched_s1_count:,} singletons.")
    
    log(f"\n[3] Writing candidate_pairs.tsv to {cand_file}...")
    with open(cand_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            c_list = sorted(list(set(test_candidates.get(sid, []))))
            if c_list:
                f.write(f"{sid}\t{','.join(c_list)}\n")
            else:
                f.write(f"{sid}\t\n")
                
    log(f"  candidate_pairs.tsv complete.")
    
    # 5. Run Official Submission Validator
    log(f"\n[4] Running official submission validator...")
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", str(match_file),
        "--candidate", str(cand_file),
        "--test-dir", str(TEST_DIR)
    ]
    log(f"  Command: {' '.join(validator_cmd)}")
    res = subprocess.run(validator_cmd, capture_output=True, text=True)
    log(res.stdout)
    if res.stderr:
        log("Validator stderr:", res.stderr)
        
    log("=" * 70)
    log(f"  TOTAL INFERENCE & VALIDATION TIME: {time.time()-t_start:.1f}s ({int((time.time()-t_start)//60)}m {int((time.time()-t_start)%60)}s)")
    log(f"  VALIDATOR RESULT: {'PASS' if res.returncode == 0 and 'PASS' in res.stdout else 'FAIL'}")
    log("=" * 70)
    
    return res.returncode == 0 and "PASS" in res.stdout

if __name__ == "__main__":
    success = run_test_inference_and_validate()
    sys.exit(0 if success else 1)
