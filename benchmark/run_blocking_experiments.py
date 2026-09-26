"""
run_blocking_experiments.py - Phase 3 & 4 Blocking Benchmark & Candidate Miss Diagnostics
========================================================================================
Evaluates blocking strategies on Validation A (10,000 S1 entities) against the FULL 10.3M
records in train_source2.tsv (5.03M) and train_source3.tsv (5.28M).

Measures:
- Candidate Recall (Overall, S2, S3, US, India)
- Reduction Ratio
- Average Candidates / S1, P95, Max
- Runtime & Memory
- Logs to experiments/experiment_log.csv
- Performs Phase 4 Candidate Miss Diagnostics
"""

import sys
import os
import re
import time
import unicodedata
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path("c:/ml challenge/student_resource")
TRAIN_DIR = BASE_DIR / "dataset/train"
BENCH_DIR = BASE_DIR / "benchmark"
EXP_DIR = BASE_DIR / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Normalization & Key Extraction Functions
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

def extract_features(name_raw, addr_raw, country_raw):
    n_clean = clean_str(name_raw)
    a_clean = clean_str(addr_raw)
    c_clean = clean_str(country_raw)
    
    n_tokens = n_clean.split()
    # Meaningful tokens
    m_tokens = [t for t in n_tokens if t not in STOPWORDS and len(t) >= 2]
    
    # Name keys
    compact_name = "".join(n_tokens)
    sorted_words = " ".join(sorted(n_tokens[:6])) if n_tokens else ""
    tok0 = m_tokens[0] if m_tokens else (n_tokens[0] if n_tokens else "")
    tok1 = m_tokens[1] if len(m_tokens) > 1 else (n_tokens[1] if len(n_tokens) > 1 else "")
    pref4 = n_clean[:4] if len(n_clean) >= 4 else n_clean
    
    # Address keys
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
        "compact_name": compact_name,
        "sorted_words": sorted_words,
        "tok0": tok0,
        "tok1": tok1,
        "pref4": pref4,
        "snum": snum,
        "sword": sword,
        "post": post,
        "country": c_clean
    }

# ----------------------------------------------------------------------
# 2. Main Benchmark Routine
# ----------------------------------------------------------------------

def run_experiments():
    print("=" * 80)
    print("  PHASE 3 & 4: COMPREHENSIVE BLOCKING BENCHMARK & MISS DIAGNOSTICS")
    print("=" * 80)
    t0_all = time.time()
    
    # 2.1 Load Validation A
    print("\n[Step 1] Loading Validation A (10,000 S1 entities)...")
    val_s1_df = pd.read_csv(BENCH_DIR / "val_a_s1.tsv", sep="\t", dtype=str)
    val_gt_df = pd.read_csv(BENCH_DIR / "val_a_gt.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    gt_map = {}
    gt_s2_map = {}
    gt_s3_map = {}
    total_true_matches = 0
    total_true_s2 = 0
    total_true_s3 = 0
    
    for sid, matches in zip(val_gt_df["source1_entity_id"], val_gt_df["matched_entity_ids"]):
        m_str = str(matches).strip()
        if m_str:
            m_list = [m.strip() for m in m_str.split(",") if m.strip()]
            m_set = set(m_list)
            gt_map[sid] = m_set
            s2_set = {m for m in m_list if "s2" in m.lower()}
            s3_set = {m for m in m_list if "s3" in m.lower()}
            gt_s2_map[sid] = s2_set
            gt_s3_map[sid] = s3_set
            total_true_matches += len(m_set)
            total_true_s2 += len(s2_set)
            total_true_s3 += len(s3_set)
        else:
            gt_map[sid] = set()
            gt_s2_map[sid] = set()
            gt_s3_map[sid] = set()
            
    print(f"  Loaded {len(val_s1_df):,} S1 entities.")
    print(f"  Total Ground Truth matches: {total_true_matches:,} (S2: {total_true_s2:,}, S3: {total_true_s3:,})")
    
    # Pre-extract Val S1 keys
    print("  Extracting keys for Validation S1...")
    val_keys = {}
    val_country_map = {}
    for _, row in val_s1_df.iterrows():
        sid = row["entity_id"]
        val_keys[sid] = extract_features(row.get("business_name", ""), row.get("business_address", ""), row.get("country", ""))
        val_country_map[sid] = str(row.get("country", "")).strip().lower()
        
    # 2.2 Indexing Train S2 & Train S3
    print("\n[Step 2] Indexing Train S2 (5.03M) and Train S3 (5.28M)...")
    t0_idx = time.time()
    
    # Indexes to populate
    # A: Exact name
    idx_exact_name = defaultdict(list)
    # B: Compact name
    idx_compact_name = defaultdict(list)
    # C: Name sorted words
    idx_name_sorted = defaultdict(list)
    # D: Name tok0 + tok1
    idx_tok01 = defaultdict(list)
    # E: Name tok0 only (baseline style)
    idx_tok0 = defaultdict(list)
    # F: Address snum + sword
    idx_addr_snum_sword = defaultdict(list)
    # G: Address snum + postal
    idx_addr_snum_post = defaultdict(list)
    # H: Cross name + postal
    idx_tok0_post = defaultdict(list)
    # I: Cross name + snum
    idx_tok0_snum = defaultdict(list)
    # J: Name prefix 4
    idx_pref4 = defaultdict(list)
    
    total_target_records = 0
    
    for filename in ["train_source2.tsv", "train_source3.tsv"]:
        path = TRAIN_DIR / filename
        print(f"  Streaming {filename}...")
        t_file = time.time()
        count = 0
        for chunk in pd.read_csv(path, sep="\t", chunksize=500_000, dtype=str):
            for eid, name, addr, country in zip(chunk["entity_id"], chunk["business_name"], chunk["business_address"], chunk["country"]):
                feat = extract_features(name, addr, country)
                c_name = feat["n_clean"]
                comp = feat["compact_name"]
                swords = feat["sorted_words"]
                t0 = feat["tok0"]
                t1 = feat["tok1"]
                snum = feat["snum"]
                sword = feat["sword"]
                post = feat["post"]
                p4 = feat["pref4"]
                
                if c_name: idx_exact_name[c_name].append(eid)
                if comp and len(comp) >= 5: idx_compact_name[comp].append(eid)
                if swords: idx_name_sorted[swords].append(eid)
                if t0 and t1: idx_tok01[f"{t0}_{t1}"].append(eid)
                if t0: idx_tok0[t0].append(eid)
                if snum and sword: idx_addr_snum_sword[f"{snum}_{sword}"].append(eid)
                if snum and post: idx_addr_snum_post[f"{snum}_{post}"].append(eid)
                if t0 and post: idx_tok0_post[f"{t0}_{post}"].append(eid)
                if t0 and snum: idx_tok0_snum[f"{t0}_{snum}"].append(eid)
                if len(p4) >= 4: idx_pref4[p4].append(eid)
                
            count += len(chunk)
            total_target_records += len(chunk)
        print(f"    Indexed {count:,} records in {time.time()-t_file:.1f}s.")
        
    try:
        import resource
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        mem_mb = 0.0
    print(f"  Current Process Memory: {mem_mb:.1f} MB")
    
    # 2.3 Define Blocking Configurations to Benchmark
    configs = [
        {
            "id": "BLK-01-Baseline",
            "desc": "Baseline (Country/Tok0 only, top-10 cap)",
            "use_tok0": True,
            "use_exact": False,
            "use_compact": False,
            "use_sorted": False,
            "use_tok01": False,
            "use_addr": False,
            "use_cross": False,
            "use_pref4": False,
            "cap_per_rule": 10,
            "max_candidates": 20
        },
        {
            "id": "BLK-02-NameOnly",
            "desc": "Name-only (Exact, Compact, Sorted, Tok01, Tok0, Pref4)",
            "use_tok0": True,
            "use_exact": True,
            "use_compact": True,
            "use_sorted": True,
            "use_tok01": True,
            "use_addr": False,
            "use_cross": False,
            "use_pref4": True,
            "cap_per_rule": 15,
            "max_candidates": 100
        },
        {
            "id": "BLK-03-AddrOnly",
            "desc": "Address-only (StreetNum+StreetWord, StreetNum+Postal)",
            "use_tok0": False,
            "use_exact": False,
            "use_compact": False,
            "use_sorted": False,
            "use_tok01": False,
            "use_addr": True,
            "use_cross": False,
            "use_pref4": False,
            "cap_per_rule": 25,
            "max_candidates": 100
        },
        {
            "id": "BLK-04-MultiKey-Cap50",
            "desc": "Multi-Key (Name + Address + Cross-Keys) Budget=50",
            "use_tok0": False, # avoid noisy tok0
            "use_exact": True,
            "use_compact": True,
            "use_sorted": True,
            "use_tok01": True,
            "use_addr": True,
            "use_cross": True,
            "use_pref4": True,
            "cap_per_rule": 10,
            "max_candidates": 50
        },
        {
            "id": "BLK-05-MultiKey-Cap100",
            "desc": "Multi-Key (Name + Address + Cross-Keys) Budget=100",
            "use_tok0": False,
            "use_exact": True,
            "use_compact": True,
            "use_sorted": True,
            "use_tok01": True,
            "use_addr": True,
            "use_cross": True,
            "use_pref4": True,
            "cap_per_rule": 15,
            "max_candidates": 100
        },
        {
            "id": "BLK-06-MultiKey-Cap200",
            "desc": "Multi-Key (Name + Address + Cross-Keys) Budget=200",
            "use_tok0": False,
            "use_exact": True,
            "use_compact": True,
            "use_sorted": True,
            "use_tok01": True,
            "use_addr": True,
            "use_cross": True,
            "use_pref4": True,
            "cap_per_rule": 20,
            "max_candidates": 200
        },
        {
            "id": "BLK-07-MultiKey-Cap300",
            "desc": "Multi-Key (Name + Address + Cross-Keys) Budget=300",
            "use_tok0": False,
            "use_exact": True,
            "use_compact": True,
            "use_sorted": True,
            "use_tok01": True,
            "use_addr": True,
            "use_cross": True,
            "use_pref4": True,
            "cap_per_rule": 30,
            "max_candidates": 300
        },
        {
            "id": "BLK-08-MultiKey-AdaptiveBalanced",
            "desc": "Multi-Key + Balanced Source Floor (Min 25 S2, Min 25 S3) Budget=150",
            "use_tok0": False,
            "use_exact": True,
            "use_compact": True,
            "use_sorted": True,
            "use_tok01": True,
            "use_addr": True,
            "use_cross": True,
            "use_pref4": True,
            "cap_per_rule": 20,
            "max_candidates": 150,
            "source_balanced": True
        }
    ]
    
    # 2.4 Run Experiment Grid
    print("\n[Step 3] Running Blocking Experiment Suite...")
    exp_results = []
    best_cand_map = None
    best_config_id = None
    best_recall = 0.0
    
    for cfg in configs:
        t_start = time.time()
        cfg_id = cfg["id"]
        cap_rule = cfg["cap_per_rule"]
        max_cands = cfg["max_candidates"]
        is_balanced = cfg.get("source_balanced", False)
        
        cand_map = {}
        total_retrieved = 0
        hits = 0
        s2_hits = 0
        s3_hits = 0
        us_hits = 0
        us_total = 0
        in_hits = 0
        in_total = 0
        
        for sid, feat in val_keys.items():
            c_set = set()
            c_s2 = []
            c_s3 = []
            
            def add_cands(cand_list):
                for cid in cand_list[:cap_rule]:
                    if cid not in c_set:
                        c_set.add(cid)
                        if "s2" in cid.lower(): c_s2.append(cid)
                        else: c_s3.append(cid)
            
            # Address keys (High Precision)
            if cfg["use_addr"]:
                snum, sword, post = feat["snum"], feat["sword"], feat["post"]
                if snum and sword: add_cands(idx_addr_snum_sword.get(f"{snum}_{sword}", []))
                if snum and post: add_cands(idx_addr_snum_post.get(f"{snum}_{post}", []))
                
            # Exact & Compact Names
            if cfg["use_exact"] and feat["n_clean"]:
                add_cands(idx_exact_name.get(feat["n_clean"], []))
            if cfg["use_compact"] and feat["compact_name"] and len(feat["compact_name"]) >= 5:
                add_cands(idx_compact_name.get(feat["compact_name"], []))
                
            # Name Sorted Words & Two-tokens
            if cfg["use_sorted"] and feat["sorted_words"]:
                add_cands(idx_name_sorted.get(feat["sorted_words"], []))
            if cfg["use_tok01"] and feat["tok0"] and feat["tok1"]:
                add_cands(idx_tok01.get(f"{feat['tok0']}_{feat['tok1']}", []))
                
            # Cross Name + Address
            if cfg["use_cross"]:
                t0, post, snum = feat["tok0"], feat["post"], feat["snum"]
                if t0 and post: add_cands(idx_tok0_post.get(f"{t0}_{post}", []))
                if t0 and snum: add_cands(idx_tok0_snum.get(f"{t0}_{snum}", []))
                
            # Fallback single token or prefix
            if cfg["use_tok0"] and feat["tok0"]:
                add_cands(idx_tok0.get(feat["tok0"], []))
            if cfg["use_pref4"] and len(c_set) < 5 and len(feat["pref4"]) >= 4:
                add_cands(idx_pref4.get(feat["pref4"], []))
                
            # Final budgeting
            if is_balanced:
                # Guarantee min 25 from S2 and S3 if available
                quota_half = max_cands // 2
                final_s2 = c_s2[:quota_half]
                final_s3 = c_s3[:quota_half]
                rem = max_cands - (len(final_s2) + len(final_s3))
                if rem > 0 and len(c_s2) > len(final_s2):
                    final_s2 += c_s2[len(final_s2):len(final_s2)+rem]
                rem = max_cands - (len(final_s2) + len(final_s3))
                if rem > 0 and len(c_s3) > len(final_s3):
                    final_s3 += c_s3[len(final_s3):len(final_s3)+rem]
                final_cands = set(final_s2 + final_s3)
            else:
                final_cands = set(list(c_set)[:max_cands])
                
            cand_map[sid] = final_cands
            total_retrieved += len(final_cands)
            
            # Ground truth hits
            gt_s = gt_map[sid]
            matched_hits = len(final_cands & gt_s)
            hits += matched_hits
            s2_hits += len(final_cands & gt_s2_map[sid])
            s3_hits += len(final_cands & gt_s3_map[sid])
            
            if val_country_map[sid] == "us":
                us_hits += matched_hits
                us_total += len(gt_s)
            else:
                in_hits += matched_hits
                in_total += len(gt_s)
                
        t_elapsed = time.time() - t_start
        cand_counts = [len(c) for c in cand_map.values()]
        avg_c = np.mean(cand_counts)
        p95_c = np.percentile(cand_counts, 95)
        max_c = np.max(cand_counts)
        
        recall = hits / total_true_matches
        s2_rec = s2_hits / total_true_s2
        s3_rec = s3_hits / total_true_s3
        us_rec = us_hits / us_total if us_total else 0
        in_rec = in_hits / in_total if in_total else 0
        red_ratio = 1.0 - (total_retrieved / (len(val_s1_df) * total_target_records))
        
        print(f"  [{cfg_id}]")
        print(f"    Recall: {recall*100:.2f}% | S2: {s2_rec*100:.2f}% | S3: {s3_rec*100:.2f}% | US: {us_rec*100:.2f}% | IN: {in_rec*100:.2f}%")
        print(f"    Avg Cands: {avg_c:.1f} | P95: {p95_c:.0f} | Max: {max_c} | Red Ratio: {red_ratio:.6f} | Time: {t_elapsed:.2f}s")
        
        exp_results.append({
            "experiment_id": cfg_id,
            "blocking_config": cfg["desc"],
            "candidate_budget": max_cands,
            "feature_version": "N/A",
            "model": "N/A",
            "weights": "N/A",
            "threshold": "N/A",
            "F0.5": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "singleton_accuracy": "N/A",
            "candidate_recall": f"{recall*100:.2f}%",
            "avg_candidates": f"{avg_c:.1f}",
            "P95_candidates": f"{p95_c:.0f}",
            "runtime": f"{t_elapsed:.2f}s",
            "RAM": f"{mem_mb:.1f}MB",
            "selected": "YES" if recall > best_recall else "NO"
        })
        
        if recall > best_recall:
            best_recall = recall
            best_config_id = cfg_id
            best_cand_map = cand_map
            
    # Save to experiments/experiment_log.csv
    log_df = pd.DataFrame(exp_results)
    log_path = EXP_DIR / "experiment_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"\n  Logged all {len(exp_results)} experiments to {log_path}")
    
    # ------------------------------------------------------------------
    # 2.5 Phase 4: Candidate Miss Diagnostics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"  PHASE 4: CANDIDATE MISS DIAGNOSTICS (On Best Config: {best_config_id})")
    print("=" * 80)
    
    miss_categories = Counter()
    miss_examples = []
    
    for sid, gt_s in gt_map.items():
        if not gt_s: continue
        cands = best_cand_map[sid]
        missed = gt_s - cands
        if not missed: continue
        
        feat = val_keys[sid]
        s1_row = val_s1_df[val_s1_df["entity_id"] == sid].iloc[0]
        s1_name = str(s1_row.get("business_name", ""))
        s1_addr = str(s1_row.get("business_address", ""))
        
        for mid in missed:
            # Diagnose reason
            # 1. Did it have address info?
            has_addr = bool(feat["snum"] or feat["post"] or feat["sword"])
            # 2. Was it a short legal name collision?
            is_short = len(feat["n_clean"]) <= 5
            # 3. Source type
            source = "S2" if "s2" in mid.lower() else "S3"
            
            if not has_addr:
                cat = "Missing Address Components in S1"
            elif is_short:
                cat = "Short / Truncated Name Collision"
            elif len(cands) >= 150: # Hit candidate budget cap
                cat = "Candidate Budget Cap Truncation"
            elif feat["tok0"] in STOPWORDS or len(feat["tok0"]) <= 2:
                cat = "Generic / Low-IDF Token Miss"
            else:
                cat = "Severe Name Variation / Transliteration Mismatch"
                
            miss_categories[cat] += 1
            if len(miss_examples) < 15:
                miss_examples.append({
                    "s1_id": sid,
                    "target_id": mid,
                    "category": cat,
                    "s1_name": s1_name,
                    "s1_addr": s1_addr
                })
                
    total_misses = sum(miss_categories.values())
    print(f"  Total True Match Misses: {total_misses:,} out of {total_true_matches:,} ({total_misses/total_true_matches*100:.2f}%)")
    print("\n  Miss Breakdown by Category:")
    for cat, count in miss_categories.most_common():
        print(f"    - {cat:<48}: {count:,} ({count/total_misses*100:.1f}%)")
        
    print("\n  Sample Miss Diagnostic Cases:")
    for ex in miss_examples[:8]:
        print(f"    [{ex['category']}] S1: {ex['s1_id']} -> Missed: {ex['target_id']}")
        print(f"      Name: '{ex['s1_name']}' | Addr: '{ex['s1_addr']}'")
        
    print("\n" + "=" * 80)
    print(f"  PHASE 3 & 4 COMPLETE IN {time.time()-t0_all:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    run_experiments()
