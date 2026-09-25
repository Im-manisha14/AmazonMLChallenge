"""
features.py - Step 5: Pairwise Feature Engineering
====================================================
For each candidate pair (S1, S2) or (S1, S3), we compute multiple similarity
features that capture DIFFERENT aspects of how similar two businesses are.

Why multiple features?
  - Name similarity alone misses address matches: "Joes Diner" at same address
  - Address alone misses name matches: different branches of same chain  
  - Embeddings capture semantic meaning but can be fooled
  - String similarity is interpretable and reliable

We combine all signals to make a robust final decision.

KEY FEATURES:
  Name features (10): edit distance, token overlap, TF-IDF, char n-grams, etc.
  Address features (10): same metrics on addresses
  Country features (2): exact match, string similarity
  Semantic (2): BGE embedding cosine similarity
  Meta (2): source pair (S1-S2 vs S1-S3), high-confidence flags
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


# ─── String Similarity Functions ─────────────────────────────────────────────

def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Levenshtein similarity = 1 - (edit_distance / max_len).
    Edit distance = minimum number of single-character edits (insert, delete, replace).
    
    Example:
      "Corporation" vs "Corp"
      edit_distance = 7, max_len = 11
      similarity = 1 - 7/11 ≈ 0.36
    
    "ABC Restaurant" vs "ABC Restarant" (typo)
      edit_distance = 1, max_len = 14
      similarity ≈ 0.93
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # fuzz.ratio returns 0-100, convert to 0-1
    return fuzz.ratio(s1, s2) / 100.0


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Jaro-Winkler similarity: focuses on common prefix agreement.
    Good for business names that often share prefixes (e.g., "ABC Corp" vs "ABC Corporation").
    Returns 0.0 to 1.0.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.WRatio(s1, s2) / 100.0


def token_jaccard(s1: str, s2: str) -> float:
    """
    Jaccard similarity on word tokens.
    
    Jaccard = |intersection| / |union|
    
    Example:
      s1 = "abc restaurant private limited" -> tokens: {abc, restaurant, private, limited}
      s2 = "abc restaurant pvt ltd"          -> tokens: {abc, restaurant, pvt, ltd}
    
    After normalization (pvt->private, ltd->limited):
      intersection: {abc, restaurant, private, limited} -> 4
      union: {abc, restaurant, private, limited}        -> 4
      Jaccard = 4/4 = 1.0 (perfect match after normalization!)
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


def token_overlap_count(s1: str, s2: str) -> int:
    """Number of common tokens between two texts."""
    t1 = set(s1.split())
    t2 = set(s2.split())
    return len(t1 & t2)


def char_ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """
    Character n-gram Jaccard similarity.
    
    Splits text into overlapping n-character windows.
    Good at catching abbreviation variants and typos.
    
    Example (n=3):
      "restaurant" -> {res, est, sta, tau, aur, ura, ran, ant}
      "restraunt" (typo) -> {res, est, str, tra, rau, aun, unt}
      overlap / union ≈ 0.4 (partial match despite typo)
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    g1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    g2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    if not g1 and not g2:
        return 1.0
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)


def token_sort_ratio(s1: str, s2: str) -> float:
    """Sort tokens alphabetically then compare (good for word-order differences)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.token_sort_ratio(s1, s2) / 100.0


def partial_ratio(s1: str, s2: str) -> float:
    """Best matching substring ratio (good for truncated or abbreviated strings)."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.partial_ratio(s1, s2) / 100.0


# ─── Address-Specific Features ────────────────────────────────────────────────

def extract_numbers(text: str) -> set:
    """Extract all numeric tokens from text."""
    if not text:
        return set()
    return set(re.findall(r'\d+', text))


def postal_code_match(addr1: str, addr2: str) -> int:
    """
    Check if both addresses share the same 5-6 digit postal code.
    Returns: 1 if match, 0 if no postal code found in either, -1 if mismatch.
    
    Postal match is a VERY strong signal (same ZIP/PIN = same area).
    """
    def find_postal(addr):
        if not addr:
            return None
        for tok in addr.split():
            if re.fullmatch(r'\d{5,6}', tok):
                return tok
        return None
    
    p1 = find_postal(addr1)
    p2 = find_postal(addr2)
    
    if p1 is None and p2 is None:
        return 0   # Can't determine
    if p1 is None or p2 is None:
        return 0   # One side missing
    return 1 if p1 == p2 else -1


def house_number_match(addr1: str, addr2: str) -> int:
    """
    Check if the first numeric token (house/building number) matches.
    Returns: 1 match, 0 missing, -1 mismatch.
    """
    def first_num(addr):
        if not addr:
            return None
        for tok in addr.split():
            t = tok.lstrip("0")
            if t and re.fullmatch(r'\d{1,6}', t):
                return t
        return None
    
    n1 = first_num(addr1)
    n2 = first_num(addr2)
    
    if n1 is None and n2 is None:
        return 0
    if n1 is None or n2 is None:
        return 0
    return 1 if n1 == n2 else -1


def numeric_token_overlap(addr1: str, addr2: str) -> float:
    """
    Jaccard similarity on numeric tokens in addresses.
    Shared numbers (house #, floor, PIN) indicate same location.
    """
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    inter = len(nums1 & nums2)
    union = len(nums1 | nums2)
    return inter / union if union > 0 else 0.0


def length_ratio(s1: str, s2: str) -> float:
    """Ratio of shorter to longer string length."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    return min(l1, l2) / max(l1, l2)


# ─── Main Feature Computation ─────────────────────────────────────────────────

def compute_pair_features(s1_row: pd.Series, cand_row: pd.Series,
                           tfidf_name_sim: float = 0.0,
                           tfidf_addr_sim: float = 0.0,
                           embedding_sim: float = 0.0) -> dict:
    """
    Compute all features for a single candidate pair (S1, candidate).
    
    Returns a dict of feature_name -> value.
    
    Args:
        s1_row: S1 record as pd.Series (must have normalized columns)
        cand_row: Candidate (S2 or S3) record as pd.Series
        tfidf_name_sim: Pre-computed TF-IDF name similarity (0-1)
        tfidf_addr_sim: Pre-computed TF-IDF address similarity (0-1)
        embedding_sim: Pre-computed BGE embedding cosine similarity (0-1)
    """
    # Get normalized texts
    s1_name = str(s1_row.get("business_name_normalized", "") or "")
    s1_addr = str(s1_row.get("business_address_normalized", "") or "")
    s1_ctry = str(s1_row.get("country_normalized", "") or "")
    s1_core = str(s1_row.get("core_name", "") or "")
    
    c_name = str(cand_row.get("business_name_normalized", "") or "")
    c_addr = str(cand_row.get("business_address_normalized", "") or "")
    c_ctry = str(cand_row.get("country_normalized", "") or "")
    c_core = str(cand_row.get("core_name", "") or "")
    
    cand_id = str(cand_row.get("entity_id", ""))
    source_pair = "S1_S2" if cand_id.startswith("S2-") else "S1_S3"
    source_pair_num = 0 if cand_id.startswith("S2-") else 1
    
    features = {}
    
    # ── NAME FEATURES ─────────────────────────────────────────────────────────
    features["name_exact"]         = int(s1_name == c_name and bool(s1_name))
    features["name_levenshtein"]   = levenshtein_similarity(s1_name, c_name)
    features["name_jaro_winkler"]  = jaro_winkler_similarity(s1_name, c_name)
    features["name_jaccard"]       = token_jaccard(s1_name, c_name)
    features["name_char3gram"]     = char_ngram_similarity(s1_name, c_name, 3)
    features["name_token_sort"]    = token_sort_ratio(s1_name, c_name)
    features["name_partial"]       = partial_ratio(s1_name, c_name)
    features["name_tfidf"]         = tfidf_name_sim
    features["name_token_overlap"] = token_overlap_count(s1_name, c_name)
    features["name_length_ratio"]  = length_ratio(s1_name, c_name)
    
    # Core name (without legal suffixes) similarity
    features["core_name_levenshtein"] = levenshtein_similarity(s1_core, c_core)
    features["core_name_jaccard"]     = token_jaccard(s1_core, c_core)
    features["core_name_exact"]       = int(s1_core == c_core and bool(s1_core))
    
    # ── ADDRESS FEATURES ──────────────────────────────────────────────────────
    features["addr_exact"]          = int(s1_addr == c_addr and bool(s1_addr))
    features["addr_levenshtein"]    = levenshtein_similarity(s1_addr, c_addr)
    features["addr_jaccard"]        = token_jaccard(s1_addr, c_addr)
    features["addr_char3gram"]      = char_ngram_similarity(s1_addr, c_addr, 3)
    features["addr_token_sort"]     = token_sort_ratio(s1_addr, c_addr)
    features["addr_partial"]        = partial_ratio(s1_addr, c_addr)
    features["addr_tfidf"]          = tfidf_addr_sim
    features["addr_numeric_overlap"]= numeric_token_overlap(s1_addr, c_addr)
    features["addr_postal_match"]   = postal_code_match(s1_addr, c_addr)
    features["addr_house_num_match"]= house_number_match(s1_addr, c_addr)
    features["addr_length_ratio"]   = length_ratio(s1_addr, c_addr)
    
    # Token overlap count
    s1_addr_toks = set(s1_addr.split())
    c_addr_toks  = set(c_addr.split())
    features["addr_token_overlap"]  = len(s1_addr_toks & c_addr_toks)
    
    # ── COUNTRY FEATURES ──────────────────────────────────────────────────────
    features["country_exact"]      = int(s1_ctry == c_ctry and bool(s1_ctry))
    features["country_lev"]        = levenshtein_similarity(s1_ctry, c_ctry)
    
    # ── SEMANTIC FEATURES ─────────────────────────────────────────────────────
    features["embedding_sim"]      = embedding_sim
    
    # ── META FEATURES ─────────────────────────────────────────────────────────
    features["source_pair"]        = source_pair_num  # 0 = S1-S2, 1 = S1-S3
    
    # High-confidence flags (very strong signal for the model)
    name_high = features["name_levenshtein"] >= 0.9
    addr_high = features["addr_levenshtein"] >= 0.8
    ctry_match = features["country_exact"] == 1
    
    features["high_conf_both"]   = int(name_high and addr_high and ctry_match)
    features["high_conf_name"]   = int(name_high and ctry_match)
    features["high_conf_addr"]   = int(addr_high and ctry_match)
    
    # Combined similarity score (simple weighted average for reference)
    features["combined_sim"] = (
        0.4 * features["name_levenshtein"] +
        0.3 * features["addr_jaccard"] +
        0.2 * features["embedding_sim"] +
        0.1 * features["country_exact"]
    )
    
    return features


def build_feature_matrix(s1_df: pd.DataFrame, cand_df: pd.DataFrame,
                          pairs: list,
                          tfidf_name_sims: dict = None,
                          tfidf_addr_sims: dict = None,
                          embedding_lookup_s1: dict = None,
                          embedding_lookup_cand: dict = None) -> pd.DataFrame:
    """
    Build feature matrix for all pairs.
    
    Args:
        s1_df: Source 1 DataFrame (preprocessed)
        cand_df: Combined S2+S3 DataFrame (preprocessed)
        pairs: list of (s1_id, candidate_id) tuples
        tfidf_name_sims: dict[(s1_id, cand_id)] -> float
        tfidf_addr_sims: dict[(s1_id, cand_id)] -> float
        embedding_lookup_s1: dict[s1_id] -> np.ndarray
        embedding_lookup_cand: dict[cand_id] -> np.ndarray
    
    Returns:
        DataFrame with one row per pair and feature columns
    """
    # Build fast lookup dicts
    s1_lookup   = s1_df.set_index("entity_id").to_dict("index")
    cand_lookup = cand_df.set_index("entity_id").to_dict("index")
    
    rows = []
    n = len(pairs)
    print(f"    Computing features for {n:,} pairs...")
    
    for i, (s1_id, cand_id) in enumerate(pairs):
        if i % 100000 == 0 and i > 0:
            print(f"      Progress: {i:,}/{n:,} ({100.0*i/n:.1f}%)")
        
        s1_row   = s1_lookup.get(s1_id, {})
        cand_row = cand_lookup.get(cand_id, {})
        
        if not s1_row or not cand_row:
            continue
        
        # Get pre-computed similarities
        tfidf_name = (tfidf_name_sims or {}).get((s1_id, cand_id), 0.0)
        tfidf_addr = (tfidf_addr_sims or {}).get((s1_id, cand_id), 0.0)
        
        # Embedding similarity
        emb_sim = 0.0
        if embedding_lookup_s1 and embedding_lookup_cand:
            emb_s1   = embedding_lookup_s1.get(s1_id)
            emb_cand = embedding_lookup_cand.get(cand_id)
            if emb_s1 is not None and emb_cand is not None:
                emb_sim = float(np.dot(emb_s1, emb_cand))
        
        feats = compute_pair_features(
            pd.Series(s1_row), pd.Series({**cand_row, "entity_id": cand_id}),
            tfidf_name_sim=tfidf_name,
            tfidf_addr_sim=tfidf_addr,
            embedding_sim=emb_sim,
        )
        feats["s1_id"]   = s1_id
        feats["cand_id"] = cand_id
        rows.append(feats)
    
    if not rows:
        return pd.DataFrame()
    
    feat_df = pd.DataFrame(rows)
    # Move ID columns to front
    cols = ["s1_id", "cand_id"] + [c for c in feat_df.columns if c not in ("s1_id", "cand_id")]
    return feat_df[cols]


def get_feature_columns(feat_df: pd.DataFrame) -> list:
    """Return list of feature columns (excluding ID columns)."""
    return [c for c in feat_df.columns if c not in ("s1_id", "cand_id", "label")]
