"""
blocking.py - Step 4: Candidate Generation / Blocking
======================================================
The core challenge: We have 1.7M test S1 entities and ~10M total S2+S3 records.
We CANNOT compare every S1 against every S2/S3 (that would be 17 TRILLION pairs!).

Blocking reduces this by grouping records that are "likely to match" and only
comparing within those groups. This is called "blocking" - we block/filter down
to only plausible candidates.

Think of it like: instead of checking if EVERY person knows EVERY other person,
you only check people in the same city, same age group, etc.

MULTIPLE BLOCKING STRATEGIES (Union = higher recall):
  Block 1: Exact normalized country match (broad, high recall)
  Block 2: Country + first name token (narrows down by name start)
  Block 3: Country + 3-char name prefix (handles short names)
  Block 4: Country + postal/PIN code (strong address signal)
  Block 5: TF-IDF retrieval (semantic similarity)
  Block 6: BGE embedding retrieval (semantic understanding)
  Block 7: Address token overlap

We take the UNION of all blocks, then score with features.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

# ─── Constants ────────────────────────────────────────────────────────────────

# Maximum candidates per S1 entity (from each blocking method)
TFIDF_TOP_K   = 30   # Top K by TF-IDF similarity
EMBED_TOP_K   = 30   # Top K by embedding similarity (if FAISS available)
MAX_TOTAL_K   = 100  # Hard cap on total candidates per S1 entity

# Artifacts directory for caching
ARTIFACTS_DIR = "artifacts"


# ─── Block 1-4: Deterministic Key-Based Blocking ─────────────────────────────

def build_blocking_index(df: pd.DataFrame, source_label: str) -> dict:
    """
    Build multiple blocking key indexes from a source DataFrame (S2 or S3).
    
    Returns a dict of dicts:
      {
        "country_name_token":   { "us_abc": [S2-001, S2-002, ...], ... },
        "country_prefix3":      { "us_abc": [...], ... },
        "country_postal":       { "us_27262": [...], ... },
        "name_first2_country":  { "ab_us": [...], ... },
      }
    
    Args:
        df: preprocessed source DataFrame (S2 or S3)
        source_label: "S2" or "S3"
    """
    indexes = {
        "country_name_token": defaultdict(list),   # Block 2
        "country_prefix3": defaultdict(list),       # Block 3
        "country_postal": defaultdict(list),        # Block 4
        "country_name2tok": defaultdict(list),      # Block 2b: first 2 tokens
        "address_token": defaultdict(list),         # Block 7: address tokens
    }
    
    for _, row in df.iterrows():
        eid     = row["entity_id"]
        country = str(row.get("country_normalized", "")).strip()
        name    = str(row.get("business_name_normalized", "")).strip()
        addr    = str(row.get("business_address_normalized", "")).strip()
        postal  = str(row.get("postal_code", "")).strip() if "postal_code" in row.index else ""
        
        name_tokens = name.split()
        addr_tokens = addr.split()
        
        # Block 2: country + first name token
        if country and name_tokens:
            key = f"{country}_{name_tokens[0]}"
            indexes["country_name_token"][key].append(eid)
        
        # Block 3: country + 3-char prefix
        if country and len(name) >= 2:
            prefix = name[:3]
            key = f"{country}_{prefix}"
            indexes["country_prefix3"][key].append(eid)
        
        # Block 4: country + postal code
        if country and postal:
            key = f"{country}_{postal}"
            indexes["country_postal"][key].append(eid)
        
        # Block 2b: country + first 2 name tokens (more specific)
        if country and len(name_tokens) >= 2:
            key = f"{country}_{name_tokens[0]}_{name_tokens[1]}"
            indexes["country_name2tok"][key].append(eid)
        
        # Block 7: address tokens (each unique token -> list of entities)
        # Only add meaningful tokens (length >= 3, not stopwords)
        for tok in addr_tokens:
            if len(tok) >= 3 and not re.fullmatch(r"\d+", tok):
                indexes["address_token"][tok].append(eid)
    
    return indexes


def add_postal_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract postal codes from addresses and add as a column."""
    import re
    def extract_postal(addr_norm):
        if pd.isna(addr_norm) or not str(addr_norm).strip():
            return ""
        tokens = str(addr_norm).split()
        for t in tokens:
            if re.fullmatch(r"\d{5,6}", t):
                return t
        return ""
    
    df = df.copy()
    df["postal_code"] = df["business_address_normalized"].apply(extract_postal)
    return df


def get_deterministic_candidates(s1_row: pd.Series, indexes_s2: dict, indexes_s3: dict,
                                  max_addr_tokens: int = 5) -> set:
    """
    Get candidate entity IDs for one S1 record using deterministic blocking.
    
    Returns a set of candidate IDs (S2-xxx and S3-xxx).
    """
    country = str(s1_row.get("country_normalized", "")).strip()
    name    = str(s1_row.get("business_name_normalized", "")).strip()
    addr    = str(s1_row.get("business_address_normalized", "")).strip()
    postal  = str(s1_row.get("postal_code", "")).strip()
    
    name_tokens = name.split()
    addr_tokens = addr.split()
    
    candidates = set()
    
    for indexes in [indexes_s2, indexes_s3]:
        # Block 2: country + first name token
        if country and name_tokens:
            key = f"{country}_{name_tokens[0]}"
            candidates.update(indexes["country_name_token"].get(key, []))
        
        # Block 3: country + 3-char prefix
        if country and len(name) >= 2:
            prefix = name[:3]
            key = f"{country}_{prefix}"
            candidates.update(indexes["country_prefix3"].get(key, []))
        
        # Block 4: country + postal code
        if country and postal:
            key = f"{country}_{postal}"
            candidates.update(indexes["country_postal"].get(key, []))
        
        # Block 2b: country + first 2 tokens
        if country and len(name_tokens) >= 2:
            key = f"{country}_{name_tokens[0]}_{name_tokens[1]}"
            candidates.update(indexes["country_name2tok"].get(key, []))
        
        # Block 7: address token overlap (limit to prevent explosion)
        meaningful_tokens = [t for t in addr_tokens if len(t) >= 3 and not re.fullmatch(r"\d+", t)]
        # Use at most max_addr_tokens to prevent too many candidates from common words
        for tok in meaningful_tokens[:max_addr_tokens]:
            candidates.update(indexes["address_token"].get(tok, []))
    
    return candidates


# ─── Block 5: TF-IDF Retrieval ───────────────────────────────────────────────

class TFIDFRetriever:
    """
    TF-IDF based approximate candidate retrieval.
    
    TF-IDF = Term Frequency-Inverse Document Frequency.
    It measures how important a word is to a document relative to a corpus.
    
    - TF: How often does the word appear in this business record?
    - IDF: How rare is this word across all business records?
    
    Rare words (like specific business names) get HIGH scores.
    Common words (like "the", "and") get LOW scores.
    
    We build a TF-IDF matrix for all S2/S3 records, then for each S1 record,
    we find the S2/S3 records with highest cosine similarity.
    """
    
    def __init__(self, top_k: int = TFIDF_TOP_K, cache_dir: str = ARTIFACTS_DIR):
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.vectorizer_name = None
        self.vectorizer_addr = None
        self.matrix_name_s2 = None
        self.matrix_name_s3 = None
        self.matrix_addr_s2 = None
        self.matrix_addr_s3 = None
        self.s2_ids = None
        self.s3_ids = None
    
    def fit_transform(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame):
        """
        Fit TF-IDF vectorizers on S2+S3 data and build sparse matrices.
        Uses character n-grams (2-5 chars) which handles abbreviation variants well.
        """
        print("    Fitting TF-IDF on name texts...")
        # Character n-gram TF-IDF: good at capturing partial matches and typos
        self.vectorizer_name = TfidfVectorizer(
            analyzer="char_wb",    # Character n-grams, respecting word boundaries
            ngram_range=(2, 5),    # 2 to 5 character n-grams
            max_features=200000,   # Limit vocabulary size for memory
            sublinear_tf=True,     # Apply log normalization (reduces impact of very frequent terms)
            min_df=2,              # Ignore terms appearing in fewer than 2 documents
        )
        
        # Combine S2 and S3 name texts to fit the vectorizer
        all_names = pd.concat([
            s2_df["business_name_normalized"].fillna(""),
            s3_df["business_name_normalized"].fillna("")
        ])
        self.vectorizer_name.fit(all_names)
        
        print("    Building name TF-IDF matrices for S2 and S3...")
        self.matrix_name_s2 = self.vectorizer_name.transform(
            s2_df["business_name_normalized"].fillna(""))
        self.matrix_name_s3 = self.vectorizer_name.transform(
            s3_df["business_name_normalized"].fillna(""))
        
        print("    Fitting TF-IDF on address texts...")
        self.vectorizer_addr = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True,
            min_df=2,
        )
        
        all_addrs = pd.concat([
            s2_df["business_address_normalized"].fillna(""),
            s3_df["business_address_normalized"].fillna("")
        ])
        self.vectorizer_addr.fit(all_addrs)
        
        self.matrix_addr_s2 = self.vectorizer_addr.transform(
            s2_df["business_address_normalized"].fillna(""))
        self.matrix_addr_s3 = self.vectorizer_addr.transform(
            s3_df["business_address_normalized"].fillna(""))
        
        self.s2_ids = s2_df["entity_id"].values
        self.s3_ids = s3_df["entity_id"].values
        
        print(f"    TF-IDF ready. Name vocab: {len(self.vectorizer_name.vocabulary_):,}  "
              f"Addr vocab: {len(self.vectorizer_addr.vocabulary_):,}")
    
    def retrieve(self, s1_batch: pd.DataFrame) -> dict:
        """
        For each S1 entity in the batch, retrieve top-K candidates from S2 and S3.
        
        Returns: { s1_id: set(candidate_ids) }
        
        Uses batched matrix multiplication for efficiency.
        """
        result = {}
        
        # Vectorize S1 batch
        s1_name_vec = self.vectorizer_name.transform(
            s1_batch["business_name_normalized"].fillna(""))
        s1_addr_vec = self.vectorizer_addr.transform(
            s1_batch["business_address_normalized"].fillna(""))
        
        s1_ids = s1_batch["entity_id"].values
        
        for i, s1_id in enumerate(s1_ids):
            cands = set()
            
            # Name similarity: S2
            name_sim_s2 = cosine_similarity(s1_name_vec[i], self.matrix_name_s2)[0]
            top_s2_idx = np.argsort(name_sim_s2)[::-1][:self.top_k]
            for idx in top_s2_idx:
                if name_sim_s2[idx] > 0.01:  # Ignore near-zero similarity
                    cands.add(self.s2_ids[idx])
            
            # Name similarity: S3
            name_sim_s3 = cosine_similarity(s1_name_vec[i], self.matrix_name_s3)[0]
            top_s3_idx = np.argsort(name_sim_s3)[::-1][:self.top_k]
            for idx in top_s3_idx:
                if name_sim_s3[idx] > 0.01:
                    cands.add(self.s3_ids[idx])
            
            # Address similarity: S2
            addr_sim_s2 = cosine_similarity(s1_addr_vec[i], self.matrix_addr_s2)[0]
            top_s2_addr = np.argsort(addr_sim_s2)[::-1][:self.top_k // 2]
            for idx in top_s2_addr:
                if addr_sim_s2[idx] > 0.05:
                    cands.add(self.s2_ids[idx])
            
            # Address similarity: S3
            addr_sim_s3 = cosine_similarity(s1_addr_vec[i], self.matrix_addr_s3)[0]
            top_s3_addr = np.argsort(addr_sim_s3)[::-1][:self.top_k // 2]
            for idx in top_s3_addr:
                if addr_sim_s3[idx] > 0.05:
                    cands.add(self.s3_ids[idx])
            
            result[s1_id] = cands
        
        return result
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "TFIDFRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── Main Blocking Function ───────────────────────────────────────────────────

def generate_candidates(s1_df: pd.DataFrame,
                        s2_df: pd.DataFrame,
                        s3_df: pd.DataFrame,
                        tfidf_retriever: TFIDFRetriever = None,
                        embed_candidates: dict = None,
                        batch_size: int = 1000) -> dict:
    """
    Generate candidate pairs for all S1 entities.
    
    Takes the UNION of:
    1. Deterministic blocking (country + name/address tokens)
    2. TF-IDF retrieval (if tfidf_retriever provided)
    3. Embedding retrieval (if embed_candidates provided)
    
    Returns: { s1_id: set(candidate_ids_from_S2_and_S3) }
    
    Args:
        s1_df: preprocessed Source 1 DataFrame
        s2_df: preprocessed Source 2 DataFrame
        s3_df: preprocessed Source 3 DataFrame
        tfidf_retriever: fitted TFIDFRetriever (optional)
        embed_candidates: pre-computed embedding candidates dict (optional)
        batch_size: batch size for TF-IDF retrieval
    """
    print("  Building blocking indexes for S2...")
    s2_df = add_postal_codes(s2_df)
    indexes_s2 = build_blocking_index(s2_df, "S2")
    
    print("  Building blocking indexes for S3...")
    s3_df = add_postal_codes(s3_df)
    indexes_s3 = build_blocking_index(s3_df, "S3")
    
    s1_df = add_postal_codes(s1_df)
    
    print("  Running deterministic blocking...")
    det_candidates = {}
    n = len(s1_df)
    for i, (_, row) in enumerate(s1_df.iterrows()):
        if i % 50000 == 0:
            print(f"    Progress: {i:,}/{n:,} ({100.0*i/n:.1f}%)")
        cands = get_deterministic_candidates(row, indexes_s2, indexes_s3)
        det_candidates[row["entity_id"]] = cands
    
    print("  Running TF-IDF retrieval...")
    tfidf_cands = {}
    if tfidf_retriever is not None:
        for start in range(0, n, batch_size):
            batch = s1_df.iloc[start:start+batch_size]
            batch_result = tfidf_retriever.retrieve(batch)
            tfidf_cands.update(batch_result)
            if start % (batch_size * 20) == 0:
                print(f"    TF-IDF progress: {start:,}/{n:,}")
    
    print("  Merging candidate sets (UNION)...")
    all_candidates = {}
    for _, row in s1_df.iterrows():
        s1_id = row["entity_id"]
        cands = set()
        
        # Add deterministic candidates
        cands.update(det_candidates.get(s1_id, set()))
        
        # Add TF-IDF candidates
        cands.update(tfidf_cands.get(s1_id, set()))
        
        # Add embedding candidates
        if embed_candidates and s1_id in embed_candidates:
            cands.update(embed_candidates[s1_id])
        
        # Remove self (S1 IDs should never be candidates)
        cands = {c for c in cands if c.startswith("S2-") or c.startswith("S3-")}
        
        # Cap at max total
        if len(cands) > MAX_TOTAL_K:
            # Prioritize: keep all deterministic, then TF-IDF top ones
            det = det_candidates.get(s1_id, set())
            extra = cands - det
            cands = det | set(list(extra)[:MAX_TOTAL_K - len(det)])
        
        all_candidates[s1_id] = cands
    
    # Stats
    sizes = [len(v) for v in all_candidates.values()]
    if sizes:
        print(f"  Candidate stats:")
        print(f"    Total S1 entities: {len(all_candidates):,}")
        print(f"    Avg candidates/S1: {np.mean(sizes):.1f}")
        print(f"    Median: {np.median(sizes):.1f}")
        print(f"    Max: {max(sizes)}")
        print(f"    Singletons (0 candidates): {sum(1 for s in sizes if s == 0):,}")
    
    return all_candidates


def compute_candidate_recall(candidates: dict, ground_truth: dict) -> float:
    """
    Compute candidate recall: of all true matches, what % are in candidates?
    
    This is the UPPER BOUND of recall the model can achieve.
    If a true match is not in candidates, the model can NEVER find it.
    
    Args:
        candidates: { s1_id: set(candidate_ids) }
        ground_truth: { s1_id: set(true_match_ids) }
    """
    total_true = 0
    total_found = 0
    
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue  # Skip singletons
        cands = candidates.get(s1_id, set())
        found = true_matches & cands
        total_true += len(true_matches)
        total_found += len(found)
    
    if total_true == 0:
        return 1.0
    return total_found / total_true
