"""
embedding_retrieval.py - Step 6: BGE Embedding Retrieval via FAISS
===================================================================
BGE (BAAI General Embedding) is a small but powerful text embedding model.
License: MIT (Apache 2.0 compatible) - complies with competition rules.
Parameters: ~33M (well under the 8B limit).

What is an embedding?
  An embedding converts text into a list of numbers (a vector) that captures
  its MEANING. Similar texts produce similar vectors, even if the words differ.

  Example:
    "ABC Restaurant Private Limited" -> [0.12, -0.43, 0.87, ...]  (384 numbers)
    "ABC Restaurant Pvt Ltd"         -> [0.11, -0.44, 0.88, ...]  (very similar!)

What is FAISS?
  FAISS (Facebook AI Similarity Search) efficiently finds the nearest vectors.
  Instead of comparing 1.7M query vectors against 10M database vectors one by one,
  FAISS uses index structures to find the top-K closest in milliseconds.
"""

import os
import numpy as np
import pandas as pd
import pickle
import json

ARTIFACTS_DIR = "artifacts"
EMBED_DIR     = os.path.join(ARTIFACTS_DIR, "embeddings")
EMBED_TOP_K   = 30  # How many nearest neighbors to retrieve


def _get_model():
    """Load BGE-small model. Cached after first load."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return model
    except Exception as e:
        print(f"  WARNING: Could not load BGE model: {e}")
        print("  Embedding retrieval will be skipped.")
        return None


def _embed_texts(model, texts: list, batch_size: int = 512, 
                 normalize: bool = True) -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    Normalizes to unit length so dot product = cosine similarity.
    
    BGE model license: MIT - legally compliant for this competition.
    Model size: ~33M parameters - well under the 8B limit.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,  # L2 normalize -> cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def compute_and_cache_embeddings(df: pd.DataFrame, cache_name: str,
                                  model=None, force_recompute: bool = False):
    """
    Compute embeddings for a DataFrame and cache to disk.
    If cache exists and force_recompute=False, loads from cache.
    
    Embeddings are computed on the combined_text field:
      "business_name_normalized [SEP] business_address_normalized [SEP] country_normalized"
    
    Returns: (np.ndarray of shape [n, 384], list of entity_ids)
    """
    os.makedirs(EMBED_DIR, exist_ok=True)
    emb_path = os.path.join(EMBED_DIR, f"{cache_name}_embeddings.npy")
    ids_path = os.path.join(EMBED_DIR, f"{cache_name}_ids.json")
    
    if not force_recompute and os.path.exists(emb_path) and os.path.exists(ids_path):
        print(f"    Loading cached embeddings from {emb_path}")
        embeddings = np.load(emb_path)
        with open(ids_path, "r") as f:
            entity_ids = json.load(f)
        print(f"    Loaded {len(entity_ids):,} embeddings of shape {embeddings.shape}")
        return embeddings, entity_ids
    
    if model is None:
        model = _get_model()
    if model is None:
        return None, None
    
    print(f"    Computing embeddings for {len(df):,} records...")
    texts = df["combined_text"].fillna("").tolist()
    entity_ids = df["entity_id"].tolist()
    
    embeddings = _embed_texts(model, texts)
    
    # Cache to disk
    np.save(emb_path, embeddings)
    with open(ids_path, "w") as f:
        json.dump(entity_ids, f)
    print(f"    Cached embeddings: {emb_path}")
    
    return embeddings, entity_ids


class FAISSRetriever:
    """
    FAISS-based nearest-neighbor retrieval for embedding similarity.
    
    Workflow:
    1. Build a FAISS index from S2+S3 embeddings (the "database")
    2. For each S1 embedding (the "query"), find top-K nearest neighbors
    3. Return the entity IDs of the neighbors
    
    Because embeddings are L2-normalized (unit length vectors),
    inner product (dot product) == cosine similarity.
    """
    
    def __init__(self, top_k: int = EMBED_TOP_K):
        self.top_k = top_k
        self.index_s2 = None
        self.index_s3 = None
        self.s2_ids = None
        self.s3_ids = None
    
    def build_indexes(self, s2_embeddings: np.ndarray, s2_ids: list,
                      s3_embeddings: np.ndarray, s3_ids: list):
        """
        Build FAISS indexes for S2 and S3.
        Uses IndexFlatIP (Inner Product) since embeddings are normalized.
        """
        import faiss
        
        dim = s2_embeddings.shape[1]  # 384 for BGE-small
        print(f"    Building FAISS indexes (dim={dim})...")
        
        # IndexFlatIP = exact search using inner product (= cosine for L2-normalized vectors)
        # For large datasets, could use IndexIVFFlat for approximate search
        self.index_s2 = faiss.IndexFlatIP(dim)
        self.index_s2.add(s2_embeddings)
        print(f"    S2 FAISS index built: {self.index_s2.ntotal:,} vectors")
        
        self.index_s3 = faiss.IndexFlatIP(dim)
        self.index_s3.add(s3_embeddings)
        print(f"    S3 FAISS index built: {self.index_s3.ntotal:,} vectors")
        
        self.s2_ids = np.array(s2_ids)
        self.s3_ids = np.array(s3_ids)
    
    def retrieve_batch(self, s1_embeddings: np.ndarray, s1_ids: list) -> dict:
        """
        For each S1 embedding, retrieve top-K candidates from S2 and S3.
        
        Returns: { s1_id: set(candidate_ids) }
        """
        if self.index_s2 is None or self.index_s3 is None:
            return {s1_id: set() for s1_id in s1_ids}
        
        result = {}
        
        # Search S2
        sims_s2, idxs_s2 = self.index_s2.search(s1_embeddings, self.top_k)
        # Search S3
        sims_s3, idxs_s3 = self.index_s3.search(s1_embeddings, self.top_k)
        
        for i, s1_id in enumerate(s1_ids):
            cands = set()
            # Add S2 candidates (filter by minimum similarity threshold)
            for rank, (idx, sim) in enumerate(zip(idxs_s2[i], sims_s2[i])):
                if idx >= 0 and sim > 0.50:  # Cosine similarity > 0.5
                    cands.add(self.s2_ids[idx])
            # Add S3 candidates
            for rank, (idx, sim) in enumerate(zip(idxs_s3[i], sims_s3[i])):
                if idx >= 0 and sim > 0.50:
                    cands.add(self.s3_ids[idx])
            result[s1_id] = cands
        
        return result
    
    def retrieve_all(self, s1_embeddings: np.ndarray, s1_ids: list,
                     batch_size: int = 5000) -> dict:
        """Retrieve candidates for all S1 records in batches."""
        all_results = {}
        n = len(s1_ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_embs = s1_embeddings[start:end]
            batch_ids  = s1_ids[start:end]
            batch_result = self.retrieve_batch(batch_embs, batch_ids)
            all_results.update(batch_result)
            if start % (batch_size * 5) == 0:
                print(f"    FAISS retrieval: {start:,}/{n:,}")
        return all_results


def get_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two (normalized) embeddings.
    Since both are L2-normalized, this is just their dot product.
    """
    return float(np.dot(emb1, emb2))


def build_embedding_lookup(embeddings: np.ndarray, entity_ids: list) -> dict:
    """Build a dict: { entity_id -> embedding_vector } for fast pairwise lookup."""
    return {eid: embeddings[i] for i, eid in enumerate(entity_ids)}
