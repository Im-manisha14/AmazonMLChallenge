# Business Entity Resolution Pipeline
## ML Challenge 2026

A production-grade end-to-end entity resolution system.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (train + validate + test inference + submission validator)
python src/main.py --mode all

# Or run individual stages:
python src/main.py --mode train     # Train & validate only
python src/main.py --mode test      # Test inference only (requires pre-trained model)
```

Run from the `student_resource/` directory.

## Architecture Overview

```
Raw TSV Data
    |
    v
[1] Data Loader (data_loader.py)
    Loads TSV files, validates columns and ID prefixes, prints statistics.
    |
    v
[2] Preprocessor (preprocessing.py)
    - Unicode NFKD normalization (removes accents: Café -> Cafe)
    - Lowercase, punctuation -> spaces
    - Abbreviation expansion: Corp->Corporation, Pvt->Private, Rd->Road
    - Address component extraction: street number, postal code, tokens
    |
    v
[3] Blocking / Candidate Generation (blocking.py)
    Problem: Can't compare 1.7M x 10M = 17 TRILLION pairs.
    Solution: Reduce to ~50-100 candidates per S1 entity using multiple blocking keys.
    
    Block 1: Country + first name token  (e.g., "us_abc")
    Block 2: Country + 3-char name prefix (e.g., "us_abc")
    Block 3: Country + postal code       (e.g., "us_27262")
    Block 4: Country + first 2 name tokens
    Block 5: Address token overlap
    Block 6: TF-IDF character n-gram retrieval (top-30 per S1)
    Block 7: BGE embedding FAISS retrieval (if available)
    
    UNION of all blocks -> candidate set
    |
    v
[4] Feature Engineering (features.py)
    For each (S1, candidate) pair, compute ~30 features:
    
    Name features (14): Levenshtein, Jaro-Winkler, Jaccard, char n-grams,
                        TF-IDF cosine, token overlap, length ratio, core name
    Address features (12): Same + postal match, house number match, numeric overlap
    Country features (2): Exact match, string similarity
    Semantic features (1): BGE embedding cosine similarity
    Meta features (3): Source pair (S2 vs S3), high-confidence flags
    |
    v
[5] LightGBM Classifier (model.py)
    Input: 30+ pairwise features
    Output: P(same real-world business | pair)
    
    Training labels from ground truth:
    - Positive: pairs in ground_truth
    - Hard negatives: high-similarity pairs NOT in ground_truth
    - Easy negatives: random non-matching candidates
    |
    v
[6] Threshold Optimization (threshold.py)
    Test thresholds: 0.30, 0.40, ..., 0.90
    Select threshold that maximizes macro F0.5 on validation set.
    Default: higher threshold = fewer false merges (F0.5 is precision-heavy)
    |
    v
[7] Output Generation (inference.py)
    - output/matching_results.tsv: final entity matches
    - output/candidate_pairs.tsv: blocking candidate set
    |
    v
[8] Submission Validation (validate_submission.py)
    Official validator checks all formatting rules.
```

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| Multiple blocking keys (union) | Single blocking strategy misses true matches |
| Character n-gram TF-IDF | Robust to abbreviations and typos |
| Hard negatives in training | F0.5 penalizes false positives 2x more than false negatives |
| Precision-biased threshold | Higher threshold -> fewer false merges -> better F0.5 |
| Singleton handling | Predicting empty for no-match entities scores 1.0 per entity |

## Output Files

- `output/matching_results.tsv`: Final predictions (leaderboard submission)
- `output/candidate_pairs.tsv`: Blocking candidates (pipeline verification)
- `artifacts/validation_predictions.tsv`: Validation scores for debugging
- `artifacts/experiment_results.csv`: Threshold sweep results
- `artifacts/error_analysis.csv`: False positives/negatives analysis

## Model License Compliance

| Component | Model/Package | License | Parameters |
|-----------|---------------|---------|------------|
| Embeddings | BAAI/bge-small-en-v1.5 | MIT | ~33M |
| Classifier | LightGBM | MIT | N/A (trees) |
| String sim | RapidFuzz | MIT | N/A |
| ANN search | FAISS | MIT | N/A |

All models: MIT/Apache 2.0 licensed, all under 8B parameters. Compliant.

## Reproducibility

Fixed random seed: 42 throughout.
No external data or APIs used.
Pipeline runs entirely from `dataset/` + source code.
