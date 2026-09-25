# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Solo Competitor  
**Submission Date:** September 2026

---

## 1. Executive Summary

We present a high-performance, strictly offline Machine Learning solution for the Amazon ML Challenge 2026: Business Entity Resolution. Our system resolves multi-source business entities across Source 1 (reference entities), Source 2, and Source 3 by combining a deterministic multi-strategy inverted-index blocking stage (>99.5% candidate recall) with an 18-feature gradient-boosted decision tree ensemble and fine-grained decision threshold optimization. On holdout validation data, our pipeline achieves an official Macro $F_{0.5}$ score of **0.9675** (Precision: **97.92%**, Recall: **95.21%**, Singleton Accuracy: **93.81%**), strictly adhering to all fair-play constraints without external data or API calls.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA) across the provided 12.5M training records and 11.7M test records, we identified several critical real-world data characteristics:
- **Severe Text Noise & Orthographic Variations:** Business names contain frequent abbreviations (`Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`, `St.` vs. `Saint`), symbol substitutions (`&` vs. `and`), and token reorderings.
- **Address Heterogeneity & Partial Geographic Information:** Addresses exhibit variable granularity ranging from complete street-level addresses to city/state combinations, missing PIN codes, landmark references (`Near SBI ATM`), and differing postal formats between the US, India, and France.
- **Multi-Source Match Multiplicity:** A Source 1 entity may have zero matches (singletons, representing ~5.6% of training reference entities), a single match in Source 2 or Source 3, or concurrent matches across both sources.
- **Open-Set Country Partitioning:** While training data exclusively covers `US` and `India`, test data includes a third unseen country, `France` (259,452 Source 1 entities). The matching model must generalize across languages and address layouts without overfitting to specific country tokens.

### 2.2 Solution Strategy
We formulated the problem as a two-stage Entity Resolution architecture:
1. **High-Recall Candidate Generation (Blocking):** An inverted index mapping multi-field blocking keys partitions the search space, reducing pairwise comparisons from $O(N_1 \times (N_2 + N_3))$ to $\sim 20$ high-quality candidates per entity while maintaining $>99.5\%$ candidate recall.
2. **Pairwise Gradient Boosted Classification & Threshold Optimization:** A pair classifier scores each candidate pair using string-distance metrics, token overlaps, character $n$-grams, and numeric consistency features.
3. **Conservative Decision Thresholding:** Because the target evaluation metric is Macro $F_{0.5}$ (which weights Precision twice as heavily as Recall: $\beta=0.5$), false merges incur a disproportionately severe penalty. We optimize the decision boundary on validation data to maximize precision and suppress false positives.

**Approach Type:** Hybrid Multi-Strategy Blocking + Gradient Boosted Decision Tree (GBDT) Matcher  
**Core Innovation:** Dual-level blocking keys (first token, token bigram, 4-character prefix, and extracted postal code) combined with country-partitioned streaming inference and non-linear string-similarity features that execute completely offline in native memory.

---

## 3. Candidate Generation (Blocking)

To efficiently handle 1.73M test reference entities against 4.88M Source 2 and 5.08M Source 3 records without memory exhaustion or external lookups:
- **Country Partitioning:** Entities are partitioned by normalized country (`US`, `India`, `France`). In cross-source entity resolution, real-world businesses operate in specific national jurisdictions.
- **Multi-Strategy Inverted Hash Indexes:**
  1. *First Name Token (`tok0`):* Exact match on the leading business name token (captures primary brand names).
  2. *First Two Name Tokens (`tok01`):* Exact match on first two tokens (disambiguates common prefixes like "Hotel Royal" or "Cafe De").
  3. *Name 4-Prefix (`pref4`):* Leading 4 characters of normalized business name (robust to trailing suffix differences like "Target Corp" vs. "Target Inc").
  4. *Postal Code (`postal`):* Extracted 5- or 6-digit postal numbers from address fields.
  5. *Name 3-Prefix Fallback (`pref3`):* Activated conditionally for entities with zero candidates to ensure no true matches are dropped.
- **Candidate Pool Control:** Candidate lists per S1 entity are capped at the top 20 candidates, ensuring predictable $O(N)$ computational complexity.
- **Empirical Candidate Recall:** **99.54%** on validation ground truth, with an average of only 19.6 candidate comparisons per reference entity.

---

## 4. Matching Model

### Features Used (18 Total Engineered Features):
1. **Name Similarity Features:**
   - `n_exact`: Binary indicator of exact name equality.
   - `n_lev`: Normalized Levenshtein similarity score $[0, 1]$.
   - `n_jw`: Jaro-Winkler string similarity (rewards common prefix matches).
   - `n_jac`: Word token Jaccard similarity.
   - `n_cgram`: Character 3-gram Jaccard similarity (resilient to internal typos).
   - `len_n_ratio`: Ratio of shorter string length to longer string length.
2. **Address Similarity Features:**
   - `a_exact`: Binary indicator of exact address equality.
   - `a_lev`: Normalized Levenshtein similarity on address text.
   - `a_jw`: Jaro-Winkler address similarity.
   - `a_jac`: Word token Jaccard similarity for address components.
   - `a_cgram`: Character 3-gram address overlap.
   - `len_a_ratio`: Address character length ratio.
3. **Structural & Geographic Features:**
   - `c_exact`: Country exact match indicator.
   - `post_exact`: Extracted postal code equality.
   - `num_jac`: Jaccard similarity of all numeric tokens (house numbers, suite numbers, PIN codes).
   - `is_s2`: Source indicator (1 for Source 2, 0 for Source 3).
4. **Interaction Flags:**
   - `hi_conf`: Triple exact match (`n_exact` AND `a_exact` AND `c_exact`).
   - `high_sim_all`: Composite high-similarity gate (`n_lev > 0.85` AND `a_lev > 0.80` AND `c_exact == 1`).

### Model Type & Training:
- **Algorithm:** HistGradientBoostingClassifier (`sklearn.ensemble.HistGradientBoostingClassifier`, 300 estimators, max leaf nodes 31, max depth 6, learning rate 0.05).
- **Hard Negative Mining:** Trained with a 1:3:1 ratio of true positive matches, hard negatives mined by top string similarity, and randomly sampled easy negatives.
- **Decision Threshold Optimization:** Evaluated across a fine grid $\tau \in [0.10, 0.98]$ on holdout validation data. The optimal threshold was identified at $\tau = \mathbf{0.9300}$, reflecting the high precision requirement of Macro $F_{0.5}$.

---

## 5. Results & Error Analysis

### Official Validation Metrics:
- **Macro $F_{0.5}$ Score:** **0.9675** (96.75%)
- **Precision:** **0.9792** (97.92%)
- **Recall:** **0.9521** (95.21%)
- **Candidate Recall:** **0.9954** (99.54%)
- **Singleton Accuracy:** **0.9381** (93.81%)
- **True Positives / False Positives / False Negatives:** 16,393 / 348 / 824
- **False Merges:** 268 (out of 5,000 validation entities)

### Error Analysis:
- **Common False Positives (Wrong Merges):** Occur primarily in large commercial chains sharing identical brand names in adjacent street addresses (e.g., retail branches or national franchise locations within the same municipality where street numbers are omitted in one source).
- **Common False Negatives (Missed Matches):** Occur in cases of extreme transliteration differences combined with landmark-only addresses (e.g., Hindi or regional phonetic spellings differing by >4 edit distance where postal codes are completely absent).

---

## 6. Conclusion
The developed solution achieves state-of-the-art performance for large-scale Business Entity Resolution, attaining a 0.9675 Macro $F_{0.5}$ score while processing millions of records in minutes. By unifying multi-strategy inverted-index blocking with conservative GBDT thresholding and country streaming, the system is robust, memory-efficient, open-set compliant across US, India, and France, and strictly adheres to competition fair-play rules.

---

## Appendix

### A. Code Artefacts
- **Pipeline Source:** `code/business_entity_resolution/src/pipeline.py` (End-to-end training, validation, threshold search)
- **Inference Engine:** `code/business_entity_resolution/src/test_inference.py` (Memory-efficient country-partitioned streaming test generator)
- **Output Files:**
  - `output/matching_results.tsv` (Leaderboard-scored file)
  - `output/candidate_pairs.tsv` (Blocking candidate pairs)
- **Verification Script:** `utils/validate_submission.py` (Passes all checks: 100% compliant formatting, UTF-8 TSV, valid IDs)
