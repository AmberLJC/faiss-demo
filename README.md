# Cross-Lingual Knowledge Alignment Analysis

**Research Question:** How consistently is the same concept represented across different languages in a multilingual embedding model?

## Motivation

Multilingual models map text into a shared embedding space where semantically equivalent concepts should have similar vectors regardless of language. This **cross-lingual alignment** enables cross-lingual retrieval, zero-shot transfer, and multilingual search—but alignment quality varies by concept type, language pair, and model architecture.

This experiment quantifies alignment using FAISS for efficient similarity search across 15 concepts in 8 languages.

## Experimental Setup

**Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages)

**Languages**: English, Spanish, French, German (Latin); Chinese (Hanzi); Japanese (Kanji/Kana); Arabic; Russian (Cyrillic)

**Concepts (15)**: Abstract (freedom, love, justice, knowledge, peace) · Concrete (water, sun, tree, house, mountain) · Scientific (gravity, energy) · Emotion (happiness, fear, anger)

### Methodology

1. Generate embeddings for each concept × language (120 vectors)
2. L2-normalize for cosine similarity via inner product
3. Build FAISS `IndexFlatIP` index
4. Compute metrics:
   - **Intra-concept similarity**: Avg pairwise cosine similarity between translations
   - **Nearest neighbor accuracy**: % of k-NN that are same concept
   - **Cross-lingual retrieval**: Accuracy retrieving correct concept across language pairs

<!-- ![Experiment Design & Methodology](fig/experiment_diagram.png) -->

## Results

### Overall Alignment Score: **0.938** (Excellent)

The model demonstrates strong cross-lingual alignment, with concepts clustering by meaning rather than by language.

### Metric Breakdown

| Metric | Score | Description |
|--------|-------|-------------|
| Intra-Concept Similarity | 0.914 | Avg cosine similarity within same concept across languages |
| Nearest Neighbor Accuracy (k=3) | 96.1% | % of nearest neighbors that are the same concept |
| Cross-Lingual Retrieval | 93.9% | Accuracy retrieving correct concept across language pairs |

### Intra-Concept Similarity by Concept

```
Concept          Similarity    Quality
─────────────────────────────────────────
justice          0.9894        Excellent
water            0.9856        Excellent
knowledge        0.9840        Excellent
energy           0.9800        Excellent
freedom          0.9798        Excellent
fear             0.9692        Excellent
happiness        0.9582        Excellent
love             0.9365        Excellent
sun              0.9349        Excellent
house            0.9330        Excellent
peace            0.8749        Good
gravity          0.8528        Good
tree             0.7966        Moderate
mountain         0.7875        Moderate
anger            0.7456        Moderate
─────────────────────────────────────────
Average          0.9139
```

### Language Centroid Distances

Distances between language centroids (lower = more overlap in embedding space):

**Closest pairs** (best aligned):
- French ↔ Russian: 0.004
- Chinese ↔ Japanese: 0.006
- Chinese ↔ Arabic: 0.011
- English ↔ French: 0.012

**Most distant pairs**:
- English ↔ Japanese: 0.045
- German ↔ Japanese: 0.036
- German ↔ Chinese: 0.034

### Misaligned Concept Pairs

Pairs with similarity < 0.5 (indicating poor alignment):

| Concept | Language Pair | Similarity |
|---------|---------------|------------|
| tree | English ↔ German | 0.308 |
| tree | German ↔ Arabic | 0.315 |
| tree | Spanish ↔ German | 0.353 |
| mountain | Spanish ↔ German | 0.363 |
| tree | French ↔ German | 0.366 |
| mountain | English ↔ German | 0.367 |
| tree | German ↔ Chinese | 0.378 |
| tree | German ↔ Russian | 0.379 |
| mountain | German ↔ Arabic | 0.399 |
| anger | English ↔ Spanish | 0.402 |

## Findings

### 1. Concept Type Matters
**Best aligned (>0.95)**: Scientific/universal concepts (water, energy, justice, knowledge) and basic emotions (fear). **Worst aligned (<0.85)**: Nature concepts with cultural variation (tree, mountain) and complex emotions (anger, peace). Universal, unambiguous meanings align better than culturally-loaded terms.

### 2. German Embeddings Show Systematic Divergence
German shows lower retrieval accuracy (80-93%) with 8 of 10 worst-aligned pairs involving German. Possible causes: compound word semantics, training data distribution, morphological complexity affecting tokenization.

### 3. Script Similarity ≠ Embedding Similarity
Languages with different scripts show strong alignment (Chinese ↔ Japanese: 0.006; French ↔ Russian: 0.004), suggesting the model learns semantic representations transcending orthographic differences.

### 4. Strong Discriminability Between Concepts
Cross-concept pairs average only 0.252 similarity vs 0.919 for same-concept cross-lingual pairs—a 0.667 discriminability gap confirming the model clusters by meaning, not language.

### 5. Emotion Concepts Show Cultural Variation
"Anger" has the lowest intra-concept similarity (0.746), consistent with research showing emotion concepts vary across cultures.

## Visualization

![Detailed Cross-Lingual Analysis](fig/cross_lingual_alignment_figure.png)

**(a)** Intra-concept similarity by category. **(b)** Cross-lingual retrieval accuracy heatmap. **(c)** Nearest-neighbor accuracy (k=3). **(d)** Most misaligned pairs (sim < 0.5). **(e)** Summary metrics. **(f)** Alignment by concept category.

## Implications

**Practitioners**: Multilingual retrieval works well for most concepts; exercise caution with German queries and culturally-loaded terms.

**Researchers**: The "German divergence" warrants investigation; emotion alignment could probe cultural bias; evaluation should cover diverse semantic categories.

## Reproducing This Experiment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python cross_lingual_alignment.py
```

To test different models, pass `model_name` to `CrossLingualAlignmentExperiment`.

## Files

| File | Description |
|------|-------------|
| `cross_lingual_alignment.py` | Main experiment code |
| `fig/experiment_diagram.png` | Experiment design diagram |
| `fig/cross_lingual_alignment_figure.png` | Results visualization |
| `alignment_results.json` | Raw results (JSON) |
| `requirements.txt` | Dependencies |

## References

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [Sentence-Transformers: Multilingual Sentence Embeddings](https://www.sbert.net/)
- [Multilingual MiniLM Paper](https://arxiv.org/abs/2002.10957)
- [On the Cross-lingual Transferability of Monolingual Representations](https://arxiv.org/abs/1910.11856)

## License

MIT
