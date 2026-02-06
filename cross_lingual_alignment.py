"""
Cross-Lingual Knowledge Alignment Experiment
=============================================
Research Question: How consistently is the same concept represented
across different languages in a multilingual model?

This experiment uses FAISS for efficient similarity search and analysis.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import json
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. EXPERIMENTAL DATA: Parallel Concepts Across Languages
# ============================================================================

# Concepts translated across 8 languages
# Format: {concept_id: {lang_code: translation}}
PARALLEL_CONCEPTS = {
    # Abstract concepts
    "freedom": {
        "en": "freedom",
        "es": "libertad",
        "fr": "liberté",
        "de": "Freiheit",
        "zh": "自由",
        "ja": "自由",
        "ar": "حرية",
        "ru": "свобода"
    },
    "love": {
        "en": "love",
        "es": "amor",
        "fr": "amour",
        "de": "Liebe",
        "zh": "爱",
        "ja": "愛",
        "ar": "حب",
        "ru": "любовь"
    },
    "justice": {
        "en": "justice",
        "es": "justicia",
        "fr": "justice",
        "de": "Gerechtigkeit",
        "zh": "正义",
        "ja": "正義",
        "ar": "عدالة",
        "ru": "справедливость"
    },
    "knowledge": {
        "en": "knowledge",
        "es": "conocimiento",
        "fr": "connaissance",
        "de": "Wissen",
        "zh": "知识",
        "ja": "知識",
        "ar": "معرفة",
        "ru": "знание"
    },
    "peace": {
        "en": "peace",
        "es": "paz",
        "fr": "paix",
        "de": "Frieden",
        "zh": "和平",
        "ja": "平和",
        "ar": "سلام",
        "ru": "мир"
    },

    # Concrete concepts
    "water": {
        "en": "water",
        "es": "agua",
        "fr": "eau",
        "de": "Wasser",
        "zh": "水",
        "ja": "水",
        "ar": "ماء",
        "ru": "вода"
    },
    "sun": {
        "en": "sun",
        "es": "sol",
        "fr": "soleil",
        "de": "Sonne",
        "zh": "太阳",
        "ja": "太陽",
        "ar": "شمس",
        "ru": "солнце"
    },
    "tree": {
        "en": "tree",
        "es": "árbol",
        "fr": "arbre",
        "de": "Baum",
        "zh": "树",
        "ja": "木",
        "ar": "شجرة",
        "ru": "дерево"
    },
    "house": {
        "en": "house",
        "es": "casa",
        "fr": "maison",
        "de": "Haus",
        "zh": "房子",
        "ja": "家",
        "ar": "بيت",
        "ru": "дом"
    },
    "mountain": {
        "en": "mountain",
        "es": "montaña",
        "fr": "montagne",
        "de": "Berg",
        "zh": "山",
        "ja": "山",
        "ar": "جبل",
        "ru": "гора"
    },

    # Scientific concepts
    "gravity": {
        "en": "gravity",
        "es": "gravedad",
        "fr": "gravité",
        "de": "Schwerkraft",
        "zh": "重力",
        "ja": "重力",
        "ar": "جاذبية",
        "ru": "гравитация"
    },
    "energy": {
        "en": "energy",
        "es": "energía",
        "fr": "énergie",
        "de": "Energie",
        "zh": "能量",
        "ja": "エネルギー",
        "ar": "طاقة",
        "ru": "энергия"
    },

    # Emotion concepts
    "happiness": {
        "en": "happiness",
        "es": "felicidad",
        "fr": "bonheur",
        "de": "Glück",
        "zh": "幸福",
        "ja": "幸せ",
        "ar": "سعادة",
        "ru": "счастье"
    },
    "fear": {
        "en": "fear",
        "es": "miedo",
        "fr": "peur",
        "de": "Angst",
        "zh": "恐惧",
        "ja": "恐怖",
        "ar": "خوف",
        "ru": "страх"
    },
    "anger": {
        "en": "anger",
        "es": "ira",
        "fr": "colère",
        "de": "Wut",
        "zh": "愤怒",
        "ja": "怒り",
        "ar": "غضب",
        "ru": "гнев"
    }
}

# Extended sentences for context-aware alignment
PARALLEL_SENTENCES = {
    "science_discovery": {
        "en": "Scientists discovered a new species in the Amazon rainforest.",
        "es": "Los científicos descubrieron una nueva especie en la selva amazónica.",
        "fr": "Les scientifiques ont découvert une nouvelle espèce dans la forêt amazonienne.",
        "de": "Wissenschaftler entdeckten eine neue Art im Amazonas-Regenwald.",
        "zh": "科学家在亚马逊雨林中发现了一个新物种。",
        "ja": "科学者たちはアマゾンの熱帯雨林で新種を発見しました。"
    },
    "climate_change": {
        "en": "Climate change is affecting ecosystems worldwide.",
        "es": "El cambio climático está afectando los ecosistemas en todo el mundo.",
        "fr": "Le changement climatique affecte les écosystèmes du monde entier.",
        "de": "Der Klimawandel beeinflusst Ökosysteme weltweit.",
        "zh": "气候变化正在影响全球生态系统。",
        "ja": "気候変動は世界中の生態系に影響を与えています。"
    },
    "technology_progress": {
        "en": "Artificial intelligence is transforming how we work and live.",
        "es": "La inteligencia artificial está transformando cómo trabajamos y vivimos.",
        "fr": "L'intelligence artificielle transforme notre façon de travailler et de vivre.",
        "de": "Künstliche Intelligenz verändert unsere Art zu arbeiten und zu leben.",
        "zh": "人工智能正在改变我们的工作和生活方式。",
        "ja": "人工知能は私たちの働き方と生活を変革しています。"
    }
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "ru": "Russian"
}


@dataclass
class AlignmentResult:
    """Results from alignment analysis"""
    concept: str
    languages: List[str]
    embeddings: np.ndarray
    intra_concept_similarity: float  # Average similarity within same concept
    nearest_neighbor_accuracy: float  # % times same concept is nearest neighbor
    cluster_purity: float  # How pure concept clusters are


class CrossLingualAlignmentExperiment:
    """
    Experiment to measure cross-lingual knowledge alignment using FAISS.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize with a multilingual embedding model.

        Popular multilingual models:
        - paraphrase-multilingual-MiniLM-L12-v2 (fast, 384 dims)
        - paraphrase-multilingual-mpnet-base-v2 (better quality, 768 dims)
        - LaBSE (Google's Language-agnostic BERT)
        """
        print(f"Loading multilingual model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

        # Storage for embeddings and metadata
        self.embeddings = None
        self.metadata = []  # [(concept, language, text), ...]
        self.concept_to_indices = defaultdict(list)
        self.language_to_indices = defaultdict(list)

    def generate_embeddings(self, concepts: Dict = PARALLEL_CONCEPTS) -> np.ndarray:
        """Generate embeddings for all concepts across all languages."""
        texts = []
        self.metadata = []
        self.concept_to_indices = defaultdict(list)
        self.language_to_indices = defaultdict(list)

        idx = 0
        for concept, translations in concepts.items():
            for lang, text in translations.items():
                texts.append(text)
                self.metadata.append((concept, lang, text))
                self.concept_to_indices[concept].append(idx)
                self.language_to_indices[lang].append(idx)
                idx += 1

        print(f"Generating embeddings for {len(texts)} texts...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = self.embeddings.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        print(f"Generated embeddings shape: {self.embeddings.shape}")
        return self.embeddings

    def build_faiss_index(self) -> faiss.Index:
        """Build FAISS index for efficient similarity search."""
        # Use IndexFlatIP for cosine similarity (on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")
        return self.index

    def measure_intra_concept_similarity(self) -> Dict[str, float]:
        """
        Measure how similar embeddings of the same concept are across languages.
        Higher = better alignment.
        """
        results = {}

        for concept, indices in self.concept_to_indices.items():
            if len(indices) < 2:
                continue

            # Get embeddings for this concept
            concept_embeddings = self.embeddings[indices]

            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(concept_embeddings)):
                for j in range(i + 1, len(concept_embeddings)):
                    sim = np.dot(concept_embeddings[i], concept_embeddings[j])
                    similarities.append(sim)

            results[concept] = np.mean(similarities)

        return results

    def measure_nearest_neighbor_accuracy(self, k: int = 5) -> Dict[str, float]:
        """
        For each embedding, check if k nearest neighbors are the same concept.
        This measures if the model groups same concepts together.
        """
        results = {}

        for concept, indices in self.concept_to_indices.items():
            correct = 0
            total = 0

            for idx in indices:
                query = self.embeddings[idx:idx+1]

                # Search for k+1 neighbors (includes self)
                distances, neighbor_indices = self.index.search(query, k + 1)

                # Skip self (first result)
                neighbors = neighbor_indices[0][1:k+1]

                # Check how many neighbors are same concept
                for neighbor_idx in neighbors:
                    if self.metadata[neighbor_idx][0] == concept:
                        correct += 1
                    total += 1

            results[concept] = correct / total if total > 0 else 0

        return results

    def measure_cross_lingual_retrieval(self) -> Dict[Tuple[str, str], float]:
        """
        For each language pair, measure retrieval accuracy.
        Query in language A, find same concept in language B.
        """
        results = {}
        languages = list(self.language_to_indices.keys())

        for lang_a in languages:
            for lang_b in languages:
                if lang_a == lang_b:
                    continue

                correct = 0
                total = 0

                for idx in self.language_to_indices[lang_a]:
                    concept = self.metadata[idx][0]
                    query = self.embeddings[idx:idx+1]

                    # Get indices for lang_b only
                    lang_b_indices = self.language_to_indices[lang_b]
                    lang_b_embeddings = self.embeddings[lang_b_indices]

                    # Build temporary index for lang_b
                    temp_index = faiss.IndexFlatIP(self.dimension)
                    temp_index.add(lang_b_embeddings)

                    # Find nearest in lang_b
                    _, neighbors = temp_index.search(query, 1)
                    retrieved_idx = lang_b_indices[neighbors[0][0]]
                    retrieved_concept = self.metadata[retrieved_idx][0]

                    if retrieved_concept == concept:
                        correct += 1
                    total += 1

                results[(lang_a, lang_b)] = correct / total if total > 0 else 0

        return results

    def analyze_language_clusters(self) -> Dict[str, np.ndarray]:
        """
        Compute centroid for each language and measure inter-language distances.
        This reveals if embeddings cluster by language rather than concept.
        """
        centroids = {}

        for lang, indices in self.language_to_indices.items():
            lang_embeddings = self.embeddings[indices]
            centroid = np.mean(lang_embeddings, axis=0)
            faiss.normalize_L2(centroid.reshape(1, -1))
            centroids[lang] = centroid

        return centroids

    def compute_alignment_score(self) -> float:
        """
        Overall alignment score combining multiple metrics.
        Range: 0-1, higher is better.
        """
        intra_sim = self.measure_intra_concept_similarity()
        nn_accuracy = self.measure_nearest_neighbor_accuracy(k=3)
        retrieval = self.measure_cross_lingual_retrieval()

        # Weighted average
        avg_intra_sim = np.mean(list(intra_sim.values()))
        avg_nn_accuracy = np.mean(list(nn_accuracy.values()))
        avg_retrieval = np.mean(list(retrieval.values()))

        # Combine metrics
        alignment_score = (
            0.3 * avg_intra_sim +      # Intra-concept similarity
            0.3 * avg_nn_accuracy +     # Nearest neighbor accuracy
            0.4 * avg_retrieval         # Cross-lingual retrieval
        )

        return alignment_score

    def run_full_analysis(self) -> Dict:
        """Run complete cross-lingual alignment analysis."""
        print("\n" + "="*60)
        print("CROSS-LINGUAL KNOWLEDGE ALIGNMENT ANALYSIS")
        print("="*60)

        # Generate embeddings and build index
        self.generate_embeddings()
        self.build_faiss_index()

        # Measure intra-concept similarity
        print("\n1. INTRA-CONCEPT SIMILARITY (same concept across languages)")
        print("-" * 50)
        intra_sim = self.measure_intra_concept_similarity()
        for concept, sim in sorted(intra_sim.items(), key=lambda x: -x[1]):
            print(f"  {concept:15s}: {sim:.4f}")
        avg_intra = np.mean(list(intra_sim.values()))
        print(f"\n  Average: {avg_intra:.4f}")

        # Measure nearest neighbor accuracy
        print("\n2. NEAREST NEIGHBOR ACCURACY (k=3)")
        print("-" * 50)
        nn_acc = self.measure_nearest_neighbor_accuracy(k=3)
        for concept, acc in sorted(nn_acc.items(), key=lambda x: -x[1]):
            print(f"  {concept:15s}: {acc:.2%}")
        avg_nn = np.mean(list(nn_acc.values()))
        print(f"\n  Average: {avg_nn:.2%}")

        # Measure cross-lingual retrieval
        print("\n3. CROSS-LINGUAL RETRIEVAL ACCURACY")
        print("-" * 50)
        retrieval = self.measure_cross_lingual_retrieval()

        # Create retrieval matrix
        languages = list(self.language_to_indices.keys())
        retrieval_matrix = np.zeros((len(languages), len(languages)))
        for i, lang_a in enumerate(languages):
            for j, lang_b in enumerate(languages):
                if lang_a != lang_b:
                    retrieval_matrix[i][j] = retrieval.get((lang_a, lang_b), 0)

        print("\n  Query \\ Target", end="")
        for lang in languages:
            print(f"  {lang:>5s}", end="")
        print()
        for i, lang_a in enumerate(languages):
            print(f"  {lang_a:14s}", end="")
            for j, lang_b in enumerate(languages):
                if lang_a == lang_b:
                    print("     -", end="")
                else:
                    print(f"  {retrieval_matrix[i][j]:5.1%}", end="")
            print()

        avg_retrieval = np.mean(list(retrieval.values()))
        print(f"\n  Average retrieval accuracy: {avg_retrieval:.2%}")

        # Analyze language clustering
        print("\n4. LANGUAGE CENTROID DISTANCES")
        print("-" * 50)
        centroids = self.analyze_language_clusters()

        print("\n  Distance between language centroids (lower = more overlap):")
        for i, (lang_a, cent_a) in enumerate(centroids.items()):
            for lang_b, cent_b in list(centroids.items())[i+1:]:
                dist = 1 - np.dot(cent_a, cent_b)
                print(f"  {lang_a}-{lang_b}: {dist:.4f}")

        # Overall alignment score
        print("\n" + "="*60)
        alignment_score = self.compute_alignment_score()
        print(f"OVERALL ALIGNMENT SCORE: {alignment_score:.4f}")
        print("="*60)

        # Interpretation
        print("\nINTERPRETATION:")
        if alignment_score > 0.8:
            print("  Excellent alignment - concepts are consistently represented")
        elif alignment_score > 0.6:
            print("  Good alignment - most concepts align well across languages")
        elif alignment_score > 0.4:
            print("  Moderate alignment - some concepts align, others diverge")
        else:
            print("  Poor alignment - significant cross-lingual inconsistency")

        return {
            "intra_concept_similarity": intra_sim,
            "nearest_neighbor_accuracy": nn_acc,
            "cross_lingual_retrieval": retrieval,
            "retrieval_matrix": retrieval_matrix,
            "languages": languages,
            "alignment_score": alignment_score
        }

    def visualize_results(self, results: Dict, save_path: str = "fig/alignment_analysis.png"):
        """Create visualizations of the alignment analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Intra-concept similarity bar chart
        ax1 = axes[0, 0]
        concepts = list(results["intra_concept_similarity"].keys())
        similarities = list(results["intra_concept_similarity"].values())
        colors = ['#2ecc71' if s > 0.7 else '#f39c12' if s > 0.5 else '#e74c3c' for s in similarities]
        ax1.barh(concepts, similarities, color=colors)
        ax1.set_xlabel("Cosine Similarity")
        ax1.set_title("Intra-Concept Similarity Across Languages")
        ax1.set_xlim(0, 1)
        ax1.axvline(x=np.mean(similarities), color='black', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
        ax1.legend()

        # 2. Cross-lingual retrieval heatmap
        ax2 = axes[0, 1]
        languages = results["languages"]
        matrix = results["retrieval_matrix"]
        sns.heatmap(matrix, annot=True, fmt='.0%', cmap='RdYlGn',
                   xticklabels=languages, yticklabels=languages, ax=ax2,
                   vmin=0, vmax=1)
        ax2.set_title("Cross-Lingual Retrieval Accuracy")
        ax2.set_xlabel("Target Language")
        ax2.set_ylabel("Query Language")

        # 3. Nearest neighbor accuracy
        ax3 = axes[1, 0]
        nn_acc = results["nearest_neighbor_accuracy"]
        concepts = list(nn_acc.keys())
        accuracies = [nn_acc[c] for c in concepts]
        colors = ['#2ecc71' if a > 0.7 else '#f39c12' if a > 0.5 else '#e74c3c' for a in accuracies]
        ax3.barh(concepts, accuracies, color=colors)
        ax3.set_xlabel("Accuracy")
        ax3.set_title("Nearest Neighbor Accuracy (k=3)")
        ax3.set_xlim(0, 1)
        ax3.axvline(x=np.mean(accuracies), color='black', linestyle='--', label=f'Mean: {np.mean(accuracies):.1%}')
        ax3.legend()

        # 4. Summary metrics
        ax4 = axes[1, 1]
        metrics = ['Intra-Concept\nSimilarity', 'NN Accuracy\n(k=3)', 'Cross-Lingual\nRetrieval', 'Overall\nAlignment']
        values = [
            np.mean(list(results["intra_concept_similarity"].values())),
            np.mean(list(results["nearest_neighbor_accuracy"].values())),
            np.mean(list(results["cross_lingual_retrieval"].values())),
            results["alignment_score"]
        ]
        colors = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']
        bars = ax4.bar(metrics, values, color=colors)
        ax4.set_ylabel("Score")
        ax4.set_title("Summary Metrics")
        ax4.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()

    def find_misaligned_concepts(self) -> List[Tuple[str, str, str, float]]:
        """
        Find concept pairs that should be similar but aren't.
        Returns: [(concept, lang1, lang2, similarity), ...]
        """
        misaligned = []

        for concept, indices in self.concept_to_indices.items():
            for i, idx_a in enumerate(indices):
                for idx_b in indices[i+1:]:
                    lang_a = self.metadata[idx_a][1]
                    lang_b = self.metadata[idx_b][1]

                    sim = np.dot(self.embeddings[idx_a], self.embeddings[idx_b])

                    if sim < 0.5:  # Threshold for misalignment
                        misaligned.append((concept, lang_a, lang_b, sim))

        return sorted(misaligned, key=lambda x: x[3])

    def compare_models(self, model_names: List[str]) -> Dict[str, float]:
        """Compare alignment scores across different multilingual models."""
        scores = {}

        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"Testing model: {model_name}")
            print('='*60)

            try:
                self.model = SentenceTransformer(model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.generate_embeddings()
                self.build_faiss_index()
                scores[model_name] = self.compute_alignment_score()
                print(f"Alignment score: {scores[model_name]:.4f}")
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                scores[model_name] = None

        return scores


def main():
    """Main experiment runner."""

    # Initialize experiment
    experiment = CrossLingualAlignmentExperiment(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Run full analysis
    results = experiment.run_full_analysis()

    # Find misaligned concepts
    print("\n" + "="*60)
    print("MISALIGNED CONCEPT PAIRS (similarity < 0.5)")
    print("="*60)
    misaligned = experiment.find_misaligned_concepts()
    if misaligned:
        for concept, lang_a, lang_b, sim in misaligned[:10]:
            print(f"  {concept}: {lang_a}-{lang_b} = {sim:.4f}")
    else:
        print("  No severely misaligned concept pairs found!")

    # Visualize results
    try:
        experiment.visualize_results(results)
    except Exception as e:
        print(f"\nVisualization skipped (matplotlib display issue): {e}")

    # Save results
    output = {
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "alignment_score": float(results["alignment_score"]),
        "intra_concept_similarity": {k: float(v) for k, v in results["intra_concept_similarity"].items()},
        "nearest_neighbor_accuracy": {k: float(v) for k, v in results["nearest_neighbor_accuracy"].items()},
        "languages": results["languages"],
        "misaligned_pairs": [(c, la, lb, float(s)) for c, la, lb, s in misaligned]
    }

    with open("alignment_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to: alignment_results.json")

    return results


if __name__ == "__main__":
    main()
