#!/usr/bin/env python3
"""
Publication-quality figure for the Cross-Lingual Knowledge Alignment experiment.
Uses colorblind-safe palettes and clean typography following scientific visualization best practices.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path.home() / '.claude/skills/scientific-visualization/assets'))
sys.path.insert(0, str(__import__('pathlib').Path.home() / '.claude/skills/scientific-visualization/scripts'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from string import ascii_lowercase

# ---------------------------------------------------------------------------
# Publication style setup
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Okabe-Ito colorblind-safe palette
OI = {
    'orange':   '#E69F00',
    'sky_blue': '#56B4E9',
    'green':    '#009E73',
    'yellow':   '#F0E442',
    'blue':     '#0072B2',
    'vermillion': '#D55E00',
    'purple':   '#CC79A7',
    'black':    '#000000',
}

# Category colors (colorblind-safe)
CAT_COLORS = {
    'Abstract':   OI['blue'],
    'Concrete':   OI['orange'],
    'Scientific': OI['green'],
    'Emotion':    OI['vermillion'],
}

CONCEPT_CATEGORY = {
    'freedom': 'Abstract', 'love': 'Abstract', 'justice': 'Abstract',
    'knowledge': 'Abstract', 'peace': 'Abstract',
    'water': 'Concrete', 'sun': 'Concrete', 'tree': 'Concrete',
    'house': 'Concrete', 'mountain': 'Concrete',
    'gravity': 'Scientific', 'energy': 'Scientific',
    'happiness': 'Emotion', 'fear': 'Emotion', 'anger': 'Emotion',
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open('alignment_results.json') as f:
    data = json.load(f)

intra_sim = data['intra_concept_similarity']
nn_acc = data['nearest_neighbor_accuracy']
languages = data['languages']
misaligned = data['misaligned_pairs']

# Retrieval matrix from README (not stored in JSON as full matrix)
retrieval_matrix = np.array([
    [0, .93, 1., .87, 1., 1., .93, 1.],
    [.93, 0, .93, .80, .93, .93, .87, .87],
    [1., .93, 0, .87, 1., 1., .93, 1.],
    [.93, .80, .93, 0, .93, .93, .80, .93],
    [1., .93, 1., .87, 0, 1., .93, 1.],
    [1., .93, 1., .87, 1., 0, .93, 1.],
    [.93, .87, .93, .80, .93, .93, 0, .93],
    [1., .93, 1., .87, 1., 1., 1., 0],
])

LANG_LABELS = ['EN', 'ES', 'FR', 'DE', 'ZH', 'JA', 'AR', 'RU']

# Sort concepts by similarity (descending)
sorted_concepts = sorted(intra_sim.items(), key=lambda x: x[1])
concept_names = [c for c, _ in sorted_concepts]
concept_sims = [s for _, s in sorted_concepts]
concept_colors = [CAT_COLORS[CONCEPT_CATEGORY[c]] for c in concept_names]

# Sort concepts for NN accuracy (same order for consistency)
nn_values = [nn_acc[c] for c in concept_names]

# ---------------------------------------------------------------------------
# Create figure  (double-column: ~183 mm = 7.2 in)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(7.2, 8.5))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.40,
             left=0.09, right=0.97, top=0.95, bottom=0.05)

# ===== Panel A: Intra-Concept Similarity =====
ax_a = fig.add_subplot(gs[0, 0])
bars_a = ax_a.barh(concept_names, concept_sims, color=concept_colors,
                   edgecolor='white', linewidth=0.3, height=0.7)
ax_a.set_xlabel('Cosine Similarity')
ax_a.set_title('Intra-Concept Similarity')
ax_a.set_xlim(0.25, 1.02)
mean_sim = np.mean(concept_sims)
ax_a.axvline(mean_sim, color=OI['black'], ls='--', lw=0.8, alpha=0.6)
ax_a.text(mean_sim + 0.01, len(concept_names) - 0.5,
          f'mean = {mean_sim:.3f}', fontsize=6.5, va='top', color='0.3')

# ===== Panel B: Cross-Lingual Retrieval Heatmap =====
ax_b = fig.add_subplot(gs[0, 1])
# Mask diagonal
masked = np.ma.masked_where(retrieval_matrix == 0, retrieval_matrix)
im = ax_b.imshow(masked, cmap='YlGnBu', vmin=0.75, vmax=1.0, aspect='equal')
ax_b.set_xticks(range(8))
ax_b.set_yticks(range(8))
ax_b.set_xticklabels(LANG_LABELS, fontsize=7)
ax_b.set_yticklabels(LANG_LABELS, fontsize=7)
ax_b.set_xlabel('Target Language')
ax_b.set_ylabel('Query Language')
ax_b.set_title('Cross-Lingual Retrieval Accuracy')

# Annotate cells
for i in range(8):
    for j in range(8):
        if i == j:
            ax_b.text(j, i, '-', ha='center', va='center', fontsize=6.5, color='0.5')
        else:
            val = retrieval_matrix[i, j]
            color = 'white' if val > 0.95 else 'black'
            ax_b.text(j, i, f'{val:.0%}', ha='center', va='center',
                      fontsize=6, color=color, fontweight='medium')

cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04, shrink=0.85)
cb.set_label('Accuracy', fontsize=7)
cb.ax.tick_params(labelsize=6)

# Draw diagonal cells in gray
for i in range(8):
    ax_b.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                  fill=True, facecolor='0.92', edgecolor='white', lw=0.5))
    ax_b.text(i, i, '-', ha='center', va='center', fontsize=6.5, color='0.5')

# ===== Panel C: Nearest-Neighbor Accuracy =====
ax_c = fig.add_subplot(gs[1, 0])
nn_colors = [CAT_COLORS[CONCEPT_CATEGORY[c]] for c in concept_names]
bars_c = ax_c.barh(concept_names, nn_values, color=nn_colors,
                   edgecolor='white', linewidth=0.3, height=0.7)
ax_c.set_xlabel('Accuracy')
ax_c.set_title('Nearest-Neighbor Accuracy (k = 3)')
ax_c.set_xlim(0.75, 1.02)
mean_nn = np.mean(nn_values)
ax_c.axvline(mean_nn, color=OI['black'], ls='--', lw=0.8, alpha=0.6)
ax_c.text(mean_nn + 0.005, len(concept_names) - 0.5,
          f'mean = {mean_nn:.1%}', fontsize=6.5, va='top', color='0.3')

# ===== Panel D: Misaligned Pairs (lollipop chart) =====
ax_d = fig.add_subplot(gs[1, 1])
pair_labels = [f'{m[0]} ({m[1].upper()}\u2013{m[2].upper()})' for m in misaligned]
pair_sims = [m[3] for m in misaligned]

# Highlight that most involve German
pair_colors = [OI['vermillion'] if 'de' in (m[1], m[2]) else OI['blue'] for m in misaligned]

ax_d.hlines(range(len(pair_labels)), 0, pair_sims, color=pair_colors,
            linewidth=1.2, alpha=0.7)
ax_d.scatter(pair_sims, range(len(pair_labels)), color=pair_colors,
             s=25, zorder=3, edgecolors='white', linewidths=0.3)
ax_d.set_yticks(range(len(pair_labels)))
ax_d.set_yticklabels(pair_labels, fontsize=6.5)
ax_d.set_xlabel('Cosine Similarity')
ax_d.set_title('Most Misaligned Pairs (sim < 0.5)')
ax_d.set_xlim(0.25, 0.52)
ax_d.axvline(0.5, color='0.7', ls=':', lw=0.7)

# Legend for German involvement
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=OI['vermillion'],
           markersize=5, label='Involves German'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=OI['blue'],
           markersize=5, label='Other pair'),
]
ax_d.legend(handles=legend_elements, loc='lower right', fontsize=6,
            frameon=True, framealpha=0.9, edgecolor='0.85')

# ===== Panel E: Summary Metrics =====
ax_e = fig.add_subplot(gs[2, 0])
metric_names = ['Intra-Concept\nSimilarity', 'NN Accuracy\n(k=3)',
                'Cross-Lingual\nRetrieval', 'Overall\nAlignment']
metric_values = [
    np.mean(list(intra_sim.values())),
    np.mean(list(nn_acc.values())),
    data['cross_lingual_retrieval_accuracy'],
    data['alignment_score'],
]
metric_colors = [OI['blue'], OI['sky_blue'], OI['orange'], OI['green']]

bars_e = ax_e.bar(metric_names, metric_values, color=metric_colors,
                  edgecolor='white', linewidth=0.5, width=0.65)
ax_e.set_ylabel('Score')
ax_e.set_title('Summary Metrics')
ax_e.set_ylim(0.85, 1.0)
for bar, val in zip(bars_e, metric_values):
    ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
              f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# ===== Panel F: Similarity by Category =====
ax_f = fig.add_subplot(gs[2, 1])

categories = ['Abstract', 'Concrete', 'Scientific', 'Emotion']
cat_means = []
cat_stds = []
cat_points = {}
for cat in categories:
    vals = [intra_sim[c] for c, ct in CONCEPT_CATEGORY.items() if ct == cat]
    cat_means.append(np.mean(vals))
    cat_stds.append(np.std(vals))
    cat_points[cat] = vals

x_pos = np.arange(len(categories))
bar_colors = [CAT_COLORS[c] for c in categories]

bars_f = ax_f.bar(x_pos, cat_means, yerr=cat_stds, color=bar_colors,
                  edgecolor='white', linewidth=0.5, width=0.55,
                  capsize=3, error_kw={'lw': 0.8, 'capthick': 0.8})

# Overlay individual points
for i, cat in enumerate(categories):
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(cat_points[cat]))
    ax_f.scatter(x_pos[i] + jitter, cat_points[cat],
                 color='black', s=12, alpha=0.5, zorder=3,
                 edgecolors='white', linewidths=0.3)

ax_f.set_xticks(x_pos)
ax_f.set_xticklabels(categories, fontsize=7)
ax_f.set_ylabel('Intra-Concept Similarity')
ax_f.set_title('Alignment by Concept Category')
ax_f.set_ylim(0.65, 1.05)

# ---------------------------------------------------------------------------
# Panel labels (a-f, bold, uppercase-style lowercase per Nature convention)
# ---------------------------------------------------------------------------
axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
for i, ax in enumerate(axes):
    ax.text(-0.12, 1.08, ascii_lowercase[i], transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

# Category legend in Panel A
from matplotlib.patches import Patch
legend_patches = [Patch(facecolor=CAT_COLORS[c], label=c) for c in categories]
ax_a.legend(handles=legend_patches, loc='lower right', fontsize=6,
            frameon=True, framealpha=0.9, edgecolor='0.85', title='Category',
            title_fontsize=6.5)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
from figure_export import save_publication_figure
save_publication_figure(fig, 'fig/cross_lingual_alignment_figure',
                        formats=['pdf', 'png'], dpi=300)

plt.close(fig)
print("\nDone. Output: fig/cross_lingual_alignment_figure.pdf / .png")
