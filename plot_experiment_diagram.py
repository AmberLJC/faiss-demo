#!/usr/bin/env python3
"""
Publication-quality experiment diagram for the Cross-Lingual Knowledge Alignment study.

Creates a schematic overview showing:
  - Input data structure (concepts x languages matrix)
  - Embedding & indexing pipeline
  - Evaluation metrics with key results

Uses colorblind-safe Okabe-Ito palette, clean sans-serif typography,
and vector-friendly rendering for journal submission.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Okabe-Ito palette
OI = {
    'orange':     '#E69F00',
    'sky_blue':   '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'blue':       '#0072B2',
    'vermillion': '#D55E00',
    'purple':     '#CC79A7',
    'black':      '#000000',
}

# Category colors
CAT = {
    'Abstract':   OI['blue'],
    'Concrete':   OI['orange'],
    'Scientific': OI['green'],
    'Emotion':    OI['vermillion'],
}

# Lighter tints for box fills
def tint(hex_color, factor=0.15):
    """Create a light tint of a hex color."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * (1 - factor))
    g = int(g + (255 - g) * (1 - factor))
    b = int(b + (255 - b) * (1 - factor))
    return f'#{r:02x}{g:02x}{b:02x}'

LIGHT_BLUE = tint(OI['blue'])
LIGHT_ORANGE = tint(OI['orange'])
LIGHT_GREEN = tint(OI['green'])
LIGHT_VERMILLION = tint(OI['vermillion'])
LIGHT_PURPLE = tint(OI['purple'])
LIGHT_SKY = tint(OI['sky_blue'])

BG_GRAY = '#f7f7f7'
BORDER_GRAY = '#cccccc'
ARROW_COLOR = '#555555'

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 9.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.5)
ax.set_aspect('equal')
ax.axis('off')

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def rounded_box(x, y, w, h, color, edge_color=None, alpha=1.0, lw=1.0,
                rounding=0.15, zorder=1):
    """Draw a rounded rectangle."""
    if edge_color is None:
        edge_color = color
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={rounding}",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def arrow(x1, y1, x2, y2, color=ARROW_COLOR, lw=1.5, style='-|>',
          connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color,
                          lw=lw, connectionstyle=connectionstyle,
                          mutation_scale=12, zorder=5)
    ax.add_patch(arr)

def section_label(x, y, text, fontsize=11):
    """Draw a bold section header."""
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold',
            color=OI['black'], va='center', ha='left')

# ===========================================================================
# TITLE
# ===========================================================================
ax.text(5.0, 12.1, 'Cross-Lingual Knowledge Alignment',
        fontsize=14, fontweight='bold', ha='center', va='center',
        color=OI['black'])
ax.text(5.0, 11.7, 'Experiment Design & Methodology',
        fontsize=9, ha='center', va='center', color='#555555')

# ===========================================================================
# SECTION 1: INPUT DATA  (y ~ 9.5 – 11.2)
# ===========================================================================
section_label(0.3, 11.15, '1  Input Data')

# Outer container
rounded_box(0.2, 8.6, 9.6, 2.4, BG_GRAY, BORDER_GRAY, lw=0.6)

# --- Concept categories (left side) ---
cat_x = 0.5
cat_w = 3.8
cat_items = [
    ('Abstract',   ['freedom', 'love', 'justice', 'knowledge', 'peace'],  CAT['Abstract']),
    ('Concrete',   ['water', 'sun', 'tree', 'house', 'mountain'],         CAT['Concrete']),
    ('Scientific', ['gravity', 'energy'],                                  CAT['Scientific']),
    ('Emotion',    ['happiness', 'fear', 'anger'],                         CAT['Emotion']),
]

cy = 10.65
for cat_name, concepts, color in cat_items:
    box_h = 0.32
    rounded_box(cat_x, cy - box_h, cat_w, box_h, tint(color, 0.20), color, lw=0.8)
    ax.text(cat_x + 0.12, cy - box_h / 2, cat_name,
            fontsize=7, fontweight='bold', color=color, va='center')
    concept_str = ', '.join(concepts)
    ax.text(cat_x + 1.35, cy - box_h / 2, concept_str,
            fontsize=6.5, color='#333333', va='center')
    cy -= box_h + 0.10

# Label below
ax.text(cat_x + cat_w / 2, 8.78, '15 concepts (4 categories)',
        fontsize=7, ha='center', color='#666666', style='italic')

# --- Multiplication sign ---
ax.text(4.75, 9.65, '\u00d7', fontsize=18, ha='center', va='center',
        color=OI['black'], fontweight='bold')

# --- Languages (right side) ---
lang_x = 5.2
lang_w = 4.4
rounded_box(lang_x, 8.85, lang_w, 1.85, 'white', BORDER_GRAY, lw=0.8)

ax.text(lang_x + lang_w / 2, 10.45, '8 Languages',
        fontsize=8, fontweight='bold', ha='center', color=OI['black'])

langs = [
    ('EN', 'English',  'Latin'),
    ('ES', 'Spanish',  'Latin'),
    ('FR', 'French',   'Latin'),
    ('DE', 'German',   'Latin'),
    ('ZH', 'Chinese',  'Hanzi'),
    ('JA', 'Japanese', 'Kana'),
    ('AR', 'Arabic',   'Arabic'),
    ('RU', 'Russian',  'Cyrillic'),
]

# 2 columns x 4 rows
for i, (code, name, script) in enumerate(langs):
    col = i // 4
    row = i % 4
    lx = lang_x + 0.3 + col * 2.2
    ly = 10.1 - row * 0.30
    # color badge
    badge_color = OI['sky_blue'] if script == 'Latin' else OI['purple']
    rounded_box(lx, ly - 0.12, 0.38, 0.24, badge_color, badge_color,
                lw=0, rounding=0.06, zorder=2)
    ax.text(lx + 0.19, ly, code, fontsize=6.5, fontweight='bold',
            color='white', ha='center', va='center', zorder=3)
    ax.text(lx + 0.50, ly, f'{name} ({script})', fontsize=6,
            color='#333333', va='center', zorder=3)

ax.text(lang_x + lang_w / 2, 8.95,
        '4 scripts \u00b7 120 total data points',
        fontsize=7, ha='center', color='#666666', style='italic')

# ===========================================================================
# ARROW: Input -> Pipeline
# ===========================================================================
arrow(5.0, 8.55, 5.0, 8.1, lw=1.8)
ax.text(5.35, 8.32, '120 texts', fontsize=6.5, color='#666666', va='center')

# ===========================================================================
# SECTION 2: PROCESSING PIPELINE  (y ~ 6.0 – 8.0)
# ===========================================================================
section_label(0.3, 7.95, '2  Processing Pipeline')

rounded_box(0.2, 5.95, 9.6, 1.85, BG_GRAY, BORDER_GRAY, lw=0.6)

# Pipeline steps
steps = [
    ('Encode',        'MiniLM-L12\nmultilingual',   OI['blue']),
    ('Normalize',     'L2 norm\n(cosine-ready)',     OI['sky_blue']),
    ('Index',         'FAISS\nIndexFlatIP',          OI['green']),
    ('Search',        'k-NN query\n(k = 3)',         OI['orange']),
]

step_w = 1.8
step_h = 1.25
gap = 0.25
total_w = len(steps) * step_w + (len(steps) - 1) * gap
start_x = (10 - total_w) / 2
step_y = 6.2

for i, (title, desc, color) in enumerate(steps):
    sx = start_x + i * (step_w + gap)
    # Step box
    rounded_box(sx, step_y, step_w, step_h, tint(color, 0.18), color,
                lw=1.0, rounding=0.12)
    # Step number circle
    circle = plt.Circle((sx + 0.25, step_y + step_h - 0.25), 0.17,
                         facecolor=color, edgecolor='white', lw=1.2, zorder=3)
    ax.add_patch(circle)
    ax.text(sx + 0.25, step_y + step_h - 0.25, str(i + 1),
            fontsize=7.5, fontweight='bold', color='white',
            ha='center', va='center', zorder=4)
    # Title
    ax.text(sx + step_w / 2, step_y + step_h - 0.28, title,
            fontsize=8.5, fontweight='bold', ha='center', va='center',
            color=color)
    # Description
    ax.text(sx + step_w / 2, step_y + 0.35, desc,
            fontsize=6.5, ha='center', va='center', color='#444444',
            linespacing=1.3)
    # Arrow to next step
    if i < len(steps) - 1:
        arrow(sx + step_w + 0.02, step_y + step_h / 2,
              sx + step_w + gap - 0.02, step_y + step_h / 2,
              color='#888888', lw=1.2)

# Model label below pipeline
rounded_box(3.2, 6.0, 3.6, 0.22, OI['blue'], OI['blue'], lw=0, rounding=0.06)
ax.text(5.0, 6.11, 'paraphrase-multilingual-MiniLM-L12-v2  (384-d)',
        fontsize=6, fontweight='bold', color='white', ha='center', va='center',
        zorder=3)

# ===========================================================================
# ARROW: Pipeline -> Evaluation
# ===========================================================================
arrow(5.0, 5.9, 5.0, 5.4, lw=1.8)

# ===========================================================================
# SECTION 3: EVALUATION METRICS  (y ~ 2.8 – 5.2)
# ===========================================================================
section_label(0.3, 5.25, '3  Evaluation Metrics')

rounded_box(0.2, 2.6, 9.6, 2.45, BG_GRAY, BORDER_GRAY, lw=0.6)

metrics = [
    ('Intra-Concept\nSimilarity',      '0.914', '30%', OI['blue'],
     'Avg cosine sim of\nsame concept across\nall language pairs'),
    ('Nearest-Neighbor\nAccuracy (k=3)', '96.1%', '30%', OI['sky_blue'],
     'Fraction of k-NN\nbelonging to the\nsame concept'),
    ('Cross-Lingual\nRetrieval',        '93.9%', '40%', OI['orange'],
     'Query lang A,\nretrieve correct\nconcept in lang B'),
]

met_w = 2.8
met_h = 2.0
met_gap = 0.35
met_total = len(metrics) * met_w + (len(metrics) - 1) * met_gap
met_start = (10 - met_total) / 2
met_y = 2.8

for i, (title, value, weight, color, desc) in enumerate(metrics):
    mx = met_start + i * (met_w + met_gap)
    # Card
    rounded_box(mx, met_y, met_w, met_h, 'white', color, lw=1.2, rounding=0.12)
    # Weight badge
    rounded_box(mx + met_w - 0.65, met_y + met_h - 0.35, 0.55, 0.25,
                color, color, lw=0, rounding=0.06, zorder=3)
    ax.text(mx + met_w - 0.375, met_y + met_h - 0.225, f'w={weight}',
            fontsize=5.5, color='white', fontweight='bold',
            ha='center', va='center', zorder=4)
    # Title
    ax.text(mx + met_w / 2, met_y + met_h - 0.45, title,
            fontsize=7.5, fontweight='bold', ha='center', va='top',
            color=color, linespacing=1.2)
    # Value
    ax.text(mx + met_w / 2, met_y + met_h / 2 - 0.15, value,
            fontsize=18, fontweight='bold', ha='center', va='center',
            color=color)
    # Description
    ax.text(mx + met_w / 2, met_y + 0.35, desc,
            fontsize=5.5, ha='center', va='center', color='#555555',
            linespacing=1.3)

    # Plus / equals signs between cards
    if i < len(metrics) - 1:
        ax.text(mx + met_w + met_gap / 2, met_y + met_h / 2, '+',
                fontsize=14, ha='center', va='center', color='#888888',
                fontweight='bold')

# ===========================================================================
# ARROW: Metrics -> Overall Score
# ===========================================================================
arrow(5.0, 2.55, 5.0, 2.1, lw=1.8)

# ===========================================================================
# SECTION 4: OVERALL SCORE  (y ~ 0.3 – 1.9)
# ===========================================================================
section_label(0.3, 1.95, '4  Result')

# Score box
rounded_box(1.5, 0.3, 7.0, 1.45, tint(OI['green'], 0.15), OI['green'],
            lw=1.5, rounding=0.18)

ax.text(3.6, 1.40, 'Overall Alignment Score', fontsize=10,
        fontweight='bold', ha='center', va='center', color=OI['green'])

ax.text(3.6, 0.82, '0.938', fontsize=28, fontweight='bold',
        ha='center', va='center', color=OI['black'])

ax.text(3.6, 0.48, 'Excellent  \u2014  consistent cross-lingual representation',
        fontsize=7, ha='center', va='center', color='#555555')

# Key findings (right side of result box)
findings_x = 6.3
ax.text(findings_x, 1.40, 'Key Findings', fontsize=7,
        fontweight='bold', color='#444444', va='center')
# Separator line
ax.plot([6.0, 6.0], [0.55, 1.50], color='#cccccc', lw=0.6, zorder=2)
findings = [
    'German shows systematic divergence',
    'Emotion concepts least aligned',
    'Scientific concepts best aligned',
]
for i, finding in enumerate(findings):
    fy = 1.15 - i * 0.25
    ax.text(findings_x, fy, f'\u2022 {finding}',
            fontsize=6, color='#555555', va='center')

# ===========================================================================
# Save
# ===========================================================================
fig.savefig('fig/experiment_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
fig.savefig('fig/experiment_diagram.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved: fig/experiment_diagram.png and fig/experiment_diagram.pdf")
