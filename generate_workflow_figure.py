#!/usr/bin/env python3
"""
Modern, visually engaging workflow figure
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import numpy as np

plt.rcParams['font.family'] = 'Arial'

fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0f172a')  # Dark background
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.set_facecolor('#0f172a')
ax.axis('off')

# Modern color palette
colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f97316', '#10b981']

def draw_node(x, y, r, color, label, sublabel, num):
    # Glow effect
    for i in range(3):
        glow = Circle((x, y), r + 0.15*(3-i), color=color, alpha=0.1*(i+1))
        ax.add_patch(glow)

    # Main circle
    circle = Circle((x, y), r, facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(circle)

    # Number badge
    badge = Circle((x - r*0.7, y + r*0.7), 0.2, facecolor='white', edgecolor=color, linewidth=2)
    ax.add_patch(badge)
    ax.text(x - r*0.7, y + r*0.7, str(num), ha='center', va='center',
            fontsize=10, fontweight='bold', color=color)

    # Labels
    ax.text(x, y + 0.05, label, ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')
    ax.text(x, y - 0.25, sublabel, ha='center', va='center', fontsize=8,
            color='#b0b0b0')

def draw_arrow(x1, y1, x2, y2, color):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='white', lw=2,
                               connectionstyle='arc3,rad=0'))

# Title
ax.text(7, 6.3, 'Cross-Lingual Knowledge Alignment', ha='center',
        fontsize=20, fontweight='bold', color='white')
ax.text(7, 5.85, 'Experiment Workflow', ha='center',
        fontsize=12, color='#94a3b8')

# Main pipeline - circular nodes
y = 4.2
r = 0.55
positions = [1.8, 4.2, 6.6, 9.0, 11.4]
labels = [('Input', '120 texts'), ('Encode', 'MiniLM'), ('Normalize', 'L2'),
          ('Index', 'FAISS'), ('Search', 'k-NN')]

for i, (pos, (lbl, sub)) in enumerate(zip(positions, labels)):
    draw_node(pos, y, r, colors[i], lbl, sub, i+1)
    if i < len(positions) - 1:
        draw_arrow(pos + r + 0.15, y, positions[i+1] - r - 0.15, y, 'white')

# Down arrow
ax.annotate('', xy=(11.4, y - r - 0.8), xytext=(11.4, y - r - 0.15),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))

# Compute node
draw_node(11.4, 2.3, r, '#06b6d4', 'Metrics', 'Compute', 6)

# Arrow to result
ax.annotate('', xy=(8.5, 2.3), xytext=(11.4 - r - 0.15, 2.3),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))

# Result box - special styling
result_bg = FancyBboxPatch((4.5, 1.5), 3.8, 1.6, boxstyle="round,rounding_size=0.3",
                           facecolor='#1e293b', edgecolor='#10b981', linewidth=3)
ax.add_patch(result_bg)

ax.text(6.4, 2.65, 'ALIGNMENT SCORE', ha='center', fontsize=9,
        fontweight='bold', color='#10b981')
ax.text(6.4, 2.05, '0.938', ha='center', fontsize=36, fontweight='bold', color='white')
ax.text(6.4, 1.65, 'Excellent', ha='center', fontsize=10, color='#10b981')

# Bottom stats cards
card_y = 0.4
card_h = 0.6
card_data = [
    ('Languages', '8', '#3b82f6'),
    ('Concepts', '15', '#8b5cf6'),
    ('Similarity', '0.914', '#ec4899'),
    ('NN Acc', '96.1%', '#f97316'),
    ('Retrieval', '93.9%', '#10b981'),
]

card_w = 2.0
start_x = 1.5
for i, (label, value, color) in enumerate(card_data):
    x = start_x + i * 2.4
    card = FancyBboxPatch((x, card_y), card_w, card_h, boxstyle="round,rounding_size=0.1",
                          facecolor='#1e293b', edgecolor=color, linewidth=2)
    ax.add_patch(card)
    ax.text(x + card_w/2, card_y + 0.42, value, ha='center', fontsize=14,
            fontweight='bold', color='white')
    ax.text(x + card_w/2, card_y + 0.15, label, ha='center', fontsize=8, color='#94a3b8')

plt.tight_layout()
plt.savefig('fig/experiment_workflow.png', dpi=300, bbox_inches='tight',
            facecolor='#0f172a', edgecolor='none')
plt.savefig('fig/experiment_workflow.pdf', bbox_inches='tight',
            facecolor='#0f172a', edgecolor='none')
print("Saved: fig/experiment_workflow.png and fig/experiment_workflow.pdf")
