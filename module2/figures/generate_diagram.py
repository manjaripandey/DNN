"""Generate architecture diagram for Module 2 report (Figure M2-1)."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def draw_box(ax, x, y, w, h, label, sublabel='', color='#AED6F1', fontsize=8):
    rect = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                   boxstyle='round,pad=0.02', linewidth=1,
                                   edgecolor='#2C3E50', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y + (0.08 if sublabel else 0), label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='#2C3E50')
    if sublabel:
        ax.text(x, y - 0.12, sublabel, ha='center', va='center',
                fontsize=6.5, color='#555555')


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2))


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    # ---------- backbone column (x=3) ----------
    bx = 3
    boxes = [
        (bx, 9.2, 'Input', '(B, 3, 32, 32)', '#F9E79F'),
        (bx, 8.1, 'CIFAR Stem', 'Conv3×3 s=1 → BN → ReLU\n(B, 64, 32, 32)', '#A9DFBF'),
        (bx, 7.0, 'Layer 1', '2 × BasicBlock(64→64, s=1)\n(B, 64, 32, 32)', '#A9DFBF'),
        (bx, 5.9, 'Layer 2', '2 × BasicBlock(64→128, s=2)\n(B, 128, 16, 16)', '#A9DFBF'),
        (bx, 4.8, 'Layer 3', '2 × BasicBlock(128→256, s=2)\n(B, 256, 8, 8)', '#A9DFBF'),
        (bx, 3.7, 'Layer 4', '2 × BasicBlock(256→512, s=2)\n(B, 512, 4, 4)', '#A9DFBF'),
        (bx, 2.6, 'Global Avg Pool', 'z ∈ ℝ⁵¹²', '#A9DFBF'),
    ]

    for (x, y, lbl, sub, col) in boxes:
        draw_box(ax, x, y, 2.5, 0.75, lbl, sub, color=col)

    for i in range(len(boxes) - 1):
        arrow(ax, boxes[i][0], boxes[i][1] - 0.38, boxes[i+1][0], boxes[i+1][1] + 0.38)

    # ---------- fork to Head A (x=7) and Head B (x=11) ----------
    fork_y = 2.6

    # Head A
    ax.annotate('', xy=(7, 1.9), xytext=(bx + 1.25, fork_y),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2,
                                connectionstyle='arc3,rad=-0.2'))
    draw_box(ax, 7, 1.55, 3.0, 0.6, 'Head A — Linear', 'Linear(512→10) → Softmax', '#AED6F1')
    draw_box(ax, 7, 0.7, 3.0, 0.55, 'q(y|x) ∈ Δ⁹', 'Head Variant A output', '#D5F5E3')

    # Head B
    ax.annotate('', xy=(11, 2.1), xytext=(bx + 1.25, fork_y),
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2,
                                connectionstyle='arc3,rad=0.2'))
    draw_box(ax, 11, 1.9, 3.5, 0.55, 'Head B — MLP (1)', 'Linear(512→256) → BN → ReLU → Dropout(0.3)', '#AED6F1')
    draw_box(ax, 11, 1.2, 3.5, 0.55, 'Head B — MLP (2)', 'Linear(256→10) → Softmax', '#AED6F1')
    draw_box(ax, 11, 0.5, 3.5, 0.55, 'q(y|x) ∈ Δ⁹', 'Head Variant B output', '#D5F5E3')
    arrow(ax, 11, 1.62, 11, 1.47)
    arrow(ax, 11, 0.92, 11, 0.77)

    arrow(ax, 7, 1.24, 7, 0.97)

    # ---------- annotations ----------
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 2.2), 2.5, 7.2,
                 boxstyle='round,pad=0.05', linewidth=1.5,
                 edgecolor='#117A65', facecolor='none', linestyle='--'))
    ax.text(0.4, 9.55, 'backbone_pretrained.pt\n(saved weights)', fontsize=7,
            color='#117A65', va='top')

    ax.add_patch(mpatches.FancyBboxPatch((5.4, 0.15), 10.0, 2.3,
                 boxstyle='round,pad=0.05', linewidth=1.5,
                 edgecolor='#922B21', facecolor='none', linestyle='--'))
    ax.text(5.5, 2.6, 'Re-initialized for fine-tuning (M3)', fontsize=7,
            color='#922B21', va='top')

    # ---------- legend ----------
    legend_elements = [
        mpatches.Patch(facecolor='#A9DFBF', edgecolor='#2C3E50', label='Backbone (pretrained)'),
        mpatches.Patch(facecolor='#AED6F1', edgecolor='#2C3E50', label='Prediction head'),
        mpatches.Patch(facecolor='#D5F5E3', edgecolor='#2C3E50', label='Output distribution'),
        mpatches.Patch(facecolor='#F9E79F', edgecolor='#2C3E50', label='Input'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=7, framealpha=0.9)

    ax.set_title('Module 2: ResNet-18 CIFAR Backbone + Prediction Heads\n'
                 'Project: Predicting Human Annotator Disagreement on CIFAR-10',
                 fontsize=11, fontweight='bold', pad=8)

    fig.tight_layout()
    fig.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight')
    print('Saved architecture_diagram.png')


if __name__ == '__main__':
    main()
