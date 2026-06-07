"""
plot_cluster_vs_baseline.py
============================
Bar chart: per-cluster RMSE & MAE vs centralized baseline (155).
Clusters below baseline are green, above are red.

Usage: python plot_cluster_vs_baseline.py
Output: plots/cluster_rmse_vs_baseline.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'xtick.labelsize': 12, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ── DATA ──
clusters = ['Cluster 0\n(n=3,770)', 'Cluster 1\n(n=5,122)',
            'Cluster 2\n(n=5,155)', 'Global\n(n=14,047)']
rmse = [167.56, 149.96, 154.85, 156.64]
mae  = [123.77, 109.07, 116.47, 115.73]
baseline = 155

# ── COLORS ──
green    = '#1D9E75'
green_lt = 'rgba(29,158,117,0.35)'
red      = '#E24B4A'
red_lt   = 'rgba(226,75,74,0.35)'

rmse_colors = [red if v > baseline else green for v in rmse]
mae_colors  = [(1.0, 0.29, 0.29, 0.35) if v > baseline else (0.11, 0.62, 0.46, 0.35) for v in rmse]
rmse_edge   = ['#A32D2D' if v > baseline else '#0F6E56' for v in rmse]
mae_edge    = ['#A32D2D' if v > baseline else '#0F6E56' for v in rmse]

# ── PLOT ──
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(clusters))
width = 0.32

bars_rmse = ax.bar(x - width/2, rmse, width, color=rmse_colors, edgecolor=rmse_edge,
                   linewidth=0.8, label='RMSE', zorder=3)
bars_mae  = ax.bar(x + width/2, mae, width, color=mae_colors, edgecolor=mae_edge,
                   linewidth=0.8, linestyle='--', label='MAE', zorder=3)

# Baseline line
ax.axhline(y=baseline, color='#BA7517', linestyle='--', linewidth=2, zorder=4)
ax.text(len(clusters) - 0.55, baseline + 1.5, f'Centralized baseline (RMSE = {baseline})',
        color='#BA7517', fontsize=11, fontweight='bold', ha='right')

# Value labels on bars
for bar, val in zip(bars_rmse, rmse):
    color = '#791F1F' if val > baseline else '#085041'
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=color)

for bar, val in zip(bars_mae, mae):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.2,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10,
            color='#555')

# Formatting
ax.set_ylabel('Error (Original Scale)')
ax.set_title('Per-Cluster Performance vs Centralized Baseline', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(clusters)
ax.set_ylim(80, 180)
ax.grid(True, alpha=0.2, axis='y')
ax.set_axisbelow(True)

# Legend
legend_elements = [
    mpatches.Patch(facecolor=green, edgecolor='#0F6E56', label='RMSE (below baseline)'),
    mpatches.Patch(facecolor=red, edgecolor='#A32D2D', label='RMSE (above baseline)'),
    mpatches.Patch(facecolor=(0.11, 0.62, 0.46, 0.35), edgecolor='#0F6E56', label='MAE (below baseline)'),
    mpatches.Patch(facecolor=(1.0, 0.29, 0.29, 0.35), edgecolor='#A32D2D', label='MAE (above baseline)'),
]
ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/cluster_rmse_vs_baseline.png')
plt.close()
print("Saved: plots/cluster_rmse_vs_baseline.png")