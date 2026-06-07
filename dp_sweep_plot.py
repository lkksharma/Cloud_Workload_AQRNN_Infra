"""
dp_sweep_plot.py
================
Reads all federated_v2_results_nm*.json files and the original federated_v2_results.json,
plots the Privacy-Utility tradeoff curve (epsilon vs RMSE).

Usage: python dp_sweep_plot.py
"""
import os, json, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Collect all result files
files = glob.glob("federated_v2_results_nm*.json")
# Also include the original run if it exists
if os.path.exists("federated_v2_results.json"):
    files.append("federated_v2_results.json")

if not files:
    print("ERROR: No result JSON files found!")
    exit(1)

points = []
for f in files:
    with open(f, 'r') as fh:
        data = json.load(fh)
    # Parse epsilon from dp_budget string "ε≤61.28, δ=1e-05"
    budget_str = data.get("dp_budget", "")
    if "≤" in budget_str:
        eps = float(budget_str.split("≤")[1].split(",")[0])
    elif "<=" in budget_str:
        eps = float(budget_str.split("<=")[1].split(",")[0])
    else:
        eps = 0
    rmse = data["rmse_orig"]
    mae = data["mae_orig"]
    smape = data["smape_orig"]
    nm = data.get("round_history", [{}])[0].get("dp_log", {})
    # Try to extract noise_mult
    noise_mult = 0
    for k, v in nm.items():
        if "noise_mult" in v:
            noise_mult = v["noise_mult"]
            break
    points.append({"eps": eps, "rmse": rmse, "mae": mae, "smape": smape,
                    "noise_mult": noise_mult, "file": f})
    print(f"  {f}: ε={eps:.1f}, RMSE={rmse:.2f}, noise_mult={noise_mult}")

# Sort by epsilon
points.sort(key=lambda x: x["eps"])

epsilons = [p["eps"] for p in points]
rmses = [p["rmse"] for p in points]
maes = [p["mae"] for p in points]
noise_mults = [p["noise_mult"] for p in points]

# === PLOT 1: Privacy-Utility Tradeoff (main plot for paper) ===
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(epsilons, rmses, 'o-', color='#e74c3c', linewidth=2.5, markersize=10,
         label='RMSE', zorder=5)
ax1.plot(epsilons, maes, 's--', color='#3498db', linewidth=2, markersize=8,
         label='MAE', alpha=0.8, zorder=4)

# Baseline reference line at RMSE=155
ax1.axhline(y=155, color='#2ecc71', linestyle=':', linewidth=2,
            label='Baseline RMSE = 155', alpha=0.8)

# Annotate each point
for p in points:
    ax1.annotate(f'σ_z={p["noise_mult"]}\nRMSE={p["rmse"]:.1f}',
                 xy=(p["eps"], p["rmse"]),
                 textcoords="offset points", xytext=(15, 10),
                 fontsize=9, ha='left',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', color='gray'))

ax1.set_xlabel('Privacy Budget ε (lower = stronger privacy)')
ax1.set_ylabel('Error (Original Scale)')
ax1.set_title('Privacy–Utility Tradeoff: AQRNN-CFL')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add secondary x-axis showing noise_mult
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
nm_ticks = epsilons
nm_labels = [f'{nm:.2f}' for nm in noise_mults]
ax2.set_xticks(nm_ticks)
ax2.set_xticklabels(nm_labels)
ax2.set_xlabel('Noise Multiplier σ_z')

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/dp_privacy_utility_tradeoff.png")
plt.close()
print(f"\nSaved: plots/dp_privacy_utility_tradeoff.png")

# === PLOT 2: Convergence curves overlaid ===
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

for i, p in enumerate(points):
    with open(p["file"], 'r') as fh:
        data = json.load(fh)
    history = data.get("round_history", [])
    rounds = [h["round"] for h in history]
    rmse_vals = [h["rmse"] for h in history]
    ax.plot(rounds, rmse_vals, 'o-', color=colors[i % len(colors)],
            linewidth=2, markersize=7,
            label=f'σ_z={p["noise_mult"]:.2f} (ε={p["eps"]:.0f})')

ax.axhline(y=155, color='gray', linestyle=':', linewidth=1.5,
           label='Baseline RMSE = 155')
ax.set_xlabel('Federated Round')
ax.set_ylabel('Validation RMSE')
ax.set_title('Convergence Under Different DP Noise Levels')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/dp_convergence_comparison.png")
plt.close()
print(f"Saved: plots/dp_convergence_comparison.png")