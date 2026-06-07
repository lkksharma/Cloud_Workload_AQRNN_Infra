"""
generate_plots.py
=================
Generates all prediction & analysis plots for the AQRNN Federated Learning paper.

Plots generated:
  1. Actual vs Predicted (time-series overlay) — per cluster + combined
  2. Scatter plot (Actual vs Predicted) with ideal line
  3. Convergence curve (RMSE over FL rounds)
  4. Residual distribution (histogram + KDE)
  5. Per-cluster bar comparison (RMSE, MAE, SMAPE)
  6. Compression & DP budget evolution over rounds
  7. Cumulative error distribution (CDF)
  8. Prediction error heatmap by hour-of-day and day-of-week

Requires: federated_v2_results.pkl, federated_v2_results.json,
          moe_experts.pkl, kmeans_model.pkl, grid5000_hybrid_clean.csv, aqrnn.py
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/SSH
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats

sys.path.append(os.getcwd())

from aqrnn import AQRNNCell, unpack_classical_weights
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pennylane as qml
from pennylane import numpy as pnp

# ── CONFIG ──
CSV_PATH       = "grid5000_hybrid_clean.csv"
RESULTS_JSON   = "federated_v2_results.json"
RESULTS_PKL    = "federated_v2_results.pkl"
KMEANS_PKL     = "kmeans_model.pkl"
SEQ_LEN        = 2
N_H            = 4
HIDDEN_DIM     = 64
N_LAYERS       = 1
PARAM_SHARING  = True
DEVICE_MODE    = "cpu"  # CPU is fine for inference-only plots
OUTPUT_DIR     = "plots"

# Plot styling
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

CLUSTER_COLORS = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
CLUSTER_NAMES  = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3'}


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING & INFERENCE
# ═══════════════════════════════════════════════════════════════

def _readouts_to_array(readouts_t, n_h, batch_size):
    out = pnp.array(readouts_t)
    if out.ndim == 1:
        return pnp.reshape(out, (1, -1))
    elif out.shape[0] == n_h and out.ndim == 2:
        return pnp.transpose(out)
    return out


def load_test_data_and_predict():
    """Load everything, run inference, return arrays for plotting."""
    print("Loading models...")
    with open(KMEANS_PKL, "rb") as f:
        kdata = pickle.load(f)
    scaler_x = kdata["scaler_x"]
    pca_obj  = kdata.get("pca")
    kmeans   = kdata.get("kmeans")

    with open(RESULTS_PKL, "rb") as f:
        experts = pickle.load(f)

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "WindowStart" in df.columns:
        df["datetime"] = pd.to_datetime(df["WindowStart"], unit='s')
        df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    if "hours" not in df.columns:
        df["hours"] = df.index.hour + df.index.minute / 60.0
        df["dow"]   = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hours"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hours"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)

    req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target_col = "TrueCPUUtil"

    X_raw = df[req_cols].values
    y_raw = df[target_col].values.reshape(-1, 1)
    timestamps = df.index

    X_scaled = scaler_x.transform(X_raw)
    if pca_obj is not None:
        X_scaled = pca_obj.transform(X_scaled)

    clusters = kmeans.predict(X_scaled) if kmeans else np.zeros(len(X_scaled), dtype=int)

    # Create sequences
    X_seq, y_seq_raw, c_seq, ts_seq = [], [], [], []
    hours_seq, dow_seq = [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i+SEQ_LEN])
        y_seq_raw.append(y_raw[i+SEQ_LEN])
        c_seq.append(clusters[i+SEQ_LEN-1])
        ts_seq.append(timestamps[i+SEQ_LEN])
        hours_seq.append(df["hours"].iloc[i+SEQ_LEN])
        dow_seq.append(df["dow"].iloc[i+SEQ_LEN])

    X_seq     = np.array(X_seq)
    y_seq_raw = np.array(y_seq_raw)
    c_seq     = np.array(c_seq)
    ts_seq    = np.array(ts_seq)
    hours_seq = np.array(hours_seq)
    dow_seq   = np.array(dow_seq)

    # Split
    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end   = train_end + int(N * 0.15)

    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    scaler_y.fit(y_seq_raw[:train_end])
    y_seq = scaler_y.transform(y_seq_raw)

    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    y_test_raw = y_seq_raw[val_end:]
    c_test = c_seq[val_end:]
    ts_test = ts_seq[val_end:]
    hours_test = hours_seq[val_end:]
    dow_test = dow_seq[val_end:]

    # Build QNode
    n_x = pca_obj.n_components_ if pca_obj is not None else 9
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    forget_vec = 0.95 * np.ones(SEQ_LEN)

    # Run inference per cluster
    print("Running inference...")
    all_preds_scaled = np.zeros(len(X_test))
    all_mask = np.zeros(len(X_test), dtype=bool)

    for k in sorted(experts.keys()):
        mask = (c_test == k)
        if not np.any(mask):
            continue
        ep = experts[k]
        X_k = X_test[mask]
        print(f"  Cluster {k}: {np.sum(mask)} samples...")

        preds_k = []
        chunk_size = 256
        for i in range(0, len(X_k), chunk_size):
            chunk = X_k[i:i+chunk_size]
            readouts_t = qnode(chunk, ep["params_q"], forget_gate=forget_vec)
            readouts = _readouts_to_array(readouts_t, N_H, len(chunk))
            W1, b1, W2, b2 = unpack_classical_weights(ep["wvec"], N_H, HIDDEN_DIM)
            z = pnp.dot(pnp.array(readouts), pnp.array(W1).T) + pnp.array(b1)
            h = pnp.maximum(0, z)
            out = pnp.dot(h, pnp.array(W2).T) + pnp.array(b2)
            preds_k.append(np.array(out[:, 0]))
        preds_k = np.concatenate(preds_k)
        all_preds_scaled[mask] = preds_k
        all_mask[mask] = True

    # Inverse transform to original scale
    y_true_orig = scaler_y.inverse_transform(y_test[all_mask].reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(all_preds_scaled[all_mask].reshape(-1, 1)).ravel()

    return {
        'y_true': y_true_orig,
        'y_pred': y_pred_orig,
        'clusters': c_test[all_mask],
        'timestamps': ts_test[all_mask],
        'hours': hours_test[all_mask],
        'dow': dow_test[all_mask],
        'mask': all_mask,
    }


def load_round_history():
    """Load the JSON results for convergence plots."""
    with open(RESULTS_JSON, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  PLOT 1: TIME-SERIES OVERLAY (Actual vs Predicted)
# ═══════════════════════════════════════════════════════════════

def plot_timeseries(data, n_points=500):
    """Actual vs Predicted over time — zoomed window for clarity."""
    y_true = data['y_true']
    y_pred = data['y_pred']
    ts     = data['timestamps']
    clusters = data['clusters']

    # Full time-series (subsampled for readability)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])

    # Top: Actual vs Predicted
    step = max(1, len(y_true) // n_points)
    idx = np.arange(0, len(y_true), step)

    axes[0].plot(ts[idx], y_true[idx], color='#2c3e50', linewidth=0.8,
                 alpha=0.9, label='Actual')
    axes[0].plot(ts[idx], y_pred[idx], color='#e74c3c', linewidth=0.8,
                 alpha=0.7, label='Predicted')
    axes[0].set_ylabel('CPU Utilization')
    axes[0].set_title('AQRNN-CFL: Actual vs Predicted CPU Utilization (Test Set)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Bottom: Error
    errors = y_true[idx] - y_pred[idx]
    axes[1].fill_between(ts[idx], errors, 0, alpha=0.4, color='#e74c3c')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Error')
    axes[1].set_xlabel('Time')
    axes[1].set_title('Prediction Error Over Time')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_timeseries_overlay.png'))
    plt.close()
    print("  ✓ 1_timeseries_overlay.png")

    # Zoomed window (first 200 points for detail)
    fig, ax = plt.subplots(figsize=(14, 5))
    zoom_n = min(200, len(y_true))
    ax.plot(range(zoom_n), y_true[:zoom_n], 'o-', color='#2c3e50',
            markersize=2, linewidth=1, label='Actual')
    ax.plot(range(zoom_n), y_pred[:zoom_n], 's-', color='#e74c3c',
            markersize=2, linewidth=1, alpha=0.7, label='Predicted')
    ax.fill_between(range(zoom_n), y_true[:zoom_n], y_pred[:zoom_n],
                    alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('CPU Utilization')
    ax.set_title('AQRNN-CFL: Zoomed Prediction Window (First 200 Test Samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1b_timeseries_zoomed.png'))
    plt.close()
    print("  ✓ 1b_timeseries_zoomed.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 2: SCATTER PLOT (Actual vs Predicted)
# ═══════════════════════════════════════════════════════════════

def plot_scatter(data):
    """Scatter with ideal line, colored by cluster."""
    y_true = data['y_true']
    y_pred = data['y_pred']
    clusters = data['clusters']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Combined scatter
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, c='#3498db', edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal (y = x)')
    ax.set_xlabel('Actual CPU Utilization')
    ax.set_ylabel('Predicted CPU Utilization')
    ax.set_title('Actual vs Predicted (All Clusters)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² and RMSE annotation
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.annotate(f'R² = {r2:.4f}\nRMSE = {rmse:.2f}',
                xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: Per-cluster scatter
    ax = axes[1]
    unique_clusters = sorted(np.unique(clusters))
    for k in unique_clusters:
        mask = clusters == k
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.2, s=8,
                   c=CLUSTER_COLORS.get(k, '#333'), label=CLUSTER_NAMES.get(k, f'C{k}'),
                   edgecolors='none')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Actual CPU Utilization')
    ax.set_ylabel('Predicted CPU Utilization')
    ax.set_title('Actual vs Predicted (Per Cluster)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_scatter_actual_vs_predicted.png'))
    plt.close()
    print("  ✓ 2_scatter_actual_vs_predicted.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 3: CONVERGENCE CURVE (RMSE over FL Rounds)
# ═══════════════════════════════════════════════════════════════

def plot_convergence(results):
    """RMSE, TopK, and DP budget over rounds."""
    history = results['round_history']
    rounds = [h['round'] for h in history]
    rmses  = [h['rmse'] for h in history]
    topks  = [h['topk'] for h in history]

    # Get epsilon from dp_log (take first expert's value)
    epsilons = []
    for h in history:
        dp = h.get('dp_log', {})
        if dp:
            first_key = list(dp.keys())[0]
            epsilons.append(dp[first_key].get('epsilon_so_far', 0))
        else:
            epsilons.append(0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # RMSE
    axes[0].plot(rounds, rmses, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[0].fill_between(rounds, rmses, alpha=0.1, color='#e74c3c')
    axes[0].set_ylabel('Validation RMSE')
    axes[0].set_title('Federated Learning Convergence')
    axes[0].grid(True, alpha=0.3)
    for i, (r, v) in enumerate(zip(rounds, rmses)):
        axes[0].annotate(f'{v:.1f}', (r, v), textcoords="offset points",
                         xytext=(0, 12), ha='center', fontsize=9)

    # TopK compression ratio
    axes[1].plot(rounds, topks, 's-', color='#2ecc71', linewidth=2, markersize=8)
    axes[1].fill_between(rounds, topks, alpha=0.1, color='#2ecc71')
    axes[1].set_ylabel('TopK Ratio')
    axes[1].set_title('Adaptive Compression (Lower = More Compression)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # DP Budget
    axes[2].plot(rounds, epsilons, 'D-', color='#9b59b6', linewidth=2, markersize=8)
    axes[2].fill_between(rounds, epsilons, alpha=0.1, color='#9b59b6')
    axes[2].set_ylabel('ε (Privacy Budget)')
    axes[2].set_xlabel('FL Round')
    axes[2].set_title('Cumulative Privacy Budget (ε)')
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_convergence_curve.png'))
    plt.close()
    print("  ✓ 3_convergence_curve.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 4: RESIDUAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def plot_residuals(data):
    """Histogram + KDE of prediction errors, per cluster."""
    y_true = data['y_true']
    y_pred = data['y_pred']
    clusters = data['clusters']
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Overall residual distribution
    ax = axes[0]
    ax.hist(residuals, bins=80, density=True, alpha=0.6, color='#3498db',
            edgecolor='white', linewidth=0.5)
    # KDE overlay
    kde_x = np.linspace(residuals.min(), residuals.max(), 300)
    kde = stats.gaussian_kde(residuals)
    ax.plot(kde_x, kde(kde_x), color='#e74c3c', linewidth=2)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=np.mean(residuals), color='#2ecc71', linestyle='--',
               linewidth=1.5, label=f'Mean = {np.mean(residuals):.2f}')
    ax.set_xlabel('Prediction Error (Actual - Predicted)')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution (All Clusters)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Per-cluster residual distributions
    ax = axes[1]
    unique_clusters = sorted(np.unique(clusters))
    for k in unique_clusters:
        mask = clusters == k
        res_k = residuals[mask]
        ax.hist(res_k, bins=50, density=True, alpha=0.35,
                color=CLUSTER_COLORS.get(k, '#333'),
                label=f'{CLUSTER_NAMES.get(k, f"C{k}")} (μ={np.mean(res_k):.1f})')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Prediction Error (Actual - Predicted)')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution (Per Cluster)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_residual_distribution.png'))
    plt.close()
    print("  ✓ 4_residual_distribution.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 5: PER-CLUSTER BAR COMPARISON
# ═══════════════════════════════════════════════════════════════

def plot_cluster_bars(results):
    """Bar chart comparing RMSE, MAE, SMAPE across clusters."""
    per_cluster = results['per_cluster']
    cluster_keys = sorted(per_cluster.keys())

    names  = [k.replace('cluster_', 'Cluster ') for k in cluster_keys]
    rmses  = [per_cluster[k]['rmse'] for k in cluster_keys]
    maes   = [per_cluster[k]['mae'] for k in cluster_keys]
    smapes = [per_cluster[k]['smape'] for k in cluster_keys]
    counts = [per_cluster[k]['n'] for k in cluster_keys]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(names))
    width = 0.5

    # RMSE
    bars = axes[0].bar(x, rmses, width, color=[CLUSTER_COLORS.get(i, '#333') for i in range(len(names))],
                       alpha=0.8, edgecolor='white')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Cluster')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{n}\n(n={c:,})' for n, c in zip(names, counts)])
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmses):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                     f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # MAE
    bars = axes[1].bar(x, maes, width, color=[CLUSTER_COLORS.get(i, '#333') for i in range(len(names))],
                       alpha=0.8, edgecolor='white')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE by Cluster')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{n}\n(n={c:,})' for n, c in zip(names, counts)])
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                     f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    # SMAPE
    bars = axes[2].bar(x, smapes, width, color=[CLUSTER_COLORS.get(i, '#333') for i in range(len(names))],
                       alpha=0.8, edgecolor='white')
    axes[2].set_ylabel('SMAPE (%)')
    axes[2].set_title('SMAPE by Cluster')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f'{n}\n(n={c:,})' for n, c in zip(names, counts)])
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, smapes):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Per-Cluster Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_cluster_bar_comparison.png'))
    plt.close()
    print("  ✓ 5_cluster_bar_comparison.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 6: CUMULATIVE ERROR DISTRIBUTION (CDF)
# ═══════════════════════════════════════════════════════════════

def plot_error_cdf(data):
    """CDF of absolute errors — shows what % of predictions are within X error."""
    y_true = data['y_true']
    y_pred = data['y_pred']
    clusters = data['clusters']
    abs_errors = np.abs(y_true - y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Overall CDF
    sorted_errors = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cdf, color='#2c3e50', linewidth=2.5, label='Overall')

    # Per-cluster CDF
    for k in sorted(np.unique(clusters)):
        mask = clusters == k
        se = np.sort(abs_errors[mask])
        cdf_k = np.arange(1, len(se) + 1) / len(se) * 100
        ax.plot(se, cdf_k, linewidth=1.5, alpha=0.7,
                color=CLUSTER_COLORS.get(k, '#333'),
                label=CLUSTER_NAMES.get(k, f'C{k}'))

    # Reference lines
    for pct in [50, 75, 90]:
        threshold = np.percentile(abs_errors, pct)
        ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.4)
        ax.annotate(f'{pct}% < {threshold:.0f}',
                    xy=(threshold, pct), xytext=(threshold + 10, pct - 5),
                    fontsize=9, color='#555')

    ax.set_xlabel('Absolute Prediction Error')
    ax.set_ylabel('Cumulative % of Predictions')
    ax.set_title('Cumulative Error Distribution (CDF)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_error_cdf.png'))
    plt.close()
    print("  ✓ 6_error_cdf.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 7: ERROR HEATMAP BY HOUR × DAY-OF-WEEK
# ═══════════════════════════════════════════════════════════════

def plot_error_heatmap(data):
    """Mean absolute error by hour-of-day and day-of-week."""
    abs_errors = np.abs(data['y_true'] - data['y_pred'])
    hours = data['hours'].astype(int) % 24
    dows  = data['dow'].astype(int) % 7

    # Build heatmap matrix
    heatmap = np.full((7, 24), np.nan)
    for d in range(7):
        for h in range(24):
            mask = (dows == d) & (hours == h)
            if np.any(mask):
                heatmap[d, h] = np.mean(abs_errors[mask])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_title('Mean Absolute Error by Time (Hour × Day)')
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.colorbar(im, ax=ax, label='MAE', shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_error_heatmap.png'))
    plt.close()
    print("  ✓ 7_error_heatmap.png")


# ═══════════════════════════════════════════════════════════════
#  PLOT 8: PER-CLUSTER TIME-SERIES
# ═══════════════════════════════════════════════════════════════

def plot_per_cluster_timeseries(data, n_points=300):
    """Separate time-series panel for each cluster."""
    y_true = data['y_true']
    y_pred = data['y_pred']
    clusters = data['clusters']

    unique_clusters = sorted(np.unique(clusters))
    n_clusters = len(unique_clusters)

    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 4 * n_clusters), sharex=False)
    if n_clusters == 1:
        axes = [axes]

    for i, k in enumerate(unique_clusters):
        ax = axes[i]
        mask = clusters == k
        yt = y_true[mask]
        yp = y_pred[mask]

        step = max(1, len(yt) // n_points)
        idx = np.arange(0, len(yt), step)

        ax.plot(idx, yt[idx], color='#2c3e50', linewidth=0.8, alpha=0.9, label='Actual')
        ax.plot(idx, yp[idx], color=CLUSTER_COLORS.get(k, '#e74c3c'),
                linewidth=0.8, alpha=0.7, label='Predicted')
        ax.fill_between(idx, yt[idx], yp[idx], alpha=0.1, color=CLUSTER_COLORS.get(k, '#e74c3c'))

        rmse_k = np.sqrt(mean_squared_error(yt, yp))
        mae_k  = mean_absolute_error(yt, yp)
        ax.set_title(f'Cluster {k} — RMSE: {rmse_k:.2f}, MAE: {mae_k:.2f}  (n={np.sum(mask):,})')
        ax.set_ylabel('CPU Utilization')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    plt.suptitle('Per-Cluster Prediction Performance', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_per_cluster_timeseries.png'))
    plt.close()
    print("  ✓ 8_per_cluster_timeseries.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("  AQRNN-CFL Plot Generator")
    print("=" * 50)

    # Load results JSON
    results = load_round_history()

    # Load data and run inference
    data = load_test_data_and_predict()

    print(f"\nGenerating plots in '{OUTPUT_DIR}/'...")
    print("-" * 40)

    plot_timeseries(data)
    plot_scatter(data)
    plot_convergence(results)
    plot_residuals(data)
    plot_cluster_bars(results)
    plot_error_cdf(data)
    plot_error_heatmap(data)
    plot_per_cluster_timeseries(data)

    print("-" * 40)
    print(f"Done! All plots saved to '{OUTPUT_DIR}/' directory.")
    print(f"Total plots: 9 PNG files at 300 DPI")


if __name__ == "__main__":
    main()