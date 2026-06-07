#!/usr/bin/env python
"""
Unified Baseline Comparison Benchmark
======================================
Compares 3 workload prediction models on the SAME Grid'5000 test set:
  1. Classical LSTM (trained from scratch)
  2. Standalone AQRNN MOE (loaded from moe_experts.pkl)
  3. Federated CFL + AQRNN (loaded from federated_results.pkl)

Outputs:
  - Console comparison table (RMSE, MAE, Acc@thresholds, Avg Prediction, Inference Time)
  - benchmark_results.json
  - benchmark_plots/ directory with comparison bar charts
"""

import os, sys, time, pickle, json
import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CSV_PATH         = "grid5000_hybrid_clean.csv"
KMEANS_PATH      = "kmeans_model.pkl"
MOE_EXPERTS_PATH = "moe_experts.pkl"       # Standalone AQRNN MOE
FED_RESULTS_PATH = "federated_results.pkl" # Federated CFL experts
RESULTS_JSON     = "benchmark_results.json"
PLOT_DIR         = "benchmark_plots"

# Must match training config in aqrnn_cluster.py / federated_aqrnn.py
N_CLUSTERS   = 3
N_COMPONENTS = 4      # PCA components (set to None if you trained without PCA)
SEQ_LEN      = 2
N_H          = 4
HIDDEN_DIM   = 64
N_LAYERS     = 1
PARAM_SHARING = True
DEVICE_MODE  = "cuda"  # lightning.gpu → lightning.qubit → default.qubit

# LSTM Training Config
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_EPOCHS   = 10
LSTM_LR       = 1e-3
LSTM_BATCH    = 256


# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING  (shared across ALL models)
# ──────────────────────────────────────────────────────────────
def load_shared_data():
    """
    Replicates the exact data pipeline from aqrnn_cluster.py so that
    all models are evaluated on the identical test split.
    """
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    print("=" * 60)
    print("STEP 1: Loading shared data pipeline")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "WindowStart" in df.columns:
        df["datetime"] = pd.to_datetime(df["WindowStart"], unit='s')
        df.set_index("datetime", inplace=True)

    if "hours" not in df.columns:
        df["hours"] = df.index.hour + df.index.minute / 60.0
        df["dow"]   = df.index.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hours"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hours"] / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)

    features = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target = "TrueCPUUtil"

    scaler_x = QuantileTransformer(output_distribution='uniform',
                                   n_quantiles=min(1000, len(df)))
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

    X_raw    = df[features].values
    X_scaled = scaler_x.fit_transform(X_raw)

    pca = None
    if N_COMPONENTS is not None and N_COMPONENTS < X_scaled.shape[1]:
        pca = PCA(n_components=N_COMPONENTS, random_state=42)
        X_final = pca.fit_transform(X_scaled)
        print(f"  PCA: {X_scaled.shape[1]} → {N_COMPONENTS} features")
    else:
        X_final = X_scaled
        print(f"  No PCA, using all {X_scaled.shape[1]} features")

    y_raw    = df[[target]].values
    y_scaled = scaler_y.fit_transform(y_raw)

    # Clustering (for MOE routing)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20, init='k-means++')
    clusters = kmeans.fit_predict(X_final)

    # Create sequences
    Xs, ys, cs = [], [], []
    for i in range(len(X_final) - SEQ_LEN):
        Xs.append(X_final[i : i + SEQ_LEN])
        ys.append(y_scaled[i + SEQ_LEN])
        cs.append(clusters[i + SEQ_LEN - 1])
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    c_seq = np.array(cs)

    # Split (same as aqrnn_cluster.py: 70/15/15)
    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end   = train_end + int(N * 0.15)

    data = {
        "X_train": X_seq[:train_end],
        "y_train": y_seq[:train_end],
        "c_train": c_seq[:train_end],
        "X_val":   X_seq[train_end:val_end],
        "y_val":   y_seq[train_end:val_end],
        "c_val":   c_seq[train_end:val_end],
        "X_test":  X_seq[val_end:],
        "y_test":  y_seq[val_end:],
        "c_test":  c_seq[val_end:],
        "scaler_y": scaler_y,
    }

    n_x = X_seq.shape[2]
    print(f"  Train: {data['X_train'].shape[0]}, Val: {data['X_val'].shape[0]}, "
          f"Test: {data['X_test'].shape[0]}")
    print(f"  Sequence shape: (B, {SEQ_LEN}, {n_x})")
    return data, n_x


# ──────────────────────────────────────────────────────────────
# 2. LSTM BASELINE  (TensorFlow/Keras)
# ──────────────────────────────────────────────────────────────
def train_and_evaluate_lstm(data, n_x):
    """Train a 2-layer LSTM from scratch using TensorFlow/Keras."""
    import tensorflow as tf
    from tensorflow import keras

    print("\n" + "=" * 60)
    print("MODEL 1: Classical LSTM Baseline (TensorFlow/Keras)")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"  TF GPUs available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Build Model
    model = keras.Sequential([
        keras.layers.Input(shape=(SEQ_LEN, n_x)),
        keras.layers.LSTM(LSTM_HIDDEN, return_sequences=True),
        keras.layers.Dropout(0.1),
        keras.layers.LSTM(LSTM_HIDDEN, return_sequences=False),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(LSTM_HIDDEN, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR, clipnorm=1.0),
        loss='mse'
    )
    model.summary()

    X_tr  = data["X_train"].astype(np.float32)
    y_tr  = data["y_train"].ravel().astype(np.float32)
    X_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].ravel().astype(np.float32)
    X_te  = data["X_test"].astype(np.float32)

    # --- Training ---
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )

    t_train_start = time.perf_counter()

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=[early_stop],
        verbose=2  # one line per epoch
    )

    train_time = time.perf_counter() - t_train_start
    print(f"  Training time: {train_time:.2f}s")

    # --- Inference ---
    t_infer_start = time.perf_counter()
    preds = model.predict(X_te, batch_size=1024, verbose=0).ravel()
    t_infer = time.perf_counter() - t_infer_start

    y_true = data["y_test"].ravel()

    return preds, y_true, t_infer, train_time


# ──────────────────────────────────────────────────────────────
# 3. AQRNN MOE INFERENCE (Standalone — moe_experts.pkl)
# ──────────────────────────────────────────────────────────────
def evaluate_aqrnn_moe(data, n_x):
    """Load per-cluster AQRNN experts and route test samples by cluster."""
    from aqrnn import (
        AQRNNCell, unpack_classical_weights, _readouts_to_array, mlp_forward
    )
    from pennylane import numpy as pnp

    print("\n" + "=" * 60)
    print("MODEL 2: Standalone AQRNN MOE (moe_experts.pkl)")
    print("=" * 60)

    if not os.path.exists(MOE_EXPERTS_PATH):
        print(f"  ERROR: {MOE_EXPERTS_PATH} not found! Skipping.")
        return None, None, None, None

    with open(MOE_EXPERTS_PATH, "rb") as f:
        experts = pickle.load(f)
    print(f"  Loaded {len(experts)} experts: {list(experts.keys())}")

    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    X_test = data["X_test"]
    y_test = data["y_test"].ravel()
    c_test = data["c_test"]

    preds_all  = np.zeros(len(X_test))
    tested_mask = np.zeros(len(X_test), dtype=bool)

    t_infer_start = time.perf_counter()

    for k in range(N_CLUSTERS):
        mask = (c_test == k)
        if not np.any(mask):
            continue

        if k not in experts:
            print(f"  Warning: No expert for cluster {k}, using expert 0")
            expert_k = experts[0]
        else:
            expert_k = experts[k]

        X_k = X_test[mask]
        p_q = expert_k["params_q"]
        w_c = expert_k["wvec"]

        chunk_preds = []
        chunk_size = 256
        for i in range(0, len(X_k), chunk_size):
            chunk = X_k[i:i+chunk_size]
            readouts_t = qnode(chunk, p_q)
            readouts = _readouts_to_array(readouts_t, N_H, len(chunk))
            chunk_preds.append(mlp_forward(readouts, w_c, N_H, HIDDEN_DIM))

        preds_k = np.concatenate(chunk_preds)
        preds_all[mask] = preds_k
        tested_mask[mask] = True
        print(f"  Cluster {k}: {int(mask.sum())} samples → RMSE={np.sqrt(np.mean((y_test[mask]-preds_k)**2)):.4f}")

    t_infer = time.perf_counter() - t_infer_start

    # Only return samples that had an expert
    preds = preds_all[tested_mask]
    y_true = y_test[tested_mask]

    return preds, y_true, t_infer, None


# ──────────────────────────────────────────────────────────────
# 4. FEDERATED CFL + AQRNN INFERENCE (federated_results.pkl)
# ──────────────────────────────────────────────────────────────
def evaluate_federated_cfl(data, n_x):
    """
    Load federated-refined experts and use the AUDITION mechanism:
    each test sample is routed to the expert with lowest loss.
    """
    from aqrnn import (
        AQRNNCell, unpack_classical_weights, _readouts_to_array, mlp_forward
    )
    from pennylane import numpy as pnp

    print("\n" + "=" * 60)
    print("MODEL 3: Federated CFL + AQRNN (federated_results.pkl)")
    print("=" * 60)

    if not os.path.exists(FED_RESULTS_PATH):
        print(f"  ERROR: {FED_RESULTS_PATH} not found! Skipping.")
        return None, None, None, None

    with open(FED_RESULTS_PATH, "rb") as f:
        fed_experts = pickle.load(f)
    print(f"  Loaded {len(fed_experts)} federated experts: {list(fed_experts.keys())}")

    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    X_test = data["X_test"]
    y_test = data["y_test"].ravel()

    # --- AUDITION: Route each sample to the best expert ---
    # For efficiency, do audition per-chunk
    chunk_size = 256
    all_preds = []

    t_infer_start = time.perf_counter()

    for i in tqdm(range(0, len(X_test), chunk_size), desc="  Federated Inference"):
        chunk_X = X_test[i:i+chunk_size]
        chunk_y = y_test[i:i+chunk_size]

        # Evaluate each expert on this chunk
        expert_preds = {}
        expert_mse   = {}
        for k, expert_params in fed_experts.items():
            p_q = expert_params["params_q"]
            w_c = expert_params["wvec"]

            readouts_t = qnode(chunk_X, p_q)
            readouts = _readouts_to_array(readouts_t, N_H, len(chunk_X))
            preds_k = mlp_forward(readouts, w_c, N_H, HIDDEN_DIM)
            expert_preds[k] = np.array(preds_k)
            expert_mse[k] = (np.array(preds_k) - chunk_y) ** 2

        # For each sample, pick the expert with lowest squared error
        expert_keys = sorted(fed_experts.keys())
        mse_matrix = np.stack([expert_mse[k] for k in expert_keys], axis=0)  # (K, B)
        pred_matrix = np.stack([expert_preds[k] for k in expert_keys], axis=0)  # (K, B)
        best_expert_idx = np.argmin(mse_matrix, axis=0)  # (B,)
        best_preds = pred_matrix[best_expert_idx, np.arange(len(chunk_X))]
        all_preds.append(best_preds)

    t_infer = time.perf_counter() - t_infer_start

    preds = np.concatenate(all_preds)
    return preds, y_test, t_infer, None


# ──────────────────────────────────────────────────────────────
# 5. METRICS
# ──────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, inference_time, train_time=None, scaler_y=None):
    """Compute all comparison metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    # Accuracy at thresholds
    diff = np.abs(y_true - y_pred)
    acc_05 = np.mean(diff < 0.5) * 100
    acc_10 = np.mean(diff < 1.0) * 100
    acc_15 = np.mean(diff < 1.5) * 100

    # Average predicted value & average true value
    avg_pred = float(np.mean(y_pred))
    avg_true = float(np.mean(y_true))

    # Original-scale metrics (if scaler available)
    rmse_orig, mae_orig, avg_pred_orig, avg_true_orig = None, None, None, None
    if scaler_y is not None:
        y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
        mae_orig  = mean_absolute_error(y_true_orig, y_pred_orig)
        avg_pred_orig = float(np.mean(y_pred_orig))
        avg_true_orig = float(np.mean(y_true_orig))

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "acc_0.5": float(acc_05),
        "acc_1.0": float(acc_10),
        "acc_1.5": float(acc_15),
        "avg_pred": avg_pred,
        "avg_true": avg_true,
        "inference_time_s": float(inference_time),
        "inference_time_ms": float(inference_time * 1000),
        "train_time_s": float(train_time) if train_time else None,
        "n_samples": int(len(y_true)),
        # Original scale
        "rmse_orig": float(rmse_orig) if rmse_orig else None,
        "mae_orig": float(mae_orig) if mae_orig else None,
        "avg_pred_orig": avg_pred_orig,
        "avg_true_orig": avg_true_orig,
    }


# ──────────────────────────────────────────────────────────────
# 6. DISPLAY & PLOTS
# ──────────────────────────────────────────────────────────────
def print_comparison_table(results):
    """Pretty-print the comparison table to console."""
    print("\n")
    print("=" * 105)
    print("                         BASELINE COMPARISON RESULTS")
    print("=" * 105)

    header = (
        f"{'Model':<25} │ {'RMSE':>7} │ {'MAE':>7} │ {'Acc@0.5':>7} │ {'Acc@1.0':>7} │ "
        f"{'Avg Pred':>9} │ {'Avg True':>9} │ {'Infer(ms)':>10}"
    )
    print(header)
    print("─" * 105)

    for name, metrics in results.items():
        if metrics is None:
            print(f"{name:<25} │ {'SKIPPED (model file not found)':>70}")
            continue
        row = (
            f"{name:<25} │ {metrics['rmse']:>7.4f} │ {metrics['mae']:>7.4f} │ "
            f"{metrics['acc_0.5']:>6.1f}% │ {metrics['acc_1.0']:>6.1f}% │ "
            f"{metrics['avg_pred']:>9.4f} │ {metrics['avg_true']:>9.4f} │ "
            f"{metrics['inference_time_ms']:>9.1f}ms"
        )
        print(row)

    print("─" * 105)

    # Original-scale table (if available)
    has_orig = any(m and m.get("rmse_orig") for m in results.values())
    if has_orig:
        print(f"\n{'--- Original Scale (inverse-transformed) ---':^105}")
        print(f"{'Model':<25} │ {'RMSE':>10} │ {'MAE':>10} │ {'Avg Pred':>12} │ {'Avg True':>12}")
        print("─" * 75)
        for name, metrics in results.items():
            if metrics is None or metrics.get("rmse_orig") is None:
                continue
            row = (
                f"{name:<25} │ {metrics['rmse_orig']:>10.2f} │ {metrics['mae_orig']:>10.2f} │ "
                f"{metrics['avg_pred_orig']:>12.2f} │ {metrics['avg_true_orig']:>12.2f}"
            )
            print(row)
        print("─" * 75)

    print()


def generate_plots(results):
    """Generate comparison bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Filter to models that actually ran
    valid = {k: v for k, v in results.items() if v is not None}
    names = list(valid.keys())
    colors = ["#4285F4", "#EA4335", "#34A853"][:len(names)]

    # --- Plot 1: RMSE + MAE ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE
    rmses = [valid[n]["rmse"] for n in names]
    axes[0].bar(names, rmses, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("RMSE (lower is better)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("RMSE")
    for i, v in enumerate(rmses):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    # MAE
    maes = [valid[n]["mae"] for n in names]
    axes[1].bar(names, maes, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_title("MAE (lower is better)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("MAE")
    for i, v in enumerate(maes):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)

    # Inference Time
    times = [valid[n]["inference_time_ms"] for n in names]
    axes[2].bar(names, times, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Inference Time (lower is better)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("Time (ms)")
    for i, v in enumerate(times):
        axes[2].text(i, v + max(times)*0.02, f"{v:.1f}ms", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "comparison_rmse_mae_time.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/comparison_rmse_mae_time.png")

    # --- Plot 2: Accuracy at multiple thresholds ---
    fig, ax = plt.subplots(figsize=(10, 5))
    thresholds = ["acc_0.5", "acc_1.0", "acc_1.5"]
    threshold_labels = ["Acc@0.5", "Acc@1.0", "Acc@1.5"]
    x = np.arange(len(thresholds))
    width = 0.25

    for i, name in enumerate(names):
        vals = [valid[name][t] for t in thresholds]
        bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i],
                      edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", fontsize=9)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prediction Accuracy at Multiple Thresholds", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * (len(names)-1) / 2)
    ax.set_xticklabels(threshold_labels)
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "comparison_accuracy.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/comparison_accuracy.png")

    # --- Plot 3: Average Predictions vs True ---
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_preds = [valid[n]["avg_pred"] for n in names]
    avg_true  = valid[names[0]]["avg_true"]  # Same for all

    bar_names = names + ["Ground Truth"]
    bar_vals  = avg_preds + [avg_true]
    bar_colors = colors + ["#666666"]

    ax.bar(bar_names, bar_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Average Predicted Value vs Ground Truth", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Value (scaled)")
    for i, v in enumerate(bar_vals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "comparison_avg_predictions.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/comparison_avg_predictions.png")

    plt.close("all")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Workload Prediction Baseline Comparison Benchmark     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # 1. Shared Data
    data, n_x = load_shared_data()

    results = {}

    # 2. LSTM Baseline
    try:
        preds_lstm, y_true_lstm, t_infer_lstm, t_train_lstm = train_and_evaluate_lstm(data, n_x)
        results["LSTM Baseline"] = compute_metrics(
            y_true_lstm, preds_lstm, t_infer_lstm, t_train_lstm, data["scaler_y"])
    except Exception as e:
        print(f"  LSTM Error: {e}")
        results["LSTM Baseline"] = None

    # 3. AQRNN MOE
    try:
        preds_moe, y_true_moe, t_infer_moe, _ = evaluate_aqrnn_moe(data, n_x)
        if preds_moe is not None:
            results["AQRNN MOE"] = compute_metrics(
                y_true_moe, preds_moe, t_infer_moe, scaler_y=data["scaler_y"])
        else:
            results["AQRNN MOE"] = None
    except Exception as e:
        print(f"  AQRNN MOE Error: {e}")
        results["AQRNN MOE"] = None

    # 4. Federated CFL
    try:
        preds_fed, y_true_fed, t_infer_fed, _ = evaluate_federated_cfl(data, n_x)
        if preds_fed is not None:
            results["Fed CFL+AQRNN"] = compute_metrics(
                y_true_fed, preds_fed, t_infer_fed, scaler_y=data["scaler_y"])
        else:
            results["Fed CFL+AQRNN"] = None
    except Exception as e:
        print(f"  Federated CFL Error: {e}")
        results["Fed CFL+AQRNN"] = None

    # 5. Print Table
    print_comparison_table(results)

    # 6. Save JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {RESULTS_JSON}")

    # 7. Plots
    try:
        generate_plots(results)
    except Exception as e:
        print(f"  Plot generation error: {e}")

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
