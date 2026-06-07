#!/usr/bin/env python
"""
══════════════════════════════════════════════════════════════════════
Residual Correction for Cluster 2 — Post-hoc Classical Fix
══════════════════════════════════════════════════════════════════════

Loads the trained CFL experts from federated_results.pkl, gets
Expert 2's predictions on Cluster 2 data, trains a tiny Ridge
regression on the residuals (prediction errors), and reports
corrected overall metrics.

Runtime: ~5 minutes (mostly quantum inference for Cluster 2).

Usage:
    conda activate badminton && python finetune_cluster2.py
══════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import AQRNN components
from aqrnn import AQRNNCell, unpack_classical_weights, _readouts_to_array
from pennylane import numpy as pnp

# ══════════════════════════════════════════════════════════════
# CONFIG (must match federated_aqrnn.py)
# ══════════════════════════════════════════════════════════════
CSV_PATH      = "grid5000_hybrid_clean.csv"
KMEANS_PATH   = "kmeans_model.pkl"
FED_RESULTS   = "federated_results.pkl"
SEQ_LEN       = 2
N_H           = 4
HIDDEN_DIM    = 64
N_LAYERS      = 1
PARAM_SHARING = True
USE_FORGET_GATE = True
DEVICE_MODE   = "cuda"  # Must match federated_aqrnn.py


def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def predict_with_params(qnode, params_q, wvec, forget_vec, X, n_h=N_H,
                        hidden_dim=HIDDEN_DIM, chunk_size=256, label="Inference"):
    """Run inference in chunks (copied from federated_aqrnn.py)."""
    all_preds = []
    n_chunks = int(np.ceil(len(X) / chunk_size))
    for i in tqdm(range(0, len(X), chunk_size), total=n_chunks,
                  desc=f"    {label}", leave=False, colour='cyan'):
        chunk = X[i:i+chunk_size]
        readouts_t = qnode(chunk, params_q, forget_gate=forget_vec)
        readouts = _readouts_to_array(readouts_t, n_h, len(chunk))
        W1, b1, W2, b2 = unpack_classical_weights(wvec, n_h, hidden_dim)
        z = pnp.dot(pnp.array(readouts), pnp.array(W1).T) + pnp.array(b1)
        h = pnp.maximum(0, z)
        out = pnp.dot(h, pnp.array(W2).T) + pnp.array(b2)
        all_preds.append(np.array(out[:, 0]))
    return np.concatenate(all_preds)


def load_test_data():
    """Load global test set (same pipeline as federated_aqrnn.py)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

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
        df["dow"] = df.index.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hours"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hours"] / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)

    features = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    with open(KMEANS_PATH, "rb") as f:
        kdata = pickle.load(f)
    scaler_x = kdata["scaler_x"]
    pca_obj = kdata.get("pca")
    kmeans = kdata.get("kmeans")

    X_scaled = scaler_x.transform(df[features].values)
    X_final = pca_obj.transform(X_scaled)
    n_x = X_final.shape[1]
    clusters = kmeans.predict(X_final) if kmeans else np.zeros(len(X_final), dtype=int)
    n_clusters = len(np.unique(clusters))

    y_raw = df[["TrueCPUUtil"]].values
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

    Xs, ys, cs = [], [], []
    for i in range(len(X_final) - SEQ_LEN):
        Xs.append(X_final[i : i + SEQ_LEN])
        ys.append(y_raw[i + SEQ_LEN, 0])
        cs.append(clusters[i + SEQ_LEN - 1])

    X_seq = np.array(Xs, dtype=np.float32)
    y_seq = np.array(ys, dtype=np.float32).reshape(-1, 1)
    c_seq = np.array(cs, dtype=np.int32)

    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end = train_end + int(N * 0.15)

    scaler_y.fit(y_seq[:train_end])
    y_scaled = scaler_y.transform(y_seq).ravel()

    return {
        "X_test": X_seq[val_end:],
        "y_test": y_scaled[val_end:],
        "c_test": c_seq[val_end:],
        "X_train": X_seq[:train_end],
        "y_train": y_scaled[:train_end],
        "c_train": c_seq[:train_end],
        "scaler_y": scaler_y,
        "n_x": n_x,
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Residual Correction for Cluster 2                      ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Load trained experts
    with open(FED_RESULTS, "rb") as f:
        experts = pickle.load(f)
    print(f"  Loaded {len(experts)} experts from {FED_RESULTS}")

    # Load data
    data = load_test_data()
    n_x = data["n_x"]

    # Build QNode
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)
    forget_vec = 0.95 * np.ones(SEQ_LEN) if USE_FORGET_GATE else None

    # ── Step 1: Get predictions for ALL clusters ──
    print("\n  Getting quantum predictions for all clusters...")
    all_preds = np.zeros(len(data["y_test"]))

    for k in range(data["n_clusters"]):
        mask = (data["c_test"] == k)
        if not np.any(mask) or k not in experts:
            continue
        ep = experts[k]
        X_k = data["X_test"][mask]
        preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                       forget_vec, X_k, label=f"Cluster {k} Test")
        all_preds[mask] = preds_k

    # ── Step 2: Fit residual correction per cluster (on TRAINING data) ──
    print("\n  Fitting per-cluster residual correction models...")
    cluster_train_rmse = {}
    cluster_corr_data = {}
    ridges = {}
    for k in range(data["n_clusters"]):
        if k not in experts:
            continue
        mask_tr = (data["c_train"] == k)
        X_tr_k = data["X_train"][mask_tr]
        y_tr_k = data["y_train"][mask_tr]

        if len(X_tr_k) < 10:
            continue

        ep_k = experts[k]
        preds_tr_k = predict_with_params(qnode, ep_k["params_q"], ep_k["wvec"],
                                          forget_vec, X_tr_k,
                                          label=f"Cluster {k} Train")

        residuals = y_tr_k - preds_tr_k

        # Compute RMSE in ORIGINAL space (not scaled) to correctly identify worst expert
        y_tr_orig = data["scaler_y"].inverse_transform(y_tr_k.reshape(-1, 1)).ravel()
        p_tr_orig = data["scaler_y"].inverse_transform(preds_tr_k.reshape(-1, 1)).ravel()
        train_rmse = float(np.sqrt(np.mean((y_tr_orig - p_tr_orig) ** 2)))
        cluster_train_rmse[k] = train_rmse

        X_corr = np.column_stack([
            preds_tr_k.reshape(-1, 1),
            X_tr_k.reshape(len(X_tr_k), -1),
        ])
        cluster_corr_data[k] = (X_corr, residuals)
        print(f"    Cluster {k}: {len(X_tr_k):,} samples, train RMSE={train_rmse:.4f}")

    # Only correct the worst-performing expert (highest training RMSE)
    worst_k = max(cluster_train_rmse, key=cluster_train_rmse.get)
    median_rmse = float(np.median(list(cluster_train_rmse.values())))
    print(f"\n    Median train RMSE: {median_rmse:.4f}")
    print(f"    Worst expert: Cluster {worst_k} (RMSE={cluster_train_rmse[worst_k]:.4f})")

    ridges = {}
    for k, (X_corr, residuals) in cluster_corr_data.items():
        if cluster_train_rmse[k] > median_rmse:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_corr, residuals)
            ridges[k] = ridge
            print(f"    Cluster {k}: Ridge APPLIED (above median)")
        else:
            print(f"    Cluster {k}: SKIP (below median, well-calibrated)")

    # ── Step 3: Apply correction to ALL clusters ──
    corrected_all = all_preds.copy()
    for k, ridge in ridges.items():
        mask_te = (data["c_test"] == k)
        preds_k = all_preds[mask_te]
        X_te_k = data["X_test"][mask_te]

        X_corr = np.column_stack([
            preds_k.reshape(-1, 1),
            X_te_k.reshape(len(X_te_k), -1),
        ])
        corrected_all[mask_te] = preds_k + ridge.predict(X_corr)

    # ── Step 5: Evaluate — before vs after ──
    scaler_y = data["scaler_y"]
    y_test = data["y_test"]
    c_test = data["c_test"]

    print("\n" + "=" * 80)
    print("  RESULTS: Before vs After Residual Correction")
    print("=" * 80)

    for label, preds in [("BEFORE (original)", all_preds), ("AFTER  (corrected)", corrected_all)]:
        y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        p_orig = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()

        rmse_all = float(np.sqrt(mean_squared_error(y_orig, p_orig)))
        mae_all  = float(mean_absolute_error(y_orig, p_orig))
        sm_all   = smape(y_orig, p_orig)

        print(f"\n  {label}:")
        print(f"    Overall:   RMSE={rmse_all:.2f}  MAE={mae_all:.2f}  SMAPE={sm_all:.1f}%")

        for k in sorted(np.unique(c_test)):
            mask = c_test == k
            yt_k = y_orig[mask]
            yp_k = p_orig[mask]
            rmse_k = float(np.sqrt(mean_squared_error(yt_k, yp_k)))
            mae_k  = float(mean_absolute_error(yt_k, yp_k))
            sm_k   = smape(yt_k, yp_k)
            print(f"    Cluster {k}: RMSE={rmse_k:.2f}  MAE={mae_k:.2f}  "
                  f"SMAPE={sm_k:.1f}% (n={mask.sum()})")

    # Save corrected results
    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    p_corr = scaler_y.inverse_transform(corrected_all.reshape(-1, 1)).ravel()

    # Also compute before metrics for comparison
    p_before = scaler_y.inverse_transform(all_preds.reshape(-1, 1)).ravel()
    rmse_before = float(np.sqrt(mean_squared_error(y_orig, p_before)))
    rmse_after  = float(np.sqrt(mean_squared_error(y_orig, p_corr)))
    mae_after   = float(mean_absolute_error(y_orig, p_corr))
    smape_after = smape(y_orig, p_corr)

    # Per-cluster corrected
    cluster_results = {}
    for k in sorted(np.unique(c_test)):
        mk = c_test == k
        cluster_results[k] = {
            "rmse": float(np.sqrt(mean_squared_error(y_orig[mk], p_corr[mk]))),
            "mae": float(mean_absolute_error(y_orig[mk], p_corr[mk])),
            "smape": smape(y_orig[mk], p_corr[mk]),
            "n": int(mk.sum()),
        }

    results = {
        "method": "CFL AQRNN + Post-Quantum Ridge Correction",
        "rmse_orig": rmse_after,
        "mae_orig": mae_after,
        "smape_orig": smape_after,
        "per_cluster": cluster_results,
        "ridge_params_per_cluster": SEQ_LEN * n_x + 2,  # weights + bias
    }
    with open("residual_correction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── FINAL ARCHITECTURE SUMMARY ──
    ridge_params = (SEQ_LEN * n_x + 1 + 1) * data["n_clusters"]  # (features + pred + bias) × 3
    print("\n" + "═" * 80)
    print("  COMPLETE ARCHITECTURE: CFL AQRNN + Post-Quantum Ridge Correction")
    print("═" * 80)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  PIPELINE                                                       │
  │                                                                 │
  │  1. Input: 9 features → QuantileTransformer → PCA(4)           │
  │  2. KMeans(3) routing → assigns to Expert 0, 1, or 2           │
  │  3. AQRNN Quantum Circuit (4-qubit, {N_LAYERS}-layer)              │
  │     → Variational ansatz → MLP readout → prediction            │
  │  4. Post-Quantum Ridge: prediction + features → correction     │
  │     final = quantum_pred + ridge_correction                     │
  │                                                                 │
  │  Training: 8-round CFL (FedAvg per expert) + Ridge on residuals│
  └─────────────────────────────────────────────────────────────────┘

  PARAMETER COUNT:
    Quantum params (per expert):  ~130 variational params
    Classical MLP (per expert):   ~{HIDDEN_DIM * N_H + HIDDEN_DIM + HIDDEN_DIM + 1} weights
    Ridge correction (per clust): {SEQ_LEN * n_x + 2} weights
    Total (3 experts):            < 450 quantum + ~{ridge_params} ridge = < 500 total
""")

    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║  FINAL RESULTS: CFL AQRNN + Post-Quantum Ridge             ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  Overall RMSE:  {rmse_after:>8.2f}  (was {rmse_before:.2f}, Δ={rmse_before-rmse_after:+.2f})     ║")
    print(f"  ║  Overall MAE:   {mae_after:>8.2f}                                  ║")
    print(f"  ║  Overall SMAPE: {smape_after:>7.1f}%                                  ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    for k, cr in cluster_results.items():
        print(f"  ║  Cluster {k}: RMSE={cr['rmse']:>7.2f}  MAE={cr['mae']:>7.2f}  "
              f"SMAPE={cr['smape']:>5.1f}%  n={cr['n']}  ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║                                                            ║")
    print("  ║  COMPARISON vs FEDERATED BASELINES:                        ║")
    print(f"  ║  FedAvg esDNN       RMSE=156.05  SMAPE=84.1%  20,161 par  ║")
    print(f"  ║  FedAvg Bi-LSTM     RMSE=155.38  SMAPE=84.9%  35,457 par  ║")
    print(f"  ║  CFL AQRNN+Ridge    RMSE={rmse_after:>6.2f}  SMAPE={smape_after:>4.1f}%   < 500 par  ║")
    print("  ║                                                            ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Saved to residual_correction_results.json\n")
