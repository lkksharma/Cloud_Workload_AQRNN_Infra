#!/usr/bin/env python
"""
===========================================================================
AQRNN Ablation Study — Fully Transferable Script (GPU-Ready)
===========================================================================
Three primary variants:
  A. FCL without MoE  — Federated Averaging with a single global model
  B. Local AQRNN      — Centralized (non-federated), single model
  C. Full MoE         — KMeans routing + per-cluster experts (current best)

Ablation grid over:
  - moe_routing       : True / False
  - param_sharing     : True / False
  - forget_gate       : True / False
  - subset_ratio      : 0.25 / 0.5 / 1.0

Metrics: RMSE, MAE, SMAPE (per-cluster breakdown for MoE variants)

Usage:
  python ablation_study.py                  # Full run (3 variants + grid)
  python ablation_study.py --quick          # Smoke test (subset=0.1, epochs=1)
  python ablation_study.py --variants-only  # Only A/B/C, skip grid
===========================================================================
"""

import os, sys, time, json, copy, glob, argparse, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

# --- WINDOWS GPU FIX ---
if sys.platform == "win32":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cuda_bin = os.path.join(conda_prefix, "Library", "bin")
        if os.path.exists(cuda_bin):
            os.add_dll_directory(cuda_bin)
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import shared AQRNN components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aqrnn import (
    AQRNNCell, AdamState, adam_update, clip_grad_norm,
    make_loss_fn, pack_classical_weights, unpack_classical_weights,
    mlp_forward, _readouts_to_array, angle_encode,
)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION (Locked across all variants)
# ══════════════════════════════════════════════════════════════
CSV_PATH       = "grid5000_hybrid_clean.csv"
DATA_DIR       = "federated_data"
N_CLUSTERS     = 3
N_COMPONENTS   = 4          # PCA components
SEQ_LEN        = 2
N_H            = 4
HIDDEN_DIM     = 64
N_LAYERS       = 1
DEVICE_MODE    = "cuda"     # lightning.gpu → lightning.qubit → default.qubit
SEED           = 42
N_EPOCHS       = 3          # Per user request
BATCH_SIZE     = 256
LR_Q           = 1e-3
LR_C           = 1e-3
GRAD_CLIP      = 1.0
L2_Q           = 1e-3
L2_C           = 1e-3

# Federated-specific
FL_ROUNDS      = 3
FL_CLIENT_MAX_SAMPLES = 512
FL_CLIENT_BATCH = 64

# Output
RESULTS_JSON   = "ablation_results.json"
PLOT_DIR       = "ablation_plots"


# ══════════════════════════════════════════════════════════════
# 1. SHARED DATA PIPELINE
# ══════════════════════════════════════════════════════════════
def load_shared_data(n_components=N_COMPONENTS, n_clusters=N_CLUSTERS):
    """
    Replicates the exact data pipeline from aqrnn_cluster.py.
    Returns dict with train/val/test splits + scalers + cluster labels.
    """
    print("=" * 60)
    print("LOADING SHARED DATA PIPELINE")
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
        df["dow"] = df.index.dayofweek
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

    X_raw = df[features].values
    X_scaled = scaler_x.fit_transform(X_raw)

    pca = None
    if n_components is not None and n_components < X_scaled.shape[1]:
        pca = PCA(n_components=n_components, random_state=42)
        X_final = pca.fit_transform(X_scaled)
        print(f"  PCA: {X_scaled.shape[1]} → {n_components} features")
    else:
        X_final = X_scaled
        print(f"  No PCA, using all {X_scaled.shape[1]} features")

    y_raw = df[[target]].values
    y_scaled = scaler_y.fit_transform(y_raw)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, init='k-means++')
    clusters = kmeans.fit_predict(X_final)

    unique, counts = np.unique(clusters, return_counts=True)
    print("  Cluster Distribution:", {int(u): int(c) for u, c in zip(unique, counts)})

    # Create sequences
    Xs, ys, cs = [], [], []
    for i in range(len(X_final) - SEQ_LEN):
        Xs.append(X_final[i : i + SEQ_LEN])
        ys.append(y_scaled[i + SEQ_LEN])
        cs.append(clusters[i + SEQ_LEN - 1])
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    c_seq = np.array(cs)

    # 70/15/15 Split
    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end = train_end + int(N * 0.15)

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
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "pca": pca,
        "kmeans": kmeans,
    }

    n_x = X_seq.shape[2]
    print(f"  Train: {data['X_train'].shape[0]:,} | Val: {data['X_val'].shape[0]:,} | "
          f"Test: {data['X_test'].shape[0]:,}")
    print(f"  Sequence shape: (B, {SEQ_LEN}, {n_x})")
    return data, n_x


# ══════════════════════════════════════════════════════════════
# 2. METRICS
# ══════════════════════════════════════════════════════════════
def smape(y_true, y_pred, eps=1e-8):
    """Symmetric Mean Absolute Percentage Error (0–100 scale)."""
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def compute_metrics(y_true, y_pred, scaler_y, train_time=0.0, infer_time=0.0):
    """Compute RMSE, MAE, SMAPE in both scaled and original space."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse_scaled = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_scaled  = float(mean_absolute_error(y_true, y_pred))
    smape_scaled = smape(y_true, y_pred)

    # Original scale
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    rmse_orig = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    mae_orig  = float(mean_absolute_error(y_true_orig, y_pred_orig))
    smape_orig = smape(y_true_orig, y_pred_orig)

    return {
        "rmse_scaled": rmse_scaled,
        "mae_scaled":  mae_scaled,
        "smape_scaled": smape_scaled,
        "rmse_orig":   rmse_orig,
        "mae_orig":    mae_orig,
        "smape_orig":  smape_orig,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 2),
        "n_samples":   len(y_true),
    }


def compute_per_cluster_metrics(y_true, y_pred, c_test, scaler_y, n_clusters):
    """Per-cluster breakdown of RMSE, MAE, SMAPE."""
    cluster_metrics = {}
    for k in range(n_clusters):
        mask = (c_test == k)
        if not np.any(mask):
            cluster_metrics[f"cluster_{k}"] = {"n_samples": 0}
            continue
        yt = np.asarray(y_true).ravel()[mask]
        yp = np.asarray(y_pred).ravel()[mask]

        yt_orig = scaler_y.inverse_transform(yt.reshape(-1, 1)).ravel()
        yp_orig = scaler_y.inverse_transform(yp.reshape(-1, 1)).ravel()

        cluster_metrics[f"cluster_{k}"] = {
            "n_samples": int(mask.sum()),
            "rmse_orig": float(np.sqrt(mean_squared_error(yt_orig, yp_orig))),
            "mae_orig":  float(mean_absolute_error(yt_orig, yp_orig)),
            "smape_orig": smape(yt_orig, yp_orig),
        }
    return cluster_metrics


# ══════════════════════════════════════════════════════════════
# 3. CORE TRAINING LOOP (reusable)
# ══════════════════════════════════════════════════════════════
def train_single_model(X_train, y_train, X_val, y_val,
                       n_x, n_epochs, batch_size, subset_ratio,
                       param_sharing, use_forget_gate,
                       device_mode=DEVICE_MODE, label="Model"):
    """
    Train a single AQRNN model. Used by Variant B and per-cluster by Variant C.
    Returns: dict with params_q, wvec, forget_vec
    """
    np.random.seed(SEED)

    # Subset
    if subset_ratio < 1.0:
        n_sub = max(32, int(len(X_train) * subset_ratio))
        X_train = X_train[:n_sub]
        y_train = y_train[:n_sub]
        print(f"  [{label}] Subsampled train → {n_sub} samples")

    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=param_sharing)
    qnode, param_shape, dev = cell.build_qnode(device_mode=device_mode)
    print(f"  [{label}] Device: {dev.name} | Params shape: {param_shape}")

    params_q = np.random.uniform(-0.01, 0.01, size=param_shape).astype(np.float32)
    W1 = (np.sqrt(2.0 / N_H) * np.random.randn(HIDDEN_DIM, N_H)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = (np.sqrt(2.0 / HIDDEN_DIM) * np.random.randn(1, HIDDEN_DIM)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    wvec = pack_classical_weights(W1, b1, W2, b2)

    forget_vec = 0.95 * np.ones(SEQ_LEN) if use_forget_gate else None

    q_state = AdamState(params_q, lr=LR_Q)
    c_state = AdamState(wvec, lr=LR_C)
    loss_fn = make_loss_fn(qnode, SEQ_LEN, n_x, N_H, HIDDEN_DIM,
                           l2_q=L2_Q, l2_c=L2_C, forget_gate=forget_vec)
    grad_q_fn = qml.grad(loss_fn, argnum=0)
    grad_c_fn = qml.grad(loss_fn, argnum=1)

    best_rmse = 1e9
    best_params = None
    patience = 0
    n_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(1, n_epochs + 1):
        perm = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        pbar = tqdm(range(n_batches), desc=f"  [{label}] Epoch {epoch}/{n_epochs}",
                    leave=True, colour='green')
        for b in pbar:
            bi = b * batch_size
            bj = min((b + 1) * batch_size, len(X_train))
            idx = perm[bi:bj]
            bX, by = X_train[idx], y_train[idx]

            g_q = grad_q_fn(params_q, wvec, bX, by)
            g_c = grad_c_fn(params_q, wvec, bX, by)
            if GRAD_CLIP > 0:
                g_q = clip_grad_norm(g_q, max_norm=GRAD_CLIP)
                g_c = clip_grad_norm(g_c, max_norm=GRAD_CLIP)
            params_q, q_state = adam_update(params_q, g_q, q_state)
            wvec, c_state = adam_update(wvec, g_c, c_state)

            lv = loss_fn(params_q, wvec, bX, by).item()
            epoch_loss += lv
            pbar.set_postfix({"Loss": f"{lv:.4f}"})

        # Validation
        if len(X_val) > 0:
            val_preds = predict_with_params(qnode, params_q, wvec, forget_vec, X_val, label=f"{label} Val")
            val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        else:
            val_rmse = epoch_loss / max(n_batches, 1)

        print(f"  [{label}] Epoch {epoch}: Val RMSE = {val_rmse:.4f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = {"params_q": params_q.copy(), "wvec": wvec.copy(),
                           "forget_vec": forget_vec}
            patience = 0
        else:
            patience += 1
        if patience >= 3:
            print(f"  [{label}] Early stopping at epoch {epoch}")
            break

    if best_params is None:
        best_params = {"params_q": params_q, "wvec": wvec, "forget_vec": forget_vec}
    return best_params, qnode


def predict_with_params(qnode, params_q, wvec, forget_vec, X, chunk_size=256, label="Inference"):
    """Run inference in chunks."""
    all_preds = []
    n_chunks = int(np.ceil(len(X) / chunk_size))
    for i in tqdm(range(0, len(X), chunk_size), total=n_chunks,
                  desc=f"    {label}", leave=False, colour='cyan'):
        chunk = X[i:i+chunk_size]
        readouts_t = qnode(chunk, params_q, forget_gate=forget_vec)
        readouts = _readouts_to_array(readouts_t, N_H, len(chunk))
        preds = mlp_forward(readouts, wvec, N_H, HIDDEN_DIM)
        all_preds.append(np.array(preds))
    return np.concatenate(all_preds)


# ══════════════════════════════════════════════════════════════
# 4. VARIANT A — FCL WITHOUT MOE (Single Global Model + FedAvg)
# ══════════════════════════════════════════════════════════════
def run_variant_a(data, n_x, n_epochs=N_EPOCHS, param_sharing=True,
                  use_forget_gate=True, subset_ratio=1.0):
    """
    Federated Averaging with a SINGLE global model — no expert routing.
    All clients train the same model, server averages all updates.
    This demonstrates "prediction pollution" when heterogeneous clusters
    share a single model without specialization.
    """
    print("\n" + "=" * 70)
    print("VARIANT A: FCL WITHOUT MoE (Single Global Model + FedAvg)")
    print("=" * 70)

    # --- Initialize Global Model ---
    np.random.seed(SEED)
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=param_sharing)
    qnode, param_shape, dev = cell.build_qnode(device_mode=DEVICE_MODE)
    print(f"  Device: {dev.name} | Qubits: {cell.total_qubits}")

    global_params_q = np.random.uniform(-0.01, 0.01, size=param_shape).astype(np.float32)
    W1 = (np.sqrt(2.0 / N_H) * np.random.randn(HIDDEN_DIM, N_H)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = (np.sqrt(2.0 / HIDDEN_DIM) * np.random.randn(1, HIDDEN_DIM)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    global_wvec = pack_classical_weights(W1, b1, W2, b2)
    forget_vec = 0.95 * np.ones(SEQ_LEN) if use_forget_gate else None

    # --- Prepare "Clients" from federated_data CSVs ---
    client_files = sorted(glob.glob(os.path.join(DATA_DIR, "client_*.csv")))
    if not client_files:
        print("  ⚠ No federated client CSVs found. Falling back to synthetic split.")
        client_files = None

    # Build per-client datasets using the SAME scaler
    scaler_x = data["scaler_x"]
    scaler_y = data["scaler_y"]
    pca_obj  = data["pca"]

    req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    clients = []
    if client_files:
        for fpath in tqdm(client_files, desc="  Loading Client CSVs", colour='yellow'):
            cid = os.path.basename(fpath).replace("client_", "").replace(".csv", "")
            df_c = pd.read_csv(fpath, index_col=0)
            if "TrueCPUUtil" not in df_c.columns:
                continue
            y_raw = df_c["TrueCPUUtil"].values.reshape(-1, 1)
            X_raw = df_c[req_cols].values
            X_scaled = scaler_x.transform(X_raw)
            if pca_obj is not None:
                X_scaled = pca_obj.transform(X_scaled)
            y_scaled = scaler_y.transform(y_raw)

            # Create sequences
            Xs, ys = [], []
            for i in range(len(X_scaled) - SEQ_LEN):
                Xs.append(X_scaled[i: i + SEQ_LEN])
                ys.append(y_scaled[i + SEQ_LEN])
            if len(Xs) < 10:
                continue
            clients.append({
                "id": cid,
                "X": np.array(Xs),
                "y": np.array(ys),
            })
        print(f"  Loaded {len(clients)} federated clients")
    else:
        # Synthetic split: divide train set into ~5 pseudo-clients
        n_per = len(data["X_train"]) // 5
        for i in range(5):
            clients.append({
                "id": f"synth_{i}",
                "X": data["X_train"][i*n_per:(i+1)*n_per],
                "y": data["y_train"][i*n_per:(i+1)*n_per],
            })
        print(f"  Created {len(clients)} synthetic clients")

    # --- FedAvg Rounds ---
    t_start = time.perf_counter()

    for r in tqdm(range(FL_ROUNDS), desc="  FedAvg Rounds", colour='magenta'):
        print(f"\n  ┌── ROUND {r+1}/{FL_ROUNDS} ──┐")
        all_updated_q = []
        all_updated_w = []

        for ci, client in enumerate(tqdm(clients, desc=f"    R{r+1} Clients", leave=False, colour='blue')):
            # Each client gets a COPY of the global model
            local_q = copy.deepcopy(global_params_q)
            local_w = copy.deepcopy(global_wvec)

            q_state = AdamState(local_q, lr=LR_Q)
            c_state = AdamState(local_w, lr=LR_C)
            loss_fn = make_loss_fn(qnode, SEQ_LEN, n_x, N_H, HIDDEN_DIM,
                                   l2_q=L2_Q, l2_c=L2_C, forget_gate=forget_vec)
            grad_q_fn = qml.grad(loss_fn, argnum=0)
            grad_c_fn = qml.grad(loss_fn, argnum=1)

            # Subset for speed
            n_use = min(FL_CLIENT_MAX_SAMPLES, len(client["X"]))
            if subset_ratio < 1.0:
                n_use = max(32, int(n_use * subset_ratio))
            idx = np.random.choice(len(client["X"]), n_use, replace=False)
            cX, cy = client["X"][idx], client["y"][idx]

            n_batches = int(np.ceil(len(cX) / FL_CLIENT_BATCH))
            perm = np.random.permutation(len(cX))

            for ep in range(n_epochs):
                for b in tqdm(range(n_batches), desc=f"      Client {client['id']} Ep{ep+1}",
                              leave=False, colour='green'):
                    bi = b * FL_CLIENT_BATCH
                    bj = min((b+1) * FL_CLIENT_BATCH, len(cX))
                    bidx = perm[bi:bj]
                    bX, by = cX[bidx], cy[bidx]

                    g_q = grad_q_fn(local_q, local_w, bX, by)
                    g_c = grad_c_fn(local_q, local_w, bX, by)
                    if GRAD_CLIP > 0:
                        g_q = clip_grad_norm(g_q, max_norm=GRAD_CLIP)
                        g_c = clip_grad_norm(g_c, max_norm=GRAD_CLIP)
                    local_q, q_state = adam_update(local_q, g_q, q_state)
                    local_w, c_state = adam_update(local_w, g_c, c_state)

            all_updated_q.append(local_q)
            all_updated_w.append(local_w)

        # --- FedAvg: Average all client params ---
        global_params_q = np.mean(all_updated_q, axis=0)
        global_wvec = np.mean(all_updated_w, axis=0)

        # === POLLUTION DIAGNOSIS PRINTS ===
        # Show how the single global model performs per-cluster AFTER averaging
        print(f"  │ Round {r+1} FedAvg Complete — Per-Cluster Performance (POLLUTION CHECK):")
        for k in tqdm(range(N_CLUSTERS), desc=f"    R{r+1} Cluster Eval", leave=False, colour='yellow'):
            mask_val = (data["c_val"] == k)
            if not np.any(mask_val):
                continue
            Xv = data["X_val"][mask_val]
            yv = data["y_val"][mask_val].ravel()
            preds_k = predict_with_params(qnode, global_params_q, global_wvec,
                                          forget_vec, Xv, label=f"Cluster {k} Val")
            rmse_k = float(np.sqrt(mean_squared_error(yv, preds_k)))
            mae_k = float(mean_absolute_error(yv, preds_k))

            # Original scale for SMAPE
            yv_orig = scaler_y.inverse_transform(yv.reshape(-1,1)).ravel()
            pk_orig = scaler_y.inverse_transform(preds_k.reshape(-1,1)).ravel()
            smape_k = smape(yv_orig, pk_orig)

            print(f"  │   Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
                  f"SMAPE={smape_k:.1f}%  (n={int(mask_val.sum())})")

        # Show overall
        all_val_preds = predict_with_params(qnode, global_params_q, global_wvec,
                                            forget_vec, data["X_val"], label="Overall Val")
        overall_rmse = float(np.sqrt(mean_squared_error(
            data["y_val"].ravel(), all_val_preds)))
        print(f"  │   OVERALL Val RMSE = {overall_rmse:.4f}")
        print(f"  │   ⚠ NOTE: If cluster RMSEs vary widely, the single model is")
        print(f"  │          POLLUTED by heterogeneous cluster data!")
        print(f"  └──────────────────────────┘")

    train_time = time.perf_counter() - t_start

    # --- Final Test Evaluation ---
    t_infer = time.perf_counter()
    test_preds = predict_with_params(qnode, global_params_q, global_wvec,
                                     forget_vec, data["X_test"], label="Variant A Test")
    infer_time = time.perf_counter() - t_infer

    metrics = compute_metrics(data["y_test"], test_preds, scaler_y,
                              train_time, infer_time)

    # Per-cluster pollution report
    print("\n  ╔═══ FINAL POLLUTION REPORT (Test Set) ═══╗")
    for k in tqdm(range(N_CLUSTERS), desc="  Test Clusters", leave=False, colour='yellow'):
        mask = (data["c_test"] == k)
        if not np.any(mask):
            continue
        yt = data["y_test"].ravel()[mask]
        yp = test_preds[mask]
        yt_orig = scaler_y.inverse_transform(yt.reshape(-1, 1)).ravel()
        yp_orig = scaler_y.inverse_transform(yp.reshape(-1, 1)).ravel()
        rmse_k = float(np.sqrt(mean_squared_error(yt_orig, yp_orig)))
        mae_k = float(mean_absolute_error(yt_orig, yp_orig))
        smape_k = smape(yt_orig, yp_orig)
        print(f"  ║ Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
              f"SMAPE={smape_k:.1f}%  (n={int(mask.sum())})")
    print(f"  ║ OVERALL:   RMSE={metrics['rmse_orig']:.4f}  "
          f"MAE={metrics['mae_orig']:.4f}  SMAPE={metrics['smape_orig']:.1f}%")
    print(f"  ╚══════════════════════════════════════════╝")

    metrics["per_cluster"] = compute_per_cluster_metrics(
        data["y_test"], test_preds, data["c_test"], scaler_y, N_CLUSTERS)

    return metrics


# ══════════════════════════════════════════════════════════════
# 5. VARIANT B — LOCAL AQRNN (Centralized, Non-Federated)
# ══════════════════════════════════════════════════════════════
def run_variant_b(data, n_x, n_epochs=N_EPOCHS, param_sharing=True,
                  use_forget_gate=True, subset_ratio=1.0):
    """
    Train a single AQRNN on the full training set (all clusters merged).
    No clustering, no routing — baseline centralized model.
    """
    print("\n" + "=" * 70)
    print("VARIANT B: LOCAL AQRNN (Centralized, Non-Federated)")
    print("=" * 70)

    scaler_y = data["scaler_y"]

    t_start = time.perf_counter()
    best_params, qnode = train_single_model(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        n_x, n_epochs, BATCH_SIZE, subset_ratio,
        param_sharing, use_forget_gate,
        label="Centralized"
    )
    train_time = time.perf_counter() - t_start

    # Inference
    t_infer = time.perf_counter()
    test_preds = predict_with_params(qnode, best_params["params_q"],
                                     best_params["wvec"],
                                     best_params["forget_vec"],
                                     data["X_test"], label="Variant B Test")
    infer_time = time.perf_counter() - t_infer

    metrics = compute_metrics(data["y_test"], test_preds, scaler_y,
                              train_time, infer_time)

    print(f"\n  Variant B Final: RMSE={metrics['rmse_orig']:.4f}  "
          f"MAE={metrics['mae_orig']:.4f}  SMAPE={metrics['smape_orig']:.1f}%")
    return metrics


# ══════════════════════════════════════════════════════════════
# 6. VARIANT C — FULL MoE (KMeans Router + Per-Cluster Experts)
# ══════════════════════════════════════════════════════════════
def run_variant_c(data, n_x, n_epochs=N_EPOCHS, param_sharing=True,
                  use_forget_gate=True, subset_ratio=1.0):
    """
    Full Mixture-of-Experts: KMeans routing + per-cluster AQRNN experts.
    Mirrors train_aqrnn_moe_robust() from aqrnn_cluster.py.
    """
    print("\n" + "=" * 70)
    print("VARIANT C: FULL MoE (KMeans Router + Per-Cluster Experts)")
    print("=" * 70)

    scaler_y = data["scaler_y"]
    experts = {}

    t_start = time.perf_counter()

    for k in tqdm(range(N_CLUSTERS), desc="  Training Experts", colour='magenta'):
        mask_train = (data["c_train"] == k)
        mask_val   = (data["c_val"] == k)

        X_tr_k = data["X_train"][mask_train]
        y_tr_k = data["y_train"][mask_train]
        X_val_k = data["X_val"][mask_val]
        y_val_k = data["y_val"][mask_val]

        if len(X_tr_k) < 32:
            print(f"  Cluster {k}: Too few samples ({len(X_tr_k)}), skipping.")
            continue

        print(f"\n  === Expert for Cluster {k} ({len(X_tr_k)} train samples) ===")
        best_params, qnode = train_single_model(
            X_tr_k, y_tr_k, X_val_k, y_val_k,
            n_x, n_epochs, BATCH_SIZE, subset_ratio,
            param_sharing, use_forget_gate,
            label=f"Expert-{k}"
        )
        experts[k] = best_params

    train_time = time.perf_counter() - t_start

    # --- MoE Inference: Route by cluster ---
    print("\n  === MoE Inference (Routed by KMeans Cluster) ===")

    # Need a qnode for inference
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=param_sharing)
    qnode_infer, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    t_infer = time.perf_counter()
    all_preds = np.zeros(len(data["X_test"]))
    all_mask  = np.zeros(len(data["X_test"]), dtype=bool)

    for k in tqdm(range(N_CLUSTERS), desc="  MoE Test Routing", colour='blue'):
        mask = (data["c_test"] == k)
        if not np.any(mask):
            continue
        if k not in experts:
            fallback = min(experts.keys()) if experts else None
            if fallback is None:
                continue
            print(f"  Cluster {k}: No expert, falling back to Expert {fallback}")
            ep = experts[fallback]
        else:
            ep = experts[k]

        X_k = data["X_test"][mask]
        preds_k = predict_with_params(qnode_infer, ep["params_q"], ep["wvec"],
                                       ep["forget_vec"], X_k, label=f"Cluster {k} Test")

        all_preds[mask] = preds_k
        all_mask[mask] = True

        # Per-cluster report
        yt_k = data["y_test"].ravel()[mask]
        yt_orig = scaler_y.inverse_transform(yt_k.reshape(-1,1)).ravel()
        yp_orig = scaler_y.inverse_transform(preds_k.reshape(-1,1)).ravel()
        rmse_k = float(np.sqrt(mean_squared_error(yt_orig, yp_orig)))
        mae_k = float(mean_absolute_error(yt_orig, yp_orig))
        smape_k = smape(yt_orig, yp_orig)
        print(f"  Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
              f"SMAPE={smape_k:.1f}%  (n={int(mask.sum())})")

    infer_time = time.perf_counter() - t_infer

    # Overall metrics (only on samples with experts)
    y_test_used = data["y_test"].ravel()[all_mask]
    preds_used  = all_preds[all_mask]

    metrics = compute_metrics(y_test_used, preds_used, scaler_y,
                              train_time, infer_time)
    metrics["per_cluster"] = compute_per_cluster_metrics(
        data["y_test"], all_preds, data["c_test"], scaler_y, N_CLUSTERS)

    print(f"\n  Variant C Final: RMSE={metrics['rmse_orig']:.4f}  "
          f"MAE={metrics['mae_orig']:.4f}  SMAPE={metrics['smape_orig']:.1f}%")
    return metrics


# ══════════════════════════════════════════════════════════════
# 7. ABLATION GRID
# ══════════════════════════════════════════════════════════════
def run_ablation_grid(data, n_x, n_epochs=N_EPOCHS):
    """
    Run ablation over: moe_routing × param_sharing × forget_gate × subset_ratio.
    """
    print("\n" + "▓" * 70)
    print("  ABLATION GRID")
    print("▓" * 70)

    grid = {
        "moe_routing":   [True, False],
        "param_sharing": [True, False],
        "forget_gate":   [True, False],
        "subset_ratio":  [0.25, 0.5, 1.0],
    }

    combos = list(product(
        grid["moe_routing"],
        grid["param_sharing"],
        grid["forget_gate"],
        grid["subset_ratio"],
    ))

    print(f"  Total combinations: {len(combos)}")
    results = []

    grid_pbar = tqdm(combos, desc="  Ablation Grid", colour='red')
    for i, (moe, ps, fg, sr) in enumerate(grid_pbar, 1):
        tag = (f"moe={moe}_ps={ps}_fg={fg}_sr={sr}")
        grid_pbar.set_postfix_str(tag)
        print(f"\n{'─'*60}")
        print(f"  [{i}/{len(combos)}] {tag}")
        print(f"{'─'*60}")

        try:
            if moe:
                m = run_variant_c(data, n_x, n_epochs=n_epochs,
                                  param_sharing=ps, use_forget_gate=fg,
                                  subset_ratio=sr)
            else:
                m = run_variant_b(data, n_x, n_epochs=n_epochs,
                                  param_sharing=ps, use_forget_gate=fg,
                                  subset_ratio=sr)
            m["config"] = {
                "moe_routing": moe,
                "param_sharing": ps,
                "forget_gate": fg,
                "subset_ratio": sr,
            }
            results.append(m)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "config": {"moe_routing": moe, "param_sharing": ps,
                           "forget_gate": fg, "subset_ratio": sr},
                "error": str(e)
            })

    return results


# ══════════════════════════════════════════════════════════════
# 8. PRETTY PRINTING
# ══════════════════════════════════════════════════════════════
def print_variant_table(results):
    """Print comparison of variants A/B/C."""
    print("\n")
    print("╔" + "═" * 95 + "╗")
    print("║" + "  ABLATION STUDY — PRIMARY VARIANT COMPARISON".center(95) + "║")
    print("╚" + "═" * 95 + "╝")

    header = (f"{'Variant':<30} │ {'RMSE':>8} │ {'MAE':>8} │ {'SMAPE':>8} │ "
              f"{'Train(s)':>9} │ {'Infer(s)':>9} │ {'N':>6}")
    print(header)
    print("─" * 95)

    for name, m in results.items():
        if m is None:
            print(f"{name:<30} │ {'FAILED':>8}")
            continue
        row = (f"{name:<30} │ {m['rmse_orig']:>8.4f} │ {m['mae_orig']:>8.4f} │ "
               f"{m['smape_orig']:>7.1f}% │ {m['train_time_s']:>9.1f} │ "
               f"{m['infer_time_s']:>9.3f} │ {m['n_samples']:>6}")
        print(row)

    print("─" * 95)

    # Per-cluster for MoE variants
    for name, m in results.items():
        if m and "per_cluster" in m:
            print(f"\n  Per-Cluster Breakdown for [{name}]:")
            print(f"  {'Cluster':<12} │ {'RMSE':>8} │ {'MAE':>8} │ {'SMAPE':>8} │ {'N':>6}")
            print(f"  {'─'*55}")
            for ck, cv in m["per_cluster"].items():
                if cv.get("n_samples", 0) == 0:
                    continue
                print(f"  {ck:<12} │ {cv['rmse_orig']:>8.4f} │ {cv['mae_orig']:>8.4f} │ "
                      f"{cv['smape_orig']:>7.1f}% │ {cv['n_samples']:>6}")

    print()


def print_grid_table(grid_results):
    """Print ablation grid results."""
    if not grid_results:
        return

    print("\n")
    print("╔" + "═" * 110 + "╗")
    print("║" + "  ABLATION GRID RESULTS".center(110) + "║")
    print("╚" + "═" * 110 + "╝")

    header = (f"{'MoE':>5} │ {'PS':>5} │ {'FG':>5} │ {'SR':>5} │ "
              f"{'RMSE':>8} │ {'MAE':>8} │ {'SMAPE':>8} │ "
              f"{'Train(s)':>9} │ {'N':>6}")
    print(header)
    print("─" * 110)

    for m in grid_results:
        cfg = m.get("config", {})
        if "error" in m:
            print(f"  {str(cfg.get('moe_routing','')):>5} │ {str(cfg.get('param_sharing','')):>5} │ "
                  f"{str(cfg.get('forget_gate','')):>5} │ {cfg.get('subset_ratio',''):>5} │ ERROR: {m['error']}")
            continue
        row = (f"{str(cfg.get('moe_routing','')):>5} │ "
               f"{str(cfg.get('param_sharing','')):>5} │ "
               f"{str(cfg.get('forget_gate','')):>5} │ "
               f"{cfg.get('subset_ratio',''):>5} │ "
               f"{m['rmse_orig']:>8.4f} │ {m['mae_orig']:>8.4f} │ "
               f"{m['smape_orig']:>7.1f}% │ "
               f"{m['train_time_s']:>9.1f} │ {m['n_samples']:>6}")
        print(row)

    print("─" * 110)
    print()


# ══════════════════════════════════════════════════════════════
# 9. PLOT GENERATION
# ══════════════════════════════════════════════════════════════
def generate_plots(variant_results, grid_results=None):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Plot 1: Variant Comparison ---
    valid = {k: v for k, v in variant_results.items() if v is not None}
    if valid:
        names = list(valid.keys())
        colors = ["#E74C3C", "#3498DB", "#2ECC71"][:len(names)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # RMSE
        vals = [valid[n]["rmse_orig"] for n in names]
        axes[0].bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[0].set_title("RMSE (Original Scale)", fontsize=13, fontweight="bold")
        for i, v in enumerate(vals):
            axes[0].text(i, v + max(vals)*0.02, f"{v:.4f}", ha="center", fontsize=10)

        # MAE
        vals = [valid[n]["mae_orig"] for n in names]
        axes[1].bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[1].set_title("MAE (Original Scale)", fontsize=13, fontweight="bold")
        for i, v in enumerate(vals):
            axes[1].text(i, v + max(vals)*0.02, f"{v:.4f}", ha="center", fontsize=10)

        # SMAPE
        vals = [valid[n]["smape_orig"] for n in names]
        axes[2].bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
        axes[2].set_title("SMAPE (%)", fontsize=13, fontweight="bold")
        for i, v in enumerate(vals):
            axes[2].text(i, v + max(vals)*0.02, f"{v:.1f}%", ha="center", fontsize=10)

        plt.suptitle("Ablation Study: Primary Variant Comparison",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        path = os.path.join(PLOT_DIR, "variant_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # --- Plot 2: Per-Cluster Breakdown (MoE variants) ---
    for vname, vm in variant_results.items():
        if vm and "per_cluster" in vm:
            pc = vm["per_cluster"]
            cluster_names = [k for k, v in pc.items() if v.get("n_samples", 0) > 0]
            if not cluster_names:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            x = np.arange(len(cluster_names))

            for ax, metric, title in zip(
                axes,
                ["rmse_orig", "mae_orig", "smape_orig"],
                ["RMSE", "MAE", "SMAPE (%)"]
            ):
                vals = [pc[cn][metric] for cn in cluster_names]
                bars = ax.bar(x, vals, color=["#E74C3C", "#3498DB", "#2ECC71"][:len(x)],
                              edgecolor="black", linewidth=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels(cluster_names, rotation=15)
                ax.set_title(f"{title}", fontsize=12, fontweight="bold")
                for j, v in enumerate(vals):
                    ax.text(j, v + max(vals)*0.02,
                            f"{v:.1f}%" if "smape" in metric else f"{v:.4f}",
                            ha="center", fontsize=9)

            plt.suptitle(f"Per-Cluster Metrics — {vname}",
                         fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            safe_name = vname.replace(" ", "_").replace("+", "").lower()
            path = os.path.join(PLOT_DIR, f"per_cluster_{safe_name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")

    # --- Plot 3: Ablation Grid Heatmap (if available) ---
    if grid_results:
        valid_grid = [m for m in grid_results if "error" not in m]
        if valid_grid:
            fig, ax = plt.subplots(figsize=(14, max(6, len(valid_grid)*0.35)))
            labels = []
            rmses = []
            for m in valid_grid:
                cfg = m["config"]
                lbl = (f"MoE={cfg['moe_routing']} PS={cfg['param_sharing']} "
                       f"FG={cfg['forget_gate']} SR={cfg['subset_ratio']}")
                labels.append(lbl)
                rmses.append(m["rmse_orig"])

            y_pos = np.arange(len(labels))
            colors_grid = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rmses)))
            sorted_idx = np.argsort(rmses)
            ax.barh([labels[i] for i in sorted_idx],
                    [rmses[i] for i in sorted_idx],
                    color=[colors_grid[i] for i in sorted_idx],
                    edgecolor="black", linewidth=0.5)
            ax.set_xlabel("RMSE (Original Scale)")
            ax.set_title("Ablation Grid — RMSE Ranking (lower is better)",
                         fontsize=13, fontweight="bold")

            for i, idx in enumerate(sorted_idx):
                ax.text(rmses[idx] + max(rmses)*0.01, i,
                        f"{rmses[idx]:.4f}", va="center", fontsize=8)

            plt.tight_layout()
            path = os.path.join(PLOT_DIR, "ablation_grid_ranking.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="AQRNN Ablation Study")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (subset=0.1, epochs=1)")
    parser.add_argument("--variants-only", action="store_true",
                        help="Run only A/B/C variants, skip ablation grid")
    parser.add_argument("--grid-only", action="store_true",
                        help="Run only the ablation grid, skip A/B/C")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"Epochs per model (default: {N_EPOCHS})")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Subset ratio for main variants (default: 1.0)")
    args = parser.parse_args()

    n_epochs = args.epochs
    subset_ratio = args.subset

    if args.quick:
        n_epochs = 1
        subset_ratio = 0.1
        print("⚡ QUICK MODE: epochs=1, subset=0.1")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          AQRNN ABLATION STUDY — GPU-READY SCRIPT           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Epochs: {n_epochs}  |  Subset: {subset_ratio}  |  Device: {DEVICE_MODE:<15}  ║")
    print(f"║  SEQ_LEN: {SEQ_LEN}  |  N_H: {N_H}  |  HIDDEN_DIM: {HIDDEN_DIM:<15}    ║")
    print(f"║  PCA: {N_COMPONENTS}  |  Clusters: {N_CLUSTERS}  |  Batch: {BATCH_SIZE:<16}    ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Load data once
    data, n_x = load_shared_data()

    variant_results = {}
    grid_results = []

    # ── Primary Variants ──
    if not args.grid_only:
        print("\n" + "█" * 70)
        print("  RUNNING PRIMARY VARIANTS (A / B / C)")
        print("█" * 70)

        # Variant A: FCL without MoE
        try:
            variant_results["A: FCL no-MoE"] = run_variant_a(
                data, n_x, n_epochs=n_epochs, subset_ratio=subset_ratio)
        except Exception as e:
            print(f"\n  ❌ Variant A FAILED: {e}")
            variant_results["A: FCL no-MoE"] = None

        # Variant B: Local AQRNN
        try:
            variant_results["B: Local AQRNN"] = run_variant_b(
                data, n_x, n_epochs=n_epochs, subset_ratio=subset_ratio)
        except Exception as e:
            print(f"\n  ❌ Variant B FAILED: {e}")
            variant_results["B: Local AQRNN"] = None

        # Variant C: Full MoE
        try:
            variant_results["C: Full MoE"] = run_variant_c(
                data, n_x, n_epochs=n_epochs, subset_ratio=subset_ratio)
        except Exception as e:
            print(f"\n  ❌ Variant C FAILED: {e}")
            variant_results["C: Full MoE"] = None

        print_variant_table(variant_results)

    # ── Ablation Grid ──
    if not args.variants_only:
        try:
            grid_results = run_ablation_grid(data, n_x, n_epochs=n_epochs)
            print_grid_table(grid_results)
        except Exception as e:
            print(f"\n  ❌ Ablation Grid FAILED: {e}")

    # ── Save Results ──
    all_results = {
        "config": {
            "n_epochs": n_epochs,
            "subset_ratio": subset_ratio,
            "seq_len": SEQ_LEN,
            "n_h": N_H,
            "hidden_dim": HIDDEN_DIM,
            "n_clusters": N_CLUSTERS,
            "n_components": N_COMPONENTS,
            "device_mode": DEVICE_MODE,
            "fl_rounds": FL_ROUNDS,
        },
        "variants": variant_results,
        "grid": grid_results,
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Results saved to {RESULTS_JSON}")

    # ── Plots ──
    try:
        generate_plots(variant_results, grid_results)
    except Exception as e:
        print(f"  Plot error: {e}")

    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()
