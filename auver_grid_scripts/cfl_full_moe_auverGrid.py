#!/usr/bin/env python
"""
===========================================================================
CFL Full MoE — AuverGrid (GWA-T-4) Dataset
===========================================================================
Identical pipeline to cfl_full_moe.py, but configured for AuverGrid.

  - Global Experts (N_CLUSTERS) initialized randomly or warm-started
  - Federated Clients loaded from federated_data_auverGrid/ CSVs
  - FL Round Loop:
      1. Audition: clients evaluate ALL experts, pick best
      2. Load Balancing: capacity-constrained assignment (regret-based)
      3. Shadow Distillation: client trains local copy of chosen expert
      4. Per-Expert Aggregation: server averages updates per expert

IMPORTANT: Run process_auverGrid.py FIRST to generate the data files.

Usage:
  python cfl_full_moe_auverGrid.py                  # Full run
  python cfl_full_moe_auverGrid.py --quick          # Smoke test
  python cfl_full_moe_auverGrid.py --rounds 5       # Custom FL rounds
  python cfl_full_moe_auverGrid.py --warmstart      # Warm-start
===========================================================================
"""

import os, sys, time, json, copy, glob, argparse, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

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

# Import shared AQRNN components (from project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from aqrnn import (
    AQRNNCell, AdamState, adam_update, clip_grad_norm,
    make_loss_fn, pack_classical_weights, unpack_classical_weights,
    mlp_forward, _readouts_to_array, angle_encode,
)


# ══════════════════════════════════════════════════════════════
# CONFIGURATION — AuverGrid Paths (everything else identical)
# ══════════════════════════════════════════════════════════════
CSV_PATH       = os.path.join(PROJECT_ROOT, "auverGrid_hybrid_clean.csv")
DATA_DIR       = os.path.join(PROJECT_ROOT, "federated_data_auverGrid")
N_CLUSTERS     = 3
N_COMPONENTS   = 4          # PCA components
SEQ_LEN        = 2
N_H            = 4
HIDDEN_DIM     = 64
N_LAYERS       = 1
DEVICE_MODE    = "cuda"
SEED           = 42
BATCH_SIZE     = 32         # Per-client batch size
LR_Q           = 1e-3
LR_C           = 1e-3
GRAD_CLIP      = 1.0

# CFL-specific
FL_ROUNDS         = 3
CLIENT_EPOCHS     = 1
MAX_CLIENT_SAMPLES = 256
EXPERT_CAPACITY   = 5
DISTILL_ALPHA     = 0.3
PARAM_SHARING     = True
USE_FORGET_GATE   = True

# Warm-start / Output (AuverGrid-specific names)
MOE_PATH       = os.path.join(PROJECT_ROOT, "moe_experts_auverGrid.pkl")
RESULTS_JSON   = os.path.join(PROJECT_ROOT, "cfl_results_auverGrid.json")


# ══════════════════════════════════════════════════════════════
# 1. SHARED DATA PIPELINE (identical logic)
# ══════════════════════════════════════════════════════════════
def load_shared_data(n_components=N_COMPONENTS, n_clusters=N_CLUSTERS):
    """
    Replicates the exact data pipeline from aqrnn_cluster.py.
    Returns dict with train/val/test splits + scalers + cluster labels.
    """
    print("=" * 60)
    print("LOADING SHARED DATA PIPELINE (AuverGrid)")
    print("=" * 60)

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

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, init='k-means++')
    clusters = kmeans.fit_predict(X_final)

    unique, counts = np.unique(clusters, return_counts=True)
    print("  Cluster Distribution:", {int(u): int(c) for u, c in zip(unique, counts)})

    Xs, ys, cs = [], [], []
    for i in range(len(X_final) - SEQ_LEN):
        Xs.append(X_final[i : i + SEQ_LEN])
        ys.append(y_scaled[i + SEQ_LEN])
        cs.append(clusters[i + SEQ_LEN - 1])
    X_seq = np.array(Xs)
    y_seq = np.array(ys)
    c_seq = np.array(cs)

    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end = train_end + int(N * 0.15)

    split_time = df.index[train_end]

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
        "split_time": split_time,
    }

    n_x = X_seq.shape[2]
    print(f"  Train: {data['X_train'].shape[0]:,} | Val: {data['X_val'].shape[0]:,} | "
          f"Test: {data['X_test'].shape[0]:,}")
    print(f"  Split Time: {split_time}")
    print(f"  Sequence shape: (B, {SEQ_LEN}, {n_x})")
    return data, n_x


# ══════════════════════════════════════════════════════════════
# 2. METRICS (identical)
# ══════════════════════════════════════════════════════════════
def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def compute_metrics(y_true, y_pred, scaler_y, train_time=0.0, infer_time=0.0):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse_scaled = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae_scaled  = float(mean_absolute_error(y_true, y_pred))
    smape_scaled = smape(y_true, y_pred)

    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    rmse_orig = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    mae_orig  = float(mean_absolute_error(y_true_orig, y_pred_orig))
    smape_orig = smape(y_true_orig, y_pred_orig)

    return {
        "rmse_scaled": rmse_scaled, "mae_scaled": mae_scaled, "smape_scaled": smape_scaled,
        "rmse_orig": rmse_orig, "mae_orig": mae_orig, "smape_orig": smape_orig,
        "train_time_s": round(train_time, 2), "infer_time_s": round(infer_time, 2),
        "n_samples": len(y_true),
    }


def compute_per_cluster_metrics(y_true, y_pred, c_test, scaler_y, n_clusters):
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
# 3. INFERENCE HELPER
# ══════════════════════════════════════════════════════════════
def predict_with_params(qnode, params_q, wvec, forget_vec, X,
                        chunk_size=256, label="Inference"):
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
# 4. FEDERATED CLIENT (identical to cfl_full_moe.py)
# ══════════════════════════════════════════════════════════════
class FederatedClient:
    def __init__(self, client_id, X_seq, y_seq):
        self.client_id = client_id
        self.X_seq = X_seq
        self.y_seq = y_seq

    def evaluate_expert(self, expert_params, qnode, n_h, hidden_dim):
        if len(self.X_seq) == 0:
            return 1e9

        n_eval = min(256, len(self.X_seq))
        idx = np.random.choice(len(self.X_seq), n_eval, replace=False)
        X_batch = self.X_seq[idx]
        y_batch = self.y_seq[idx]

        params_q = expert_params["params_q"]
        wvec     = expert_params["wvec"]

        readouts_t = qnode(X_batch, params_q)

        out = pnp.array(readouts_t)
        if out.ndim == 1:
            readouts = pnp.reshape(out, (1, -1))
        elif out.shape[0] == n_h and out.ndim == 2:
            readouts = pnp.transpose(out)
        else:
            readouts = out

        W1, b1, W2, b2 = unpack_classical_weights(wvec, n_h, hidden_dim)
        W1 = pnp.array(W1); b1 = pnp.array(b1)
        W2 = pnp.array(W2); b2 = pnp.array(b2)
        readouts = pnp.array(readouts)

        z = pnp.dot(readouts, W1.T) + b1
        h = pnp.maximum(0, z)
        out = pnp.dot(h, W2.T) + b2
        preds = out[:, 0]

        y_flat = y_batch.ravel()
        mse = float(np.mean((np.array(preds) - y_flat) ** 2))
        return mse

    def train_epoch(self, best_expert_params, qnode, n_h, hidden_dim,
                    lr_q, lr_c, alpha=DISTILL_ALPHA, n_epochs=CLIENT_EPOCHS,
                    max_samples=MAX_CLIENT_SAMPLES, batch_size=BATCH_SIZE):
        params_q = copy.deepcopy(best_expert_params["params_q"])
        wvec     = copy.deepcopy(best_expert_params["wvec"])

        teacher_params_q = best_expert_params["params_q"]
        teacher_wvec     = best_expert_params["wvec"]

        tW1, tb1, tW2, tb2 = unpack_classical_weights(teacher_wvec, n_h, hidden_dim)
        tW1 = pnp.array(tW1); tb1 = pnp.array(tb1)
        tW2 = pnp.array(tW2); tb2 = pnp.array(tb2)

        q_state = AdamState(params_q, lr=lr_q)
        c_state = AdamState(wvec, lr=lr_c)

        def distillation_loss(p_q, w_v, batch_X, batch_y):
            readouts_student = qnode(batch_X, p_q)
            out_s = pnp.array(readouts_student)
            if out_s.ndim == 1:
                out_s = pnp.reshape(out_s, (1, -1))
            elif out_s.shape[0] == n_h and out_s.ndim == 2:
                out_s = pnp.transpose(out_s)

            SW1, Sb1, SW2, Sb2 = unpack_classical_weights(w_v, n_h, hidden_dim)
            sz = pnp.dot(out_s, pnp.array(SW1).T) + pnp.array(Sb1)
            sh = pnp.maximum(0, sz)
            student_pred = pnp.dot(sh, pnp.array(SW2).T) + pnp.array(Sb2)
            student_pred = student_pred[:, 0]

            readouts_teacher = qnode(batch_X, teacher_params_q)
            out_t = pnp.array(readouts_teacher)
            if out_t.ndim == 1:
                out_t = pnp.reshape(out_t, (1, -1))
            elif out_t.shape[0] == n_h and out_t.ndim == 2:
                out_t = pnp.transpose(out_t)

            tz = pnp.dot(out_t, tW1.T) + tb1
            th = pnp.maximum(0, tz)
            teacher_pred = pnp.dot(th, tW2.T) + tb2
            teacher_pred = teacher_pred[:, 0]

            loss_data = pnp.mean((student_pred - pnp.reshape(batch_y, (-1,))) ** 2)
            loss_distill = pnp.mean((student_pred - teacher_pred) ** 2)

            return (1 - alpha) * loss_data + alpha * loss_distill

        grad_fn = qml.grad(distillation_loss, argnums=[0, 1])

        if len(self.X_seq) == 0:
            return {"params_q": params_q, "wvec": wvec}, 0

        total_loss = 0
        total_batches = 0

        for ep in range(n_epochs):
            if len(self.X_seq) > max_samples:
                epoch_idx = np.random.choice(len(self.X_seq), max_samples, replace=False)
            else:
                epoch_idx = np.arange(len(self.X_seq))

            perm = np.random.permutation(epoch_idx)
            n_batches = int(np.ceil(len(perm) / batch_size))

            for b in tqdm(range(n_batches),
                          desc=f"      Client {self.client_id} Ep{ep+1}",
                          leave=False, colour='green'):
                bi = b * batch_size
                bj = min((b + 1) * batch_size, len(perm))
                batch_idx = perm[bi:bj]
                batch_X = self.X_seq[batch_idx]
                batch_y = self.y_seq[batch_idx]

                g_q, g_c = grad_fn(params_q, wvec, batch_X, batch_y)

                if GRAD_CLIP > 0:
                    g_q = clip_grad_norm(g_q, max_norm=GRAD_CLIP)
                    g_c = clip_grad_norm(g_c, max_norm=GRAD_CLIP)

                params_q, q_state = adam_update(params_q, g_q, q_state)
                wvec, c_state = adam_update(wvec, g_c, c_state)

                loss_val = float(distillation_loss(params_q, wvec, batch_X, batch_y))
                total_loss += loss_val
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        return {"params_q": params_q, "wvec": wvec}, avg_loss


# ══════════════════════════════════════════════════════════════
# 5. LOAD FEDERATED CLIENTS
# ══════════════════════════════════════════════════════════════
def load_federated_clients(data):
    client_files = sorted(glob.glob(os.path.join(DATA_DIR, "client_*.csv")))
    scaler_x = data["scaler_x"]
    scaler_y = data["scaler_y"]
    pca_obj  = data["pca"]
    split_time = data.get("split_time")

    req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    clients = []
    if client_files:
        print(f"  Found {len(client_files)} client CSVs")
        for fpath in tqdm(client_files, desc="  Loading Clients", colour='yellow'):
            cid = os.path.basename(fpath).replace("client_", "").replace(".csv", "")
            try:
                df_c = pd.read_csv(fpath, index_col=0)

                if split_time is not None:
                    if "datetime" in df_c.columns:
                         df_c["datetime"] = pd.to_datetime(df_c["datetime"])
                         df_c = df_c[df_c["datetime"] <= split_time]
                    elif "WindowStart" in df_c.columns:
                         df_c["datetime"] = pd.to_datetime(df_c["WindowStart"], unit='s')
                         df_c = df_c[df_c["datetime"] <= split_time]

                if "TrueCPUUtil" not in df_c.columns or len(df_c) == 0:
                    continue

                y_raw = df_c["TrueCPUUtil"].values.reshape(-1, 1)
                X_raw = df_c[req_cols].values
                X_scaled = scaler_x.transform(X_raw)
                if pca_obj is not None:
                    X_scaled = pca_obj.transform(X_scaled)
                y_scaled = scaler_y.transform(y_raw)

                Xs, ys = [], []
                for i in range(len(X_scaled) - SEQ_LEN):
                    Xs.append(X_scaled[i: i + SEQ_LEN])
                    ys.append(y_scaled[i + SEQ_LEN])

                if len(Xs) < 10:
                    continue

                clients.append(FederatedClient(
                    client_id=cid,
                    X_seq=np.array(Xs),
                    y_seq=np.array(ys),
                ))
            except Exception as e:
                print(f"    Error loading {cid}: {e}")
    else:
        print("  ⚠ No federated client CSVs found. Using synthetic split.")
        n_per = len(data["X_train"]) // 5
        for i in range(5):
            clients.append(FederatedClient(
                client_id=f"synth_{i}",
                X_seq=data["X_train"][i*n_per:(i+1)*n_per],
                y_seq=data["y_train"][i*n_per:(i+1)*n_per],
            ))

    print(f"  Loaded {len(clients)} federated clients")
    return clients


# ══════════════════════════════════════════════════════════════
# 6. INITIALIZE GLOBAL EXPERTS
# ══════════════════════════════════════════════════════════════
def init_experts(n_x, n_clusters, param_sharing, use_forget_gate, warmstart=False, moe_path=MOE_PATH):
    experts = {}

    if warmstart and os.path.exists(moe_path):
        print(f"  Warm-starting experts from {moe_path}")
        with open(moe_path, "rb") as f:
            loaded = pickle.load(f)
        for k in range(n_clusters):
            if k in loaded:
                experts[k] = {
                    "params_q": loaded[k]["params_q"].copy(),
                    "wvec":     loaded[k]["wvec"].copy(),
                }
            else:
                experts[k] = _random_expert(n_x, param_sharing, use_forget_gate)
        print(f"  Loaded {len(experts)} experts from checkpoint")
    else:
        if warmstart:
            print(f"  ⚠ {moe_path} not found, using random init")
        for k in range(n_clusters):
            experts[k] = _random_expert(n_x, param_sharing, use_forget_gate)
        print(f"  Initialized {n_clusters} random experts")

    return experts


def _random_expert(n_x, param_sharing, use_forget_gate):
    np.random.seed(None)

    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=param_sharing)
    _, param_shape, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    params_q = np.random.uniform(-0.01, 0.01, size=param_shape).astype(np.float32)
    W1 = (np.sqrt(2.0 / N_H) * np.random.randn(HIDDEN_DIM, N_H)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = (np.sqrt(2.0 / HIDDEN_DIM) * np.random.randn(1, HIDDEN_DIM)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    wvec = pack_classical_weights(W1, b1, W2, b2)

    return {"params_q": params_q, "wvec": wvec}


# ══════════════════════════════════════════════════════════════
# 7. CFL SIMULATION (identical logic)
# ══════════════════════════════════════════════════════════════
def run_cfl_simulation(data, n_x, experts, clients, qnode,
                       n_rounds=FL_ROUNDS, client_epochs=CLIENT_EPOCHS,
                       expert_capacity=EXPERT_CAPACITY, alpha=DISTILL_ALPHA):
    print("\n" + "=" * 70)
    print("  CLUSTERED FEDERATED LEARNING SIMULATION (AuverGrid)")
    print("=" * 70)
    print(f"  Rounds: {n_rounds} | Client Epochs: {client_epochs} | "
          f"Alpha: {alpha} | Capacity: {expert_capacity}")
    print(f"  Clients: {len(clients)} | Experts: {len(experts)}")

    scaler_y = data["scaler_y"]
    t_start = time.perf_counter()

    for r in tqdm(range(n_rounds), desc="  CFL Rounds", colour='magenta'):
        print(f"\n  ┌── ROUND {r+1}/{n_rounds} ──┐")

        updates = {k: [] for k in experts.keys()}

        # Step 1: AUDITION
        print(f"  │ Audition Phase...")
        client_losses = {}
        for c in tqdm(clients, desc=f"    R{r+1} Audition", leave=False, colour='blue'):
            losses_for_c = {}
            for k, expert_params in experts.items():
                loss = c.evaluate_expert(expert_params, qnode, N_H, HIDDEN_DIM)
                losses_for_c[k] = loss
            client_losses[c.client_id] = losses_for_c

        # Step 2: LOAD BALANCING
        client_best = {}
        client_regrets = {}

        for cid, losses in client_losses.items():
            best_k = min(losses, key=losses.get)
            best_loss = losses[best_k]
            client_best[cid] = (best_k, best_loss)
            client_regrets[cid] = {k: losses[k] - best_loss for k in losses}

        def get_flexibility(cid):
            regrets = client_regrets[cid]
            sorted_regrets = sorted(regrets.values())
            if len(sorted_regrets) >= 2:
                return sorted_regrets[1] - sorted_regrets[0]
            return 0

        sorted_clients = sorted(clients, key=lambda c: get_flexibility(c.client_id),
                                reverse=True)

        expert_capacity_remaining = {k: expert_capacity for k in experts.keys()}
        client_assignment = {}

        for c in sorted_clients:
            cid = c.client_id
            best_k, _ = client_best[cid]

            if expert_capacity_remaining[best_k] > 0:
                client_assignment[cid] = best_k
                expert_capacity_remaining[best_k] -= 1
            else:
                available = [k for k, cap in expert_capacity_remaining.items() if cap > 0]
                if available:
                    regrets = client_regrets[cid]
                    assigned_k = min(available, key=lambda k: regrets[k])
                    client_assignment[cid] = assigned_k
                    expert_capacity_remaining[assigned_k] -= 1
                else:
                    print(f"  │   ⚠ All experts full, forcing client {cid} → Expert {best_k}")
                    client_assignment[cid] = best_k

        print(f"  │ Expert Assignment (load-balanced):")
        for k in experts.keys():
            assigned = sum(1 for v in client_assignment.values() if v == k)
            print(f"  │   Expert {k}: {assigned} clients")

        avg_local_loss = np.mean([loss for _, loss in client_best.values()])
        print(f"  │ Avg Local Val Loss: {avg_local_loss:.4f} (Audition Best)")

        # Step 3: SHADOW DISTILLATION TRAINING
        print(f"  │ Distillation Training...")
        for c in tqdm(clients, desc=f"    R{r+1} Training", leave=False, colour='green'):
            assigned_k = client_assignment[c.client_id]
            chosen_params = experts[assigned_k]

            updated_params, train_loss = c.train_epoch(
                chosen_params, qnode, N_H, HIDDEN_DIM,
                lr_q=LR_Q, lr_c=LR_C, alpha=alpha,
                n_epochs=client_epochs, max_samples=MAX_CLIENT_SAMPLES,
                batch_size=BATCH_SIZE,
            )
            updates[assigned_k].append(updated_params)

        # Step 4: PER-EXPERT AGGREGATION
        print(f"  │ Aggregating...")
        for k, update_list in updates.items():
            if not update_list:
                print(f"  │   Expert {k}: No updates (0 clients)")
                continue
            new_pq = np.mean([u["params_q"] for u in update_list], axis=0)
            new_wv = np.mean([u["wvec"] for u in update_list], axis=0)
            experts[k] = {"params_q": new_pq, "wvec": new_wv}
            print(f"  │   Expert {k}: Aggregated {len(update_list)} updates")

        # Validation check
        forget_vec = 0.95 * np.ones(SEQ_LEN) if USE_FORGET_GATE else None
        print(f"  │ Validation (per-cluster):")
        for k in range(N_CLUSTERS):
            mask_val = (data["c_val"] == k)
            if not np.any(mask_val):
                continue
            if k not in experts:
                continue
            Xv = data["X_val"][mask_val]
            yv = data["y_val"][mask_val].ravel()
            ep = experts[k]
            preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                          forget_vec, Xv, label=f"Val Cluster {k}")
            rmse_k = float(np.sqrt(mean_squared_error(yv, preds_k)))
            mae_k = float(mean_absolute_error(yv, preds_k))

            yv_orig = scaler_y.inverse_transform(yv.reshape(-1, 1)).ravel()
            pk_orig = scaler_y.inverse_transform(preds_k.reshape(-1, 1)).ravel()
            smape_k = smape(yv_orig, pk_orig)

            print(f"  │   Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
                  f"SMAPE={smape_k:.1f}%  (n={int(mask_val.sum())})")

        print(f"  └──────────────────────────┘")

    train_time = time.perf_counter() - t_start
    return experts, train_time


# ══════════════════════════════════════════════════════════════
# 8. FINAL EVALUATION
# ══════════════════════════════════════════════════════════════
def evaluate_cfl(data, n_x, experts, qnode, train_time):
    print("\n" + "=" * 70)
    print("  CFL FINAL EVALUATION (AuverGrid — KMeans Routing)")
    print("=" * 70)

    scaler_y = data["scaler_y"]
    forget_vec = 0.95 * np.ones(SEQ_LEN) if USE_FORGET_GATE else None

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
            ep = experts[fallback]
        else:
            ep = experts[k]

        X_k = data["X_test"][mask]
        preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                       forget_vec, X_k, label=f"Cluster {k} Test")

        all_preds[mask] = preds_k
        all_mask[mask] = True

        yt_k = data["y_test"].ravel()[mask]
        yt_orig = scaler_y.inverse_transform(yt_k.reshape(-1, 1)).ravel()
        yp_orig = scaler_y.inverse_transform(preds_k.reshape(-1, 1)).ravel()
        rmse_k = float(np.sqrt(mean_squared_error(yt_orig, yp_orig)))
        mae_k = float(mean_absolute_error(yt_orig, yp_orig))
        smape_k = smape(yt_orig, yp_orig)
        print(f"  Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
              f"SMAPE={smape_k:.1f}%  (n={int(mask.sum())})")

    infer_time = time.perf_counter() - t_infer

    y_test_used = data["y_test"].ravel()[all_mask]
    preds_used  = all_preds[all_mask]

    metrics = compute_metrics(y_test_used, preds_used, scaler_y,
                              train_time, infer_time)
    metrics["per_cluster"] = compute_per_cluster_metrics(
        data["y_test"], all_preds, data["c_test"], scaler_y, N_CLUSTERS)

    print(f"\n  ╔═══ CFL FINAL RESULTS (AuverGrid — KMeans Routing) ═══╗")
    print(f"  ║ RMSE  (orig) = {metrics['rmse_orig']:.4f}")
    print(f"  ║ MAE   (orig) = {metrics['mae_orig']:.4f}")
    print(f"  ║ SMAPE (orig) = {metrics['smape_orig']:.1f}%")
    print(f"  ║ Train Time   = {metrics['train_time_s']:.1f}s")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    # Oracle Evaluation
    print("\n  Running Oracle Evaluation (Best Expert per Sample)...")
    t_oracle = time.perf_counter()
    best_preds = np.zeros(len(data["X_test"]))
    best_losses = np.full(len(data["X_test"]), np.inf)

    for k, ep in experts.items():
        preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                      forget_vec, data["X_test"], label=f"Oracle Exp {k}")
        se_k = (preds_k - data["y_test"].ravel()) ** 2
        better_mask = (se_k < best_losses)
        best_preds[better_mask] = preds_k[better_mask]
        best_losses[better_mask] = se_k[better_mask]

    oracle_infer_time = time.perf_counter() - t_oracle
    metrics_oracle = compute_metrics(data["y_test"], best_preds, scaler_y, train_time, oracle_infer_time)

    print(f"\n  ╔═══ CFL ORACLE RESULTS (Best Expert Selection) ═══╗")
    print(f"  ║ RMSE  (orig) = {metrics_oracle['rmse_orig']:.4f}")
    print(f"  ║ MAE   (orig) = {metrics_oracle['mae_orig']:.4f}")
    print(f"  ║ SMAPE (orig) = {metrics_oracle['smape_orig']:.1f}%")
    print(f"  ╚══════════════════════════════════════════════════╝")

    metrics["oracle"] = metrics_oracle

    # Per-cluster table
    print(f"\n  Per-Cluster Breakdown (KMeans Routing):")
    print(f"  {'Cluster':<12} │ {'RMSE':>8} │ {'MAE':>8} │ {'SMAPE':>8} │ {'N':>6}")
    print(f"  {'─'*55}")
    for ck, cv in metrics["per_cluster"].items():
        if cv.get("n_samples", 0) == 0:
            continue
        print(f"  {ck:<12} │ {cv['rmse_orig']:>8.4f} │ {cv['mae_orig']:>8.4f} │ "
              f"{cv['smape_orig']:>7.1f}% │ {cv['n_samples']:>6}")
    print()

    return metrics


# ══════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CFL Full MoE — AuverGrid (GWA-T-4)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 1 round, 1 epoch")
    parser.add_argument("--rounds", type=int, default=FL_ROUNDS,
                        help=f"FL rounds (default {FL_ROUNDS})")
    parser.add_argument("--epochs", type=int, default=CLIENT_EPOCHS,
                        help=f"Client epochs per round (default {CLIENT_EPOCHS})")
    parser.add_argument("--alpha", type=float, default=DISTILL_ALPHA,
                        help=f"Distillation alpha (default {DISTILL_ALPHA})")
    parser.add_argument("--capacity", type=int, default=EXPERT_CAPACITY,
                        help=f"Expert capacity (default {EXPERT_CAPACITY})")
    parser.add_argument("--warmstart", action="store_true",
                        help="Warm-start experts from moe_experts_auverGrid.pkl")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of client data to use (default 1.0)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate checkpoint")
    parser.add_argument("--checkpoint", type=str, default=MOE_PATH,
                        help="Checkpoint path for warmstart/eval")
    parser.add_argument("--no-forget-gate", action="store_true",
                        help="Disable forget gate")
    args = parser.parse_args()

    if args.quick:
        args.rounds = 1
        args.epochs = 1

    global USE_FORGET_GATE
    if args.no_forget_gate:
        USE_FORGET_GATE = False
        print("⚠ Forget Gate DISABLED via --no-forget-gate")

    np.random.seed(SEED)

    # 1. Load shared data
    data, n_x = load_shared_data()

    # 2. Init experts
    if args.eval_only:
        print("\n" + "=" * 60)
        print(f"LOADING CHECKPOINT FOR EVAL: {args.checkpoint}")
        print("=" * 60)
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found!")
        with open(args.checkpoint, "rb") as f:
            experts = pickle.load(f)
        print(f"  Loaded {len(experts)} experts.")
        train_time = 0.0
    else:
        print("\n" + "=" * 60)
        print("INITIALIZING EXPERTS")
        print("=" * 60)
        experts = init_experts(n_x, N_CLUSTERS, PARAM_SHARING,
                               USE_FORGET_GATE, warmstart=args.warmstart,
                               moe_path=args.checkpoint)

    # 3. Load clients
    if not args.eval_only:
        print("\n" + "=" * 60)
        print("LOADING FEDERATED CLIENTS")
        print("=" * 60)
        clients = load_federated_clients(data)

        if args.subset < 1.0:
            for c in clients:
                n_sub = max(32, int(len(c.X_seq) * args.subset))
                c.X_seq = c.X_seq[:n_sub]
                c.y_seq = c.y_seq[:n_sub]
            print(f"  Subsampled clients to {args.subset*100:.0f}%")

    # 4. Build shared QNode
    print("\n  Building QNode...")
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, dev = cell.build_qnode(device_mode=DEVICE_MODE)
    print(f"  Device: {dev.name} | Qubits: {cell.total_qubits}")

    # 5. Run CFL simulation
    if not args.eval_only:
        experts, train_time = run_cfl_simulation(
            data, n_x, experts, clients, qnode,
            n_rounds=args.rounds, client_epochs=args.epochs,
            expert_capacity=args.capacity, alpha=args.alpha,
        )

    # 6. Final evaluation
    metrics = evaluate_cfl(data, n_x, experts, qnode, train_time)
    metrics["config"] = {
        "dataset": "AuverGrid (GWA-T-4)",
        "method": "CFL_Full_MoE",
        "n_rounds": args.rounds,
        "client_epochs": args.epochs,
        "alpha": args.alpha,
        "capacity": args.capacity,
        "warmstart": args.warmstart,
        "param_sharing": PARAM_SHARING,
        "forget_gate": USE_FORGET_GATE,
    }

    # 7. Save results
    with open(RESULTS_JSON, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_JSON}")

    # 8. Save updated experts
    updated_experts_path = os.path.join(PROJECT_ROOT, "cfl_experts_updated_auverGrid.pkl")
    with open(updated_experts_path, "wb") as f:
        pickle.dump(experts, f)
    print(f"  Updated experts saved to {updated_experts_path}")

    print("\n" + "=" * 60)
    print("  CFL FULL MoE (AuverGrid) COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
