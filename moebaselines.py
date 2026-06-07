"""
ablation_baselines.py
=====================
Runs ALL baseline variants for the AQRNN-CFL paper.

AQRNN Variants:
  A: FedAvg single-model (no clustering, single global AQRNN)
  B: Centralized AQRNN (full data, no federation, no routing)
  C: Centralized MoE (KMeans routing, no federation)

Classical Federated Baselines:
  FedAvg SimpleRNN
  FedAvg GRU-Only
  FedAvg Bi-LSTM
  FedAvg esDNN (stacked dense)

CFL-LSTM:
  FedAvg LSTM vs CFL LSTM

Usage:
    python ablation_baselines.py --variant A
    python ablation_baselines.py --variant B
    python ablation_baselines.py --variant C
    python ablation_baselines.py --variant classical
    python ablation_baselines.py --variant cfl_lstm
    python ablation_baselines.py --variant all

Requires: grid5000_hybrid_clean.csv, kmeans_model.pkl, federated_data/client_*.csv, aqrnn.py
"""

import os, sys, json, time, pickle, glob, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.getcwd())

# ── CONFIG ──
CSV_PATH    = "grid5000_hybrid_clean.csv"
KMEANS_PKL  = "kmeans_model.pkl"
DATA_DIR    = "federated_data"
SEQ_LEN     = 2
N_H         = 4
HIDDEN_DIM  = 64
N_LAYERS    = 1
PARAM_SHARING = True
BATCH_SIZE  = 256
DEVICE_MODE = "cuda"
SEED        = 42

# ═══════════════════════════════════════════════════════════════
#  SHARED: Data Loading
# ═══════════════════════════════════════════════════════════════

def load_global_data():
    """Load Grid5000, apply scalers from kmeans_model.pkl, return train/val/test splits."""
    with open(KMEANS_PKL, "rb") as f:
        kdata = pickle.load(f)
    scaler_x = kdata["scaler_x"]
    pca_obj  = kdata.get("pca")
    kmeans   = kdata.get("kmeans")

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

    X_raw = df[req_cols].values
    y_raw = df["TrueCPUUtil"].values.reshape(-1, 1)

    X_scaled = scaler_x.transform(X_raw)
    if pca_obj is not None:
        X_scaled = pca_obj.transform(X_scaled)

    clusters = kmeans.predict(X_scaled) if kmeans else np.zeros(len(X_scaled), dtype=int)

    X_seq, y_seq_raw, c_seq = [], [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i+SEQ_LEN])
        y_seq_raw.append(y_raw[i+SEQ_LEN])
        c_seq.append(clusters[i+SEQ_LEN-1])

    X_seq     = np.array(X_seq)
    y_seq_raw = np.array(y_seq_raw)
    c_seq     = np.array(c_seq)

    N = len(X_seq)
    tr = int(N * 0.7)
    va = tr + int(N * 0.15)

    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    scaler_y.fit(y_seq_raw[:tr])
    y_seq = scaler_y.transform(y_seq_raw)

    n_x = X_scaled.shape[1]

    return {
        'X_train': X_seq[:tr], 'y_train': y_seq[:tr], 'c_train': c_seq[:tr],
        'X_val':   X_seq[tr:va], 'y_val': y_seq[tr:va], 'c_val': c_seq[tr:va],
        'X_test':  X_seq[va:], 'y_test': y_seq[va:], 'c_test': c_seq[va:],
        'scaler_y': scaler_y, 'n_x': n_x, 'kmeans': kmeans,
    }


def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def evaluate(y_true_scaled, y_pred_scaled, scaler_y, label=""):
    yt = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
    yp = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae  = float(mean_absolute_error(yt, yp))
    sm   = smape(yt, yp)
    print(f"  {label}: RMSE={rmse:.2f}  MAE={mae:.2f}  SMAPE={sm:.1f}%")
    return {"rmse": rmse, "mae": mae, "smape": sm}


# ═══════════════════════════════════════════════════════════════
#  AQRNN INFERENCE HELPER
# ═══════════════════════════════════════════════════════════════

def aqrnn_predict(qnode, params_q, wvec, forget_vec, X, n_h=N_H, hidden_dim=HIDDEN_DIM):
    from aqrnn import unpack_classical_weights
    from pennylane import numpy as pnp

    def _readouts_to_array(readouts_t, nh, bs):
        out = pnp.array(readouts_t)
        if out.ndim == 1: return pnp.reshape(out, (1, -1))
        elif out.shape[0] == nh and out.ndim == 2: return pnp.transpose(out)
        return out

    all_preds = []
    for i in range(0, len(X), 256):
        chunk = X[i:i+256]
        readouts_t = qnode(chunk, params_q, forget_gate=forget_vec)
        readouts = _readouts_to_array(readouts_t, n_h, len(chunk))
        W1, b1, W2, b2 = unpack_classical_weights(wvec, n_h, hidden_dim)
        z = pnp.dot(pnp.array(readouts), pnp.array(W1).T) + pnp.array(b1)
        h = pnp.maximum(0, z)
        out = pnp.dot(h, pnp.array(W2).T) + pnp.array(b2)
        all_preds.append(np.array(out[:, 0]))
    return np.concatenate(all_preds)


# ═══════════════════════════════════════════════════════════════
#  VARIANT A: FedAvg Single-Model AQRNN (no clustering)
# ═══════════════════════════════════════════════════════════════

def run_variant_a():
    """Single global AQRNN trained via FedAvg across all clients (no MoE routing)."""
    from aqrnn import (AQRNNCell, AdamState, adam_update, make_loss_fn,
                       pack_classical_weights, clip_grad_norm)
    import pennylane as qml
    from pennylane import numpy as pnp

    print("\n" + "="*60)
    print("  VARIANT A: FedAvg Single-Model AQRNN")
    print("="*60)

    data = load_global_data()
    n_x = data['n_x']

    # Build single QNode
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, param_shape, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    # Init params
    np.random.seed(SEED)
    params_q = np.random.uniform(-0.01, 0.01, size=param_shape).astype(np.float32)
    W1 = (np.sqrt(2.0/N_H) * np.random.randn(HIDDEN_DIM, N_H)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = (np.sqrt(2.0/HIDDEN_DIM) * np.random.randn(1, HIDDEN_DIM)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    wvec = pack_classical_weights(W1, b1, W2, b2)
    forget_vec = 0.95 * np.ones(SEQ_LEN)

    q_state = AdamState(params_q, lr=1e-3)
    c_state = AdamState(wvec, lr=1e-3)
    loss_fn = make_loss_fn(qnode, SEQ_LEN, n_x, N_H, HIDDEN_DIM,
                           l2_q=1e-3, l2_c=1e-3, forget_gate=forget_vec)
    grad_q_fn = qml.grad(loss_fn, argnum=0)
    grad_c_fn = qml.grad(loss_fn, argnum=1)

    # Load client data for FedAvg simulation
    client_files = sorted(glob.glob(os.path.join(DATA_DIR, "client_*.csv")))
    print(f"  Clients: {len(client_files)}")

    N_ROUNDS = 8
    t_start = time.perf_counter()

    for r in range(1, N_ROUNDS + 1):
        print(f"\n  Round {r}/{N_ROUNDS}")
        all_delta_q = []
        all_delta_w = []

        for cf in tqdm(client_files, desc=f"  R{r} Clients", leave=False):
            # Load client
            df_c = pd.read_csv(cf, index_col=0)
            req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                        "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            X_raw_c = df_c[req_cols].values
            y_raw_c = df_c["TrueCPUUtil"].values.reshape(-1, 1)

            with open(KMEANS_PKL, "rb") as f:
                kd = pickle.load(f)
            X_sc = kd["scaler_x"].transform(X_raw_c)
            if kd.get("pca") is not None:
                X_sc = kd["pca"].transform(X_sc)

            scaler_y_c = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
            scaler_y_c.fit(y_raw_c)
            y_sc = scaler_y_c.transform(y_raw_c)

            Xs, ys = [], []
            for i in range(len(X_sc) - SEQ_LEN):
                Xs.append(X_sc[i:i+SEQ_LEN])
                ys.append(y_sc[i+SEQ_LEN])
            if len(Xs) < 10: continue
            Xs = np.array(Xs); ys = np.array(ys)

            # Local training (1 epoch, limited samples)
            local_pq = params_q.copy()
            local_wv = wvec.copy()
            lq = AdamState(local_pq, lr=1e-3)
            lc = AdamState(local_wv, lr=1e-3)

            n_use = min(1024, len(Xs))
            idx = np.random.choice(len(Xs), n_use, replace=False)
            for b in range(0, n_use, BATCH_SIZE):
                bi = idx[b:b+BATCH_SIZE]
                bX, bY = Xs[bi], ys[bi]
                gq = grad_q_fn(local_pq, local_wv, bX, bY)
                gc = grad_c_fn(local_pq, local_wv, bX, bY)
                gq = clip_grad_norm(gq, 1.0)
                gc = clip_grad_norm(gc, 1.0)
                local_pq, lq = adam_update(local_pq, gq, lq)
                local_wv, lc = adam_update(local_wv, gc, lc)

            all_delta_q.append(local_pq - params_q)
            all_delta_w.append(local_wv - wvec)

        # FedAvg aggregate
        params_q = params_q + 0.5 * np.mean(all_delta_q, axis=0)
        wvec     = wvec     + 0.5 * np.mean(all_delta_w, axis=0)

        # Quick val
        val_pred = aqrnn_predict(qnode, params_q, wvec, forget_vec, data['X_val'])
        val_rmse = np.sqrt(mean_squared_error(data['y_val'], val_pred))
        print(f"    Val RMSE (scaled): {val_rmse:.4f}")

    train_time = time.perf_counter() - t_start

    # Final eval
    test_pred = aqrnn_predict(qnode, params_q, wvec, forget_vec, data['X_test'])
    metrics = evaluate(data['y_test'].ravel(), test_pred, data['scaler_y'], "Variant A")
    metrics['train_time'] = round(train_time, 2)
    metrics['variant'] = 'A_FedAvg_Single'
    metrics['description'] = 'FedAvg, single global model'

    with open("results_variant_A.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: results_variant_A.json (Train: {train_time:.0f}s)")
    return metrics


# ═══════════════════════════════════════════════════════════════
#  VARIANT B: Centralized AQRNN (no federation, no routing)
# ═══════════════════════════════════════════════════════════════

def run_variant_b():
    """Single AQRNN trained on ALL data centrally (no federation, no clustering)."""
    from aqrnn import (AQRNNCell, AdamState, adam_update, make_loss_fn,
                       pack_classical_weights, clip_grad_norm)
    import pennylane as qml
    from pennylane import numpy as pnp

    print("\n" + "="*60)
    print("  VARIANT B: Centralized AQRNN (No Federation)")
    print("="*60)

    data = load_global_data()
    n_x = data['n_x']
    X_tr, y_tr = data['X_train'], data['y_train']

    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, param_shape, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    np.random.seed(SEED)
    params_q = np.random.uniform(-0.01, 0.01, size=param_shape).astype(np.float32)
    W1 = (np.sqrt(2.0/N_H) * np.random.randn(HIDDEN_DIM, N_H)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = (np.sqrt(2.0/HIDDEN_DIM) * np.random.randn(1, HIDDEN_DIM)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    wvec = pack_classical_weights(W1, b1, W2, b2)
    forget_vec = 0.95 * np.ones(SEQ_LEN)

    q_state = AdamState(params_q, lr=1e-3)
    c_state = AdamState(wvec, lr=1e-3)
    loss_fn = make_loss_fn(qnode, SEQ_LEN, n_x, N_H, HIDDEN_DIM,
                           l2_q=1e-3, l2_c=1e-3, forget_gate=forget_vec)
    grad_q_fn = qml.grad(loss_fn, argnum=0)
    grad_c_fn = qml.grad(loss_fn, argnum=1)

    N_EPOCHS = 5
    t_start = time.perf_counter()

    for epoch in range(1, N_EPOCHS + 1):
        perm = np.random.permutation(len(X_tr))
        n_batches = int(np.ceil(len(X_tr) / BATCH_SIZE))
        epoch_loss = 0
        for b in tqdm(range(n_batches), desc=f"  Epoch {epoch}/{N_EPOCHS}", leave=True):
            bi = b * BATCH_SIZE
            bj = min((b+1) * BATCH_SIZE, len(X_tr))
            idx = perm[bi:bj]
            bX, bY = X_tr[idx], y_tr[idx]

            gq = grad_q_fn(params_q, wvec, bX, bY)
            gc = grad_c_fn(params_q, wvec, bX, bY)
            gq = clip_grad_norm(gq, 1.0)
            gc = clip_grad_norm(gc, 1.0)
            params_q, q_state = adam_update(params_q, gq, q_state)
            wvec, c_state = adam_update(wvec, gc, c_state)
            epoch_loss += float(loss_fn(params_q, wvec, bX, bY))

        val_pred = aqrnn_predict(qnode, params_q, wvec, forget_vec, data['X_val'])
        val_rmse = np.sqrt(mean_squared_error(data['y_val'], val_pred))
        print(f"    Epoch {epoch} Val RMSE (scaled): {val_rmse:.4f}")

    train_time = time.perf_counter() - t_start

    test_pred = aqrnn_predict(qnode, params_q, wvec, forget_vec, data['X_test'])
    metrics = evaluate(data['y_test'].ravel(), test_pred, data['scaler_y'], "Variant B")
    metrics['train_time'] = round(train_time, 2)
    metrics['variant'] = 'B_Centralized'
    metrics['description'] = 'Centralized, no federation'

    with open("results_variant_B.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: results_variant_B.json (Train: {train_time:.0f}s)")
    return metrics


# ═══════════════════════════════════════════════════════════════
#  VARIANT C: Centralized MoE (KMeans routing, no federation)
# ═══════════════════════════════════════════════════════════════

def run_variant_c():
    """MoE with KMeans routing, trained centrally on full data (no federation)."""
    from aqrnn import (AQRNNCell, AdamState, adam_update, make_loss_fn,
                       pack_classical_weights, clip_grad_norm)
    import pennylane as qml
    from pennylane import numpy as pnp

    print("\n" + "="*60)
    print("  VARIANT C: Centralized MoE (KMeans Routing)")
    print("="*60)

    data = load_global_data()
    n_x = data['n_x']
    n_clusters = len(np.unique(data['c_train']))

    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, param_shape, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    experts = {}
    N_EPOCHS = 5
    t_start = time.perf_counter()

    for k in range(n_clusters):
        print(f"\n  Training Expert {k}...")
        mask_tr = data['c_train'] == k
        mask_va = data['c_val'] == k
        X_k = data['X_train'][mask_tr]
        y_k = data['y_train'][mask_tr]
        X_vk = data['X_val'][mask_va]
        y_vk = data['y_val'][mask_va]
        print(f"    Samples: {len(X_k)} train, {len(X_vk)} val")

        np.random.seed(SEED + k)
        params_q = np.random.uniform(-0.01, 0.01, size=param_shape).astype(np.float32)
        W1 = (np.sqrt(2.0/N_H) * np.random.randn(HIDDEN_DIM, N_H)).astype(np.float32)
        b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        W2 = (np.sqrt(2.0/HIDDEN_DIM) * np.random.randn(1, HIDDEN_DIM)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        wvec = pack_classical_weights(W1, b1, W2, b2)
        forget_vec = 0.95 * np.ones(SEQ_LEN)

        q_state = AdamState(params_q, lr=1e-3)
        c_state = AdamState(wvec, lr=1e-3)
        loss_fn = make_loss_fn(qnode, SEQ_LEN, n_x, N_H, HIDDEN_DIM,
                               l2_q=1e-3, l2_c=1e-3, forget_gate=forget_vec)
        grad_q_fn = qml.grad(loss_fn, argnum=0)
        grad_c_fn = qml.grad(loss_fn, argnum=1)

        best_rmse = 1e9
        best_params = None

        for epoch in range(1, N_EPOCHS + 1):
            perm = np.random.permutation(len(X_k))
            n_batches = int(np.ceil(len(X_k) / BATCH_SIZE))
            for b in tqdm(range(n_batches), desc=f"  Expert {k} Epoch {epoch}", leave=False):
                bi = b * BATCH_SIZE
                bj = min((b+1)*BATCH_SIZE, len(X_k))
                idx = perm[bi:bj]
                bX, bY = X_k[idx], y_k[idx]
                gq = grad_q_fn(params_q, wvec, bX, bY)
                gc = grad_c_fn(params_q, wvec, bX, bY)
                gq = clip_grad_norm(gq, 1.0)
                gc = clip_grad_norm(gc, 1.0)
                params_q, q_state = adam_update(params_q, gq, q_state)
                wvec, c_state = adam_update(wvec, gc, c_state)

            if len(X_vk) > 0:
                vp = aqrnn_predict(qnode, params_q, wvec, forget_vec, X_vk)
                vr = np.sqrt(mean_squared_error(y_vk, vp))
                print(f"    Epoch {epoch} Val RMSE: {vr:.4f}")
                if vr < best_rmse:
                    best_rmse = vr
                    best_params = {"params_q": params_q.copy(), "wvec": wvec.copy()}

        if best_params is None:
            best_params = {"params_q": params_q, "wvec": wvec}
        experts[k] = best_params

    train_time = time.perf_counter() - t_start

    # Evaluate per-cluster
    forget_vec = 0.95 * np.ones(SEQ_LEN)
    all_preds = np.zeros(len(data['X_test']))
    all_mask  = np.zeros(len(data['X_test']), dtype=bool)

    for k in range(n_clusters):
        mask = data['c_test'] == k
        if not np.any(mask): continue
        ep = experts[k]
        preds = aqrnn_predict(qnode, ep["params_q"], ep["wvec"], forget_vec, data['X_test'][mask])
        all_preds[mask] = preds
        all_mask[mask]  = True

    metrics = evaluate(data['y_test'].ravel()[all_mask], all_preds[all_mask],
                       data['scaler_y'], "Variant C")
    metrics['train_time'] = round(train_time, 2)
    metrics['variant'] = 'C_Centralized_MoE'
    metrics['description'] = 'Centralized, KMeans routing'

    with open("results_variant_C.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: results_variant_C.json (Train: {train_time:.0f}s)")
    return metrics


# ═══════════════════════════════════════════════════════════════
#  CLASSICAL FEDERATED BASELINES (SimpleRNN, GRU, Bi-LSTM, esDNN)
# ═══════════════════════════════════════════════════════════════

def run_classical_baselines():
    """Train 4 classical models via FedAvg on the same client shards."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("ERROR: PyTorch required for classical baselines. Install via:")
        print("  pip install torch --break-system-packages")
        return

    print("\n" + "="*60)
    print("  CLASSICAL FEDERATED BASELINES")
    print("="*60)

    data = load_global_data()
    n_x = data['n_x']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── MODEL DEFINITIONS ──
    class SimpleRNNModel(nn.Module):
        def __init__(self, input_dim, hidden=32):
            super().__init__()
            self.rnn = nn.RNN(input_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            _, h = self.rnn(x)
            return self.fc(h.squeeze(0)).squeeze(-1)

    class GRUModel(nn.Module):
        def __init__(self, input_dim, hidden=32):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(h.squeeze(0)).squeeze(-1)

    class BiLSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden * 2, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    class EsDNNModel(nn.Module):
        def __init__(self, input_dim, seq_len):
            super().__init__()
            flat_dim = input_dim * seq_len
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(flat_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

    models_config = {
        "SimpleRNN": lambda: SimpleRNNModel(n_x),
        "GRU":       lambda: GRUModel(n_x),
        "BiLSTM":    lambda: BiLSTMModel(n_x),
        "esDNN":     lambda: EsDNNModel(n_x, SEQ_LEN),
    }

    # Load client data
    client_files = sorted(glob.glob(os.path.join(DATA_DIR, "client_*.csv")))
    print(f"  Clients: {len(client_files)}")

    # Prepare client tensors
    client_data = []
    for cf in client_files:
        df_c = pd.read_csv(cf, index_col=0)
        req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                    "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        X_raw_c = df_c[req_cols].values
        y_raw_c = df_c["TrueCPUUtil"].values.reshape(-1, 1)

        with open(KMEANS_PKL, "rb") as f:
            kd = pickle.load(f)
        X_sc = kd["scaler_x"].transform(X_raw_c)
        if kd.get("pca") is not None:
            X_sc = kd["pca"].transform(X_sc)

        scaler_y_c = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        scaler_y_c.fit(y_raw_c)
        y_sc = scaler_y_c.transform(y_raw_c)

        Xs, ys = [], []
        for i in range(len(X_sc) - SEQ_LEN):
            Xs.append(X_sc[i:i+SEQ_LEN])
            ys.append(y_sc[i+SEQ_LEN, 0])
        if len(Xs) < 10: continue
        client_data.append((np.array(Xs), np.array(ys)))

    # Test tensors
    X_test_t = torch.FloatTensor(data['X_test']).to(device)
    y_test_np = data['y_test'].ravel()

    N_ROUNDS = 10
    results = {}

    for model_name, model_fn in models_config.items():
        print(f"\n  ── {model_name} ──")
        torch.manual_seed(SEED)
        global_model = model_fn().to(device)
        n_params = sum(p.numel() for p in global_model.parameters())
        print(f"    Parameters: {n_params:,}")

        t_start = time.perf_counter()

        for r in range(1, N_ROUNDS + 1):
            client_deltas = []
            global_state = {k: v.clone() for k, v in global_model.state_dict().items()}

            for X_c, y_c in client_data:
                local_model = model_fn().to(device)
                local_model.load_state_dict(global_state)
                optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
                loss_fn = nn.MSELoss()

                X_t = torch.FloatTensor(X_c).to(device)
                y_t = torch.FloatTensor(y_c).to(device)
                ds = TensorDataset(X_t, y_t)
                dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

                local_model.train()
                for bX, bY in dl:
                    optimizer.zero_grad()
                    pred = local_model(bX)
                    loss = loss_fn(pred, bY)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                    optimizer.step()

                delta = {k: local_model.state_dict()[k] - global_state[k]
                         for k in global_state}
                client_deltas.append(delta)

            # FedAvg
            avg_delta = {k: torch.mean(torch.stack([d[k] for d in client_deltas]), dim=0)
                         for k in global_state}
            new_state = {k: global_state[k] + 0.5 * avg_delta[k] for k in global_state}
            global_model.load_state_dict(new_state)

            # Val
            global_model.eval()
            with torch.no_grad():
                vp = global_model(X_test_t).cpu().numpy()
            vr = np.sqrt(mean_squared_error(y_test_np, vp))
            print(f"    R{r} Val RMSE (scaled): {vr:.4f}")

        train_time = time.perf_counter() - t_start

        global_model.eval()
        with torch.no_grad():
            test_pred = global_model(X_test_t).cpu().numpy()
        metrics = evaluate(y_test_np, test_pred, data['scaler_y'], f"FedAvg {model_name}")
        metrics['train_time'] = round(train_time, 2)
        metrics['parameters'] = n_params
        metrics['model'] = model_name
        results[model_name] = metrics

    with open("results_classical_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results_classical_baselines.json")
    return results


# ═══════════════════════════════════════════════════════════════
#  CFL-LSTM: Compare FedAvg LSTM vs CFL LSTM
# ═══════════════════════════════════════════════════════════════

def run_cfl_lstm():
    """Compare plain FedAvg LSTM vs CFL LSTM with MoE routing."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("ERROR: PyTorch required. pip install torch --break-system-packages")
        return

    print("\n" + "="*60)
    print("  CFL-LSTM: FedAvg vs CFL Comparison")
    print("="*60)

    data = load_global_data()
    n_x = data['n_x']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clusters = len(np.unique(data['c_train']))

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    # Load client data with cluster assignments
    client_files = sorted(glob.glob(os.path.join(DATA_DIR, "client_*.csv")))
    client_data = []
    for cf in client_files:
        df_c = pd.read_csv(cf, index_col=0)
        req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                    "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        X_raw_c = df_c[req_cols].values
        y_raw_c = df_c["TrueCPUUtil"].values.reshape(-1, 1)

        with open(KMEANS_PKL, "rb") as f:
            kd = pickle.load(f)
        X_sc = kd["scaler_x"].transform(X_raw_c)
        if kd.get("pca") is not None:
            X_sc = kd["pca"].transform(X_sc)
        kmeans = kd.get("kmeans")
        clusters_c = kmeans.predict(X_sc) if kmeans else np.zeros(len(X_sc), dtype=int)

        scaler_y_c = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        scaler_y_c.fit(y_raw_c)
        y_sc = scaler_y_c.transform(y_raw_c)

        Xs, ys, cs = [], [], []
        for i in range(len(X_sc) - SEQ_LEN):
            Xs.append(X_sc[i:i+SEQ_LEN])
            ys.append(y_sc[i+SEQ_LEN, 0])
            cs.append(clusters_c[i+SEQ_LEN-1])
        if len(Xs) < 10: continue
        # Find dominant cluster for this client
        dom_cluster = int(np.bincount(cs).argmax())
        client_data.append((np.array(Xs), np.array(ys), dom_cluster))

    X_test_t = torch.FloatTensor(data['X_test']).to(device)
    y_test_np = data['y_test'].ravel()
    N_ROUNDS = 10

    results = {}

    # --- FedAvg LSTM (single model) ---
    print("\n  ── FedAvg LSTM ──")
    torch.manual_seed(SEED)
    global_model = LSTMModel(n_x).to(device)
    n_params = sum(p.numel() for p in global_model.parameters())
    print(f"    Parameters: {n_params:,}")

    t_start = time.perf_counter()
    for r in range(1, N_ROUNDS + 1):
        global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
        deltas = []
        for X_c, y_c, _ in client_data:
            local = LSTMModel(n_x).to(device)
            local.load_state_dict(global_state)
            opt = torch.optim.Adam(local.parameters(), lr=1e-3)
            X_t = torch.FloatTensor(X_c).to(device)
            y_t = torch.FloatTensor(y_c).to(device)
            local.train()
            for b in range(0, len(X_c), BATCH_SIZE):
                bX = X_t[b:b+BATCH_SIZE]; bY = y_t[b:b+BATCH_SIZE]
                opt.zero_grad()
                nn.MSELoss()(local(bX), bY).backward()
                torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                opt.step()
            deltas.append({k: local.state_dict()[k] - global_state[k] for k in global_state})

        avg_d = {k: torch.mean(torch.stack([d[k] for d in deltas]), dim=0) for k in global_state}
        global_model.load_state_dict({k: global_state[k] + 0.5 * avg_d[k] for k in global_state})

    train_time_fedavg = time.perf_counter() - t_start
    global_model.eval()
    with torch.no_grad():
        pred_fa = global_model(X_test_t).cpu().numpy()
    m_fa = evaluate(y_test_np, pred_fa, data['scaler_y'], "FedAvg LSTM")
    m_fa['train_time'] = round(train_time_fedavg, 2)
    m_fa['parameters'] = n_params
    results['FedAvg_LSTM'] = m_fa

    # --- CFL LSTM (one expert per cluster) ---
    print("\n  ── CFL LSTM ──")
    experts_lstm = {}
    t_start = time.perf_counter()

    for k in range(n_clusters):
        torch.manual_seed(SEED + k)
        experts_lstm[k] = LSTMModel(n_x).to(device)

    for r in range(1, N_ROUNDS + 1):
        for k in range(n_clusters):
            global_state = {kk: v.clone() for kk, v in experts_lstm[k].state_dict().items()}
            deltas = []
            for X_c, y_c, dom_k in client_data:
                if dom_k != k: continue
                local = LSTMModel(n_x).to(device)
                local.load_state_dict(global_state)
                opt = torch.optim.Adam(local.parameters(), lr=1e-3)
                X_t = torch.FloatTensor(X_c).to(device)
                y_t = torch.FloatTensor(y_c).to(device)
                local.train()
                for b in range(0, len(X_c), BATCH_SIZE):
                    bX = X_t[b:b+BATCH_SIZE]; bY = y_t[b:b+BATCH_SIZE]
                    opt.zero_grad()
                    nn.MSELoss()(local(bX), bY).backward()
                    torch.nn.utils.clip_grad_norm_(local.parameters(), 1.0)
                    opt.step()
                deltas.append({kk: local.state_dict()[kk] - global_state[kk] for kk in global_state})

            if deltas:
                avg_d = {kk: torch.mean(torch.stack([d[kk] for d in deltas]), dim=0) for kk in global_state}
                experts_lstm[k].load_state_dict({kk: global_state[kk] + 0.5 * avg_d[kk] for kk in global_state})

    train_time_cfl = time.perf_counter() - t_start

    # CFL eval: route by cluster
    all_preds_cfl = np.zeros(len(data['X_test']))
    for k in range(n_clusters):
        mask = data['c_test'] == k
        if not np.any(mask): continue
        experts_lstm[k].eval()
        with torch.no_grad():
            preds_k = experts_lstm[k](torch.FloatTensor(data['X_test'][mask]).to(device)).cpu().numpy()
        all_preds_cfl[mask] = preds_k

    m_cfl = evaluate(y_test_np, all_preds_cfl, data['scaler_y'], "CFL LSTM")
    m_cfl['train_time'] = round(train_time_cfl, 2)
    m_cfl['parameters'] = n_params
    results['CFL_LSTM'] = m_cfl

    # Per-cluster CFL results
    for k in range(n_clusters):
        mask = data['c_test'] == k
        if not np.any(mask): continue
        yt_k = data['scaler_y'].inverse_transform(y_test_np[mask].reshape(-1, 1)).ravel()
        yp_k = data['scaler_y'].inverse_transform(all_preds_cfl[mask].reshape(-1, 1)).ravel()
        print(f"    CFL LSTM Cluster {k}: RMSE={np.sqrt(mean_squared_error(yt_k, yp_k)):.2f}  "
              f"MAE={mean_absolute_error(yt_k, yp_k):.2f}  SMAPE={smape(yt_k, yp_k):.1f}%  (n={mask.sum()})")

    with open("results_cfl_lstm.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results_cfl_lstm.json")
    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQRNN-CFL Ablation Baselines")
    parser.add_argument("--variant", type=str, default="all",
                        choices=["A", "B", "C", "classical", "cfl_lstm", "all"],
                        help="Which variant to run")
    args = parser.parse_args()

    if args.variant in ("A", "all"):
        run_variant_a()
    if args.variant in ("B", "all"):
        run_variant_b()
    if args.variant in ("C", "all"):
        run_variant_c()
    if args.variant in ("classical", "all"):
        run_classical_baselines()
    if args.variant in ("cfl_lstm", "all"):
        run_cfl_lstm()

    print("\n" + "="*60)
    print("  ALL BASELINES COMPLETE")
    print("="*60)