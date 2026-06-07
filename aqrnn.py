#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- WINDOWS GPU FIX: Add Conda DLLs to Path ---
if sys.platform == "win32":
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cuda_bin = os.path.join(conda_prefix, "Library", "bin")
        if os.path.exists(cuda_bin):
            os.add_dll_directory(cuda_bin)
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

import pennylane as qml
from pennylane import numpy as pnp

from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONSTANTS ---
DEFAULT_SCALE_ANGLE = np.pi
RNG = np.random.RandomState(42)

# --- 1. DATA EXTRACTION & CLUSTERING ---
csv_path = "grid5000_hybrid_clean.csv"

def load_and_process_data_clustered(k=3, n_components=None, save_kmeans_path="kmeans_model.pkl"):
    """
    Loads data. Uses QuantileTransformer to ensure balanced clustering.
    Retains ALL features if n_components is None.
    """
    if not os.path.exists(csv_path):
        print("Error: CSV not found. Please ensure 'grid5000_hybrid_clean.csv' exists.")
        sys.exit(1)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "WindowStart" in df.columns:
        df["datetime"] = pd.to_datetime(df["WindowStart"], unit='s')
        df.set_index("datetime", inplace=True)

    # Feature Engineering (Cyclical Time)
    if "hours" not in df.columns:
        df["hours"] = df.index.hour + df.index.minute / 60.0
        df["dow"] = df.index.dayofweek
        df["hour_sin"] = np.sin(2*np.pi* df["hours"]/24)
        df["hour_cos"] = np.cos(2*np.pi*df["hours"]/24)
        df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
        df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    # ORIGINAL FEATURES (9 Total)
    features = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem", "UserDiversity",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target = "TrueCPUUtil"

    # --- KEY CHANGE: Use QuantileTransformer for X ---
    # This spreads the data distribution out, preventing "Super Clusters"
    print("Applying QuantileTransformer to inputs (Fixes clustering imbalance)...")
    scaler_x = QuantileTransformer(output_distribution='uniform', n_quantiles=min(1000, len(df)))
    
    # Target still uses Normal distribution for easier regression
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

    X_raw = df[features].values
    X_scaled = scaler_x.fit_transform(X_raw)
    
    # APPLY PCA IF REQUESTED (Default None = Use All Features)
    pca = None
    if n_components is not None and n_components < X_scaled.shape[1]:
        print(f"Applying PCA: Reducing {X_scaled.shape[1]} features to {n_components} components...")
        pca = PCA(n_components=n_components, random_state=42)
        X_final = pca.fit_transform(X_scaled)
    else:
        print(f"Retaining all {X_scaled.shape[1]} features (No PCA).")
        X_final = X_scaled
    
    y_raw = df[[target]].values
    y_scaled = scaler_y.fit_transform(y_raw)

    # --- CLUSTERING ---
    # Using k-means++ init and n_init=20 for better stability
    print(f"Training K-Means (K={k}) on features...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, init='k-means++')
    clusters = kmeans.fit_predict(X_final)
    
    # Print Cluster Distribution
    unique, counts = np.unique(clusters, return_counts=True)
    print("\n--- CLUSTER DISTRIBUTION ---")
    for u, c in zip(unique, counts):
        print(f"Cluster {u}: {c} samples ({c/len(clusters)*100:.1f}%)")
    print("----------------------------\n")
    
    with open(save_kmeans_path, "wb") as f:
        pickle.dump({"kmeans": kmeans, "scaler_x": scaler_x, "pca": pca}, f)
    print(f"Models saved to {save_kmeans_path}")

    return X_final, y_scaled, clusters, scaler_y, kmeans

def create_dataset_clustered(X, y, clusters, time_steps=1):
    Xs, ys, cs = [], [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
        cs.append(clusters[i + time_steps - 1]) 
    return np.array(Xs), np.array(ys), np.array(cs)

# --- AQRNN COMPONENTS ---
def angle_encode(x, input_wires, scale = DEFAULT_SCALE_ANGLE):
    for i, w in enumerate(input_wires):
        if pnp.ndim(x) > 1: val = x[:, i]
        else: val = x[i]
        qml.RY(val * scale, wires=w)

def _validate_topology(top):
    valid = {"ladder", "start", "all_to_all"}
    if top not in valid: raise ValueError(f"give correct topology name from: {valid}")
    return top

def local_rotations(params_block, wires):
    for i, w in enumerate(wires):
        ry = params_block[i, 0]
        rz = params_block[i, 1]
        qml.RY(ry, wires = w)
        qml.RZ(rz, wires = w)

def entangler_layer(wires, topology="ladder"):
    topology = _validate_topology(topology)
    if topology == "all_to_all":
        for i in range(len(wires)):
            for j in range(len(wires)):
                if i != j: qml.CNOT(wires=[wires[i],wires[j]])
    elif topology == "start":
        center = wires[0]
        for w in wires[1:]:
            qml.CNOT(wires = [center, w])
    else:
        for i, j in zip(wires[:-1], wires[1:]):
            qml.CNOT(wires=[i,j])

def cry_gates(input_wires, hidden_wires, angles):
    n = min(len(input_wires), len(hidden_wires))
    if pnp.ndim(angles) == 0:
        for i in range(n): qml.CRY(angles, wires = [input_wires[i], hidden_wires[i]])
    else:
        for i in range(n): qml.CRY(angles[i], wires = [input_wires[i], hidden_wires[i]])

class AQRNNCell:
    def __init__(self, n_x, n_h, seq_len, n_layers=1, n_ancilla=1, topology="ladder", param_sharing=False, angle_scale=np.pi):
        self.n_x = n_x; self.n_h = n_h; self.seq_len = seq_len
        self.n_layers = max(1, int(n_layers)); self.n_ancilla = n_ancilla
        self.topology = _validate_topology(topology)
        self.param_sharing = bool(param_sharing); self.angle_scale = float(angle_scale)
        self.input_wires = list(range(0, n_x))
        self.hidden_wires = list(range(n_x, n_x + n_h))
        self.ancilla_wires = list(range(n_x + n_h, n_x + n_h + n_ancilla))
        self.total_qubits = n_x + n_h + n_ancilla 
        self.total_wires = self.input_wires + self.hidden_wires + self.ancilla_wires
        n_circuit_wires = len(self.total_wires)
        if self.param_sharing: self.param_shape = (self.n_layers, n_circuit_wires, 2)
        else: self.param_shape = (self.seq_len, self.n_layers, n_circuit_wires, 2)

    def _pqc_block(self, params_block):
        local_rotations(params_block, self.total_wires)
        entangler_layer(self.total_wires, self.topology)
        gate_angle = pnp.mean(params_block[:, 0]) * 0.1
        cry_gates(self.input_wires, self.hidden_wires, gate_angle)

    def build_qnode(self, device=None, device_mode="cuda", shots=400, noise_model=None):
        if device is None:
            if device_mode == "cpu" or device_mode == "simulator":
                try:
                    dev = qml.device("lightning.qubit", wires=self.total_qubits)
                    print(f"DEBUG: Using lightning.qubit (C++ CPU) for {self.total_qubits} wires")
                except:
                    dev = qml.device("default.qubit", wires=self.total_qubits)
                    print("DEBUG: Fallback to default.qubit (Python CPU)")
            elif device_mode == "cuda": 
                try:
                    dev = qml.device("lightning.gpu", wires=self.total_qubits)
                except:
                    print("Warning: lightning.gpu not found. Falling back to lightning.qubit.")
                    try:
                        dev = qml.device("lightning.qubit", wires=self.total_qubits)
                    except:
                        dev = qml.device("default.qubit", wires=self.total_qubits)
            elif device_mode == "noisy": 
                dev = qml.device("qiskit.aer", wires=self.total_qubits, backend="qasm_simulator", shots=shots, noise_model=noise_model)
        else: dev = device
        
        diff_method = "adjoint" if (dev.name in ["lightning.qubit", "lightning.gpu"]) else "best"

        @qml.qnode(dev, interface="autograd", diff_method=diff_method)
        def sequence_qnode(sequence, params, forget_gate=None):
            for t in range(self.seq_len):
                if pnp.ndim(sequence) == 3: x_t = sequence[:, t, :]
                else: x_t = sequence[t]
                angle_encode(x_t, self.input_wires, scale = self.angle_scale)
                if self.param_sharing:
                    for layer in range(self.n_layers): self._pqc_block(params[layer])
                else:
                    for layer in range(self.n_layers): self._pqc_block(params[t, layer])
                if forget_gate is not None:
                    fg = forget_gate[t]
                    if pnp.ndim(fg) == 0:
                        angle = float((1.0 - fg) * 0.25)
                        for hw in self.hidden_wires:
                            qml.RY(angle, wires=hw)
                    else:
                        for i, hw in enumerate(self.hidden_wires):
                            angle = float((1.0 - float(fg[i])) * 0.25)
                            qml.RY(angle, wires=hw)
                qml.adjoint(angle_encode)(x_t, self.input_wires, scale=self.angle_scale)
            return [qml.expval(qml.PauliZ(x)) for x in self.hidden_wires]
        return sequence_qnode, self.param_shape, dev

# --- OPTIMIZERS ---
class AdamState:
    def __init__(self, shape, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        dtype = shape.dtype if hasattr(shape, 'dtype') else float
        self.m = np.zeros_like(shape, dtype=dtype)
        self.v = np.zeros_like(shape, dtype=dtype)
        self.beta1 = beta1; self.beta2 = beta2; self.lr = lr; self.eps = eps; self.t = 0

def adam_update(param, grad, state: AdamState):
    state.t += 1
    state.m = state.beta1 * state.m + (1 - state.beta1) * grad
    state.v = state.beta2 * state.v + (1 - state.beta2) * (grad ** 2)
    m_hat = state.m / (1 - state.beta1 ** state.t)
    v_hat = state.v / (1 - state.beta2 ** state.t)
    step = state.lr * m_hat / (np.sqrt(v_hat) + state.eps)
    return param - step, state

def clip_grad_norm(grad, max_norm=1.0):
    norm = np.sqrt(np.sum(np.array(grad).astype(np.float64) ** 2))
    if norm > max_norm and norm > 1e-12:
        return (np.array(grad, copy=False) * (max_norm / float(norm))).astype(grad.dtype)
    return grad

def pack_classical_weights(W1, b1, W2, b2):
    return np.concatenate([W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()])

def unpack_classical_weights(vec, n_h, hidden_dim):
    idx = 0
    sW1 = hidden_dim* n_h; W1 = vec[idx: idx + sW1].reshape((hidden_dim, n_h)); idx += sW1
    sb1 = hidden_dim; b1 = vec[idx: idx + sb1].reshape((hidden_dim,)); idx += sb1
    sW2 = hidden_dim * 1; W2 = vec[idx: idx + sW2].reshape((1, hidden_dim)); idx += sW2
    sb2 = 1; b2 = vec[idx: idx + sb2].reshape((1,)); idx += sb2
    return W1, b1, W2, b2

def mlp_forward(readout, wvec, n_h, hidden_dim):
    W1, b1, W2, b2 = unpack_classical_weights(wvec, n_h, hidden_dim)
    W1 = pnp.array(W1); b1 = pnp.array(b1); W2 = pnp.array(W2); b2 = pnp.array(b2); readout = pnp.array(readout)
    if readout.ndim == 1:
        h = pnp.maximum(0, pnp.dot(W1, readout) + b1)
        out = pnp.dot(W2, h) + b2
        return out[0]
    else:
        z = pnp.dot(readout, W1.T) + b1; h = pnp.maximum(0, z)
        out = pnp.dot(h, W2.T) + b2
        return out[:, 0]

def _readouts_to_array(readouts_t, n_h, batch_size):
    out = pnp.array(readouts_t)
    if out.ndim == 1: return pnp.reshape(out, (1, -1))
    if out.shape[0] == n_h and out.ndim == 2: return pnp.transpose(out)
    return out

def make_loss_fn(qnode, seq_len, n_x, n_h, hidden_dim, l2_q=1e-3, l2_c=1e-3, forget_gate=None):
    def loss(params_q, wvec, batch_X, batch_y):
        B = batch_X.shape[0]
        readouts_t = qnode(batch_X, params_q, forget_gate=forget_gate)
        readouts = _readouts_to_array(readouts_t, n_h, B)
        preds = mlp_forward(readouts, wvec, n_h, hidden_dim)
        preds_flat = pnp.reshape(preds, (-1,)); y_flat = pnp.reshape(batch_y, (-1,))
        mse = pnp.mean((preds_flat - y_flat) ** 2)
        l2_q_val = l2_q * pnp.sum(params_q ** 2)
        l2_c_val = l2_c * pnp.sum(wvec ** 2)
        return mse + l2_q_val + l2_c_val
    return loss

def calc_accuracy_multi(y_true, y_pred, thresholds=(0.5, 1.0, 1.5)):
    out = {}
    diff = np.abs(y_true - y_pred)
    for t in thresholds:
        correct = diff < t
        out[t] = np.mean(correct) * 100.0
    return out

# --- MIXTURE OF EXPERTS TRAINING ---

def train_aqrnn_moe_robust(
    n_clusters=3,
    n_components=None,  # CHANGED: None = Use ALL features
    seq_len=2,          # Keep 2 for CPU speed
    n_h=4,
    hidden_dim=64, 
    n_layers=1,
    param_sharing=True,
    n_epochs=5,
    batch_size=256,     
    lr_q=1e-3,
    lr_c=1e-3,
    device_mode="cuda", 
    seed=42,
    subset_ratio=1.0,   
    grad_clip=1.0,
    l2_q=1e-3,
    l2_c=1e-3
):
    np.random.seed(seed)
    
    # 1. Prepare Data
    X_s, y_s, clusters_s, scaler_y, kmeans = load_and_process_data_clustered(k=n_clusters, n_components=n_components)
    X_seq, y_seq, c_seq = create_dataset_clustered(X_s, y_s, clusters_s, time_steps=seq_len)
    
    # Split
    N = len(X_seq)
    train_size = int(N * 0.7)
    val_size = int(N * 0.15)
    
    X_train, y_train, c_train = X_seq[:train_size], y_seq[:train_size], c_seq[:train_size]
    X_val, y_val, c_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size], c_seq[train_size:train_size+val_size]
    X_test, y_test, c_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:], c_seq[train_size+val_size:]
    
    # Subset
    if subset_ratio < 1.0:
        n_sub = int(len(X_train) * subset_ratio)
        X_train, y_train, c_train = X_train[:n_sub], y_train[:n_sub], c_train[:n_sub]
        print(f"Subsampled train set to {n_sub} samples (Ratio: {subset_ratio})")

    experts = {}
    n_x = X_train.shape[2] 
    print(f"Input Features: {n_x}")
    print(f"Total Qubits: {n_x} (input) + {n_h} (hidden) + 1 (ancilla) = {n_x + n_h + 1}")
    
    # 2. Train Expert for Each Cluster
    for k in range(n_clusters):
        print(f"\n=== Training Expert for Cluster {k} ===")
        mask_train = (c_train == k)
        mask_val = (c_val == k)
        X_tr_k = X_train[mask_train]
        y_tr_k = y_train[mask_train]
        X_val_k = X_val[mask_val]
        y_val_k = y_val[mask_val]
        
        current_batch_size = batch_size
        if len(X_tr_k) < batch_size:
            if len(X_tr_k) < 32:
                print(f"Error: Cluster {k} size too small. Skipping.")
                continue
            current_batch_size = max(32, len(X_tr_k))
            print(f"-> Adjusted batch_size to {current_batch_size}")
            
        print(f"Cluster {k} Samples: {len(X_tr_k)}")
        cell = AQRNNCell(n_x=n_x, n_h=n_h, seq_len=seq_len, n_layers=n_layers, param_sharing=param_sharing)
        qnode, param_shape, dev = cell.build_qnode(device_mode=device_mode)
        print(f"Using Device: {dev.name}", flush=True)
        
        params_q = np.random.uniform(low=-0.01, high=0.01, size=param_shape).astype(np.float32)
        W1 = (np.sqrt(2.0 / n_h) * np.random.randn(hidden_dim, n_h)).astype(np.float32)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        W2 = (np.sqrt(2.0 / hidden_dim) * np.random.randn(1, hidden_dim)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        wvec = pack_classical_weights(W1, b1, W2, b2)
        forget_vec = 0.95 * np.ones(seq_len)

        q_state = AdamState(params_q, lr=lr_q)
        c_state = AdamState(wvec, lr=lr_c)
        loss_fn = make_loss_fn(qnode, seq_len, n_x, n_h, hidden_dim, l2_q=l2_q, l2_c=l2_c, forget_gate=forget_vec)
        grad_q_fn = qml.grad(loss_fn, argnum=0)
        grad_c_fn = qml.grad(loss_fn, argnum=1)
        
        best_rmse = 1e9
        best_params_k = None
        patience_counter = 0
        n_batches = int(np.ceil(len(X_tr_k) / current_batch_size))
        
        print("  (Compiling circuit on first batch...)", flush=True)
        for epoch in range(1, n_epochs + 1):
             perm = np.random.permutation(len(X_tr_k))
             epoch_loss = 0.0
             pbar = tqdm(range(n_batches), desc=f"Cluster {k} | Epoch {epoch}/{n_epochs}", leave=True, colour='green')
             for b in pbar:
                 bi = b * current_batch_size; bj = min((b+1)*current_batch_size, len(X_tr_k))
                 idx = perm[bi:bj]
                 batch_X, batch_y = X_tr_k[idx], y_tr_k[idx]
                 g_q = grad_q_fn(params_q, wvec, batch_X, batch_y)
                 g_c = grad_c_fn(params_q, wvec, batch_X, batch_y)
                 if grad_clip > 0:
                     g_q = clip_grad_norm(g_q, max_norm=grad_clip)
                     g_c = clip_grad_norm(g_c, max_norm=grad_clip)
                 params_q, q_state = adam_update(params_q, g_q, q_state)
                 wvec, c_state = adam_update(wvec, g_c, c_state)
                 loss_val = loss_fn(params_q, wvec, batch_X, batch_y).item()
                 epoch_loss += loss_val
                 pbar.set_postfix({"Loss": f"{loss_val:.4f}"})
             
             def predict_k(Xvar):
                 chunk_size=256
                 preds=[]
                 for i in range(0, len(Xvar), chunk_size):
                     chunk = Xvar[i:i+chunk_size]
                     readouts_t = qnode(chunk, params_q, forget_gate=forget_vec)
                     readouts = _readouts_to_array(readouts_t, n_h, len(chunk))
                     preds.append(mlp_forward(readouts, wvec, n_h, hidden_dim))
                 return np.concatenate(preds)
             
             if len(X_val_k) > 0:
                 val_preds = predict_k(X_val_k)
                 val_rmse = np.sqrt(mean_squared_error(y_val_k, val_preds))
             else: val_rmse = 0.0
             
             if val_rmse < best_rmse:
                 best_rmse = val_rmse
                 best_params_k = {"params_q": params_q, "wvec": wvec}
                 patience_counter = 0
             else: patience_counter += 1
             print(f"  Epoch {epoch} Done: Val RMSE {val_rmse:.4f}")
             if patience_counter >= 3:
                 print(f"  Early stopping at epoch {epoch}")
                 break
        
        if best_params_k is None: best_params_k = {"params_q": params_q, "wvec": wvec}
        experts[k] = best_params_k
    
    # 3. Router Inference
    print("\n" + "="*40)
    print("=== FINAL STEP: Mixture of Experts Inference ===")
    print("="*40)
    
    final_preds = []
    final_true = []
    
    cell = AQRNNCell(n_x=n_x, n_h=n_h, seq_len=seq_len, n_layers=n_layers, param_sharing=param_sharing)
    qnode, _, _ = cell.build_qnode(device_mode=device_mode)
    
    for k in range(n_clusters):
        mask_test = (c_test == k)
        if not np.any(mask_test): continue
            
        X_test_k = X_test[mask_test]
        y_test_k = y_test[mask_test]
        
        print(f"\nRouting Test Cluster {k} through Expert {k}...")
        if k in experts: model_params = experts[k]
        elif 0 in experts: model_params = experts[0]
        else: continue 

        preds_k = []
        p_q = model_params["params_q"]
        w_c = model_params["wvec"]
        
        chunk_size = 256
        n_test_chunks = int(np.ceil(len(X_test_k) / chunk_size))
        
        test_pbar = tqdm(range(n_test_chunks), desc=f"Evaluating Cluster {k}", colour='blue')
        for i in test_pbar:
             start_idx = i * chunk_size
             end_idx = min((i + 1) * chunk_size, len(X_test_k))
             chunk = X_test_k[start_idx:end_idx]
             
             readouts_t = qnode(chunk, p_q)
             readouts = _readouts_to_array(readouts_t, n_h, len(chunk))
             preds_k.append(mlp_forward(readouts, w_c, n_h, hidden_dim))
        
        preds_k = np.concatenate(preds_k)
        rmse_k = np.sqrt(mean_squared_error(y_test_k, preds_k))
        mae_k = mean_absolute_error(y_test_k, preds_k)
        acc_k = calc_accuracy_multi(y_test_k, preds_k)
        acc_str = f"Acc@0.5: {acc_k[0.5]:.1f}% | Acc@1.0: {acc_k[1.0]:.1f}%"
        
        print(f"Cluster {k} Test Results: RMSE={rmse_k:.4f} | MAE={mae_k:.4f} | {acc_str}")
        final_preds.append(preds_k)
        final_true.append(y_test_k)

    if len(final_preds) > 0:
        final_preds = np.concatenate(final_preds).ravel()
        final_true = np.concatenate(final_true).ravel()
        rmse = np.sqrt(mean_squared_error(final_true, final_preds))
        mae = mean_absolute_error(final_true, final_preds)
        print(f"\n>>> Global Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    
    with open("moe_experts.pkl", "wb") as f:
        pickle.dump(experts, f)
    print("\nExpert Models Saved successfully.")
    return experts

if __name__ == "__main__":
    # Optimized settings: 3 Clusters, ALL 9 Features (No PCA), Batch 256, Force CPU
    train_aqrnn_moe_robust(n_epochs=2, subset_ratio=1.0, n_clusters=3, n_components=4, batch_size=256)