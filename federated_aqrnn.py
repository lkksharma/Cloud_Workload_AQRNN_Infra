
import os
import sys
import pickle
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import copy

# Import AQRNN components from existing script
# Ensure current dir is in path
sys.path.append(os.getcwd())
try:
    from aqrnn import (
        AQRNNCell, AdamState, adam_update, make_loss_fn, 
        pack_classical_weights, unpack_classical_weights
    )
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as e:
    print(f"Error importing aqrnn_cluster: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
DATA_DIR = "federated_data"
MOE_PATH = "moe_experts.pkl"
RESULTS_PATH = "federated_results.pkl"

# FL Hyperparameters
N_ROUNDS = 8
CLIENT_EPOCHS = 3
BATCH_SIZE = 256
LR_Q = 1e-3
LR_C = 1e-3
SEQ_LEN = 2
N_H = 4
HIDDEN_DIM = 64
N_LAYERS = 1
PARAM_SHARING = True
DEVICE_MODE = "cuda" # Tries lightning.gpu -> lightning.qubit -> default.qubit
CSV_PATH = "grid5000_hybrid_clean.csv"
USE_FORGET_GATE = True
GRAD_CLIP = 1.0

# Adaptive distillation: alpha is inversely proportional to cluster variance.
# High-variance (volatile) clusters get less distillation automatically.
ALPHA_BASE = 0.3   # Maximum alpha for the most stable cluster
ALPHA_MIN  = 0.05  # Floor to prevent zero distillation


# ══════════════════════════════════════════════════════════════
# HELPER: Load Global Test Set (matching ablation_study pipeline)
# ══════════════════════════════════════════════════════════════
def load_global_test_set(scaler_x, pca_obj, seq_len=SEQ_LEN):
    """
    Load grid5000_hybrid_clean.csv, apply temporal 70/15/15 split,
    return test arrays + split_time for client filtering.
    Fits its own scaler_y on the training portion for consistency.
    """
    print("\n  Loading Global Test Set from grid5000_hybrid_clean.csv...")
    df = pd.read_csv(CSV_PATH)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "WindowStart" in df.columns:
        df["datetime"] = pd.to_datetime(df["WindowStart"], unit='s')
        df.set_index("datetime", inplace=True)

    df.sort_index(inplace=True)

    # Feature engineering
    if "hours" not in df.columns:
        df["hours"] = df.index.hour + df.index.minute / 60.0
        df["dow"] = df.index.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hours"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hours"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)

    req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target_col = "TrueCPUUtil"

    X_raw = df[req_cols].values
    y_raw = df[target_col].values.reshape(-1, 1)

    X_scaled = scaler_x.transform(X_raw)
    if pca_obj is not None:
        X_scaled = pca_obj.transform(X_scaled)

    # KMeans cluster assignment
    from sklearn.cluster import KMeans
    kmeans_path = "kmeans_model.pkl"
    with open(kmeans_path, "rb") as f:
        kdata = pickle.load(f)
    kmeans = kdata.get("kmeans")
    if kmeans is not None:
        clusters = kmeans.predict(X_scaled)
    else:
        clusters = np.zeros(len(X_scaled), dtype=int)

    # Create sequences
    X_seq, y_seq_raw, c_seq = [], [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq_raw.append(y_raw[i+seq_len])
        c_seq.append(clusters[i+seq_len-1])
    X_seq = np.array(X_seq)
    y_seq_raw = np.array(y_seq_raw)   # shape (N, 1), unscaled
    c_seq = np.array(c_seq)

    # 70/15/15 split
    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end = train_end + int(N * 0.15)

    split_time = df.index[train_end]

    # Fit scaler_y on TRAINING portion only (correct practice)
    from sklearn.preprocessing import QuantileTransformer
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    scaler_y.fit(y_seq_raw[:train_end])

    # Scale all y
    y_seq = scaler_y.transform(y_seq_raw)

    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    c_test = c_seq[val_end:]

    n_clusters = len(np.unique(c_test))
    print(f"  Split Time: {split_time}")
    print(f"  Test Set: {len(X_test):,} samples | {n_clusters} clusters")

    return X_test, y_test, c_test, split_time, n_clusters, scaler_y


# ══════════════════════════════════════════════════════════════
# HELPER: Metrics (matching ablation_study.py)
# ══════════════════════════════════════════════════════════════
def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def _readouts_to_array(readouts_t, n_h, batch_size):
    out = pnp.array(readouts_t)
    if out.ndim == 1:
        return pnp.reshape(out, (1, -1))
    elif out.shape[0] == n_h and out.ndim == 2:
        return pnp.transpose(out)
    return out


def predict_with_params(qnode, params_q, wvec, forget_vec, X, n_h=N_H,
                        hidden_dim=HIDDEN_DIM, chunk_size=256, label="Inference"):
    """Run inference in chunks."""
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


class FederatedClient:
    def __init__(self, client_id, csv_path):
        self.client_id = client_id
        self.data_path = csv_path
        self.load_data()
        
    def load_data(self):
        # Load and process similar to aqrnn_cluster
        df = pd.read_csv(self.data_path, index_col=0)
        
        # Features used in original script:
        # ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem", "UserDiversity",
        #  "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        # Our processed CSVs have these (renamed slightly in process script, need to match)
        
        # Check columns
        req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem", "UserDiversity",
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        
        # Load Raw Y (Target) immediately so it exists for Server init
        self.y_raw = df["TrueCPUUtil"].values.reshape(-1, 1)
        self.df = df[req_cols].copy()
        
    def prepare_batches(self, scaler_x, pca=None, seq_len=2, batch_size=128):
        # Apply Global Scaler
        X_raw = self.df.values
        X_scaled = scaler_x.transform(X_raw)
        
        # Apply PCA if provided (CRITICAL FIX)
        if pca is not None:
            X_scaled = pca.transform(X_scaled)
        
        # Create sequences
        self.X_scaled = X_scaled
        # self.y_raw already set in load_data
        
    def create_dataset(self, scaler_y, seq_len=2):
        y_scaled = scaler_y.transform(self.y_raw)
        
        Xs, ys = [], []
        for i in range(len(self.X_scaled) - seq_len):
            Xs.append(self.X_scaled[i : i + seq_len])
            ys.append(y_scaled[i + seq_len])
        
        self.X_seq = np.array(Xs)
        self.y_seq = np.array(ys)
        
    def evaluate_expert(self, expert_params, qnode, n_x, n_h, hidden_dim):
        """
        Evaluate a single expert on local data (Subset for speed)
        Returns: MSE Loss
        """
        if len(self.X_seq) == 0: return 1e9
        
        idx = np.random.choice(len(self.X_seq), min(256, len(self.X_seq)), replace=False)
        X_batch = self.X_seq[idx]
        y_batch = self.y_seq[idx]
        
        params_q = expert_params["params_q"]
        wvec = expert_params["wvec"]
        
        # Run inference logic
        readouts_t = qnode(X_batch, params_q)
        
        out = pnp.array(readouts_t)
        if out.ndim == 1: readouts = pnp.reshape(out, (1, -1))
        elif out.shape[0] == n_h and out.ndim == 2: readouts = pnp.transpose(out)
        else: readouts = out
            
        W1, b1, W2, b2 = unpack_classical_weights(wvec, n_h, hidden_dim)
        W1 = pnp.array(W1); b1 = pnp.array(b1); W2 = pnp.array(W2); b2 = pnp.array(b2)
        readouts = pnp.array(readouts)
        
        z = pnp.dot(readouts, W1.T) + b1; h = pnp.maximum(0, z)
        out = pnp.dot(h, W2.T) + b2
        preds = out[:, 0]
        
        y_flat = y_batch.ravel()
        mse = np.mean((preds - y_flat) ** 2)
        return mse

    def train_epoch(self, best_expert_params, qnode, n_x, n_h, hidden_dim, lr_q, lr_c, alpha=0.3):
        """
        Trains the chosen expert on local data for 1 epoch (or subset) using SHADOW DISTILLATION.
        Alpha: Weight of distillation (0.3 means 30% trust in global teacher)
        """
        # Student Params (Will be updated)
        params_q = copy.deepcopy(best_expert_params["params_q"])
        wvec = copy.deepcopy(best_expert_params["wvec"])
        
        # Teacher Params (FROZEN - Global Expert)
        teacher_params_q = best_expert_params["params_q"]
        teacher_wvec = best_expert_params["wvec"]
        
        # Unpack Teacher Weights once for speed
        tW1, tb1, tW2, tb2 = unpack_classical_weights(teacher_wvec, n_h, hidden_dim)
        tW1 = pnp.array(tW1); tb1 = pnp.array(tb1); tW2 = pnp.array(tW2); tb2 = pnp.array(tb2)
        
        q_state = AdamState(params_q, lr=lr_q)
        c_state = AdamState(wvec, lr=lr_c)
        
        # Helper for MLP forward
        def run_mlp(readouts_in, w_vec_in):
            r_arr = pnp.array(readouts_in)
            if r_arr.ndim == 1: r_shp = pnp.reshape(r_arr, (1, -1))
            elif r_arr.shape[0] == n_h and r_arr.ndim == 2: r_shp = pnp.transpose(r_arr)
            else: r_shp = r_arr
            
            U1, u1, U2, u2 = unpack_classical_weights(w_vec_in, n_h, hidden_dim)
            zz = pnp.dot(r_shp, pnp.array(U1).T) + pnp.array(u1)
            hh = pnp.maximum(0, zz)
            oo = pnp.dot(hh, pnp.array(U2).T) + pnp.array(u2)
            return oo[:, 0]

        # Custom Distillation Loss
        def distillation_loss(p_q, w_v, batch_X, batch_y):
            # 1. Student Prediction
            readouts_student = qnode(batch_X, p_q)
            
            # --- Student MLP ---
            out_s = pnp.array(readouts_student)
            if out_s.ndim == 1: out_s = pnp.reshape(out_s, (1, -1))
            elif out_s.shape[0] == n_h and out_s.ndim == 2: out_s = pnp.transpose(out_s)
            
            SW1, Sb1, SW2, Sb2 = unpack_classical_weights(w_v, n_h, hidden_dim)
            sz = pnp.dot(out_s, pnp.array(SW1).T) + pnp.array(Sb1)
            sh = pnp.maximum(0, sz)
            student_pred = pnp.dot(sh, pnp.array(SW2).T) + pnp.array(Sb2)
            student_pred = student_pred[:, 0]

            # 2. Teacher Prediction (No Gradients)
            # Use frozen weights
            readouts_teacher = qnode(batch_X, teacher_params_q)
            
            out_t = pnp.array(readouts_teacher)
            if out_t.ndim == 1: out_t = pnp.reshape(out_t, (1, -1))
            elif out_t.shape[0] == n_h and out_t.ndim == 2: out_t = pnp.transpose(out_t)
            
            tz = pnp.dot(out_t, tW1.T) + tb1
            th = pnp.maximum(0, tz)
            teacher_pred = pnp.dot(th, tW2.T) + tb2
            teacher_pred = teacher_pred[:, 0]
            
            # 3. Combined Loss
            loss_data = pnp.mean((student_pred - pnp.reshape(batch_y, (-1,))) ** 2)
            loss_distill = pnp.mean((student_pred - teacher_pred) ** 2)
            
            return (1 - alpha) * loss_data + alpha * loss_distill
        
        grad_fn = qml.grad(distillation_loss, argnums=[0, 1])
        
        # --- OPTIMIZATION FIX: MATCH BASELINE ---
        # Increased to match Variant C training volume
        MAX_SAMPLES = 2048 
        BATCH_SIZE = 256 
        
        if len(self.X_seq) == 0: return {"params_q": params_q, "wvec": wvec}, 0
        
        # Choose indices
        if len(self.X_seq) > MAX_SAMPLES:
            epoch_idx = np.random.choice(len(self.X_seq), MAX_SAMPLES, replace=False)
        else:
            epoch_idx = np.arange(len(self.X_seq))
            
        n_batches = int(np.ceil(len(epoch_idx) / BATCH_SIZE))
        
        epoch_loss = 0
        # Optional: Outer loop handles tqdm, so we don't need another tqdm here to avoid spam
        # But we iterate over the subset now
        
        start_idx = 0
        perm = np.random.permutation(epoch_idx) # Shuffle the subset
        
        # Inner progress bar for batches
        # disable=False ensures it shows up even if nested
        for b in tqdm(range(n_batches), desc=f"Client {self.client_id}", leave=False):
            batch_idx = perm[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            batch_X, batch_y = self.X_seq[batch_idx], self.y_seq[batch_idx]
            
            g_q, g_c = grad_fn(params_q, wvec, batch_X, batch_y)
            
            params_q, q_state = adam_update(params_q, g_q, q_state)
            wvec, c_state = adam_update(wvec, g_c, c_state)
            
            loss_val = distillation_loss(params_q, wvec, batch_X, batch_y)
            epoch_loss += loss_val
            
        return {"params_q": params_q, "wvec": wvec}, epoch_loss / n_batches

class FederatedServer:
    def __init__(self, moe_path):
        if not os.path.exists(moe_path):
            raise FileNotFoundError(f"MOE Experts not found at {moe_path}")
            
        print(f"Server: Loading experts from {moe_path}...")
        with open(moe_path, "rb") as f:
            self.experts = pickle.load(f)
            # self.experts is {0: {'params_q':..., 'wvec':...}, 1: ...}
            
        # Load scalers from kmeans_model.pkl
        kmeans_path = "kmeans_model.pkl"
        if not os.path.exists(kmeans_path):
             raise FileNotFoundError(f"{kmeans_path} needed for scalers and PCA!")
             
        print(f"Server: Loading scalers from {kmeans_path}...")
        with open(kmeans_path, "rb") as f:
            kdata = pickle.load(f)
            self.scaler_x = kdata["scaler_x"]
            self.pca = kdata.get("pca") # Use .get() to avoid key errors if PCA was None
            
        # Re-initialize Global Y Scaler logic
        from sklearn.preprocessing import QuantileTransformer
        self.scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
            # We'll fit it on dummy data 0..100 to initialize, or fit on all client data in init.
            
    def aggregate_updates(self, updates_by_expert):
        """
        updates_by_expert: { k: [ {params}, {params} ] }
        """
        print("Server: Aggregating updates...")
        for k, update_list in updates_by_expert.items():
            if not update_list: continue
            
            n_updates = len(update_list)
            print(f"  Expert {k}: Aggregating {n_updates} client updates.")
            
            # Simple Average
            new_params_q = np.mean([u["params_q"] for u in update_list], axis=0)
            new_wvec = np.mean([u["wvec"] for u in update_list], axis=0)
            
            self.experts[k] = {"params_q": new_params_q, "wvec": new_wvec}


def run_simulation():
    # 1. Initialize Server
    server = FederatedServer(MOE_PATH)
    
    # 2. Initialize Clients
    client_files = glob.glob(os.path.join(DATA_DIR, "client_*.csv"))
    if not client_files:
        print(f"ERROR: No client files found in '{DATA_DIR}'!")
        print("Please run 'python process_sqlite_to_federated.py' first to generate the cluster CSVs.")
        sys.exit(1)
        
    # 2b. Load Global Test Set + get split_time for client filtering
    # Fit Global Y Scaler on ALL client data first
    all_y = []
    temp_clients = []
    for f_path in client_files:
        cid = os.path.basename(f_path).replace("client_", "").replace(".csv", "")
        c = FederatedClient(cid, f_path)
        temp_clients.append(c)
        all_y.append(c.y_raw)
    
    print("Fitting Global Y Scaler...")
    y_concat = np.concatenate(all_y)
    server.scaler_y.fit(y_concat)
    
    X_test, y_test, c_test, split_time, n_clusters, eval_scaler_y = load_global_test_set(
        server.scaler_x, server.pca, seq_len=SEQ_LEN
    )
    
    # 2c. Filter client data by split_time & prepare
    clients = []
    print(f"\nFiltering & preparing {len(temp_clients)} clients (cutoff: {split_time})...")
    for c in tqdm(temp_clients, desc="  Loading Clients", colour='yellow'):
        # TEMPORAL FILTER: Drop rows after split_time
        if "datetime" in c.df.columns:
            pass  # df is already feature-only; use y_raw's original df
        # We need the raw df to filter by time. Re-load with timestamp.
        df_raw = pd.read_csv(c.data_path, index_col=0)
        if "WindowStart" in df_raw.columns:
            df_raw["datetime"] = pd.to_datetime(df_raw["WindowStart"], unit='s')
            n_before = len(df_raw)
            df_raw = df_raw[df_raw["datetime"] <= split_time]
            n_after = len(df_raw)
            # Update client's data
            c.y_raw = df_raw["TrueCPUUtil"].values.reshape(-1, 1)
            req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                        "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            c.df = df_raw[req_cols].copy()
        elif "datetime" in df_raw.columns:
            df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
            n_before = len(df_raw)
            df_raw = df_raw[df_raw["datetime"] <= split_time]
            n_after = len(df_raw)
            c.y_raw = df_raw["TrueCPUUtil"].values.reshape(-1, 1)
            req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                        "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            c.df = df_raw[req_cols].copy()
        else:
            n_before = len(c.df)
            n_after = n_before  # No timestamp to filter
        
        if len(c.df) < SEQ_LEN + 1:
            continue
            
        c.prepare_batches(server.scaler_x, pca=server.pca)
        c.create_dataset(server.scaler_y, seq_len=SEQ_LEN)
        
        if len(c.X_seq) >= 10:
            clients.append(c)
    
    print(f"  {len(clients)} clients ready (filtered to training period)")

    # Compute adaptive distillation alpha per expert (inversely proportional to variance)
    cluster_variances = {}
    for c in clients:
        # Determine which expert this client would most likely belong to
        # Use y_raw variance as a proxy for cluster volatility
        var = float(np.var(c.y_raw))
        # We'll aggregate variances after audition in the loop, but for initialization
        # compute from all clients assigned to each expert in round 1
        cluster_variances[c.client_id] = var

    # We'll compute cluster_alpha dynamically per round after audition,
    # but set initial defaults based on global variance
    all_vars = list(cluster_variances.values())
    min_var = min(all_vars) if all_vars else 1.0
    print(f"  Client variance range: [{min(all_vars):.6f}, {max(all_vars):.6f}]")
    
    # 3. Build QNode (Reused across clients for simulation speed)
    print("Building QNode...")
    n_features = 9 # original features
    # But wait, did we use PCA? In aqrnn_cluster, n_components=4.
    # Check `kmeans_model.pkl` content.
    if server.pca is not None:
        n_x = server.pca.n_components_
    else:
        n_x = n_features
        
    # Initialize Cell to get QNode
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN, n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)
    
    # --- LOAD BALANCING CONFIG ---
    # Hard cap: max clients per expert per round
    EXPERT_CAPACITY = 5
    
    # 4. Simulation Loop
    t_start = time.perf_counter()
    for r in range(N_ROUNDS):
        print(f"\n=== ROUND {r+1}/{N_ROUNDS} ===")
        
        # Storage for updates: { expert_id: [params, ...] }
        updates = {k: [] for k in server.experts.keys()}
        cluster_choices = {k: 0 for k in server.experts.keys()}
        
        # =========================================================
        # BATCH ASSIGNMENT: Hard-Cap with Second-Best Margin Reassignment
        # =========================================================
        # Step 1: Evaluate ALL clients against ALL experts
        print("Evaluating all clients against all experts...")
        client_losses = {}  # { client_id: { expert_k: loss } }
        
        for c in tqdm(clients, desc="Expert Evaluation"):
            losses_for_c = {}
            for k, expert_params in server.experts.items():
                loss = c.evaluate_expert(expert_params, qnode, n_x, N_H, HIDDEN_DIM)
                losses_for_c[k] = loss
            client_losses[c.client_id] = losses_for_c
        
        # Step 2: Compute best expert and regret for each client
        client_best = {}  # { client_id: (best_k, best_loss) }
        client_regrets = {}  # { client_id: { expert_k: regret } }
        
        for cid, losses in client_losses.items():
            best_k = min(losses, key=losses.get)
            best_loss = losses[best_k]
            client_best[cid] = (best_k, best_loss)
            
            # Regret = L_c,k - L_c,k* (how much worse than best)
            regrets = {k: losses[k] - best_loss for k in losses}
            client_regrets[cid] = regrets
        
        # Step 3: Batch Assignment with Capacity Constraints
        expert_capacity_remaining = {k: EXPERT_CAPACITY for k in server.experts.keys()}
        client_assignment = {}  # { client_id: assigned_expert_k }
        
        # Sort clients by their "flexibility" (those with smallest regret gap should be assigned first)
        # Actually, for optimal assignment: assign least flexible clients first
        # "Flexibility" = difference between best regret (0) and second-best regret
        def get_flexibility(cid):
            regrets = client_regrets[cid]
            sorted_regrets = sorted(regrets.values())
            if len(sorted_regrets) >= 2:
                return sorted_regrets[1] - sorted_regrets[0]  # Gap to second-best
            return 0
        
        # Assign least flexible (highest gap) clients first - they suffer most from reassignment
        sorted_clients = sorted(clients, key=lambda c: get_flexibility(c.client_id), reverse=True)
        
        for c in sorted_clients:
            cid = c.client_id
            best_k, _ = client_best[cid]
            
            if expert_capacity_remaining[best_k] > 0:
                # Best expert has capacity - assign directly
                client_assignment[cid] = best_k
                expert_capacity_remaining[best_k] -= 1
            else:
                # Best expert is FULL - find expert with minimal regret among available
                available_experts = [k for k, cap in expert_capacity_remaining.items() if cap > 0]
                
                if available_experts:
                    # Assign to the available expert with smallest regret
                    regrets = client_regrets[cid]
                    assigned_k = min(available_experts, key=lambda k: regrets[k])
                    client_assignment[cid] = assigned_k
                    expert_capacity_remaining[assigned_k] -= 1
                else:
                    # All experts full - force assign to best anyway (overflow)
                    print(f"  WARNING: All experts at capacity, client {cid} forced to expert {best_k}")
                    client_assignment[cid] = best_k
        
        # Log assignment stats
        print("Expert Assignment (after load balancing):")
        for k in server.experts.keys():
            assigned_count = sum(1 for v in client_assignment.values() if v == k)
            cluster_choices[k] = assigned_count
            print(f"  Expert {k}: {assigned_count} clients")
        
        # Step 4: Compute adaptive alpha per expert (variance-based)
        expert_alphas = {}
        for k in server.experts.keys():
            assigned_clients = [c for c in clients if client_assignment[c.client_id] == k]
            if assigned_clients:
                expert_var = float(np.mean([np.var(c.y_raw) for c in assigned_clients]))
                # alpha = ALPHA_BASE * (min_var / expert_var), clamped to [ALPHA_MIN, ALPHA_BASE]
                expert_alphas[k] = max(ALPHA_MIN, ALPHA_BASE * (min_var / max(expert_var, 1e-8)))
            else:
                expert_alphas[k] = ALPHA_BASE
            print(f"  Expert {k}: alpha={expert_alphas[k]:.3f}")

        # Step 5: Train each client on their ASSIGNED expert
        for c in tqdm(clients, desc="Clients Training"):
            assigned_k = client_assignment[c.client_id]
            
            # Train with adaptive alpha
            chosen_params = server.experts[assigned_k]
            alpha_k = expert_alphas[assigned_k]
            updated_params, train_loss = c.train_epoch(chosen_params, qnode, n_x, N_H, HIDDEN_DIM, LR_Q, LR_C, alpha=alpha_k)
            
            # Step C: Upload update
            updates[assigned_k].append(updated_params)
            
        print("Round Summary (Client Choices):")
        for k, count in cluster_choices.items():
            print(f"  Expert {k}: chosen by {count} clients")
            
        # Server Aggregates
        server.aggregate_updates(updates)
        
    train_time = time.perf_counter() - t_start
    print(f"\nSimulation Complete. Training Time: {train_time:.1f}s")
    
    # Save Final Models
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(server.experts, f)
    print(f"Saved federated models to {RESULTS_PATH}")
    
    # ══════════════════════════════════════════════════════════════
    # FINAL EVALUATION on Global Test Set
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION (Global Test Set)")
    print("=" * 70)
    
    forget_vec = 0.95 * np.ones(SEQ_LEN) if USE_FORGET_GATE else None
    
    t_infer = time.perf_counter()
    all_preds = np.zeros(len(X_test))
    all_mask  = np.zeros(len(X_test), dtype=bool)
    
    cluster_metrics = {}
    for k in tqdm(range(n_clusters), desc="  MoE Test Routing", colour='blue'):
        mask = (c_test == k)
        if not np.any(mask) or k not in server.experts:
            continue
        
        ep = server.experts[k]
        X_k = X_test[mask]
        preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                       forget_vec, X_k, label=f"Cluster {k} Test")
        
        all_preds[mask] = preds_k
        all_mask[mask] = True
        
        # Per-cluster metrics
        yt_k = y_test.ravel()[mask]
        yt_orig = eval_scaler_y.inverse_transform(yt_k.reshape(-1, 1)).ravel()
        yp_orig = eval_scaler_y.inverse_transform(preds_k.reshape(-1, 1)).ravel()
        rmse_k = float(np.sqrt(mean_squared_error(yt_orig, yp_orig)))
        mae_k = float(mean_absolute_error(yt_orig, yp_orig))
        smape_k = smape(yt_orig, yp_orig)
        cluster_metrics[f"cluster_{k}"] = {
            "rmse": rmse_k, "mae": mae_k, "smape": smape_k, "n": int(mask.sum())
        }
        print(f"  Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
              f"SMAPE={smape_k:.1f}%  (n={int(mask.sum())})")
    
    infer_time = time.perf_counter() - t_infer
    
    # Overall metrics
    y_used = y_test.ravel()[all_mask]
    p_used = all_preds[all_mask]
    y_orig = eval_scaler_y.inverse_transform(y_used.reshape(-1, 1)).ravel()
    p_orig = eval_scaler_y.inverse_transform(p_used.reshape(-1, 1)).ravel()
    
    rmse_overall = float(np.sqrt(mean_squared_error(y_orig, p_orig)))
    mae_overall  = float(mean_absolute_error(y_orig, p_orig))
    smape_overall = smape(y_orig, p_orig)
    
    print(f"\n  ╔═══ CFL FINAL RESULTS ═══╗")
    print(f"  ║ RMSE  (orig) = {rmse_overall:.4f}")
    print(f"  ║ MAE   (orig) = {mae_overall:.4f}")
    print(f"  ║ SMAPE (orig) = {smape_overall:.1f}%")
    print(f"  ║ Train Time   = {train_time:.1f}s")
    print(f"  ║ Infer Time   = {infer_time:.3f}s")
    print(f"  ╚══════════════════════════╝")
    
    # Save to JSON
    results = {
        "method": "Federated_CFL",
        "n_rounds": N_ROUNDS,
        "client_epochs": CLIENT_EPOCHS,
        "rmse_orig": rmse_overall,
        "mae_orig": mae_overall,
        "smape_orig": smape_overall,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 2),
        "per_cluster": cluster_metrics,
    }
    with open("federated_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Metrics saved to federated_results.json")


if __name__ == "__main__":
    run_simulation()
