"""
federated_solution_v2.py
========================
Enhanced Clustered Federated Learning for AQRNN Cloud Workload Prediction.

Enhancements over v1:
  1. DYNAMIC MoE        – Experts split/merge based on demand & error signals.
  2. ADAPTIVE COMPRESSION – TopK sparsification with adaptive K per round.
  3. DIFFERENTIAL PRIVACY – Gaussian mechanism (clip + noise) on client deltas.

Authors : Krish et al.
Requires: aqrnn.py, moe_experts.pkl, kmeans_model.pkl, federated_data/client_*.csv
"""

import os, sys, copy, json, time, math, pickle, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.getcwd())
try:
    from aqrnn import (
        AQRNNCell, AdamState, adam_update,
        pack_classical_weights, unpack_classical_weights
    )
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.cluster import KMeans
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as e:
    print(f"Import error: {e}"); sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
DATA_DIR          = "federated_data"
MOE_PATH          = "moe_experts.pkl"
RESULTS_PATH      = "federated_v2_results.pkl"
CSV_PATH          = "grid5000_hybrid_clean.csv"

# FL Core
N_ROUNDS          = 20
BATCH_SIZE        = 256
MAX_SAMPLES       = 2048
LR_Q              = 1e-3
LR_C              = 1e-3
SEQ_LEN           = 2
N_H               = 4
HIDDEN_DIM        = 64
N_LAYERS          = 1
PARAM_SHARING     = True
DEVICE_MODE       = "cuda"
USE_FORGET_GATE   = True
GRAD_CLIP         = 1.0

# Server-side learning rate with cosine decay
# Starts aggressive for fast early convergence, decays to prevent late-round divergence
SERVER_LR_INIT    = 0.5       # Starting server LR
SERVER_LR_MIN     = 0.2       # Floor — never go below this

# Distillation
ALPHA_BASE        = 0.3
ALPHA_MIN         = 0.05

# --- [FEATURE 1] DYNAMIC MoE ---
DYNAMIC_MOE       = True
SPLIT_ERROR_THR   = 2.0      # Split if expert RMSE > global_mean * this (was 1.5 — too aggressive)
SPLIT_MIN_CLIENTS = 4        # Must have at least this many clients to consider split
MERGE_MAX_CLIENTS = 1        # Merge if expert gets <= this many clients for 2 rounds
MAX_EXPERTS       = 5        # Upper cap on total experts
MIN_EXPERTS       = 2        # Lower cap on total experts
MERGE_PATIENCE    = 3        # Consecutive underutilized rounds before merge (was 2 — too eager)
MOE_WARMUP_ROUNDS = 4        # No split/merge in first N rounds (let model stabilize)

# --- [FEATURE 2] ADAPTIVE COMPRESSION ---
COMPRESSION       = True
TOPK_INIT         = 0.6      # Initial: send top 60% of delta values (was 0.5)
TOPK_MIN          = 0.2      # Floor: never go below 20% (was 0.1 — too aggressive)
TOPK_MAX          = 0.8      # Ceiling: cap boost at 80% (NEW — prevents disabling compression)
TOPK_DECAY        = 0.95     # Per-round decay (was 0.9 — slower decay = more stable)
TOPK_ERROR_BOOST  = 1.1      # Boost on error increase (was 1.3 — caused spiral)

# --- [FEATURE 3] DIFFERENTIAL PRIVACY ---
DP_ENABLED        = True
DP_CLIP_NORM      = 0.7      # Max L2-norm of each client's delta (was 1.0 — tighter = less signal but less noise needed)
DP_NOISE_MULT     = 0.25     # ε ∝ 1/noise_mult. Was 0.1 → gave ε≈153. Now 0.25 → gives ε≈61 (was 0.1)
DP_DELTA          = 1e-5     # δ parameter for (ε, δ)-DP accounting

# Load-balancing
EXPERT_CAPACITY   = 5


# ═══════════════════════════════════════════════════════════════
#  [FEATURE 3] DIFFERENTIAL PRIVACY UTILITIES
# ═══════════════════════════════════════════════════════════════

def dp_clip_delta(delta_dict, max_norm):
    """
    Clip the L2 norm of a client's parameter delta.
    delta_dict = {'params_q': np.array, 'wvec': np.array}
    Returns clipped delta_dict (new copy).
    """
    # Flatten all params into one vector to compute global norm
    flat = np.concatenate([delta_dict['params_q'].ravel(),
                           delta_dict['wvec'].ravel()])
    norm = float(np.linalg.norm(flat))
    clip_factor = min(1.0, max_norm / max(norm, 1e-12))

    return {
        'params_q': delta_dict['params_q'] * clip_factor,
        'wvec':     delta_dict['wvec']     * clip_factor,
    }


def dp_add_noise(aggregated_delta, n_clients, clip_norm, noise_mult, rng=None):
    """
    Add calibrated Gaussian noise AFTER aggregation (central DP model).
    σ = clip_norm * noise_mult / n_clients
    Noise is added to the *mean* delta, so we scale by 1/n_clients.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sigma = clip_norm * noise_mult / max(n_clients, 1)

    noised = {}
    for key in ('params_q', 'wvec'):
        noise = rng.normal(0.0, sigma, size=aggregated_delta[key].shape).astype(
            aggregated_delta[key].dtype)
        noised[key] = aggregated_delta[key] + noise
    return noised, sigma


def compute_dp_epsilon(noise_mult, delta=DP_DELTA, n_rounds=N_ROUNDS):
    """
    Correct (ε, δ)-DP bound via Gaussian mechanism + advanced composition.

    Key insight: for the AVERAGE of n clipped deltas:
      - Sensitivity Δf = clip_norm / n_clients
      - Noise σ         = clip_norm * noise_mult / n_clients
      - These share the same (clip_norm / n_clients) factor, so they cancel:
        ε_round = Δf * sqrt(2·ln(1.25/δ)) / σ = sqrt(2·ln(1.25/δ)) / noise_mult

    ε depends ONLY on noise_mult and δ, not on clip_norm or n_clients.
    Composed over T rounds via advanced composition: ε_total ≈ ε_round * sqrt(T).
    """
    if noise_mult < 1e-12:
        return float('inf')
    eps_round = math.sqrt(2.0 * math.log(1.25 / delta)) / noise_mult
    eps_total = eps_round * math.sqrt(n_rounds)  # advanced composition
    return eps_total


# ═══════════════════════════════════════════════════════════════
#  [FEATURE 2] ADAPTIVE COMPRESSION UTILITIES
# ═══════════════════════════════════════════════════════════════

def topk_compress(delta_dict, k_ratio):
    """
    TopK sparsification: keep only top k_ratio fraction of values by magnitude.
    Returns: compressed dict with (indices, values, shape) per key.
    """
    compressed = {}
    total_original = 0
    total_sent = 0
    for key in ('params_q', 'wvec'):
        flat = delta_dict[key].ravel()
        total_original += flat.size

        n_keep = max(1, int(len(flat) * k_ratio))
        # Get indices of top-K absolute values
        abs_vals = np.abs(flat)
        topk_idx = np.argpartition(abs_vals, -n_keep)[-n_keep:]
        topk_vals = flat[topk_idx]

        total_sent += n_keep
        compressed[key] = {
            'indices': topk_idx.astype(np.int32),
            'values':  topk_vals,
            'shape':   delta_dict[key].shape,
            'dtype':   delta_dict[key].dtype,
        }
    ratio = total_sent / max(total_original, 1) * 100
    return compressed, ratio


def topk_decompress(compressed_dict):
    """Reconstruct full delta arrays from compressed representation."""
    delta = {}
    for key in ('params_q', 'wvec'):
        c = compressed_dict[key]
        full = np.zeros(np.prod(c['shape']), dtype=c['dtype'])
        full[c['indices']] = c['values']
        delta[key] = full.reshape(c['shape'])
    return delta


def adaptive_topk_ratio(current_k, prev_error, curr_error, decay, error_boost,
                        k_min, k_max=TOPK_MAX):
    """
    Adapt TopK ratio based on error trend:
    - If error decreased: decay K (compress more aggressively)
    - If error increased: boost K (send more information), capped at k_max
    """
    if curr_error < prev_error:
        new_k = current_k * decay
    else:
        new_k = current_k * error_boost
    return max(k_min, min(new_k, k_max))


# ═══════════════════════════════════════════════════════════════
#  [FEATURE 1] DYNAMIC MoE UTILITIES
# ═══════════════════════════════════════════════════════════════

def should_split_expert(expert_id, expert_rmse, global_mean_rmse,
                        n_clients, expert_count):
    """Check if an expert should be SPLIT into two."""
    if expert_count >= MAX_EXPERTS:
        return False
    if n_clients < SPLIT_MIN_CLIENTS:
        return False
    if expert_rmse > global_mean_rmse * SPLIT_ERROR_THR:
        return True
    return False


def split_expert(experts, expert_id, perturbation_scale=0.05, rng=None):
    """
    Split one expert into two by cloning + symmetric perturbation.
    Returns new expert_id for the spawned expert.
    """
    if rng is None:
        rng = np.random.default_rng()

    parent = experts[expert_id]
    new_id = max(experts.keys()) + 1

    # Child A stays at expert_id (slightly perturbed +)
    # Child B gets new_id (slightly perturbed -)
    noise_q = rng.normal(0, perturbation_scale,
                         size=parent['params_q'].shape).astype(parent['params_q'].dtype)
    noise_w = rng.normal(0, perturbation_scale,
                         size=parent['wvec'].shape).astype(parent['wvec'].dtype)

    experts[expert_id] = {
        'params_q': parent['params_q'] + noise_q,
        'wvec':     parent['wvec']     + noise_w,
    }
    experts[new_id] = {
        'params_q': parent['params_q'] - noise_q,
        'wvec':     parent['wvec']     - noise_w,
    }
    return new_id


def should_merge_expert(expert_id, n_clients, underutil_tracker, expert_count):
    """Check if an expert should be MERGED due to persistent underutilization."""
    if expert_count <= MIN_EXPERTS:
        return False
    if n_clients <= MERGE_MAX_CLIENTS:
        underutil_tracker[expert_id] = underutil_tracker.get(expert_id, 0) + 1
        if underutil_tracker[expert_id] >= MERGE_PATIENCE:
            return True
    else:
        underutil_tracker[expert_id] = 0
    return False


def merge_experts(experts, remove_id, merge_into_id):
    """
    Merge remove_id INTO merge_into_id by averaging their parameters.
    Deletes remove_id from experts dict.
    """
    a = experts[merge_into_id]
    b = experts[remove_id]
    experts[merge_into_id] = {
        'params_q': (a['params_q'] + b['params_q']) / 2.0,
        'wvec':     (a['wvec']     + b['wvec'])     / 2.0,
    }
    del experts[remove_id]


# ═══════════════════════════════════════════════════════════════
#  CORE HELPERS (from v1, kept intact)
# ═══════════════════════════════════════════════════════════════

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


def predict_with_params(qnode, params_q, wvec, forget_vec, X,
                        n_h=N_H, hidden_dim=HIDDEN_DIM,
                        chunk_size=256, label="Inference"):
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


# ═══════════════════════════════════════════════════════════════
#  GLOBAL TEST SET LOADER
# ═══════════════════════════════════════════════════════════════

def load_global_test_set(scaler_x, pca_obj, seq_len=SEQ_LEN):
    print("\n  Loading Global Test Set from grid5000_hybrid_clean.csv...")
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

    X_scaled = scaler_x.transform(X_raw)
    if pca_obj is not None:
        X_scaled = pca_obj.transform(X_scaled)

    # Cluster assignment
    with open("kmeans_model.pkl", "rb") as f:
        kdata = pickle.load(f)
    kmeans = kdata.get("kmeans")
    clusters = kmeans.predict(X_scaled) if kmeans is not None else np.zeros(len(X_scaled), dtype=int)

    X_seq, y_seq_raw, c_seq = [], [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i:i+seq_len])
        y_seq_raw.append(y_raw[i+seq_len])
        c_seq.append(clusters[i+seq_len-1])
    X_seq     = np.array(X_seq)
    y_seq_raw = np.array(y_seq_raw)
    c_seq     = np.array(c_seq)

    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end   = train_end + int(N * 0.15)
    split_time = df.index[train_end]

    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    scaler_y.fit(y_seq_raw[:train_end])
    y_seq = scaler_y.transform(y_seq_raw)

    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    c_test = c_seq[val_end:]

    n_clusters = len(np.unique(c_test))
    print(f"  Split Time: {split_time}")
    print(f"  Test Set: {len(X_test):,} samples | {n_clusters} clusters")
    return X_test, y_test, c_test, split_time, n_clusters, scaler_y


# ═══════════════════════════════════════════════════════════════
#  FEDERATED CLIENT (Enhanced)
# ═══════════════════════════════════════════════════════════════

class FederatedClientV2:
    def __init__(self, client_id, csv_path):
        self.client_id = client_id
        self.data_path = csv_path
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path, index_col=0)
        req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                    "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        self.y_raw = df["TrueCPUUtil"].values.reshape(-1, 1)
        self.df = df[req_cols].copy()

    def prepare_batches(self, scaler_x, pca=None):
        X_raw = self.df.values
        X_scaled = scaler_x.transform(X_raw)
        if pca is not None:
            X_scaled = pca.transform(X_scaled)
        self.X_scaled = X_scaled

    def create_dataset(self, scaler_y, seq_len=SEQ_LEN):
        y_scaled = scaler_y.transform(self.y_raw)
        Xs, ys = [], []
        for i in range(len(self.X_scaled) - seq_len):
            Xs.append(self.X_scaled[i:i+seq_len])
            ys.append(y_scaled[i+seq_len])
        self.X_seq = np.array(Xs)
        self.y_seq = np.array(ys)

    def evaluate_expert(self, expert_params, qnode, n_x, n_h, hidden_dim):
        if len(self.X_seq) == 0:
            return 1e9
        idx = np.random.choice(len(self.X_seq), min(256, len(self.X_seq)), replace=False)
        X_batch = self.X_seq[idx]
        y_batch = self.y_seq[idx]

        readouts_t = qnode(X_batch, expert_params["params_q"])
        readouts = _readouts_to_array(readouts_t, n_h, len(X_batch))
        W1, b1, W2, b2 = unpack_classical_weights(expert_params["wvec"], n_h, hidden_dim)
        z = pnp.dot(pnp.array(readouts), pnp.array(W1).T) + pnp.array(b1)
        h = pnp.maximum(0, z)
        out = pnp.dot(h, pnp.array(W2).T) + pnp.array(b2)
        preds = out[:, 0]
        return float(np.mean((preds - y_batch.ravel()) ** 2))

    def train_epoch(self, global_expert_params, qnode, n_x, n_h, hidden_dim,
                    lr_q, lr_c, alpha=0.3):
        """
        Train on local data with shadow distillation.
        Returns: (delta_dict, train_loss)
            delta_dict = {'params_q': Δq, 'wvec': Δw}  (update DELTAS, not full params)
        """
        # Student = copy of global (will be updated)
        params_q = copy.deepcopy(global_expert_params["params_q"])
        wvec     = copy.deepcopy(global_expert_params["wvec"])

        # Teacher = frozen global
        teacher_pq = global_expert_params["params_q"]
        teacher_wv = global_expert_params["wvec"]
        tW1, tb1, tW2, tb2 = unpack_classical_weights(teacher_wv, n_h, hidden_dim)
        tW1 = pnp.array(tW1); tb1 = pnp.array(tb1)
        tW2 = pnp.array(tW2); tb2 = pnp.array(tb2)

        q_state = AdamState(params_q, lr=lr_q)
        c_state = AdamState(wvec,     lr=lr_c)

        def distillation_loss(p_q, w_v, batch_X, batch_y):
            # Student forward
            out_s = pnp.array(qnode(batch_X, p_q))
            if out_s.ndim == 1: out_s = pnp.reshape(out_s, (1, -1))
            elif out_s.shape[0] == n_h and out_s.ndim == 2: out_s = pnp.transpose(out_s)
            SW1, Sb1, SW2, Sb2 = unpack_classical_weights(w_v, n_h, hidden_dim)
            sz = pnp.dot(out_s, pnp.array(SW1).T) + pnp.array(Sb1)
            student_pred = pnp.dot(pnp.maximum(0, sz), pnp.array(SW2).T) + pnp.array(Sb2)
            student_pred = student_pred[:, 0]

            # Teacher forward (frozen)
            out_t = pnp.array(qnode(batch_X, teacher_pq))
            if out_t.ndim == 1: out_t = pnp.reshape(out_t, (1, -1))
            elif out_t.shape[0] == n_h and out_t.ndim == 2: out_t = pnp.transpose(out_t)
            tz = pnp.dot(out_t, tW1.T) + tb1
            teacher_pred = pnp.dot(pnp.maximum(0, tz), tW2.T) + tb2
            teacher_pred = teacher_pred[:, 0]

            loss_data    = pnp.mean((student_pred - pnp.reshape(batch_y, (-1,))) ** 2)
            loss_distill = pnp.mean((student_pred - teacher_pred) ** 2)
            return (1 - alpha) * loss_data + alpha * loss_distill

        grad_fn = qml.grad(distillation_loss, argnums=[0, 1])

        if len(self.X_seq) == 0:
            return {'params_q': np.zeros_like(params_q),
                    'wvec':     np.zeros_like(wvec)}, 0.0

        n_use = min(MAX_SAMPLES, len(self.X_seq))
        epoch_idx = (np.random.choice(len(self.X_seq), n_use, replace=False)
                     if len(self.X_seq) > n_use else np.arange(len(self.X_seq)))
        perm = np.random.permutation(epoch_idx)
        n_batches = int(np.ceil(len(perm) / BATCH_SIZE))

        epoch_loss = 0.0
        for b in tqdm(range(n_batches), desc=f"Client {self.client_id}", leave=False):
            bi = perm[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            bX, bY = self.X_seq[bi], self.y_seq[bi]

            g_q, g_c = grad_fn(params_q, wvec, bX, bY)
            if GRAD_CLIP > 0:
                g_q = np.array(g_q); g_c = np.array(g_c)
                nq = np.linalg.norm(g_q)
                nc = np.linalg.norm(g_c)
                if nq > GRAD_CLIP: g_q = g_q * (GRAD_CLIP / nq)
                if nc > GRAD_CLIP: g_c = g_c * (GRAD_CLIP / nc)

            params_q, q_state = adam_update(params_q, g_q, q_state)
            wvec,     c_state = adam_update(wvec,     g_c, c_state)
            epoch_loss += float(distillation_loss(params_q, wvec, bX, bY))

        # *** Return DELTAS (not full params) — needed for DP clipping + compression ***
        delta = {
            'params_q': params_q - global_expert_params["params_q"],
            'wvec':     wvec     - global_expert_params["wvec"],
        }
        return delta, epoch_loss / max(n_batches, 1)


# ═══════════════════════════════════════════════════════════════
#  FEDERATED SERVER (Enhanced)
# ═══════════════════════════════════════════════════════════════

class FederatedServerV2:
    def __init__(self, moe_path):
        if not os.path.exists(moe_path):
            raise FileNotFoundError(f"MOE experts not found: {moe_path}")

        print(f"Server: Loading experts from {moe_path}...")
        with open(moe_path, "rb") as f:
            self.experts = pickle.load(f)

        print(f"Server: Loading scalers from kmeans_model.pkl...")
        with open("kmeans_model.pkl", "rb") as f:
            kdata = pickle.load(f)
            self.scaler_x = kdata["scaler_x"]
            self.pca      = kdata.get("pca")

        self.scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

        # Track underutilization for merge decisions
        self.underutil_tracker = {}
        # DP RNG (separate from training RNG for reproducibility)
        self.dp_rng = np.random.default_rng(seed=7)

    def aggregate_updates(self, deltas_by_expert, round_num, server_lr):
        """
        Enhanced aggregation pipeline:
          1. Decompress (if compression enabled)
          2. DP-clip each delta
          3. Average deltas
          4. Add DP noise to mean delta
          5. Apply mean delta to global expert (with decaying server LR)
        """
        dp_log = {}

        for k, delta_list in deltas_by_expert.items():
            if not delta_list:
                continue

            n_clients = len(delta_list)
            print(f"  Expert {k}: Aggregating {n_clients} deltas", end="")

            # --- Step 1: Decompress ---
            if COMPRESSION:
                delta_list = [topk_decompress(d) for d in delta_list]

            # --- Step 2: DP Clip each delta ---
            if DP_ENABLED:
                delta_list = [dp_clip_delta(d, DP_CLIP_NORM) for d in delta_list]

            # --- Step 3: Average ---
            mean_delta = {
                'params_q': np.mean([d['params_q'] for d in delta_list], axis=0),
                'wvec':     np.mean([d['wvec']     for d in delta_list], axis=0),
            }

            # --- Step 4: Add DP noise ---
            if DP_ENABLED:
                mean_delta, sigma = dp_add_noise(
                    mean_delta, n_clients, DP_CLIP_NORM, DP_NOISE_MULT,
                    rng=self.dp_rng)
                eps = compute_dp_epsilon(DP_NOISE_MULT, DP_DELTA, round_num + 1)
                dp_log[k] = {'sigma': sigma, 'epsilon_so_far': eps, 'noise_mult': DP_NOISE_MULT}
                print(f" | DP: σ={sigma:.4f}, ε≤{eps:.2f}", end="")

            # --- Step 5: Apply delta to global expert (with decaying server LR) ---
            self.experts[k] = {
                'params_q': self.experts[k]['params_q'] + server_lr * mean_delta['params_q'],
                'wvec':     self.experts[k]['wvec']     + server_lr * mean_delta['wvec'],
            }
            print(f" | server_lr={server_lr:.3f}", end="")
            print()  # newline

        return dp_log

    def dynamic_moe_step(self, expert_errors, expert_client_counts, round_num, rng=None):
        """
        After each round: decide if any expert should split or merge.
        expert_errors  = {k: RMSE_on_local_eval}
        expert_client_counts = {k: n_clients_assigned}
        """
        if not DYNAMIC_MOE:
            return []

        # Warmup: no structural changes in early rounds (let model stabilize)
        if round_num < MOE_WARMUP_ROUNDS:
            return [f"(warmup: round {round_num+1} < {MOE_WARMUP_ROUNDS}, skipping split/merge)"]

        actions = []
        current_count = len(self.experts)
        all_errors = [v for v in expert_errors.values() if v > 0]
        mean_error = np.mean(all_errors) if all_errors else 1.0

        # --- CHECK SPLITS ---
        keys_snapshot = list(self.experts.keys())
        for k in keys_snapshot:
            err = expert_errors.get(k, 0)
            nc  = expert_client_counts.get(k, 0)
            if should_split_expert(k, err, mean_error, nc, current_count):
                new_id = split_expert(self.experts, k, rng=rng)
                current_count += 1
                actions.append(f"SPLIT expert {k} → {k} + {new_id} "
                               f"(RMSE={err:.4f} > {mean_error*SPLIT_ERROR_THR:.4f}, "
                               f"clients={nc})")
                # Reset underutil tracker for both
                self.underutil_tracker.pop(k, None)
                self.underutil_tracker.pop(new_id, None)

        # --- CHECK MERGES ---
        keys_snapshot = list(self.experts.keys())
        for k in keys_snapshot:
            if k not in self.experts:
                continue
            nc = expert_client_counts.get(k, 0)
            if should_merge_expert(k, nc, self.underutil_tracker, current_count):
                # Find closest expert to merge into (by param similarity)
                best_target = None
                best_dist   = float('inf')
                for other_k in self.experts:
                    if other_k == k:
                        continue
                    dist = (np.linalg.norm(self.experts[k]['params_q'] -
                                           self.experts[other_k]['params_q']) +
                            np.linalg.norm(self.experts[k]['wvec'] -
                                           self.experts[other_k]['wvec']))
                    if dist < best_dist:
                        best_dist = dist
                        best_target = other_k

                if best_target is not None:
                    merge_experts(self.experts, remove_id=k, merge_into_id=best_target)
                    current_count -= 1
                    self.underutil_tracker.pop(k, None)
                    actions.append(f"MERGE expert {k} → {best_target} "
                                   f"(clients={nc}, patience hit)")

        return actions


# ═══════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════

def run_simulation_v2():
    print("=" * 70)
    print("  FEDERATED AQRNN v2")
    print("  Dynamic MoE | Adaptive Compression | Differential Privacy")
    print("=" * 70)
    feature_status = (f"  Dynamic MoE: {'ON' if DYNAMIC_MOE else 'OFF'} | "
                      f"Compression: {'ON' if COMPRESSION else 'OFF'} | "
                      f"DP: {'ON' if DP_ENABLED else 'OFF'}")
    print(feature_status)
    if DP_ENABLED:
        print(f"  DP Config: clip={DP_CLIP_NORM}, noise_mult={DP_NOISE_MULT}, δ={DP_DELTA}")
    if COMPRESSION:
        print(f"  Compression Config: TopK init={TOPK_INIT}, min={TOPK_MIN}, max={TOPK_MAX}, decay={TOPK_DECAY}")
    print(f"  Server LR: {SERVER_LR_INIT}→{SERVER_LR_MIN} (cosine decay) | MoE Warmup: {MOE_WARMUP_ROUNDS} rounds")
    print()

    # 1. Server Init
    server = FederatedServerV2(MOE_PATH)
    rng = np.random.default_rng(42)

    # 2. Load Clients
    client_files = glob.glob(os.path.join(DATA_DIR, "client_*.csv"))
    if not client_files:
        print(f"ERROR: No client files in '{DATA_DIR}/'")
        sys.exit(1)

    temp_clients = []
    all_y = []
    for fp in client_files:
        cid = os.path.basename(fp).replace("client_", "").replace(".csv", "")
        c = FederatedClientV2(cid, fp)
        temp_clients.append(c)
        all_y.append(c.y_raw)

    print("Fitting Global Y Scaler...")
    server.scaler_y.fit(np.concatenate(all_y))

    X_test, y_test, c_test, split_time, n_clusters_test, eval_scaler_y = \
        load_global_test_set(server.scaler_x, server.pca, seq_len=SEQ_LEN)

    # Filter clients by split_time
    clients = []
    print(f"\nFiltering clients (cutoff: {split_time})...")
    for c in tqdm(temp_clients, desc="  Loading Clients", colour='yellow'):
        df_raw = pd.read_csv(c.data_path, index_col=0)
        ts_col = None
        if "WindowStart" in df_raw.columns:
            df_raw["datetime"] = pd.to_datetime(df_raw["WindowStart"], unit='s')
            ts_col = "datetime"
        elif "datetime" in df_raw.columns:
            df_raw["datetime"] = pd.to_datetime(df_raw["datetime"])
            ts_col = "datetime"

        if ts_col:
            df_raw = df_raw[df_raw[ts_col] <= split_time]
            c.y_raw = df_raw["TrueCPUUtil"].values.reshape(-1, 1)
            req_cols = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem",
                        "UserDiversity", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
            c.df = df_raw[req_cols].copy()

        if len(c.df) < SEQ_LEN + 1:
            continue
        c.prepare_batches(server.scaler_x, pca=server.pca)
        c.create_dataset(server.scaler_y, seq_len=SEQ_LEN)
        if len(c.X_seq) >= 10:
            clients.append(c)

    print(f"  {len(clients)} clients ready\n")

    # 3. Build shared QNode
    n_x = server.pca.n_components_ if server.pca is not None else 9
    cell = AQRNNCell(n_x=n_x, n_h=N_H, seq_len=SEQ_LEN,
                     n_layers=N_LAYERS, param_sharing=PARAM_SHARING)
    qnode, _, _ = cell.build_qnode(device_mode=DEVICE_MODE)

    # 4. State tracking
    topk_ratio      = TOPK_INIT
    prev_round_error = float('inf')
    round_history    = []

    # ── TRAINING LOOP ──
    t_start = time.perf_counter()

    for r in range(N_ROUNDS):
        print(f"\n{'═'*60}")
        print(f"  ROUND {r+1}/{N_ROUNDS}   |   Experts: {list(server.experts.keys())}   "
              f"|   TopK: {topk_ratio:.2f}")
        print(f"{'═'*60}")

        expert_keys = list(server.experts.keys())
        deltas_by_expert    = {k: [] for k in expert_keys}
        expert_client_count = {k: 0  for k in expert_keys}

        # ── A. Expert Audition (all clients × all experts) ──
        print("  Audition: evaluating clients against experts...")
        client_losses = {}
        for c in tqdm(clients, desc="  Audition", leave=False):
            losses = {}
            for k in expert_keys:
                losses[k] = c.evaluate_expert(server.experts[k], qnode,
                                              n_x, N_H, HIDDEN_DIM)
            client_losses[c.client_id] = losses

        # ── B. Load-Balanced Assignment ──
        client_best = {}
        client_regrets = {}
        for cid, losses in client_losses.items():
            best_k = min(losses, key=losses.get)
            client_best[cid] = (best_k, losses[best_k])
            client_regrets[cid] = {k: losses[k] - losses[best_k] for k in losses}

        cap_remaining = {k: EXPERT_CAPACITY for k in expert_keys}

        def flexibility(cid):
            vals = sorted(client_regrets[cid].values())
            return vals[1] - vals[0] if len(vals) >= 2 else 0.0

        sorted_clients = sorted(clients,
                                key=lambda c: flexibility(c.client_id), reverse=True)
        assignment = {}
        for c in sorted_clients:
            cid = c.client_id
            best_k, _ = client_best[cid]
            if cap_remaining.get(best_k, 0) > 0:
                assignment[cid] = best_k
                cap_remaining[best_k] -= 1
            else:
                avail = [k for k, v in cap_remaining.items() if v > 0]
                if avail:
                    assignment[cid] = min(avail, key=lambda k: client_regrets[cid][k])
                    cap_remaining[assignment[cid]] -= 1
                else:
                    assignment[cid] = best_k

        for k in expert_keys:
            cnt = sum(1 for v in assignment.values() if v == k)
            expert_client_count[k] = cnt
            print(f"    Expert {k}: {cnt} clients")

        # Adaptive alpha per expert
        min_var = min(float(np.var(c.y_raw)) for c in clients) or 1.0
        expert_alphas = {}
        for k in expert_keys:
            assigned = [c for c in clients if assignment.get(c.client_id) == k]
            if assigned:
                ev = float(np.mean([np.var(c.y_raw) for c in assigned]))
                expert_alphas[k] = max(ALPHA_MIN,
                                       ALPHA_BASE * (min_var / max(ev, 1e-8)))
            else:
                expert_alphas[k] = ALPHA_BASE

        # ── C. Client Training ──
        for c in tqdm(clients, desc="  Training", colour='green'):
            k = assignment[c.client_id]
            alpha = expert_alphas[k]
            delta, loss = c.train_epoch(server.experts[k], qnode, n_x,
                                        N_H, HIDDEN_DIM, LR_Q, LR_C, alpha)

            # [FEATURE 3] DP: Clip delta on client side before upload
            if DP_ENABLED:
                delta = dp_clip_delta(delta, DP_CLIP_NORM)

            # [FEATURE 2] Compress delta before upload
            if COMPRESSION:
                delta, comp_pct = topk_compress(delta, topk_ratio)
                # (delta is now a compressed dict)

            deltas_by_expert[k].append(delta)

        # ── D. Server Aggregation (decompress → clip → average → noise) ──
        # Cosine decay: start at SERVER_LR_INIT, decay to SERVER_LR_MIN
        server_lr = SERVER_LR_MIN + 0.5 * (SERVER_LR_INIT - SERVER_LR_MIN) * (
            1 + math.cos(math.pi * r / N_ROUNDS))
        dp_log = server.aggregate_updates(deltas_by_expert, round_num=r,
                                          server_lr=server_lr)

        # ── E. Quick validation error for adaptive compression ──
        forget_vec = 0.95 * np.ones(SEQ_LEN) if USE_FORGET_GATE else None
        round_preds = np.zeros(len(X_test))
        round_mask  = np.zeros(len(X_test), dtype=bool)
        for k in server.experts:
            mask = (c_test == k)
            if not np.any(mask):
                continue
            ep = server.experts[k]
            preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                          forget_vec, X_test[mask],
                                          label=f"Val k={k}")
            round_preds[mask] = preds_k
            round_mask[mask]  = True

        if np.any(round_mask):
            yt = eval_scaler_y.inverse_transform(
                y_test.ravel()[round_mask].reshape(-1, 1)).ravel()
            yp = eval_scaler_y.inverse_transform(
                round_preds[round_mask].reshape(-1, 1)).ravel()
            round_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        else:
            round_rmse = prev_round_error

        print(f"\n  Round {r+1} Validation RMSE (orig scale): {round_rmse:.4f}")

        # [FEATURE 2] Adapt compression ratio
        if COMPRESSION:
            topk_ratio = adaptive_topk_ratio(
                topk_ratio, prev_round_error, round_rmse,
                TOPK_DECAY, TOPK_ERROR_BOOST, TOPK_MIN)
            print(f"  → Next round TopK ratio: {topk_ratio:.3f}")

        # [FEATURE 1] Dynamic MoE: split / merge
        # Build per-expert error map for split/merge decision
        expert_errors = {}
        for k in server.experts:
            mask = (c_test == k)
            if np.any(mask) and np.any(round_mask & mask):
                sub_yt = eval_scaler_y.inverse_transform(
                    y_test.ravel()[mask].reshape(-1, 1)).ravel()
                sub_yp = eval_scaler_y.inverse_transform(
                    round_preds[mask].reshape(-1, 1)).ravel()
                expert_errors[k] = float(np.sqrt(mean_squared_error(sub_yt, sub_yp)))
            else:
                expert_errors[k] = 0.0

        moe_actions = server.dynamic_moe_step(expert_errors, expert_client_count, round_num=r, rng=rng)
        if moe_actions:
            print("  ┌── Dynamic MoE Actions ──")
            for a in moe_actions:
                print(f"  │  {a}")
            print("  └──────────────────────────")

        prev_round_error = round_rmse
        round_history.append({
            'round':       r + 1,
            'rmse':        round_rmse,
            'topk':        topk_ratio,
            'n_experts':   len(server.experts),
            'dp_log':      dp_log,
            'moe_actions': moe_actions,
        })

    train_time = time.perf_counter() - t_start

    # ═══════════════════════════════════════════════════════════
    #  FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  FINAL EVALUATION (Global Test Set)")
    print(f"{'═'*70}")

    forget_vec = 0.95 * np.ones(SEQ_LEN) if USE_FORGET_GATE else None
    t_infer = time.perf_counter()

    all_preds = np.zeros(len(X_test))
    all_mask  = np.zeros(len(X_test), dtype=bool)
    cluster_metrics = {}

    for k in tqdm(sorted(server.experts.keys()), desc="  MoE Routing", colour='blue'):
        mask = (c_test == k)
        if not np.any(mask) or k not in server.experts:
            continue
        ep = server.experts[k]
        preds_k = predict_with_params(qnode, ep["params_q"], ep["wvec"],
                                      forget_vec, X_test[mask],
                                      label=f"Cluster {k}")
        all_preds[mask] = preds_k
        all_mask[mask]  = True

        yt_k = eval_scaler_y.inverse_transform(
            y_test.ravel()[mask].reshape(-1, 1)).ravel()
        yp_k = eval_scaler_y.inverse_transform(preds_k.reshape(-1, 1)).ravel()
        rmse_k  = float(np.sqrt(mean_squared_error(yt_k, yp_k)))
        mae_k   = float(mean_absolute_error(yt_k, yp_k))
        smape_k = smape(yt_k, yp_k)
        cluster_metrics[f"cluster_{k}"] = {
            "rmse": rmse_k, "mae": mae_k, "smape": smape_k, "n": int(mask.sum())
        }
        print(f"  Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
              f"SMAPE={smape_k:.1f}%  (n={int(mask.sum())})")

    infer_time = time.perf_counter() - t_infer

    y_orig = eval_scaler_y.inverse_transform(
        y_test.ravel()[all_mask].reshape(-1, 1)).ravel()
    p_orig = eval_scaler_y.inverse_transform(
        all_preds[all_mask].reshape(-1, 1)).ravel()

    rmse_all  = float(np.sqrt(mean_squared_error(y_orig, p_orig)))
    mae_all   = float(mean_absolute_error(y_orig, p_orig))
    smape_all = smape(y_orig, p_orig)

    # Compute final DP budget
    dp_budget_str = "N/A"
    if DP_ENABLED:
        eps_total = compute_dp_epsilon(DP_NOISE_MULT, DP_DELTA, N_ROUNDS)
        dp_budget_str = f"ε≤{eps_total:.2f}, δ={DP_DELTA}"

    print(f"\n  ╔═══ CFL v2 FINAL RESULTS ═══╗")
    print(f"  ║ RMSE  (orig)  = {rmse_all:.4f}")
    print(f"  ║ MAE   (orig)  = {mae_all:.4f}")
    print(f"  ║ SMAPE (orig)  = {smape_all:.1f}%")
    print(f"  ║ Train Time    = {train_time:.1f}s")
    print(f"  ║ Infer Time    = {infer_time:.3f}s")
    print(f"  ║ Final Experts = {sorted(server.experts.keys())}")
    if DP_ENABLED:
        print(f"  ║ DP Budget     = {dp_budget_str}")
    if COMPRESSION:
        print(f"  ║ Final TopK    = {topk_ratio:.3f}")
    print(f"  ╚═════════════════════════════╝")

    # Save
    results = {
        "method":         "Federated_CFL_v2",
        "features":       {"dynamic_moe": DYNAMIC_MOE,
                           "compression": COMPRESSION,
                           "dp": DP_ENABLED},
        "n_rounds":       N_ROUNDS,
        "rmse_orig":      rmse_all,
        "mae_orig":       mae_all,
        "smape_orig":     smape_all,
        "train_time_s":   round(train_time, 2),
        "infer_time_s":   round(infer_time, 2),
        "dp_budget":      dp_budget_str,
        "final_topk":     topk_ratio,
        "final_experts":  sorted(server.experts.keys()),
        "per_cluster":    cluster_metrics,
        "round_history":  round_history,
    }

    with open("federated_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(server.experts, f)
    print(f"\n  Saved: federated_v2_results.json + {RESULTS_PATH}")


if __name__ == "__main__":
    run_simulation_v2()