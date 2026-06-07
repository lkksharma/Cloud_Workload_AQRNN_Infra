#!/usr/bin/env python
"""
══════════════════════════════════════════════════════════════════════
Federated LSTM Comparison — Apples-to-Apples vs CFL AQRNN
══════════════════════════════════════════════════════════════════════

Partitions the GLOBAL CSV into 13 temporal shards as synthetic
federated clients. All data uses the same source and same scaler.

Two experiments:
  F1. FedAvg LSTM (single global)   — same as AQRNN Variant A
  F2. Clustered Federated LSTM      — same as AQRNN Variant D (CFL)

Usage:
    conda activate badminton && python federated_lstm_comparison.py
══════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, glob, pickle, copy
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CSV_PATH      = "grid5000_hybrid_clean.csv"
KMEANS_PATH   = "kmeans_model.pkl"
N_COMPONENTS  = 4
SEQ_LEN       = 2
SEED          = 42
N_CLIENTS     = 13

N_ROUNDS      = 5
CLIENT_EPOCHS = 3
BATCH_SIZE    = 256
LR            = 1e-3
LSTM_HIDDEN   = 50
EXPERT_CAPACITY = 5

QUICK_TEST    = False   # ← Set to False for the real full run

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ══════════════════════════════════════════════════════════════
# DATA: Partition global CSV into synthetic federated clients
# ══════════════════════════════════════════════════════════════
def load_and_partition():
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

    y_raw = df[["TrueCPUUtil"]].values
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

    Xs, ys, cs = [], [], []
    for i in trange(len(X_final) - SEQ_LEN, desc="  Building sequences"):
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

    X_train, y_train = X_seq[:train_end], y_scaled[:train_end]

    shard_size = len(X_train) // N_CLIENTS
    client_data = []
    for i in range(N_CLIENTS):
        start = i * shard_size
        end = start + shard_size if i < N_CLIENTS - 1 else len(X_train)
        client_data.append({"X": X_train[start:end], "y": y_train[start:end], "id": f"shard_{i}"})
        print(f"    Client shard_{i}: {end - start:,} sequences")

    test_data = {
        "X_test": X_seq[val_end:], "y_test": y_scaled[val_end:],
        "c_test": c_seq[val_end:], "scaler_y": scaler_y,
    }
    print(f"  Test set: {len(test_data['X_test']):,} samples")
    return client_data, test_data, n_x


def build_lstm(n_x):
    model = Sequential([
        LSTM(LSTM_HIDDEN, activation='tanh', input_shape=(SEQ_LEN, n_x)),
        Dense(1)
    ])
    model.compile(loss='mae', optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def evaluate_on_test(weights_or_dict, test_data, n_x, is_clustered=False):
    scaler_y = test_data["scaler_y"]
    X_test, y_test, c_test = test_data["X_test"], test_data["y_test"], test_data["c_test"]

    if is_clustered:
        preds = np.zeros(len(y_test))
        for k, weights in weights_or_dict.items():
            mask = (c_test == k)
            if not np.any(mask): continue
            model = build_lstm(n_x)
            model.set_weights(weights)
            preds[mask] = model.predict(X_test[mask], batch_size=BATCH_SIZE, verbose=0).ravel()
    else:
        model = build_lstm(n_x)
        model.set_weights(weights_or_dict)
        preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()

    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    p_orig = scaler_y.inverse_transform(np.clip(preds, -5, 5).reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_orig, p_orig)))
    mae  = float(mean_absolute_error(y_orig, p_orig))
    sm   = smape(y_orig, p_orig)

    cluster_metrics = {}
    print(f"\n  Per-Cluster Breakdown:")
    for k in sorted(np.unique(c_test)):
        mask = c_test == k
        yt_k, yp_k = y_orig[mask], p_orig[mask]
        rmse_k = float(np.sqrt(mean_squared_error(yt_k, yp_k)))
        cluster_metrics[int(k)] = {"rmse": rmse_k, "n": int(mask.sum())}
        print(f"    Cluster {k}: RMSE={rmse_k:.2f} (n={mask.sum()})")

    return {"rmse_orig": rmse, "mae_orig": mae, "smape_orig": sm, "cluster_breakdown": cluster_metrics}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Federated LSTM Comparison — vs CFL AQRNN (Variant D)  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

    client_data, test_data, n_x = load_and_partition()

    n_rounds = 1 if QUICK_TEST else N_ROUNDS
    if QUICK_TEST:
        client_data = client_data[:3]
        print(f"\n  ⚡ QUICK TEST MODE: {n_rounds} round, {len(client_data)} clients")

    results = {}

    # ── F1: FedAvg LSTM ──
    print("\n" + "=" * 60 + "\n  F1: FedAvg LSTM (Single Global Model)\n" + "=" * 60)
    global_model = build_lstm(n_x)
    global_weights = global_model.get_weights()
    n_params = global_model.count_params()
    print(f"  Params: {n_params:,}")

    t0 = time.perf_counter()
    for r in trange(n_rounds, desc="  LSTM FedAvg", colour='green'):
        updates = []
        for cd in tqdm(client_data, desc=f"    Round {r+1}", leave=False, colour='blue'):
            model = build_lstm(n_x)
            model.set_weights(global_weights)
            model.fit(cd["X"], cd["y"], epochs=CLIENT_EPOCHS,
                      batch_size=BATCH_SIZE, shuffle=False, verbose=0)
            updates.append(model.get_weights())
        global_weights = [np.mean([u[li] for u in updates], axis=0)
                          for li in range(len(global_weights))]
    train_time = time.perf_counter() - t0

    m = evaluate_on_test(global_weights, test_data, n_x)
    m.update({"method": "FedAvg LSTM", "params": n_params, "train_time_s": round(train_time, 2)})
    results["F1_FedAvg_LSTM"] = m
    print(f"\n  ╔═══ F1 RESULT: RMSE = {m['rmse_orig']:.2f} ═══╗")

    keras.backend.clear_session()

    # ── F2: Clustered Federated LSTM (CFL-style) ──
    print("\n" + "=" * 60 + "\n  F2: Clustered Federated LSTM (CFL-style)\n" + "=" * 60)
    experts = {k: build_lstm(n_x).get_weights() for k in range(3)}
    print(f"  3× LSTM experts, {n_params:,} params each")

    t0 = time.perf_counter()
    for r in trange(n_rounds, desc="  LSTM CFL", colour='magenta'):
        # Audition: each client evaluates all experts
        client_losses = {}
        for cd in client_data:
            losses = {}
            for k, w in experts.items():
                model = build_lstm(n_x)
                model.set_weights(w)
                p = model.predict(cd["X"], batch_size=BATCH_SIZE, verbose=0).ravel()
                losses[k] = float(np.mean((p - cd["y"]) ** 2))
            client_losses[cd["id"]] = losses

        # Assign clients to best expert (with capacity)
        assignment = {}
        capacity = {k: EXPERT_CAPACITY for k in experts}
        for cd in client_data:
            best_k = min(client_losses[cd["id"]], key=client_losses[cd["id"]].get)
            if capacity[best_k] > 0:
                assignment[cd["id"]] = best_k
                capacity[best_k] -= 1
            else:
                avail = [k for k, c in capacity.items() if c > 0]
                ak = min(avail, key=lambda k: client_losses[cd["id"]][k]) if avail else best_k
                assignment[cd["id"]] = ak
                capacity[ak] -= 1

        # Train + aggregate per expert
        expert_updates = {k: [] for k in experts}
        for cd in client_data:
            ak = assignment[cd["id"]]
            model = build_lstm(n_x)
            model.set_weights(experts[ak])
            model.fit(cd["X"], cd["y"], epochs=CLIENT_EPOCHS,
                      batch_size=BATCH_SIZE, shuffle=False, verbose=0)
            expert_updates[ak].append(model.get_weights())

        for k, u_list in expert_updates.items():
            if u_list:
                experts[k] = [np.mean([u[li] for u in u_list], axis=0)
                              for li in range(len(experts[k]))]

    train_time = time.perf_counter() - t0
    m2 = evaluate_on_test(experts, test_data, n_x, is_clustered=True)
    m2.update({"method": "Clustered Fed LSTM", "params": n_params * 3, "train_time_s": round(train_time, 2)})
    results["F2_Clustered_Fed_LSTM"] = m2
    print(f"\n  ╔═══ F2 RESULT: RMSE = {m2['rmse_orig']:.2f} ═══╗")

    # Summary
    print("\n" + "=" * 70)
    print("  COMPARISON: Federated LSTM vs CFL AQRNN")
    print("=" * 70)
    print(f"  {'Model':<40} {'RMSE':>8} {'MAE':>8} {'SMAPE':>8}")
    print("  " + "-" * 66)
    for name, m in results.items():
        print(f"  {m['method']:<40} {m['rmse_orig']:>8.2f} {m['mae_orig']:>8.2f} {m['smape_orig']:>7.1f}%")
    print("  " + "-" * 66)
    print(f"  {'CFL AQRNN Variant D (reference)':<40} {'159.58':>8} {'114.88':>8} {'69.2':>7}%")

    with open("federated_lstm_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to federated_lstm_results.json\n")
