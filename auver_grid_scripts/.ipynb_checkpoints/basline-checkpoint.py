#!/usr/bin/env python
"""
======================================================================
Classical Baselines for AuverGrid - FedAvg + 500-Param MLP
======================================================================

Part 1: Federated DL Baselines (FedAvg)
  - esDNN       : Conv1D(32), GRU(64), Dense(16, swish), Dense(1)
  - GRU-Only    : GRU(64), Dense(1)
  - Bi-LSTM     : Bidirectional LSTM(64), Dense(1)
  - SimpleRNN   : SimpleRNN(64), Dense(1)

Part 2: Quantum Advantage Baseline
  - MLP-508     : Dense(23, relu), Dense(12, relu), Dense(1)  [508 params]
    Trained CENTRALIZED (all data, no federation) - giving it every
    advantage. If AQRNN (<500 params, federated) still wins, quantum
    advantage is demonstrated.

Data: auverGrid_hybrid_clean.csv  (5 sites, ~505K 5-min windows)

Usage:
    python classical_baselines_auverGrid.py
======================================================================
"""

import os, sys, time, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import (Conv1D, GRU, LSTM, SimpleRNN, Bidirectional,
                          Dense, Dropout, Flatten, Input)
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==============================================================
# CONFIG
# ==============================================================
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH      = os.path.join(PROJECT_ROOT, "auverGrid_hybrid_clean.csv")

N_COMPONENTS  = 4
SEQ_LEN       = 2
SEED          = 42
N_CLIENTS     = 5       # AuverGrid has 5 sites

# FedAvg config (matching CFL run)
N_ROUNDS      = 3
CLIENT_EPOCHS = 3
BATCH_SIZE    = 256
LR            = 1e-3

# MLP config
MLP_EPOCHS    = 200
MLP_PATIENCE  = 20

np.random.seed(SEED)
tf.random.set_seed(SEED)

RESULTS_JSON  = os.path.join(PROJECT_ROOT, "classical_baselines_auverGrid.json")


# ==============================================================
# DATA PIPELINE
# ==============================================================
def load_shared_data():
    """Load AuverGrid CSV, fit scalers/PCA/KMeans securely on train, build sequences."""
    print("=" * 60)
    print("LOADING SHARED DATA PIPELINE (AuverGrid) - LEAKAGE FIXED")
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

    # Split df BEFORE fitting any transformations (70 / 15 / 15 split)
    N = len(df) - SEQ_LEN
    if N <= 0: raise ValueError("Dataset too short")
    train_end = int(N * 0.7)
    val_end   = train_end + int(N * 0.15)

    X_raw = df[features].values
    y_raw = df[[target]].values

    # Sequences
    Xs, ys = [], []
    for i in trange(len(X_raw) - SEQ_LEN, desc="  Building raw sequences"):
        Xs.append(X_raw[i : i + SEQ_LEN])
        ys.append(y_raw[i + SEQ_LEN, 0])
    X_seq_raw = np.array(Xs)
    y_seq_raw = np.array(ys).reshape(-1, 1)

    X_train_raw = X_seq_raw[:train_end]
    y_train_raw = y_seq_raw[:train_end]
    
    X_val_raw   = X_seq_raw[train_end:val_end]
    y_val_raw   = y_seq_raw[train_end:val_end]
    
    X_test_raw  = X_seq_raw[val_end:]
    y_test_raw  = y_seq_raw[val_end:]

    # Flatten train sets to fit transformations
    flat_X_train = X_train_raw.reshape(-1, len(features))

    scaler_x = QuantileTransformer(output_distribution='uniform',
                                   n_quantiles=min(1000, len(flat_X_train)))
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

    # Fit on Train only
    flat_X_train_scaled = scaler_x.fit_transform(flat_X_train)
    y_train_scaled      = scaler_y.fit_transform(y_train_raw)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    flat_X_train_final = pca.fit_transform(flat_X_train_scaled)
    print(f"  PCA: {flat_X_train_scaled.shape[1]} to {N_COMPONENTS} features (fit on train)")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, init='k-means++')
    kmeans.fit(flat_X_train_final)
    print("  KMeans fit on training data.")

    # Apply transformations to Train/Val/Test
    def process_split(X_split_raw, y_split_raw):
        flat_X = X_split_raw.reshape(-1, len(features))
        flat_X_sc = scaler_x.transform(flat_X)
        flat_X_fin = pca.transform(flat_X_sc)
            
        c_flat = kmeans.predict(flat_X_fin)
        
        # Reshape X back to sequences: (B, SEQ_LEN, attrs)
        B = X_split_raw.shape[0]
        n_attrs = flat_X_fin.shape[1]
        X_out = flat_X_fin.reshape(B, SEQ_LEN, n_attrs)
        
        y_out = scaler_y.transform(y_split_raw).ravel()
        
        # Cluster assignments
        last_step_fin = flat_X_fin.reshape(B, SEQ_LEN, n_attrs)[:, -1, :]
        c_out = kmeans.predict(last_step_fin).astype(np.int32)

        return X_out.astype(np.float32), y_out.astype(np.float32), c_out

    X_train, y_train, c_train = process_split(X_train_raw, y_train_raw)
    X_val, y_val, c_val       = process_split(X_val_raw, y_val_raw)
    X_test, y_test, c_test    = process_split(X_test_raw, y_test_raw)

    unique, counts = np.unique(c_train, return_counts=True)
    print(f"  Train Cluster Distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "c_train": c_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "c_val":   c_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "c_test":  c_test,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "pca": pca,
        "kmeans": kmeans,
    }

    n_x = X_train.shape[2]
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"  Sequence shape: (B, {SEQ_LEN}, {n_x})")
    return data, n_x


# ==============================================================
# METRICS
# ==============================================================
def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def evaluate_on_test(model, data, label=""):
    """Evaluate a Keras model on the test set, return metrics dict."""
    scaler_y = data["scaler_y"]
    X_test, y_test, c_test = data["X_test"], data["y_test"], data["c_test"]

    t0 = time.perf_counter()
    preds = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    infer_time = time.perf_counter() - t0

    y_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    p_orig = scaler_y.inverse_transform(
        np.clip(preds, -5, 5).reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_orig, p_orig)))
    mae  = float(mean_absolute_error(y_orig, p_orig))
    sm   = smape(y_orig, p_orig)

    # Per-cluster
    cluster_metrics = {}
    print(f"\n  Per-Cluster Breakdown ({label}):")
    for k in sorted(np.unique(c_test)):
        mask = c_test == k
        yt_k, yp_k = y_orig[mask], p_orig[mask]
        rmse_k = float(np.sqrt(mean_squared_error(yt_k, yp_k)))
        mae_k  = float(mean_absolute_error(yt_k, yp_k))
        smape_k = smape(yt_k, yp_k)
        cluster_metrics[int(k)] = {
            "rmse": rmse_k, "mae": mae_k, "smape": smape_k, "n": int(mask.sum())
        }
        print(f"    Cluster {k}: RMSE={rmse_k:.4f}  MAE={mae_k:.4f}  "
              f"SMAPE={smape_k:.1f}% (n={mask.sum()})")

    return {
        "rmse_orig": rmse, "mae_orig": mae, "smape_orig": sm,
        "infer_time_s": round(infer_time, 2),
        "cluster_breakdown": cluster_metrics,
    }


# ==============================================================
# MODEL BUILDERS (matching federated_esdnn_comparison.py)
# ==============================================================
def build_esdnn(n_x):
    model = Sequential([
        Conv1D(32, kernel_size=min(5, SEQ_LEN), padding='causal',
               activation='relu', input_shape=(SEQ_LEN, n_x)),
        GRU(64, return_sequences=False),
        Dense(16, activation='swish'),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model

def build_gru_only(n_x):
    model = Sequential([
        GRU(64, input_shape=(SEQ_LEN, n_x)),
        Dense(1),
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model

def build_bilstm(n_x):
    model = Sequential([
        Bidirectional(LSTM(64), input_shape=(SEQ_LEN, n_x)),
        Dense(1),
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model

def build_simplernn(n_x):
    model = Sequential([
        SimpleRNN(64, input_shape=(SEQ_LEN, n_x), activation='tanh'),
        Dense(1),
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LR))
    return model


# ==============================================================
# PART 1: FEDAVG RUNNER
# ==============================================================
def run_fedavg(name, build_fn, data, n_x, n_rounds):
    """Run FedAvg for a given model builder on AuverGrid data."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # Partition training data into N_CLIENTS temporal shards
    X_train, y_train = data["X_train"], data["y_train"]
    shard_size = len(X_train) // N_CLIENTS
    client_data = []
    for i in range(N_CLIENTS):
        start = i * shard_size
        end = start + shard_size if i < N_CLIENTS - 1 else len(X_train)
        client_data.append({"X": X_train[start:end], "y": y_train[start:end]})

    model = build_fn(n_x)
    global_weights = model.get_weights()
    n_params = model.count_params()
    print(f"  Params: {n_params:,} | Rounds: {n_rounds} | Clients: {N_CLIENTS}")

    t0 = time.perf_counter()
    for r in trange(n_rounds, desc=f"  {name}", colour='green'):
        updates = []
        for cd in tqdm(client_data, desc=f"    Round {r+1}", leave=False, colour='blue'):
            m = build_fn(n_x)
            m.set_weights(global_weights)
            m.fit(cd["X"], cd["y"], epochs=CLIENT_EPOCHS,
                  batch_size=BATCH_SIZE, shuffle=False, verbose=0)
            updates.append(m.get_weights())

        # FedAvg aggregation
        global_weights = [np.mean([u[li] for u in updates], axis=0)
                          for li in range(len(global_weights))]
    train_time = time.perf_counter() - t0

    # Final evaluation
    final_model = build_fn(n_x)
    final_model.set_weights(global_weights)
    metrics = evaluate_on_test(final_model, data, label=name)
    metrics.update({
        "method": name, "params": n_params,
        "train_time_s": round(train_time, 2),
    })

    print(f"\n  +=== {name} ===+")
    print(f"  | RMSE  = {metrics['rmse_orig']:.4f}")
    print(f"  | MAE   = {metrics['mae_orig']:.4f}")
    print(f"  | SMAPE = {metrics['smape_orig']:.1f}%")
    print(f"  | Train = {train_time:.1f}s | Params = {n_params:,}")
    print(f"  +{'=' * (len(name) + 8)}+")

    keras.backend.clear_session()
    return metrics


# ==============================================================
# PART 2: 505-PARAMETER MLP (Quantum Advantage Baseline)
# ==============================================================
def run_mlp_508(data, n_x):
    """
    Train a ~508-parameter MLP CENTRALIZED on all data.
    Architecture: Flatten(2x4=8), Dense(23, relu), Dense(12, relu), Dense(1)
    Params: 8*23+23 + 23*12+12 + 12*1+1 = 207 + 288 + 13 = 508
    """
    print(f"\n{'=' * 60}")
    print(f"  QUANTUM ADVANTAGE BASELINE: MLP-508 (Centralized)")
    print(f"{'=' * 60}")

    model = Sequential([
        Flatten(input_shape=(SEQ_LEN, n_x)),     # (2, 4) = 8
        Dense(23, activation='relu'),             # 8*23 + 23 = 207
        Dense(12, activation='relu'),             # 23*12 + 12 = 288
        Dense(1),                                 # 12*1 + 1  = 13
    ])                                            # Total: 207+288+13 = 508

    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0))

    n_params = model.count_params()
    print(f"  Architecture: Flatten({SEQ_LEN}x{n_x}), Dense(23,relu), Dense(12,relu), Dense(1)")
    print(f"  Parameters: {n_params}")
    print(f"  Training: CENTRALIZED (full dataset, giving MLP every advantage)")
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', patience=MLP_PATIENCE,
                       restore_best_weights=True, verbose=1)

    t0 = time.perf_counter()
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=MLP_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[es],
        verbose=2,
    )
    train_time = time.perf_counter() - t0

    metrics = evaluate_on_test(model, data, label="MLP-508 (Centralized)")
    metrics.update({
        "method": "MLP-508 (Centralized)",
        "params": n_params,
        "train_time_s": round(train_time, 2),
    })

    print(f"\n  +=== MLP-508 RESULTS ===+")
    print(f"  | RMSE  = {metrics['rmse_orig']:.4f}")
    print(f"  | MAE   = {metrics['mae_orig']:.4f}")
    print(f"  | SMAPE = {metrics['smape_orig']:.1f}%")
    print(f"  | Train = {train_time:.1f}s | Params = {n_params}")
    print(f"  +=========================+")
    print(f"\n  [!] NOTE: This MLP was trained CENTRALIZED (all data, no federation).")
    print(f"          AQRNN achieves its results with FEDERATED training (<500 params).")
    print(f"          If AQRNN matches or beats this MLP, quantum advantage is confirmed.")

    keras.backend.clear_session()
    return metrics


# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print("+==============================================================+")
    print("|  Classical Baselines - AuverGrid (FedAvg + MLP-505)        |")
    print("+==============================================================+\n")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPU detected: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("  WARNING: No GPU detected, running on CPU")

    # -- Load Data --
    data, n_x = load_shared_data()
    results = {}

    # ==========================================================
    # PART 1: FEDERATED BASELINES
    # ==========================================================
    print("\n" + "#" * 70)
    print("  PART 1: FEDERATED DL BASELINES (FedAvg)")
    print("#" * 70)

    experiments = [
        ("FedAvg esDNN",      build_esdnn),
        ("FedAvg GRU-Only",   build_gru_only),
        ("FedAvg Bi-LSTM",    build_bilstm),
        ("FedAvg SimpleRNN",  build_simplernn),
    ]

    for name, build_fn in experiments:
        results[name] = run_fedavg(name, build_fn, data, n_x, N_ROUNDS)

    # ==========================================================
    # PART 2: 505-PARAMETER MLP
    # ==========================================================
    print("\n" + "#" * 70)
    print("  PART 2: QUANTUM ADVANTAGE BASELINE (MLP-508)")
    print("#" * 70)

    results["MLP-508 (Centralized)"] = run_mlp_508(data, n_x)

    # ==========================================================
    # SUMMARY
    # ==========================================================
    print("\n" + "=" * 90)
    print("  CLASSICAL BASELINES - AUVERGRID FINAL SUMMARY")
    print("=" * 90)
    print(f"  {'Model':<35} {'RMSE':>8} {'MAE':>8} {'SMAPE':>8} "
          f"{'Params':>8} {'Train(s)':>10}")
    print("  " + "-" * 85)
    for name, m in results.items():
        print(f"  {m['method']:<35} {m['rmse_orig']:>8.4f} {m['mae_orig']:>8.4f} "
              f"{m['smape_orig']:>7.1f}% {m['params']:>8,} {m['train_time_s']:>10.1f}")
    print("  " + "-" * 85)
    print(f"  {'CFL AQRNN (Reference)':35} {'0.2761':>8} {'0.0996':>8} "
          f"{'45.4':>7}% {'< 500':>8} {'6,111':>10}")
    print("  " + "-" * 85)

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_JSON}")
    print("\n[OK] All classical baselines complete!")