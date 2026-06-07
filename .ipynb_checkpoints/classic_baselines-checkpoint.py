#!/usr/bin/env python
"""
══════════════════════════════════════════════════════════════════════
Classical DL Baselines for AQRNN Ablation Study  (TensorFlow / Keras)
══════════════════════════════════════════════════════════════════════

D. Cloud-Forecast LSTM  (Christofidi et al., EuroMLSys '23)
   Cloned: baselines/cloud-forecast-lstm/
   Architecture: LSTM(50, activation='tanh') + Dense(1), MAE loss
   NOTE: The original paper used activation='softmax' which is
   incorrect for regression — softmax normalises hidden states to a
   probability simplex, destroying representational capacity for
   targets outside [0,1].  We correct this to the standard 'tanh'
   for a fair comparison.

E. WGAN-GP Transformer  (Arbat et al., IAAI '22 / AAAI-22)
   Cloned: baselines/wgan-gp-transformer/
   Keras translation of: Transformer Enc-Dec (6L) + MLP Critic, WGAN-GP

Same Grid5000 pipeline: QuantileTransformer → PCA(4) → 70/15/15 split

Usage:
    conda activate badminton && python classical_baselines.py
══════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, math
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import (LSTM, Dense, Input, LayerNormalization,
                          MultiHeadAttention, Dropout, Add, Flatten)
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CSV_PATH     = "grid5000_hybrid_clean.csv"
N_COMPONENTS = 4
SEQ_LEN      = 2
SEED         = 42

# LSTM config (corrected from original)
LSTM_HIDDEN   = 50       # Original: LSTM(50, ...)
LSTM_EPOCHS   = 200      # Original: 1000 — capped, early stopping handles rest
LSTM_BATCH    = 256
LSTM_PATIENCE = 20       # Original: patience=20

# WGAN-GP Transformer config
TF_D_MODEL    = 64
TF_NHEAD      = 4
TF_LAYERS     = 6
TF_DROPOUT    = 0.1
TF_EPOCHS     = 100
TF_LR_G       = 1e-4
TF_LR_D       = 1e-4
TF_BATCH      = 256
TF_CRITIC_ITERS = 5
TF_LAMBDA_GP  = 10.0
TF_PATIENCE   = 15

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ══════════════════════════════════════════════════════════════
# SHARED DATA PIPELINE (matching ablation_study.py)
# ══════════════════════════════════════════════════════════════
def load_shared_data():
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

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_final = pca.fit_transform(X_scaled)
    print(f"  PCA: {X_scaled.shape[1]} → {N_COMPONENTS} features")

    y_raw = df[[target]].values
    y_scaled = scaler_y.fit_transform(y_raw)

    # Cluster labels (same as ablation_study.py)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, init='k-means++')
    clusters = kmeans.fit_predict(X_final)
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  Cluster distribution: { {int(u): int(c) for u,c in zip(unique, counts)} }")

    # Create sequences
    Xs, ys, cs = [], [], []
    for i in trange(len(X_final) - SEQ_LEN, desc="  Building sequences"):
        Xs.append(X_final[i : i + SEQ_LEN])
        ys.append(y_scaled[i + SEQ_LEN, 0])
        cs.append(clusters[i + SEQ_LEN - 1])   # cluster of last input timestep
    X_seq = np.array(Xs, dtype=np.float32)
    y_seq = np.array(ys, dtype=np.float32)
    c_seq = np.array(cs, dtype=np.int32)

    # 70/15/15 temporal split
    N = len(X_seq)
    train_end = int(N * 0.7)
    val_end = train_end + int(N * 0.15)

    data = {
        "X_train": X_seq[:train_end],
        "y_train": y_seq[:train_end],
        "X_val":   X_seq[train_end:val_end],
        "y_val":   y_seq[train_end:val_end],
        "X_test":  X_seq[val_end:],
        "y_test":  y_seq[val_end:],
        "c_test":  c_seq[val_end:],    # cluster labels for test set
        "scaler_y": scaler_y,
    }

    n_x = X_seq.shape[2]
    print(f"  Train: {data['X_train'].shape[0]:,} | Val: {data['X_val'].shape[0]:,} | "
          f"Test: {data['X_test'].shape[0]:,}")
    print(f"  Sequence shape: (B, {SEQ_LEN}, {n_x})")
    return data, n_x


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(num / den) * 100.0)


def compute_metrics(y_true_scaled, y_pred_scaled, scaler_y,
                    train_time=0.0, infer_time=0.0):
    y_orig = scaler_y.inverse_transform(
        y_true_scaled.reshape(-1, 1)).ravel()
    p_orig = scaler_y.inverse_transform(
        np.clip(y_pred_scaled, -5, 5).reshape(-1, 1)).ravel()
    rmse = float(np.sqrt(mean_squared_error(y_orig, p_orig)))
    mae  = float(mean_absolute_error(y_orig, p_orig))
    sm   = smape(y_orig, p_orig)
    return {
        "rmse_orig": rmse, "mae_orig": mae, "smape_orig": sm,
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 2),
    }


def per_cluster_metrics(y_true_scaled, y_pred_scaled, c_labels, scaler_y):
    """
    Break down RMSE/MAE/SMAPE per cluster.
    This empirically shows where a global model struggles on
    heterogeneous workloads — no speculation needed.
    """
    y_orig = scaler_y.inverse_transform(
        y_true_scaled.reshape(-1, 1)).ravel()
    p_orig = scaler_y.inverse_transform(
        np.clip(y_pred_scaled, -5, 5).reshape(-1, 1)).ravel()

    n_clusters = len(np.unique(c_labels))
    cluster_metrics = {}

    print(f"\n  Per-Cluster Breakdown:")
    print(f"  {'Cluster':>8} {'n':>6} {'RMSE':>10} {'MAE':>10} {'SMAPE':>10}")
    print("  " + "-" * 48)

    for k in sorted(np.unique(c_labels)):
        mask = c_labels == k
        yt_k = y_orig[mask]
        yp_k = p_orig[mask]
        rmse_k = float(np.sqrt(mean_squared_error(yt_k, yp_k)))
        mae_k  = float(mean_absolute_error(yt_k, yp_k))
        smap_k = smape(yt_k, yp_k)
        cluster_metrics[int(k)] = {
            "rmse": rmse_k, "mae": mae_k, "smape": smap_k, "n": int(mask.sum())
        }
        print(f"  {'Cluster '+str(k):>8} {mask.sum():>6,} {rmse_k:>10.4f} "
              f"{mae_k:>10.4f} {smap_k:>9.1f}%")

    return cluster_metrics


# ══════════════════════════════════════════════════════════════
# MODEL D: CLOUD-FORECAST LSTM  (CORRECTED)
# ══════════════════════════════════════════════════════════════
#
# Original (baselines/cloud-forecast-lstm/train_lstms.py):
#   model.add(LSTM(50, activation='softmax'))  ← BUG
#   model.add(Dense(1))
#   model.compile(loss='mae', optimizer='adam')
#
# Correction: activation='tanh' (the LSTM default)
#
# The original paper used 'softmax' which normalises the hidden
# state to a probability simplex — destroying the LSTM's capacity
# for regression on normally-distributed targets.  The model early-
# stops at epoch 1 because training makes predictions worse.
#
# We correct to 'tanh' for a scientifically honest comparison.
# ══════════════════════════════════════════════════════════════

class TQDMKerasCallback(keras.callbacks.Callback):
    """tqdm progress bar for Keras training."""
    def __init__(self, total_epochs, desc="Training"):
        super().__init__()
        self.total_epochs = total_epochs
        self.desc = desc
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc=self.desc,
                         colour='green', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        self.pbar.set_postfix({'loss': f'{loss:.4f}',
                               'val_loss': f'{val_loss:.4f}'})

    def on_train_end(self, logs=None):
        self.pbar.close()


def run_lstm_baseline(data, n_x):
    print("\n" + "=" * 60)
    print("  MODEL D: Cloud-Forecast LSTM (Corrected)")
    print("  Ref: Christofidi et al., EuroMLSys '23")
    print("  Source: baselines/cloud-forecast-lstm/train_lstms.py")
    print("  Fix: activation='softmax' → 'tanh' (see docstring)")
    print("=" * 60)

    # ── Build model: original architecture, corrected activation ──
    model = Sequential()
    model.add(LSTM(LSTM_HIDDEN,
                   input_shape=(SEQ_LEN, n_x),
                   activation='tanh'))           # CORRECTED from 'softmax'
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])

    es = EarlyStopping(monitor='val_loss', mode='min',
                       verbose=1, patience=LSTM_PATIENCE,
                       restore_best_weights=True)
    tqdm_cb = TQDMKerasCallback(LSTM_EPOCHS, desc="  LSTM Training")

    n_params = model.count_params()
    print(f"  Architecture: LSTM({LSTM_HIDDEN}, tanh) → Dense(1)")
    print(f"  Loss: MAE  |  Optimizer: Adam")
    print(f"  Params: {n_params:,}")
    model.summary()

    # ── Train ──
    t0 = time.perf_counter()
    history = model.fit(
        x=data["X_train"], y=data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        shuffle=False,
        callbacks=[es, tqdm_cb],
        verbose=0,          # Suppress default logs, tqdm handles it
    )
    train_time = time.perf_counter() - t0

    # ── Inference ──
    print("  Running inference on test set...")
    t1 = time.perf_counter()
    preds = model.predict(data["X_test"], batch_size=LSTM_BATCH, verbose=0)
    infer_time = time.perf_counter() - t1
    preds = preds.ravel()

    metrics = compute_metrics(data["y_test"], preds, data["scaler_y"],
                              train_time, infer_time)
    metrics["method"] = "Cloud-Forecast LSTM (corrected)"
    metrics["params"] = n_params
    metrics["cluster_breakdown"] = per_cluster_metrics(
        data["y_test"], preds, data["c_test"], data["scaler_y"]
    )

    print(f"\n  ╔═══ LSTM RESULTS ═══╗")
    print(f"  ║ RMSE  = {metrics['rmse_orig']:.4f}")
    print(f"  ║ MAE   = {metrics['mae_orig']:.4f}")
    print(f"  ║ SMAPE = {metrics['smape_orig']:.1f}%")
    print(f"  ║ Train = {train_time:.1f}s | Infer = {infer_time:.3f}s")
    print(f"  ║ Params = {n_params:,}")
    print(f"  ╚══════════════════════╝")
    return metrics


# ══════════════════════════════════════════════════════════════
# MODEL E: WGAN-GP TRANSFORMER  (Keras translation)
# ══════════════════════════════════════════════════════════════
#
# Original (PyTorch): baselines/wgan-gp-transformer/quaesita/model.py
#   Transformer_EncoderDecoder_Seq2Seq:
#     - PositionalEncoding (sinusoidal)
#     - TransformerEncoder (6 layers, d_model, nhead)
#     - TransformerDecoder (6 layers, cross-attention)
#     - Linear → Tanh output
#   SequenceCritic:
#     - Linear(1, d_model) → Tanh
#     - Linear(d_model, d_model*2) → Tanh
#     - Linear(d_model*2, 1) → Tanh
#   Training: WGAN-GP loss + L1 reconstruction
# ══════════════════════════════════════════════════════════════

def positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding (matching original PositionalEncoding.py)."""
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                      (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term[:d_model//2])
    return pe


def build_transformer_generator(n_x, seq_len=SEQ_LEN, d_model=TF_D_MODEL,
                                 nhead=TF_NHEAD, n_layers=TF_LAYERS,
                                 dropout=TF_DROPOUT):
    """
    Keras Functional API translation of Transformer_EncoderDecoder_Seq2Seq.
    """
    inp = Input(shape=(seq_len, n_x), name="input_seq")

    # Project to d_model
    x = Dense(d_model, name="input_proj")(inp)

    # Add positional encoding
    pe = positional_encoding(seq_len, d_model)
    pe_tensor = tf.constant(pe[np.newaxis, :, :])
    x = x + pe_tensor

    # ── Encoder (6 layers) ──
    for i in range(n_layers):
        attn_out = MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead,
            dropout=dropout, name=f"enc_mha_{i}"
        )(x, x)
        attn_out = Dropout(dropout)(attn_out)
        x = Add()([x, attn_out])
        x = LayerNormalization(name=f"enc_ln1_{i}")(x)

        ff = Dense(d_model * 4, activation='relu', name=f"enc_ff1_{i}")(x)
        ff = Dense(d_model, name=f"enc_ff2_{i}")(ff)
        ff = Dropout(dropout)(ff)
        x = Add()([x, ff])
        x = LayerNormalization(name=f"enc_ln2_{i}")(x)

    enc_out = x

    # ── Decoder (6 layers) ──
    dec_in = x[:, -1:, :]

    for i in range(n_layers):
        dec_attn = MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead,
            dropout=dropout, name=f"dec_self_mha_{i}"
        )(dec_in, dec_in)
        dec_attn = Dropout(dropout)(dec_attn)
        dec_in = Add()([dec_in, dec_attn])
        dec_in = LayerNormalization(name=f"dec_ln1_{i}")(dec_in)

        cross_attn = MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead,
            dropout=dropout, name=f"dec_cross_mha_{i}"
        )(dec_in, enc_out)
        cross_attn = Dropout(dropout)(cross_attn)
        dec_in = Add()([dec_in, cross_attn])
        dec_in = LayerNormalization(name=f"dec_ln2_{i}")(dec_in)

        ff = Dense(d_model * 4, activation='relu', name=f"dec_ff1_{i}")(dec_in)
        ff = Dense(d_model, name=f"dec_ff2_{i}")(ff)
        ff = Dropout(dropout)(ff)
        dec_in = Add()([dec_in, ff])
        dec_in = LayerNormalization(name=f"dec_ln3_{i}")(dec_in)

    # Output: Linear → Tanh (matching original)
    out = Dense(1, name="output_proj")(dec_in)
    out = keras.activations.tanh(out)
    out = Flatten()(out)

    return Model(inp, out, name="TransformerGenerator")


def build_critic(d_model=64):
    """
    Keras translation of SequenceCritic (quaesita/transformerGANs.py).
    """
    inp = Input(shape=(1,), name="critic_input")
    x = Dense(d_model, activation='tanh', name="critic_fc1")(inp)
    x = Dense(d_model * 2, activation='tanh', name="critic_fc2")(x)
    x = Dense(1, activation='tanh', name="critic_out")(x)
    return Model(inp, x, name="SequenceCritic")


def gradient_penalty(critic, real, fake, batch_size):
    """Improved gradient penalty (matching utils/optimizer.py)."""
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    interpolated = alpha * real + (1.0 - alpha) * fake

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = tape.gradient(pred, interpolated)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
    return tf.reduce_mean((grad_norm - 1.0) ** 2)


def run_wgan_transformer_baseline(data, n_x):
    print("\n" + "=" * 60)
    print("  MODEL E: WGAN-GP Transformer (Keras Translation)")
    print("  Ref: Arbat et al., IAAI '22 / AAAI-22")
    print("  Source: baselines/wgan-gp-transformer/")
    print("=" * 60)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_test  = data["X_test"]

    generator = build_transformer_generator(n_x)
    critic    = build_critic(d_model=64)

    opt_G = Adam(learning_rate=TF_LR_G, beta_1=0.5, beta_2=0.9)
    opt_D = Adam(learning_rate=TF_LR_D, beta_1=0.5, beta_2=0.9)

    g_params = generator.count_params()
    c_params = critic.count_params()

    print(f"  Generator: Transformer Enc-Dec ({TF_LAYERS}L, d={TF_D_MODEL}, h={TF_NHEAD})")
    print(f"  Critic: SequenceCritic (3-layer MLP, Tanh)")
    print(f"  Loss: WGAN-GP (λ={TF_LAMBDA_GP}) + L1")
    print(f"  Generator Params: {g_params:,}  |  Critic Params: {c_params:,}")
    generator.summary()

    n_batches = len(X_train) // TF_BATCH
    best_val_loss = float('inf')
    best_weights = None
    patience_count = 0

    t0 = time.perf_counter()
    epoch_pbar = trange(1, TF_EPOCHS + 1, desc="  WGAN-GP Training", colour='magenta')

    for epoch in epoch_pbar:
        perm = np.random.permutation(len(X_train))
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        g_losses, d_losses = [], []

        batch_pbar = trange(n_batches, desc=f"    Epoch {epoch}", leave=False,
                            colour='cyan')
        for b in batch_pbar:
            s = b * TF_BATCH
            e = s + TF_BATCH
            xb = tf.constant(X_shuf[s:e])
            yb = tf.constant(y_shuf[s:e].reshape(-1, 1))
            bs = xb.shape[0]

            # ═══ Critic Training ═══
            for _ in range(TF_CRITIC_ITERS):
                with tf.GradientTape() as d_tape:
                    fake_pred = generator(xb, training=False)
                    fake_pred = tf.reshape(fake_pred, [-1, 1])

                    d_real = critic(yb, training=True)
                    d_fake = critic(fake_pred, training=True)

                    gp = gradient_penalty(critic, yb, fake_pred, bs)
                    d_loss = (tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
                              + TF_LAMBDA_GP * gp)

                d_grads = d_tape.gradient(d_loss, critic.trainable_variables)
                opt_D.apply_gradients(zip(d_grads, critic.trainable_variables))
                d_losses.append(float(d_loss))

            # ═══ Generator Training ═══
            with tf.GradientTape() as g_tape:
                fake_pred = generator(xb, training=True)
                fake_pred_r = tf.reshape(fake_pred, [-1, 1])

                g_adv = -tf.reduce_mean(critic(fake_pred_r, training=False))
                g_recon = tf.reduce_mean(tf.abs(fake_pred_r - yb))
                g_loss = g_recon + g_adv

            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_grads = [tf.clip_by_norm(g, 1.0) for g in g_grads]
            opt_G.apply_gradients(zip(g_grads, generator.trainable_variables))
            g_losses.append(float(g_loss))

            batch_pbar.set_postfix({'G': f'{g_losses[-1]:.4f}',
                                    'D': f'{d_losses[-1]:.4f}'})

        # Validation
        val_pred = generator.predict(X_val, batch_size=TF_BATCH, verbose=0)
        val_mse = float(np.mean((val_pred.ravel() - y_val) ** 2))

        epoch_pbar.set_postfix({'val_mse': f'{val_mse:.5f}',
                                'G_loss': f'{np.mean(g_losses):.4f}',
                                'D_loss': f'{np.mean(d_losses):.4f}'})

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_weights = generator.get_weights()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= TF_PATIENCE:
                tqdm.write(f"  Early stop at epoch {epoch} (patience={TF_PATIENCE})")
                break

    train_time = time.perf_counter() - t0

    generator.set_weights(best_weights)

    print("  Running inference on test set...")
    t1 = time.perf_counter()
    preds = generator.predict(X_test, batch_size=TF_BATCH, verbose=0).ravel()
    infer_time = time.perf_counter() - t1

    metrics = compute_metrics(data["y_test"], preds, data["scaler_y"],
                              train_time, infer_time)
    metrics["method"] = "WGAN-GP Transformer"
    metrics["params"] = g_params
    metrics["cluster_breakdown"] = per_cluster_metrics(
        data["y_test"], preds, data["c_test"], data["scaler_y"]
    )

    print(f"\n  ╔═══ WGAN-GP TRANSFORMER RESULTS ═══╗")
    print(f"  ║ RMSE  = {metrics['rmse_orig']:.4f}")
    print(f"  ║ MAE   = {metrics['mae_orig']:.4f}")
    print(f"  ║ SMAPE = {metrics['smape_orig']:.1f}%")
    print(f"  ║ Train = {train_time:.1f}s | Infer = {infer_time:.3f}s")
    print(f"  ║ Params = {g_params:,}")
    print(f"  ╚═══════════════════════════════════════╝")
    return metrics


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Classical DL Baselines — AQRNN Ablation Study  ║")
    print("║  TensorFlow / Keras on GPU                      ║")
    print("╚══════════════════════════════════════════════════╝\n")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  GPU detected: {gpus[0].name}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        print("  WARNING: No GPU detected, running on CPU")

    data, n_x = load_shared_data()
    results = {}

    # D. LSTM (corrected)
    print("\n" + "─" * 60)
    lstm_metrics = run_lstm_baseline(data, n_x)
    results["D_LSTM"] = lstm_metrics

    # Clear session between models
    keras.backend.clear_session()

    # E. WGAN-GP Transformer
    print("\n" + "─" * 60)
    wgan_metrics = run_wgan_transformer_baseline(data, n_x)
    results["E_WGAN_GP_Transformer"] = wgan_metrics

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY — Classical DL Baselines")
    print("=" * 70)
    print(f"  {'Model':<30} {'RMSE':>10} {'MAE':>10} {'SMAPE':>10} "
          f"{'Train(s)':>10} {'Params':>10}")
    print("  " + "-" * 80)
    for name, m in results.items():
        print(f"  {m['method']:<30} {m['rmse_orig']:>10.4f} {m['mae_orig']:>10.4f} "
              f"{m['smape_orig']:>9.1f}% {m['train_time_s']:>10.1f} "
              f"{m.get('params','?'):>10}")

    out_path = "classical_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")
