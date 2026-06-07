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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONSTANTS ---
RNG = np.random.RandomState(42)
torch.manual_seed(42)

# --- 1. DATA EXTRACTION & CLUSTERING (Identical to AQRNN) ---
csv_path = "grid5000_hybrid_clean.csv"

def load_and_process_data_clustered(k=3, n_components=4, save_kmeans_path="kmeans_baseline.pkl"):
    if not os.path.exists(csv_path):
        print("Error: CSV not found.")
        sys.exit(1)

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
    elif "WindowStart" in df.columns:
        df["datetime"] = pd.to_datetime(df["WindowStart"], unit='s')
        df.set_index("datetime", inplace=True)

    if "hours" not in df.columns:
        df["hours"] = df.index.hour + df.index.minute / 60.0
        df["dow"] = df.index.dayofweek
        df["hour_sin"] = np.sin(2*np.pi* df["hours"]/24)
        df["hour_cos"] = np.cos(2*np.pi*df["hours"]/24)
        df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
        df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    features = ["TotalJobs", "TotalReqCPUs", "AvgReqTime", "TotalReqMem", "UserDiversity",
                "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target = "TrueCPUUtil"

    # --- Match AQRNN Scaling Strategy ---
    print("Applying QuantileTransformer to inputs...")
    scaler_x = QuantileTransformer(output_distribution='uniform', n_quantiles=min(1000, len(df)))
    scaler_y = QuantileTransformer(output_distribution='normal', n_quantiles=1000)

    X_raw = df[features].values
    X_scaled = scaler_x.fit_transform(X_raw)
    
    # PCA
    if n_components is not None and n_components < X_scaled.shape[1]:
        print(f"Applying PCA: Reducing {X_scaled.shape[1]} features to {n_components} components...")
        pca = PCA(n_components=n_components, random_state=42)
        X_final = pca.fit_transform(X_scaled)
    else:
        X_final = X_scaled
    
    y_raw = df[[target]].values
    y_scaled = scaler_y.fit_transform(y_raw)

    print(f"Training K-Means (K={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, init='k-means++')
    clusters = kmeans.fit_predict(X_final)
    
    return X_final, y_scaled, clusters, scaler_y

def create_dataset_clustered(X, y, clusters, time_steps=2):
    Xs, ys, cs = [], [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
        cs.append(clusters[i + time_steps - 1]) 
    return np.array(Xs), np.array(ys), np.array(cs)

# --- 2. LSTM MODEL DEFINITION ---
class LSTMExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=1):
        super(LSTMExpert, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Standard LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Classical Head (MLP) matching AQRNN capacity
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM output
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        # MLP Head
        out = self.fc(out)
        return out

def calc_accuracy_multi(y_true, y_pred, thresholds=(0.5, 1.0, 1.5)):
    out = {}
    diff = np.abs(y_true - y_pred)
    for t in thresholds:
        correct = diff < t
        out[t] = np.mean(correct) * 100.0
    return out

# --- 3. TRAINING ENGINE ---
def train_lstm_moe(
    n_clusters=3,
    n_components=4,
    seq_len=2,
    hidden_dim=64,
    n_epochs=10,
    batch_size=256,
    lr=1e-3,
    device_mode="cuda",
    seed=42
):
    # Set Device with Explicit Fallback Warning
    if device_mode == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using Device: NVIDIA GPU (CUDA)")
        else:
            device = torch.device("cpu")
            print("Using Device: CPU (CUDA requested but not available!)")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # Load Data
    X_s, y_s, clusters_s, scaler_y = load_and_process_data_clustered(k=n_clusters, n_components=n_components)
    X_seq, y_seq, c_seq = create_dataset_clustered(X_s, y_s, clusters_s, time_steps=seq_len)
    
    # Train/Test Split (70/15/15)
    N = len(X_seq)
    train_size = int(N * 0.7)
    val_size = int(N * 0.15)
    
    X_train = torch.tensor(X_seq[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y_seq[:train_size], dtype=torch.float32)
    c_train = c_seq[:train_size]
    
    X_val = torch.tensor(X_seq[train_size:train_size+val_size], dtype=torch.float32)
    y_val = torch.tensor(y_seq[train_size:train_size+val_size], dtype=torch.float32)
    c_val = c_seq[train_size:train_size+val_size]
    
    X_test = torch.tensor(X_seq[train_size+val_size:], dtype=torch.float32)
    y_test = torch.tensor(y_seq[train_size+val_size:], dtype=torch.float32)
    c_test = c_seq[train_size+val_size:]

    experts = {}
    input_dim = X_train.shape[2]
    
    print(f"\nBaseline Config: LSTM (Hidden={hidden_dim}) | Clusters={n_clusters} | Input Features={input_dim}")

    # --- TRAIN EXPERTS ---
    for k in range(n_clusters):
        print(f"\n=== Training LSTM Expert {k} ===")
        
        # Filter Data by Cluster
        mask_train = (c_train == k)
        mask_val = (c_val == k)
        
        X_tr_k = X_train[mask_train].to(device)
        y_tr_k = y_train[mask_train].to(device)
        X_val_k = X_val[mask_val].to(device)
        y_val_k = y_val[mask_val].to(device)
        
        if len(X_tr_k) < 32:
            print(f"Skipping Cluster {k} (Insufficient Data: {len(X_tr_k)})")
            continue
            
        print(f"Cluster {k} Samples: {len(X_tr_k)}")
        
        # Create DataLoader
        dataset = TensorDataset(X_tr_k, y_tr_k)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize Model
        model = LSTMExpert(input_dim, hidden_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training Loop
        best_val_rmse = float('inf')
        patience = 0
        
        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Cluster {k} Epoch {epoch}", leave=False)
            for batch_x, batch_y in pbar:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if len(X_val_k) > 0:
                    val_preds = model(X_val_k)
                    val_rmse = torch.sqrt(criterion(val_preds, y_val_k)).item()
                else:
                    val_rmse = 0.0
            
            print(f"  Epoch {epoch}: Train Loss {epoch_loss/len(loader):.4f} | Val RMSE {val_rmse:.4f}")
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), f"lstm_expert_{k}.pth")
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    print("  Early stopping.")
                    break
        
        # Load best weights
        model.load_state_dict(torch.load(f"lstm_expert_{k}.pth"))
        experts[k] = model

    # --- EVALUATION (ROUTER) ---
    print("\n" + "="*40)
    print("=== FINAL STEP: Mixture of Experts Inference ===")
    print("="*40)
    
    final_preds = []
    final_true = []
    
    for k in range(n_clusters):
        mask_test = (c_test == k)
        if not np.any(mask_test): continue
        
        X_test_k = X_test[mask_test].to(device)
        y_test_k = y_test[mask_test].cpu().numpy()
        
        # Route
        if k in experts:
            model = experts[k]
        elif 0 in experts:
            model = experts[0] # Fallback
        else:
            continue
            
        model.eval()
        with torch.no_grad():
            preds_k = model(X_test_k).cpu().numpy()
            
        # Metrics
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
        print(f"\n>>> Global Baseline RMSE: {rmse:.4f} | MAE: {mae:.4f}")

if __name__ == "__main__":
    train_lstm_moe(n_epochs=10, n_clusters=3, n_components=4, batch_size=256, device_mode="cuda")