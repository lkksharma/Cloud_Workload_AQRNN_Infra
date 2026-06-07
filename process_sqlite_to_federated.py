
import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime
import tqdm

# --- CONFIGURATION ---
DB_PATH = "gwa_t_2_anon_jobs_sqlite/Grid5000.sqlite"
OUTPUT_DIR = "federated_data"
RESAMPLE_FREQ = "5min"
MIN_JOBS_PER_CLUSTER = 1000  # Skip tiny testing clusters if any

# Define Feature Engineering columns
# We need to map raw SQL columns to our model features
# Feature Map:
# TotalJobs -> Count(JobID)
# TotalReqCPUs -> Sum(ReqNProcs)
# AvgReqTime -> Mean(ReqTime)
# TotalReqMem -> Sum(ReqMemory)  (Note: UsedMemory might be -1 or null often, Req is safer)
# UserDiversity -> Count(Distinct UserID)
# TrueCPUUtil -> (Inferred, possibly UsedCPUTime / RunTime / NProc) - but standardized later

def process_cluster_data(db_path, output_dir):
    """
    Extracts job data from SQLite, splits by LastRunSiteID (Physical Cluster),
    resamples to time-series, and saves as per-client CSVs.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    conn = sqlite3.connect(db_path)
    
    # 1. Get List of Clusters
    print("Scanning for clusters (LastRunSiteID)...")
    clusters_df = pd.read_sql_query("SELECT LastRunSiteID, COUNT(*) as Count FROM jobs GROUP BY LastRunSiteID", conn)
    valid_clusters = clusters_df[clusters_df["Count"] > MIN_JOBS_PER_CLUSTER]["LastRunSiteID"].tolist()
    
    print(f"Found {len(valid_clusters)} valid clusters: {valid_clusters}")

    # 2. Process each cluster
    for cluster_id in tqdm.tqdm(valid_clusters, desc="Processing Clusters"):
        safe_cluster_name = cluster_id.replace("/", "_").replace(" ", "")
        
        # Query raw jobs for this cluster
        query = f"""
            SELECT 
                SubmitTime, 
                ReqNProcs, 
                ReqTime, 
                ReqMemory, 
                UserID,
                CASE WHEN RunTime > 0 THEN UsedCPUTime / RunTime ELSE 0 END as CalcCPUUtil
            FROM jobs 
            WHERE LastRunSiteID = '{cluster_id}' 
            AND SubmitTime > 0
            ORDER BY SubmitTime ASC
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Convert SubmitTime to DateTime
        df["datetime"] = pd.to_datetime(df["SubmitTime"], unit='s')
        df.set_index("datetime", inplace=True)
        
        # Resample logic
        # We aggregate into time windows
        resampled = df.resample(RESAMPLE_FREQ).agg({
            "ReqNProcs": "sum",       # Total CPUS requested in this window
            "ReqTime": "mean",        # Avg Job Duration requested
            "ReqMemory": "sum",       # Total Mem requested
            "UserID": "nunique",      # User Diversity
            "CalcCPUUtil": "mean",    # Avg CPU Utilization (proxy)
            "SubmitTime": "count"     # Total Job Count (using any non-null col)
        })
        
        resampled.rename(columns={"SubmitTime": "TotalJobs", "ReqNProcs": "TotalReqCPUs", "ReqTime": "AvgReqTime", "ReqMemory": "TotalReqMem", "UserID": "UserDiversity", "CalcCPUUtil": "TrueCPUUtil"}, inplace=True)
        
        # Fill NaNs (empty windows have 0 load)
        resampled.fillna(0, inplace=True)
        
        # Add Cyclical Time Features
        resampled["hours"] = resampled.index.hour + resampled.index.minute / 60.0
        resampled["dow"] = resampled.index.dayofweek
        resampled["hour_sin"] = np.sin(2 * np.pi * resampled["hours"] / 24)
        resampled["hour_cos"] = np.cos(2 * np.pi * resampled["hours"] / 24)
        resampled["dow_sin"] = np.sin(2 * np.pi * resampled["dow"] / 7)
        resampled["dow_cos"] = np.cos(2 * np.pi * resampled["dow"] / 7)
        
        # Drop raw time cols if preferred, but keep for debug
        resampled.drop(columns=["hours", "dow"], inplace=True)
        
        # Save
        if len(resampled) > 50: # Ensure minimal data size
            out_path = os.path.join(output_dir, f"client_{safe_cluster_name}.csv")
            resampled.to_csv(out_path)
            # print(f"Saved {safe_cluster_name}: {len(resampled)} rows")
    
    conn.close()
    print("Processing Complete.")

if __name__ == "__main__":
    process_cluster_data(DB_PATH, OUTPUT_DIR)
