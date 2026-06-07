#!/usr/bin/env python
"""
===========================================================================
AuverGrid (GWA-T-4) Data Processing Pipeline
===========================================================================
Converts the AuverGrid SQLite database into:
  1. Per-site CSV files in federated_data_auverGrid/ (for federated training)
  2. A merged centralized CSV auverGrid_hybrid_clean.csv (for ablation)

Schema mapping (identical to Grid'5000 process_sqlite_to_federated.py):
  TotalJobs     → COUNT(JobID)
  TotalReqCPUs  → SUM(ReqNProcs)
  AvgReqTime    → MEAN(ReqTime)
  TotalReqMem   → SUM(ReqMemory)
  UserDiversity → COUNT(DISTINCT UserID)
  TrueCPUUtil   → MEAN(UsedCPUTime / RunTime)  [proxy]

Usage:
  python process_auverGrid.py
===========================================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime
import tqdm

# --- CONFIGURATION ---
# Relative to the ExQ project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "gwa_t_4_anon_jobs_sqlite", "anon_jobs.db3")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "federated_data_auverGrid")
MERGED_CSV = os.path.join(PROJECT_ROOT, "auverGrid_hybrid_clean.csv")
RESAMPLE_FREQ = "5min"
MIN_JOBS_PER_CLUSTER = 1000  # Skip tiny testing clusters if any


def process_cluster_data(db_path, output_dir, merged_csv_path):
    """
    Extracts job data from AuverGrid SQLite, splits by LastRunSiteID,
    resamples to time-series, and saves as per-client CSVs + one merged CSV.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    conn = sqlite3.connect(db_path)

    # 1. Get List of Clusters (Sites)
    print("Scanning for clusters (LastRunSiteID)...")
    clusters_df = pd.read_sql_query(
        "SELECT LastRunSiteID, COUNT(*) as Count FROM Jobs GROUP BY LastRunSiteID",
        conn
    )
    valid_clusters = clusters_df[clusters_df["Count"] > MIN_JOBS_PER_CLUSTER]["LastRunSiteID"].tolist()
    print(f"Found {len(valid_clusters)} valid clusters: {valid_clusters}")
    print(f"Job counts: {dict(zip(clusters_df['LastRunSiteID'], clusters_df['Count']))}")

    all_site_dfs = []

    # 2. Process each cluster
    for cluster_id in tqdm.tqdm(valid_clusters, desc="Processing Clusters"):
        safe_cluster_name = cluster_id.replace("/", "_").replace(" ", "")

        # Query raw jobs for this cluster
        # Filter: SubmitTime > 0, RunTime > 0 (exclude cancelled jobs)
        query = f"""
            SELECT 
                SubmitTime, 
                ReqNProcs, 
                ReqTime, 
                ReqMemory, 
                UserID,
                CASE WHEN RunTime > 0 THEN UsedCPUTime / RunTime ELSE 0 END as CalcCPUUtil
            FROM Jobs 
            WHERE LastRunSiteID = '{cluster_id}' 
            AND SubmitTime > 0
            ORDER BY SubmitTime ASC
        """

        df = pd.read_sql_query(query, conn)

        # Convert SubmitTime to DateTime
        df["datetime"] = pd.to_datetime(df["SubmitTime"], unit='s')
        df.set_index("datetime", inplace=True)

        # Resample logic — aggregate into time windows
        resampled = df.resample(RESAMPLE_FREQ).agg({
            "ReqNProcs": "sum",       # Total CPUs requested in this window
            "ReqTime": "mean",        # Avg Job Duration requested
            "ReqMemory": "sum",       # Total Mem requested
            "UserID": "nunique",      # User Diversity
            "CalcCPUUtil": "mean",    # Avg CPU Utilization (proxy)
            "SubmitTime": "count"     # Total Job Count
        })

        resampled.rename(columns={
            "SubmitTime": "TotalJobs",
            "ReqNProcs": "TotalReqCPUs",
            "ReqTime": "AvgReqTime",
            "ReqMemory": "TotalReqMem",
            "UserID": "UserDiversity",
            "CalcCPUUtil": "TrueCPUUtil"
        }, inplace=True)

        # Fill NaNs (empty windows have 0 load)
        resampled.fillna(0, inplace=True)

        # Add Cyclical Time Features
        resampled["hours"] = resampled.index.hour + resampled.index.minute / 60.0
        resampled["dow"] = resampled.index.dayofweek
        resampled["hour_sin"] = np.sin(2 * np.pi * resampled["hours"] / 24)
        resampled["hour_cos"] = np.cos(2 * np.pi * resampled["hours"] / 24)
        resampled["dow_sin"] = np.sin(2 * np.pi * resampled["dow"] / 7)
        resampled["dow_cos"] = np.cos(2 * np.pi * resampled["dow"] / 7)

        # Drop raw time cols
        resampled.drop(columns=["hours", "dow"], inplace=True)

        # Save per-site CSV
        if len(resampled) > 50:
            out_path = os.path.join(output_dir, f"client_{safe_cluster_name}.csv")
            resampled.to_csv(out_path)
            print(f"  Saved {safe_cluster_name}: {len(resampled)} rows")

            # Tag with site for merged CSV
            site_df = resampled.copy()
            site_df["SiteID"] = cluster_id
            all_site_dfs.append(site_df)

    conn.close()

    # 3. Build merged centralized CSV
    if all_site_dfs:
        merged = pd.concat(all_site_dfs, axis=0)
        merged.sort_index(inplace=True)
        # Drop the SiteID column for the centralized version
        merged.drop(columns=["SiteID"], inplace=True)
        merged.to_csv(merged_csv_path)
        print(f"\nMerged centralized CSV: {merged_csv_path} ({len(merged)} rows)")
    else:
        print("ERROR: No site data was processed!")

    print("Processing Complete.")


if __name__ == "__main__":
    print(f"DB Path:    {DB_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Merged CSV: {MERGED_CSV}")
    print()
    process_cluster_data(DB_PATH, OUTPUT_DIR, MERGED_CSV)
