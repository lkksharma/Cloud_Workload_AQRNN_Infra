# CFL-AQRNN-RC: Regime-Decomposed Federated Quantum Experts for Multi-Tenant Cloud Workload Prediction

[![Paper](https://img.shields.io/badge/Paper-ICDM%202026-blue)](#citation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.34%2B-orange)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reference implementation of **CFL-AQRNN-RC**, a regime-aware federated framework for multi-tenant cloud workload forecasting that combines a Mixture of Quantum Experts with Clustered Federated Learning and auto-adaptive residual correction. Submitted to **IEEE ICDM 2026 Applied Track**.

---

## Headline Results

| Method | RMSE ↓ | MAE ↓ | Trainable Params |
|---|---:|---:|---:|
| FedAvg SimpleRNN | 156.28 | 114.08 | 4,481 |
| FedAvg GRU-Only | 155.63 | 113.37 | 13,505 |
| FedAvg esDNN | 156.05 | 113.50 | 20,161 |
| FedAvg Bi-LSTM | 155.38 | 113.22 | 35,457 |
| Clustered Fed LSTM | 153.46 | 114.05 | 33,153 |
| **CFL-AQRNN-RC (ours)** | **147.21** | **105.54** | **<500** |

Grid'5000 Hybrid corpus, 13 federated clients, 8 rounds × 3 local epochs, 14,047 test samples, identical federated protocol across all methods.

**5.3% lower RMSE than the strongest classical baseline, with ~70× fewer trainable parameters.**

---

## One-Command Reproduction

```bash
# Clone, install, and reproduce the headline number
git clone https://github.com/lkksharma/Cloud_Workload_AQRNN_Infra.git
cd Cloud_Workload_AQRNN_Infra
pip install -r requirements.txt
python federated_aqrnn.py --dataset grid5000 --rounds 8 --epochs 3 --clients 13 --output results/main_run/
```

Expected output: `RMSE 147.21, MAE 105.54` in `results/main_run/metrics.json` (±0.8 RMSE across seeds).

---

## Installation

### Requirements

- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ (for `pennylane.lightning.gpu`)
- ~8 GB GPU memory (tested on A100; runs on smaller GPUs with reduced batch size)

### Setup

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies (full list in `requirements.txt`):
- `pennylane>=0.34`, `pennylane-lightning-gpu`
- `torch>=2.0`
- `scikit-learn>=1.3`
- `numpy`, `pandas`, `scipy`, `matplotlib`

---

## Reproducing Each Table and Figure

| Paper Element | Command |
|---|---|
| **Table IV** (AQRNN variants A–D) | `python ablation_study.py --variants A B C D` |
| **Table V** (Classical FedAvg baselines) | `python classical_baselines.py --all` |
| **Table VI** (CFL on LSTM vs AQRNN) | `python federated_lstm_comparison.py && python federated_aqrnn.py` |
| **Table VII** (Cumulative ablation A→E) | `python ablation_study.py --cumulative` |
| **Table VIII** (Final comparison) | `python benchmark_baselines.py --final-table` |
| **Table IX** (Per-cluster breakdown) | `python federated_aqrnn.py --per-cluster` |
| **Fig. 1** (K-Means cluster validation) | `python analyze_clustering.py --plot` |
| **Fig. 5** (Final comparison plot) | `python generate_plots.py --method-comparison` |

Each script writes to `results/<table_name>/` and logs to stdout. Total walltime ≈ 6 GPU-hours for the full reproduction on a single A100.

---

## Repository Structure

```
.
├── federated_aqrnn.py          # Main CFL-AQRNN-RC training entry point (Table VII Stage D)
├── cfl_full_moe.py             # Full pipeline: routing + CFL + Ridge correction (Stage E)
├── aqrnn.py                    # AQRNN cell: variational quantum circuit with amplitude-damping channel
├── aqrnn_cluster.py            # Mixture-of-Quantum-Experts router
├── post_quantum_ridge.py       # Auto-adaptive Ridge residual corrector
│
├── classical_baselines.py      # FedAvg esDNN / GRU / Bi-LSTM / SimpleRNN baselines (Table V)
├── federated_lstm_comparison.py # CFL on classical LSTM backbone (Table VI)
├── federated_esdnn_comparison.py
├── benchmark_baselines.py      # Cross-method benchmark harness
│
├── ablation_study.py           # Stage-by-stage ablation (Table IV, Table VII)
├── analyze_clustering.py       # K-Means validation metrics (Fig. 1)
├── generate_plots.py           # Reproduces Fig. 5 (method_comparison.pdf)
│
├── grid5000_hybrid_clean.csv   # Processed Grid'5000 workload traces (13 sites)
├── auverGrid_hybrid_clean.csv  # AuverGrid corpus (planned extension)
├── federated_data/             # Pre-sharded train/val/test splits per client
│
├── aqrnn_final.pkl             # Trained quantum experts (408 params)
├── cfl_experts_updated.pkl     # Federated CFL checkpoints
├── kmeans_model.pkl            # Fitted K-Means router (3 centroids)
│
├── requirements.txt
└── README.md
```

---

## Hyperparameters

All defaults match Table III of the paper:

| Parameter | Value |
|---|---|
| Federated rounds | 8 |
| Local epochs per round | 3 |
| Federated clients | 13 (Grid'5000 site shards) |
| Aggregation | FedAvg, per-expert under CFL |
| Mixture-of-Experts | K = 3 (Stable / Variable / Bursty) |
| Per-expert capacity (C_max) | 5 clients |
| Distillation strength (α) | 0.3 |
| Amplitude-damping retention (f) | 0.95 |
| Quantum backend | PennyLane `lightning.gpu` |
| Optimizer | Adam, gradient-norm clip δ = 1.0 |
| L2 regularization | λ_Q = λ_C = 1e-3 |
| Train / Val / Test split | 70 / 15 / 15, temporal |

Pass `--config configs/custom.yaml` to any training script to override.

---

## Dataset

The Grid'5000 multi-site corpus (CPU utilization traces from 13 geographically distributed clusters) is the primary benchmark. Cleaned and pre-sharded CSVs live under `federated_data/`. AuverGrid is included as a second-dataset reference for the journal extension.

Citation for the Grid'5000 testbed:

> D. Balouek et al., *"Adding virtualization capabilities to the Grid'5000 testbed,"* in *Cloud Computing and Services Science*, CCIS vol. 367, Springer, 2013, pp. 3–20. https://www.grid5000.fr

---

## Hardware Footprint

- **Trainable quantum parameters:** 408 (three AQRNN experts of ~130 angles each, plus three 5→1 MLP heads)
- **Per-round federated payload:** ~2 KB (FP32) — versus ~140 KB for Bi-LSTM
- **NISQ requirements:** 6 qubits per expert (4 input + 2 hidden), circuit depth ~20 including adjoint uncomputing. Compatible with mid-circuit measurement primitives on IBM Heron, Quantinuum H-2, IonQ Forte-Enterprise.

---

## Citation

If this work is useful in your research, please cite the paper:

```bibtex
@inproceedings{sharma2026heterogeneity,
  title     = {When Heterogeneity Is the Signal: Regime-Decomposed Federated Quantum Experts for Multi-Tenant Cloud Workload Prediction},
  author    = {Sharma, Lakksh and Sharma, Krish and Bedi, Jatin},
  booktitle = {IEEE International Conference on Data Mining (ICDM), Applied Track},
  year      = {2026},
  address   = {Shenyang, China}
}
```

---

## Acknowledgments

The authors gratefully acknowledge the Grid'5000 community for releasing the multi-site workload traces used throughout this study. Grid'5000 is supported by a scientific interest group hosted by Inria and including CNRS, RENATER, and several universities and other organisations. See https://www.grid5000.fr.

---

## License

Released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or to report issues, please open a GitHub issue or contact:

- Lakksh Sharma — `lksharma_be23@thapar.edu`
- Krish Sharma — `ksharma_be23@thapar.edu`
- Jatin Bedi — `jatin.bedi@thapar.edu`

Thapar Institute of Engineering and Technology, Patiala, Punjab, India.
