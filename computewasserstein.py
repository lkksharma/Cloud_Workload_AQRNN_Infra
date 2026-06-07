"""
==============================================================================
Path A: Compute Wasserstein Distances Between Language Embedding Clouds
==============================================================================

WHAT THIS DOES:
1. Loads XLM-RoBERTa-base (the pre-trained model, NOT your fine-tuned one)
2. Extracts token-level embeddings for each language's WikiANN test set
3. Computes W2 (Wasserstein-2) distance between English and every other language
4. Plots W2 distance vs F1 gap — if this is linear, the OT theory is validated

WHY USE THE PRE-TRAINED MODEL (not fine-tuned)?
The W2 distance measures the INHERENT distance between language representations
in the multilingual space. Fine-tuning on English NER would bias the space
toward English. The pre-trained model gives the "natural" language distances.

WHAT IS W2 DISTANCE?
Imagine English embeddings as a cloud of points, Hindi as another cloud.
W2 measures the minimum "effort" to move one cloud into the shape of the other.
Formally: W2 = (min over couplings ∫ ||x-y||² dπ(x,y))^(1/2)

COMPUTATIONAL TRICK:
Exact W2 in high dimensions is expensive. We use two approaches:
1. Sinkhorn approximation (entropic OT) — fast, differentiable, standard
2. Sliced Wasserstein — project to 1D, compute exact W1/W2, average over projections

Both give similar rankings. Sliced is faster; Sinkhorn is what the paper would use.

EXPECTED RUNTIME: ~30-40 minutes on RTX A4000
- Embedding extraction: ~2 min per language
- W2 computation: ~1 min per language pair

PREREQUISITE:
    pip install POT    (Python Optimal Transport library)
"""

import os
import json
import torch
import numpy as np
import time
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    model_name = "xlm-roberta-base"
    
    # Languages — must match what you ran in eval_more_languages.py
    source_lang = "en"
    all_langs = ["en", "fr", "de", "pt", "es", "ar", "hi", "bn", "zh", "ta"]
    
    # Embedding extraction
    max_samples = 5000      # Samples per language (more = better estimate, slower)
    max_length = 128
    batch_size = 64         # Large batch OK — inference only
    embedding_layer = -1    # Last hidden layer (standard choice)
    
    # W2 computation
    sinkhorn_reg = 0.05     # Entropic regularization (smaller = closer to true W2)
    n_sliced_projections = 500  # For sliced Wasserstein
    subsample_for_w2 = 2000    # Subsample embeddings for W2 (full is too slow)
    
    output_dir = "./ner_baseline_results"


# ============================================================================
# STEP 1: EXTRACT EMBEDDINGS
# ============================================================================

def extract_embeddings(model, tokenizer, dataset, config, device):
    """
    Extract token-level embeddings from XLM-R for a given dataset.
    
    WHAT WE EXTRACT:
    For each sentence, XLM-R produces a hidden state vector (768-dim) for
    every token position. We collect ALL token embeddings (excluding padding
    and special tokens) into one big matrix.
    
    The result is a cloud of points in R^768 — one point per token.
    Each language produces a different cloud. W2 measures cloud distance.
    
    RETURNS: numpy array of shape (num_tokens, 768)
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    all_embeddings = []
    
    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=config.max_length,
            is_split_into_words=True,
            padding="max_length",
            return_tensors=None,
        )
    
    # Limit samples
    if len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format("torch")
    loader = DataLoader(tokenized, batch_size=config.batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get hidden states from the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
            
            # Take the last hidden layer
            hidden = outputs.hidden_states[config.embedding_layer]  # (batch, seq_len, 768)
            
            # Collect only real tokens (not padding, not special tokens)
            # Special tokens: position 0 (<s>) and wherever </s> appears
            for i in range(hidden.size(0)):
                mask = attention_mask[i].bool()
                ids = input_ids[i]
                
                # Skip <s> (position 0) and </s> and <pad>
                valid = mask.clone()
                valid[0] = False  # skip <s>
                # Find </s> token (id=2 for XLM-R) and skip it
                eos_positions = (ids == 2).nonzero(as_tuple=True)[0]
                for pos in eos_positions:
                    valid[pos] = False
                
                token_embeds = hidden[i][valid].cpu().numpy()
                all_embeddings.append(token_embeds)
    
    # Concatenate all token embeddings into one big matrix
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


# ============================================================================
# STEP 2: COMPUTE WASSERSTEIN DISTANCES
# ============================================================================

def compute_sinkhorn_w2(embeddings_a, embeddings_b, reg=0.05, max_samples=2000):
    """
    Compute approximate W2 distance using Sinkhorn (entropic OT).
    
    WHY SINKHORN:
    Exact W2 between n points is O(n³) — too slow for thousands of embeddings.
    Sinkhorn adds entropy regularization, making it O(n² × iterations).
    With reg → 0, it converges to true W2. reg=0.05 is a good tradeoff.
    
    Uses the POT (Python Optimal Transport) library.
    """
    import ot
    
    # Subsample if too many points
    if len(embeddings_a) > max_samples:
        idx = np.random.choice(len(embeddings_a), max_samples, replace=False)
        embeddings_a = embeddings_a[idx]
    if len(embeddings_b) > max_samples:
        idx = np.random.choice(len(embeddings_b), max_samples, replace=False)
        embeddings_b = embeddings_b[idx]
    
    n_a = len(embeddings_a)
    n_b = len(embeddings_b)
    
    # Uniform weights (each point has equal mass)
    a_weights = np.ones(n_a) / n_a
    b_weights = np.ones(n_b) / n_b
    
    # Cost matrix: squared Euclidean distance between all pairs
    # This is the C(x,y) = ||x - y||² in the W2 formula
    cost_matrix = ot.dist(embeddings_a, embeddings_b, metric='sqeuclidean')
    
    # Normalize cost matrix for numerical stability
    cost_matrix = cost_matrix / cost_matrix.max()
    
    # Sinkhorn algorithm
    transport_cost = ot.sinkhorn2(a_weights, b_weights, cost_matrix, reg=reg)
    
    # W2 = sqrt(transport_cost)
    w2 = np.sqrt(float(transport_cost))
    
    return w2


def compute_sliced_wasserstein(embeddings_a, embeddings_b, n_projections=500, max_samples=2000):
    """
    Compute Sliced Wasserstein distance.
    
    WHY SLICED:
    Instead of computing W2 in 768 dimensions (expensive), project both clouds
    onto random 1D directions and compute exact W1 in 1D (just sorting!).
    Average over many random directions. Much faster than Sinkhorn.
    
    Good as a sanity check — rankings should match Sinkhorn.
    """
    # Subsample
    if len(embeddings_a) > max_samples:
        idx = np.random.choice(len(embeddings_a), max_samples, replace=False)
        embeddings_a = embeddings_a[idx]
    if len(embeddings_b) > max_samples:
        idx = np.random.choice(len(embeddings_b), max_samples, replace=False)
        embeddings_b = embeddings_b[idx]
    
    d = embeddings_a.shape[1]  # 768
    
    # Random projection directions (unit vectors on the sphere)
    projections = np.random.randn(n_projections, d)
    projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
    
    distances = []
    for proj in projections:
        # Project both clouds onto this 1D direction
        proj_a = embeddings_a @ proj  # shape: (n_a,)
        proj_b = embeddings_b @ proj  # shape: (n_b,)
        
        # Sort and compute 1D Wasserstein (just the difference of sorted values)
        proj_a_sorted = np.sort(proj_a)
        proj_b_sorted = np.sort(proj_b)
        
        # Interpolate to same number of points if different sizes
        if len(proj_a_sorted) != len(proj_b_sorted):
            n = min(len(proj_a_sorted), len(proj_b_sorted))
            proj_a_sorted = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(proj_a_sorted)),
                proj_a_sorted,
            )
            proj_b_sorted = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(proj_b_sorted)),
                proj_b_sorted,
            )
        
        # 1D W2 distance = sqrt(mean of squared differences of sorted values)
        w2_1d = np.sqrt(np.mean((proj_a_sorted - proj_b_sorted) ** 2))
        distances.append(w2_1d)
    
    # Average over all projections
    return np.mean(distances)


# ============================================================================
# STEP 3: MAIN
# ============================================================================

def main():
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    np.random.seed(42)
    
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ---- Load PRE-TRAINED model (not fine-tuned!) ----
    # We want the natural multilingual space, not the English-biased one
    print(f"\nLoading pre-trained {config.model_name}...")
    model = AutoModel.from_pretrained(config.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model.eval()
    print("Model loaded.")
    
    # ---- Extract embeddings for each language ----
    print(f"\n{'='*70}")
    print(f"STEP 1: EXTRACTING EMBEDDINGS")
    print(f"{'='*70}\n")
    
    embeddings = {}
    
    for lang in config.all_langs:
        print(f"  {lang.upper()}: ", end="", flush=True)
        start = time.time()
        
        try:
            ds = load_dataset("wikiann", lang, split="test")
        except Exception as e:
            print(f"ERROR loading: {e}")
            continue
        
        emb = extract_embeddings(model, tokenizer, ds, config, device)
        embeddings[lang] = emb
        
        elapsed = time.time() - start
        print(f"{emb.shape[0]} tokens extracted ({emb.shape[1]}d) in {elapsed:.1f}s")
    
    # ---- Compute W2 distances ----
    print(f"\n{'='*70}")
    print(f"STEP 2: COMPUTING WASSERSTEIN DISTANCES")
    print(f"{'='*70}\n")
    
    source_emb = embeddings[config.source_lang]
    
    w2_sinkhorn = {}
    w2_sliced = {}
    
    for lang in config.all_langs:
        if lang == config.source_lang:
            w2_sinkhorn[lang] = 0.0
            w2_sliced[lang] = 0.0
            continue
        
        if lang not in embeddings:
            continue
        
        print(f"  en → {lang}: ", end="", flush=True)
        start = time.time()
        
        # Sinkhorn W2
        w2_s = compute_sinkhorn_w2(
            source_emb, embeddings[lang],
            reg=config.sinkhorn_reg,
            max_samples=config.subsample_for_w2,
        )
        w2_sinkhorn[lang] = w2_s
        
        # Sliced W2 (sanity check)
        w2_sl = compute_sliced_wasserstein(
            source_emb, embeddings[lang],
            n_projections=config.n_sliced_projections,
            max_samples=config.subsample_for_w2,
        )
        w2_sliced[lang] = w2_sl
        
        elapsed = time.time() - start
        print(f"Sinkhorn W2 = {w2_s:.4f} | Sliced W2 = {w2_sl:.4f} ({elapsed:.1f}s)")
    
    # ---- Load F1 results from previous evaluation ----
    print(f"\n{'='*70}")
    print(f"STEP 3: CORRELATING W2 WITH F1 GAPS")
    print(f"{'='*70}\n")
    
    # Try loading saved results
    results_path = os.path.join(config.output_dir, "multilingual_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            f1_results = json.load(f)
        print(f"  Loaded F1 results from {results_path}")
    else:
        # Fallback: use the original results
        orig_path = os.path.join(config.output_dir, "results.json")
        if os.path.exists(orig_path):
            with open(orig_path) as f:
                data = json.load(f)
            f1_results = data.get("results", {})
            print(f"  Loaded F1 results from {orig_path}")
        else:
            print("  WARNING: No F1 results found. Run eval_more_languages.py first.")
            f1_results = {}
    
    # ---- Build the correlation table ----
    source_f1 = f1_results.get(config.source_lang, {}).get("f1", None)
    
    print(f"\n  {'Lang':<6} {'F1':>8} {'Gap':>8} {'W2 (Sink)':>10} {'W2 (Sliced)':>12}")
    print(f"  {'-'*50}")
    
    # Collect data points for correlation
    gaps = []
    w2s = []
    lang_labels = []
    
    for lang in config.all_langs:
        if lang not in w2_sinkhorn or lang not in f1_results:
            continue
        
        f1 = f1_results[lang]["f1"]
        gap = (source_f1 - f1) if source_f1 else 0
        w2 = w2_sinkhorn[lang]
        w2_sl = w2_sliced.get(lang, 0)
        
        marker = " ← source" if lang == config.source_lang else ""
        print(f"  {lang:<6} {f1:>8.4f} {gap:>+8.4f} {w2:>10.4f} {w2_sl:>12.4f}{marker}")
        
        if lang != config.source_lang:
            gaps.append(gap)
            w2s.append(w2)
            lang_labels.append(lang)
    
    # ---- Compute correlation ----
    if len(gaps) >= 2:
        gaps_arr = np.array(gaps)
        w2s_arr = np.array(w2s)
        
        # Pearson correlation
        correlation = np.corrcoef(w2s_arr, gaps_arr)[0, 1]
        
        # Linear regression: gap ≈ a * W2 + b
        if len(gaps) >= 2:
            coeffs = np.polyfit(w2s_arr, gaps_arr, 1)
            slope, intercept = coeffs
        
        print(f"\n  {'='*50}")
        print(f"  CORRELATION ANALYSIS")
        print(f"  {'='*50}")
        print(f"  Pearson correlation (W2 vs F1 gap): {correlation:.4f}")
        print(f"  Linear fit: gap ≈ {slope:.4f} × W2 + {intercept:.4f}")
        
        if correlation > 0.7:
            print(f"\n  ✓ STRONG CORRELATION — the OT theory is well-supported!")
            print(f"    W2 distance PREDICTS cross-lingual transfer quality.")
            print(f"    This is the key empirical validation for your paper.")
        elif correlation > 0.4:
            print(f"\n  ~ MODERATE CORRELATION — promising, worth investigating.")
            print(f"    May improve with better distance metrics (geodesic instead of Euclidean).")
        else:
            print(f"\n  ✗ WEAK CORRELATION — Euclidean W2 may not be sufficient.")
            print(f"    This actually MOTIVATES the manifold-geodesic version in the paper!")
    
    # ---- Save everything ----
    output = {
        "w2_sinkhorn": w2_sinkhorn,
        "w2_sliced": w2_sliced,
        "f1_results": f1_results,
        "gaps": {lang: gap for lang, gap in zip(lang_labels, gaps)},
        "correlation": float(correlation) if len(gaps) >= 2 else None,
    }
    
    output_path = os.path.join(config.output_dir, "wasserstein_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  All results saved to {output_path}")
    
    # ---- Print what goes in the paper ----
    print(f"\n{'='*70}")
    print(f"FOR YOUR PAPER")
    print(f"{'='*70}")
    print("\n    Figure: Plot W2 (x-axis) vs F1 gap (y-axis) with language labels.")
    print("    Table: The full results table above is Table 1 of the paper.")
    
    return output


if __name__ == "__main__":
    main()