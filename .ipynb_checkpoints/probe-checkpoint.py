#!/usr/bin/env python3
"""
Information-Theoretic Probing and Visualization for VIB-OT
Generates the core empirical proofs for the EMNLP submission.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from transformers import AutoTokenizer
from datasets import load_dataset
import random

# Import your model class and config from your training script
# Assuming you saved your previous script as vib_ot_train.py
from ottrain import VIBTokenClassifier, Config, tokenize_labeled, collate_fields

def extract_bottleneck_embeddings(model, loader, device, lang_label):
    """Passes data through the frozen model to extract the bottleneck mu."""
    model.eval()
    all_mu = []
    all_ner_tags = []
    all_lang_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # Extract mu from the VIB
            outputs = model(batch["input_ids"], batch["attention_mask"])
            mu = outputs["mu"].cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            valid_mask = batch["valid_mask"].cpu().numpy().astype(bool)

            # Flatten and filter out padding/ignored tokens (-100)
            for i in range(mu.shape[0]):
                valid_idx = valid_mask[i] & (labels[i] != -100)
                if valid_idx.any():
                    all_mu.append(mu[i][valid_idx])
                    all_ner_tags.append(labels[i][valid_idx])
                    all_lang_labels.append(np.full(valid_idx.sum(), lang_label))

    return np.concatenate(all_mu), np.concatenate(all_ner_tags), np.concatenate(all_lang_labels)

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Locking in on: {device}")

    # 1. Load the trained VIB-OT model
    model_path = "./runs/ablation_no_vib/best_model.pt" # Adjust if you saved it differently
    if not os.path.exists(model_path):
        print(f"Error: Could not find trained model at {model_path}. Make sure you run phase 1 & 2 first.")
        # For testing the script without the saved model, uncomment the next line to use untrained weights
        # model_path = None 
    
    model = VIBTokenClassifier(cfg, num_labels=7).to(device)
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Successfully loaded VIB-OT weights.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # 2. Load Evaluation Data (English and Hindi)
    print("Loading datasets...")
    en_ds = load_dataset(cfg.dataset_name, "en", split="test").select(range(1000)) # Subsample for speed
    hi_ds = load_dataset(cfg.dataset_name, "hi", split="test").select(range(1000))

    en_tok = en_ds.map(lambda ex: tokenize_labeled(ex, tokenizer, cfg), batched=True).remove_columns(en_ds.column_names)
    hi_tok = hi_ds.map(lambda ex: tokenize_labeled(ex, tokenizer, cfg), batched=True).remove_columns(hi_ds.column_names)
    en_tok.set_format("python")
    hi_tok.set_format("python")

    from torch.utils.data import DataLoader
    en_loader = DataLoader(en_tok, batch_size=32, collate_fn=collate_fields)
    hi_loader = DataLoader(hi_tok, batch_size=32, collate_fn=collate_fields)

    # 3. Extract Latent Embeddings (mu)
    print("Extracting VIB bottleneck representations...")
    en_mu, en_ner, en_lang = extract_bottleneck_embeddings(model, en_loader, device, lang_label=0)
    hi_mu, hi_ner, hi_lang = extract_bottleneck_embeddings(model, hi_loader, device, lang_label=1)

    # Combine for probing
    X = np.vstack((en_mu, hi_mu))
    y_ner = np.concatenate((en_ner, hi_ner))
    y_lang = np.concatenate((en_lang, hi_lang))

    # Shuffle
    indices = np.random.permutation(len(X))
    X, y_ner, y_lang = X[indices], y_ner[indices], y_lang[indices]

    # Split Train/Test for probes
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_ner_train, y_ner_test = y_ner[:split], y_ner[split:]
    y_lang_train, y_lang_test = y_lang[:split], y_lang[split:]

    print("\n--- INTRINSIC EVALUATION: INFORMATION PROBING ---")
    
    # 4. The Language Adversary Probe
    lang_probe = LogisticRegression(max_iter=1000)
    lang_probe.fit(X_train, y_lang_train)
    lang_acc = accuracy_score(y_lang_test, lang_probe.predict(X_test))
    print(f"Language-ID Probe Accuracy: {lang_acc:.4f} (Closer to 0.50 is BETTER - means syntax is erased)")

    # 5. The Task Probe
    ner_probe = LogisticRegression(max_iter=1000)
    ner_probe.fit(X_train, y_ner_train)
    ner_acc = accuracy_score(y_ner_test, ner_probe.predict(X_test))
    print(f"NER Task Probe Accuracy:    {ner_acc:.4f} (Closer to 1.0 is BETTER - means semantics are preserved)")

    # 6. Silhouette Score (Mathematical measure of cluster overlap)
    # Filter to only look at actual entities (ignore 'O' tags which dominate)
    entity_mask = y_ner > 0
    X_entities = X[entity_mask][:5000] # Subsample for memory
    y_lang_entities = y_lang[entity_mask][:5000]
    
    if len(np.unique(y_lang_entities)) > 1:
        # A score near 0 means the English and Hindi manifolds are perfectly overlapping.
        sil_score = silhouette_score(X_entities, y_lang_entities, metric='euclidean')
        print(f"Cross-Lingual Silhouette Score: {sil_score:.4f} (Closer to 0 is BETTER - means perfect overlap)")

    print("\n--- VISUALIZATION: THE EMNLP MONEY PLOT ---")
    print("Running UMAP dimensionality reduction (this takes a minute)...")
    
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    X_2d = reducer.fit_transform(X_entities)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_lang_entities, cmap='coolwarm', alpha=0.6, s=10)
    
    # Customizing the plot for publication
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['English', 'Hindi'])
    plt.title('UMAP Projection of VIB Latent Space ($Z$)\nPerfect overlap indicates successful causal disentanglement', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plot_path = "./runs/ablation_no_vib/manifold_overlap_failed_smoothie.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Publication-ready plot saved to: {plot_path}")

if __name__ == "__main__":
    main()