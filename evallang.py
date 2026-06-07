"""
==============================================================================
Path B: Evaluate Saved Model on More Languages (NO retraining)
==============================================================================

WHAT THIS DOES:
- Loads your already-trained best model from ./ner_baseline_results/best_model/
- Evaluates it zero-shot on 7 new languages + re-runs en/es/hi for consistency
- More languages = more data points on the W2 vs F1-gap plot
- Each language takes ~2-3 minutes (inference only)

LANGUAGES CHOSEN FOR MAXIMUM DIVERSITY:
- French (fr)     — Romance, Latin script, close to English
- German (de)     — Germanic, Latin script, close to English
- Portuguese (pt) — Romance, Latin script, close to English/Spanish
- Arabic (ar)     — Semitic, Arabic script, RTL, very different
- Chinese (zh)    — Sino-Tibetan, logographic, extremely different
- Tamil (ta)      — Dravidian, Tamil script, very different
- Bengali (bn)    — Indo-Aryan, Bengali script, different but related to Hindi

This gives us a spectrum from "very close to English" to "maximally distant"
— exactly what we need for the W2 correlation plot.

EXPECTED RUNTIME: ~20-25 minutes total on RTX A4000

EXPECTED PATTERN:
- Close languages (fr, de, pt): small gap (5-10%)
- Medium languages (ar): medium gap (12-18%)
- Distant languages (zh, ta, bn): large gap (15-25%)
- This gradient is what W2 should predict
"""

import os
import json
import torch
import numpy as np
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    model_name = "xlm-roberta-base"
    saved_model_path = "./ner_baseline_results/best_model"
    
    # All languages to evaluate
    # Source language first, then targets ordered roughly by expected similarity
    source_lang = "en"
    all_eval_langs = [
        "en",   # English    — source (upper bound)
        "fr",   # French     — Romance, very close
        "de",   # German     — Germanic, close
        "pt",   # Portuguese — Romance, close
        "es",   # Spanish    — Romance, close (already have this)
        "ar",   # Arabic     — Semitic, distant
        "hi",   # Hindi      — Indo-Aryan, distant (already have this)
        "bn",   # Bengali    — Indo-Aryan, distant
        "zh",   # Chinese    — Sino-Tibetan, very distant
        "ta",   # Tamil      — Dravidian, very distant
    ]
    
    batch_size = 32     # Larger batch for inference (no gradients = less memory)
    max_length = 128
    output_dir = "./ner_baseline_results"


def tokenize_and_align(examples, tokenizer, max_length):
    """Same tokenization as training — align word labels to subword tokens."""
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding="max_length",
    )
    
    all_labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id
        
        all_labels.append(label_ids)
    
    tokenized["labels"] = all_labels
    return tokenized


def evaluate(model, dataloader, label_names, device):
    """Evaluate model, return entity-level F1/precision/recall."""
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            for pred_seq, label_seq in zip(predictions, labels):
                pred_list = []
                label_list = []
                for p, l in zip(pred_seq, label_seq):
                    if l.item() != -100:
                        pred_list.append(label_names[p.item()])
                        label_list.append(label_names[l.item()])
                all_preds.append(pred_list)
                all_labels.append(label_list)
    
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report,
    }


def main():
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    
    config = Config()
    
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Running on CPU")
    
    # ---- Load saved model ----
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    num_labels = len(label_names)
    
    print(f"\nLoading saved model from {config.saved_model_path}...")
    model = AutoModelForTokenClassification.from_pretrained(
        config.saved_model_path, num_labels=num_labels
    ).to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # ---- Load tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # ---- Evaluate each language ----
    print(f"\n{'='*70}")
    print(f"MULTILINGUAL ZERO-SHOT NER EVALUATION")
    print(f"Model trained on: English only")
    print(f"Evaluating on: {len(config.all_eval_langs)} languages")
    print(f"{'='*70}\n")
    
    results = {}
    
    for lang in config.all_eval_langs:
        print(f"--- {lang.upper()} ---")
        
        # Load dataset
        print(f"  Loading WikiANN/{lang}...")
        try:
            ds = load_dataset("wikiann", lang, split="test")
        except Exception as e:
            print(f"  ERROR loading {lang}: {e}")
            print(f"  Skipping {lang}.\n")
            continue
        
        print(f"  Test set: {len(ds)} examples")
        
        # Tokenize
        tokenized = ds.map(
            lambda ex: tokenize_and_align(ex, tokenizer, config.max_length),
            batched=True,
            remove_columns=ds.column_names,
        )
        tokenized.set_format("torch")
        
        loader = DataLoader(tokenized, batch_size=config.batch_size, shuffle=False)
        
        # Evaluate
        metrics = evaluate(model, loader, label_names, device)
        results[lang] = {
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"\n{metrics['report']}")
    
    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"SUMMARY: CROSS-LINGUAL TRANSFER GAPS")
    print(f"{'='*70}")
    print(f"{'Language':<12} {'F1':>8} {'Gap':>8} {'Relative Drop':>15}")
    print(f"{'-'*45}")
    
    source_f1 = results[config.source_lang]["f1"]
    
    # Sort by F1 descending for clean presentation
    sorted_langs = sorted(results.keys(), key=lambda l: results[l]["f1"], reverse=True)
    
    for lang in sorted_langs:
        f1 = results[lang]["f1"]
        gap = source_f1 - f1
        rel_drop = (gap / source_f1) * 100 if source_f1 > 0 else 0
        
        marker = " ← source" if lang == config.source_lang else ""
        print(f"  {lang:<10} {f1:>8.4f} {gap:>+8.4f} {rel_drop:>13.1f}%{marker}")
    
    # ---- Language families for context ----
    print(f"\n{'='*70}")
    print(f"LANGUAGE FAMILY GROUPING")
    print(f"{'='*70}")
    
    families = {
        "Germanic (closest)": ["en", "de"],
        "Romance": ["fr", "es", "pt"],
        "Indo-Aryan": ["hi", "bn"],
        "Semitic": ["ar"],
        "Sino-Tibetan": ["zh"],
        "Dravidian": ["ta"],
    }
    
    for family, langs in families.items():
        present = [l for l in langs if l in results]
        if present:
            avg_f1 = np.mean([results[l]["f1"] for l in present])
            avg_gap = source_f1 - avg_f1
            print(f"  {family:<25} Avg F1: {avg_f1:.4f} | Avg Gap: {avg_gap:.4f}")
    
    print(f"\n  PREDICTION: W2 distance should correlate with these gaps.")
    print(f"  If it does, the OT theory is validated before writing a single proof.\n")
    
    # ---- Save results ----
    results_path = os.path.join(config.output_dir, "multilingual_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()