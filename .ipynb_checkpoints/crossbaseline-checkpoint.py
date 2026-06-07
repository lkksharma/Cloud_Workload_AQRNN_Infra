"""
==============================================================================
Cross-Lingual NER Baseline: Train on English, Test Zero-Shot on Other Languages
==============================================================================

WHAT THIS SCRIPT DOES:
1. Downloads WikiANN NER dataset (English, Spanish, Hindi)
2. Fine-tunes XLM-RoBERTa-base on ENGLISH training data only
3. Evaluates on English test set (upper bound — how good the model is)
4. Evaluates on Spanish and Hindi test sets WITHOUT any training on them
   (zero-shot cross-lingual transfer)
5. Reports F1 scores — the gap between English and target languages
   is exactly what the OT paper would shrink

HARDWARE:
- RTX 4000 (8GB): works with batch_size=8, max_length=128
- RTX A4000 (16GB): works with batch_size=16, max_length=128
- Any GPU with ≥6GB VRAM should work with small batch sizes

EXPECTED RUNTIME on RTX 4000:
- Training: ~2-3 hours (10 epochs)
- Evaluation: ~10 minutes per language
- Total: ~3-4 hours

EXPECTED RESULTS (approximate):
- English F1: ~82-86%
- Spanish F1: ~72-77% (moderate drop)
- Hindi F1: ~58-67% (large drop — this is the gap to fix)

INSTALL FIRST:
    pip install torch transformers datasets seqeval accelerate --break-system-packages
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

# ============================================================================
# STEP 0: CONFIGURATION
# ============================================================================
# Everything you might want to change is here in one place.

class Config:
    # Model
    model_name = "xlm-roberta-base"  # 270M params, fits on any modern GPU
    
    # Data
    dataset_name = "wikiann"  # Standard cross-lingual NER benchmark
    source_lang = "en"        # Train on English
    target_langs = ["es", "hi"]  # Evaluate zero-shot on Spanish, Hindi
    # You can add more: "fr", "de", "ar", "zh", "pt", "ta" (Tamil), "bn" (Bengali)
    
    # Training hyperparameters
    # -- If you have 8GB GPU (Turing RTX 4000): use batch_size=8
    # -- If you have 16GB GPU (A4000): use batch_size=16
    # -- If you have 20GB GPU (Ada RTX 4000): use batch_size=16 or 24
    batch_size = 16
    learning_rate = 2e-5      # Standard for fine-tuning transformers
    num_epochs = 3           # WikiANN is small, needs more epochs
    max_length = 128          # Max tokens per sentence (WikiANN sentences are short)
    warmup_ratio = 0.1        # Warmup first 10% of training steps
    weight_decay = 0.01       # L2 regularization
    fp16 = True               # Mixed precision — 2x faster, uses less memory
    seed = 42                 # For reproducibility
    
    # Checkpointing
    output_dir = "./ner_baseline_results"
    save_best_model = True
    
    # Evaluation
    eval_every_epoch = True   # Evaluate on all languages after each epoch


# ============================================================================
# STEP 1: SET UP ENVIRONMENT
# ============================================================================

def setup_environment(config):
    """
    Set random seeds, detect GPU, print hardware info.
    
    WHY THIS MATTERS:
    - Reproducibility: same seed → same results (mostly — GPU ops have tiny nondeterminism)
    - GPU detection: we need to know what hardware we're working with
    """
    # Set seeds everywhere
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Auto-adjust batch size based on VRAM
        if gpu_mem < 10:  # 8GB cards
            config.batch_size = 8
            print(f"  → Auto-reduced batch_size to {config.batch_size} for {gpu_mem:.0f}GB GPU")
        elif gpu_mem < 18:  # 16GB cards
            config.batch_size = 16
        else:  # 20GB+ cards
            config.batch_size = 24
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Training will be very slow.")
        config.batch_size = 4
        config.fp16 = False  # fp16 on CPU is pointless
    
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Mixed precision (fp16): {config.fp16}")
    
    return device


# ============================================================================
# STEP 2: LOAD AND PREPARE DATA
# ============================================================================

def load_datasets(config):
    """
    Load WikiANN NER data for all languages.
    
    WikiANN STRUCTURE:
    - Each example has: tokens (list of words), ner_tags (list of integer labels)
    - Labels: O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6
    - Available in 100+ languages with identical label schema
    
    WHY WikiANN:
    - Standard benchmark used by XTREME, XTREME-R, every cross-lingual NER paper
    - Same 3 entity types across all languages → fair comparison
    - Free and instantly downloadable via HuggingFace
    """
    from datasets import load_dataset
    
    all_langs = [config.source_lang] + config.target_langs
    datasets = {}
    
    for lang in all_langs:
        print(f"Loading WikiANN for '{lang}'...")
        ds = load_dataset("wikiann", lang)
        datasets[lang] = ds
        print(f"  Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")
    
    # The label names are the same for all languages
    # WikiANN uses: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    
    print(f"\nLabel schema: {label_names}")
    print(f"Number of labels: {len(label_names)}")
    
    return datasets, label_names


# ============================================================================
# STEP 3: TOKENIZATION WITH LABEL ALIGNMENT
# ============================================================================

def tokenize_and_align(examples, tokenizer, max_length, label_all_tokens=False):
    """
    THE TRICKIEST PART OF NER WITH TRANSFORMERS.
    
    PROBLEM:
    Raw data has one label per WORD. But XLM-R splits words into SUBWORDS:
    
        Word:     "Sharma"
        Subwords: ["▁Sh", "ar", "ma"]     ← 3 tokens for 1 word
        Label:    B-PER
    
    We need to decide: what label does each subword get?
    
    SOLUTION (standard approach):
    - First subword of each word → gets the real label (B-PER)
    - All other subwords → get label -100 (special "ignore" value)
    
    PyTorch's CrossEntropyLoss automatically skips positions with label=-100.
    So the model only predicts one label per word, which is what we want.
    
    EXAMPLE:
        Words:    ["Mr.", "Sharma", "visited", "Delhi"]
        Labels:   [  O,   B-PER,      O,      B-LOC ]
        
        After tokenization:
        Tokens:   ["<s>", "▁Mr", ".",  "▁Sh", "ar", "ma", "▁visited", "▁Del", "hi", "</s>"]
        Labels:   [-100,    0,   -100,   1,    -100, -100,    0,          5,    -100, -100]
                   ^^^^                        ^^^^  ^^^^                       ^^^^  ^^^^
                   special                     not first subword               not first / special
    """
    # Tokenize all sentences in the batch
    # is_split_into_words=True tells the tokenizer that input is already word-tokenized
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding="max_length",
    )
    
    all_labels = []
    
    for i, word_labels in enumerate(examples["ner_tags"]):
        # word_ids() maps each token position to the original word index
        # Special tokens (<s>, </s>, <pad>) get None
        word_ids = tokenized.word_ids(batch_index=i)
        
        label_ids = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # Special token → ignore
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # First subword of a new word → real label
                label_ids.append(word_labels[word_id])
            else:
                # Subsequent subword of same word → ignore
                # (Alternative: copy the label. Both work, -100 is more standard)
                label_ids.append(-100)
            
            previous_word_id = word_id
        
        all_labels.append(label_ids)
    
    tokenized["labels"] = all_labels
    return tokenized


# ============================================================================
# STEP 4: BUILD THE MODEL
# ============================================================================

def build_model(config, num_labels, device):
    """
    Load XLM-RoBERTa-base and add a token classification head.
    
    ARCHITECTURE:
    
    Input tokens → XLM-R encoder (12 layers, 768-dim) → hidden states per token
                                                              ↓
                                                    Linear(768 → num_labels)
                                                              ↓
                                                    Softmax → predicted label
    
    The Linear layer is the "classification head." It's randomly initialized
    and trained from scratch. The XLM-R encoder is pre-trained and fine-tuned.
    
    WHY XLM-RoBERTa-base (not large):
    - 270M params vs 550M for large
    - Fits comfortably on 8GB GPU
    - Good enough for baseline numbers
    - Large model results can come later on A100
    """
    from transformers import AutoModelForTokenClassification
    
    print(f"Loading {config.model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.1f}M")
    
    return model


# ============================================================================
# STEP 5: EVALUATION — COMPUTING NER F1
# ============================================================================

def evaluate(model, dataloader, label_names, device):
    """
    Evaluate NER model and compute entity-level F1.
    
    CRITICAL DISTINCTION: Token F1 vs Entity F1
    
    Token F1 counts each token independently:
        - "B-PER" predicted correctly = +1 to both precision and recall
    
    Entity F1 (what we use) requires the ENTIRE entity span to be correct:
        - True entity: "Sharma" = [B-PER]
        - Predicted:   "Sharma" = [B-PER] → correct (full match)
        - Predicted:   "Sharma" = [B-ORG] → wrong (type mismatch)
        - Predicted:   "Mr. Sharma" = [B-PER, I-PER] → wrong (boundary mismatch)
    
    Entity F1 is MUCH harder and is the standard metric.
    This is the same Strict F1 vs CHR distinction you know from MultiClinAI.
    
    We use the seqeval library which handles all of this correctly.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)
            
            # Get predicted labels (argmax over label dimension)
            predictions = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)
            
            # Convert to lists, filtering out -100 positions
            for pred_seq, label_seq in zip(predictions, labels):
                pred_list = []
                label_list = []
                
                for p, l in zip(pred_seq, label_seq):
                    if l.item() != -100:  # Only evaluate real labels
                        pred_list.append(label_names[p.item()])
                        label_list.append(label_names[l.item()])
                
                all_preds.append(pred_list)
                all_labels.append(label_list)
    
    # Compute entity-level metrics using seqeval
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "report": report
    }


# ============================================================================
# STEP 6: TRAINING LOOP
# ============================================================================

def train(config, model, train_dataloader, val_dataloader, label_names, device):
    """
    Standard fine-tuning loop with:
    - AdamW optimizer (weight decay for regularization)
    - Linear warmup then linear decay learning rate schedule
    - Optional fp16 mixed precision
    - Validation after each epoch
    - Save best model by validation F1
    
    WHAT HAPPENS EACH STEP:
    1. Forward pass: tokens → model → logits (predictions)
    2. Loss computation: cross-entropy between logits and true labels
    3. Backward pass: compute gradients via backpropagation
    4. Optimizer step: update weights using gradients
    5. (Optional) Evaluate on validation set
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    # Optimizer — AdamW is the standard for transformer fine-tuning
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate schedule: warmup then linear decay to 0
    total_steps = len(train_dataloader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision scaler (for fp16)
    scaler = torch.amp.GradScaler("cuda") if config.fp16 and device.type == "cuda" else None
    
    print(f"\n{'='*60}")
    print(f"TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Steps per epoch: {len(train_dataloader)}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"{'='*60}\n")
    
    best_val_f1 = 0.0
    training_log = []
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass (with optional mixed precision)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,  # When labels are provided, model computes loss
                    )
                    loss = outputs.loss
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                avg_loss = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{config.num_epochs} | Step {step+1}/{len(train_dataloader)} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        # Validate
        if config.eval_every_epoch and val_dataloader is not None:
            val_metrics = evaluate(model, val_dataloader, label_names, device)
            val_f1 = val_metrics["f1"]
            
            print(f"\nEpoch {epoch+1}/{config.num_epochs} | Loss: {avg_epoch_loss:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.0f}s")
            
            # Save best model
            if val_f1 > best_val_f1 and config.save_best_model:
                best_val_f1 = val_f1
                save_path = os.path.join(config.output_dir, "best_model")
                model.save_pretrained(save_path)
                print(f"  → New best model saved! F1: {val_f1:.4f}")
            
            training_log.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "val_f1": val_f1,
                "time_seconds": epoch_time,
            })
        else:
            print(f"\nEpoch {epoch+1}/{config.num_epochs} | Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.0f}s")
            training_log.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "time_seconds": epoch_time,
            })
    
    print(f"\nTraining complete. Best validation F1: {best_val_f1:.4f}")
    return training_log, best_val_f1


# ============================================================================
# STEP 7: MAIN — PUT IT ALL TOGETHER
# ============================================================================

def main():
    """
    Full pipeline:
    1. Setup environment
    2. Load data for all languages
    3. Tokenize everything
    4. Build model
    5. Train on English
    6. Evaluate on ALL languages (English + targets)
    7. Print the cross-lingual gap
    """
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    
    # ---- Environment ----
    device = setup_environment(config)
    
    # ---- Data ----
    datasets, label_names = load_datasets(config)
    num_labels = len(label_names)
    
    # ---- Tokenizer ----
    print(f"\nLoading tokenizer for {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # ---- Tokenize all splits for all languages ----
    print("\nTokenizing datasets...")
    tokenized_datasets = {}
    
    for lang, ds in datasets.items():
        tokenized_datasets[lang] = {}
        for split in ["train", "validation", "test"]:
            tokenized = ds[split].map(
                lambda examples: tokenize_and_align(examples, tokenizer, config.max_length),
                batched=True,
                remove_columns=ds[split].column_names,
            )
            tokenized.set_format("torch")
            tokenized_datasets[lang][split] = tokenized
            print(f"  {lang}/{split}: {len(tokenized)} examples")
    
    # ---- DataLoaders ----
    # Training: only English
    train_loader = DataLoader(
        tokenized_datasets[config.source_lang]["train"],
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    # Validation: only English (for model selection)
    val_loader = DataLoader(
        tokenized_datasets[config.source_lang]["validation"],
        batch_size=config.batch_size * 2,  # Can use larger batch for eval (no gradients)
        shuffle=False,
    )
    
    # Test loaders: ALL languages
    test_loaders = {}
    for lang in [config.source_lang] + config.target_langs:
        test_loaders[lang] = DataLoader(
            tokenized_datasets[lang]["test"],
            batch_size=config.batch_size * 2,
            shuffle=False,
        )
    
    # ---- Model ----
    model = build_model(config, num_labels, device)
    
    # ---- Train ----
    training_log, best_val_f1 = train(
        config, model, train_loader, val_loader, label_names, device
    )
    
    # ---- Load best model for final evaluation ----
    if config.save_best_model:
        from transformers import AutoModelForTokenClassification
        best_model_path = os.path.join(config.output_dir, "best_model")
        if os.path.exists(best_model_path):
            print(f"\nLoading best model from {best_model_path}...")
            model = AutoModelForTokenClassification.from_pretrained(
                best_model_path, num_labels=num_labels
            ).to(device)
    
    # ---- Final Evaluation on ALL languages ----
    print(f"\n{'='*60}")
    print("CROSS-LINGUAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Trained on: {config.source_lang} (English)")
    print(f"Evaluated on: {[config.source_lang] + config.target_langs}")
    print(f"{'='*60}\n")
    
    results = {}
    for lang in [config.source_lang] + config.target_langs:
        print(f"\n--- {lang.upper()} ---")
        metrics = evaluate(model, test_loaders[lang], label_names, device)
        results[lang] = metrics
        
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"\n  Per-entity breakdown:")
        print(metrics['report'])
    
    # ---- Print the gap (this is what the paper is about) ----
    print(f"\n{'='*60}")
    print("CROSS-LINGUAL TRANSFER GAP")
    print(f"{'='*60}")
    
    source_f1 = results[config.source_lang]["f1"]
    for lang in config.target_langs:
        target_f1 = results[lang]["f1"]
        gap = source_f1 - target_f1
        relative_gap = (gap / source_f1) * 100
        print(f"  {config.source_lang} → {lang}: {source_f1:.4f} → {target_f1:.4f} | Gap: {gap:.4f} ({relative_gap:.1f}% relative drop)")
    
    print(f"\n  These gaps are what the OT-regularized method would shrink.")
    print(f"  The Wasserstein distance W2(en, lang) should predict the gap size.")
    
    # ---- Save results ----
    results_summary = {
        "config": {
            "model": config.model_name,
            "source_lang": config.source_lang,
            "target_langs": config.target_langs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "seed": config.seed,
        },
        "training_log": training_log,
        "results": {
            lang: {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"]}
            for lang, m in results.items()
        },
    }
    
    results_path = os.path.join(config.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


if __name__ == "__main__":
    main()