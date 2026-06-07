from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

@dataclass
class Config:
    model_name: str = "xlm-roberta-base"
    source_lang: str = "English (EN)"
    # Defaulting to MultiCoNER targets. Can be overridden via command line.
    target_langs: Tuple[str, ...] = ("Spanish (ES)", "Chinese (ZH)", "Hindi (HI)", "Bangla (BN)") 
    dataset_name: str = "MultiCoNER/multiconer_v2"
    output_dir: str = "./vib_spot_runs"

    max_length: int = 160
    train_batch_size: int = 16  
    eval_batch_size: int = 32
    grad_accum_steps: int = 2
    num_workers: int = 0
    seed: int = 42

    source_epochs: int = 10
    pseudo_epochs: int = 0     
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    layerwise_lr_decay: float = 0.9
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    dropout: float = 0.1

    freeze_embeddings: bool = True
    freeze_bottom_k_layers: int = 4
    scalar_mix_top_k: int = 4
    gradient_checkpointing: bool = True

    # VIB & SPOT NOVELTY HYPERPARAMS
    bottleneck_dim: int = 512  # EXPANDED: Better for 30+ classes
    vib_weight: float = 0.0001  # RESTORED: VIB is back online
    ot_weight: float = 0.05
    
    consistency_weight: float = 0.05
    ema_decay: float = 0.995
    pseudo_confidence: float = 0.95 # INCREASED: Stricter filtering for noisy datasets
    max_pseudo_per_lang: int = 10000

    fp16: bool = True
    save_everything: bool = True


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    for field in Config.__dataclass_fields__:
        default = getattr(Config, field)
        if isinstance(default, tuple):
            parser.add_argument(f"--{field}", nargs="+", default=list(default))
        elif isinstance(default, bool):
            parser.add_argument(f"--{field}", type=bool, default=default)
        else:
            parser.add_argument(f"--{field}", type=type(default), default=default)
    args = parser.parse_args()
    base = Config()
    for key, value in vars(args).items():
        if isinstance(getattr(base, key), tuple):
            setattr(base, key, tuple(value))
        else:
            setattr(base, key, value)
    return base

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NOVELTY: VIB MODEL ARCHITECTURE ---
class VIBTokenClassifier(nn.Module):
    def __init__(self, cfg: Config, num_labels: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        if cfg.gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self.encoder.config.use_cache = False
        
        hidden_size = self.encoder.config.hidden_size
        self.scalar_mix_top_k = min(cfg.scalar_mix_top_k, self.encoder.config.num_hidden_layers + 1)
        self.layer_weights = nn.Parameter(torch.zeros(self.scalar_mix_top_k))
        self.layer_gamma = nn.Parameter(torch.ones(1))
        
        # Variational Information Bottleneck (VIB) Projection
        self.vib_mu = nn.Linear(hidden_size, cfg.bottleneck_dim)
        self.vib_logvar = nn.Linear(hidden_size, cfg.bottleneck_dim)
        
        self.classifier = nn.Linear(cfg.bottleneck_dim, num_labels)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-self.scalar_mix_top_k :]
        mix = torch.softmax(self.layer_weights, dim=0)
        mixed = self.layer_gamma * sum(weight * hs for weight, hs in zip(mix, hidden_states))
        
        # VIB Computation
        mu = self.vib_mu(mixed)
        logvar = self.vib_logvar(mixed)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        # KL Divergence for the bottleneck
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            
        logits = self.classifier(self.dropout(z))
        
        return {
            "logits": logits, 
            "mu": mu,          # SPOT operates strictly on the deterministic manifold
            "kl_loss": kl_loss.mean()
        }

def freeze_for_transfer(model: VIBTokenClassifier, cfg: Config) -> None:
    if cfg.freeze_embeddings and hasattr(model.encoder, "embeddings"):
        for param in model.encoder.embeddings.parameters():
            param.requires_grad = False
    layers = getattr(model.encoder, "encoder", None)
    layers = getattr(layers, "layer", None)
    if layers is not None:
        for layer in layers[: cfg.freeze_bottom_k_layers]:
            for param in layer.parameters():
                param.requires_grad = False

def build_optimizer(model: VIBTokenClassifier, cfg: Config) -> AdamW:
    no_decay = ("bias", "LayerNorm.weight")
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        lr = cfg.encoder_lr if "encoder" in name else cfg.head_lr
        decay = 0.0 if any(nd in name for nd in no_decay) else cfg.weight_decay
        param_groups.append({"params": [param], "lr": lr, "weight_decay": decay})
        
    return AdamW(param_groups)

# --- STANDARD TOKENIZATION & DATA LOADING ---
def tokenize_labeled(batch, tokenizer, cfg: Config, label_names: list = None):
    tokenized = tokenizer(
        [list(map(str, tokens)) for tokens in batch["tokens"]],
        is_split_into_words=True, truncation=True,
        padding="max_length", max_length=cfg.max_length,
    )
    
    # Fast translation dictionary for string labels
    label2id = {n: i for i, n in enumerate(label_names)} if label_names else {}
    
    labels, valid_mask = [], []
    for idx, word_labels in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=idx)
        aligned, valid, previous_word = [], [], None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100); valid.append(0)
            elif word_id != previous_word:
                tag = word_labels[word_id]
                # If the tag is a string text (like "B-PER"), map it safely
                if isinstance(tag, str) and not str(tag).lstrip('-').isdigit():
                    aligned.append(label2id.get(tag, 0)) 
                else:
                    aligned.append(int(tag))
                valid.append(1)
            else:
                aligned.append(-100); valid.append(0)
            previous_word = word_id
        labels.append(aligned); valid_mask.append(valid)
    tokenized["labels"] = labels
    tokenized["valid_mask"] = valid_mask
    return tokenized

def tokenize_unlabeled(batch, tokenizer, cfg: Config):
    tokenized = tokenizer(
        [list(map(str, tokens)) for tokens in batch["tokens"]],
        is_split_into_words=True, truncation=True,
        padding="max_length", max_length=cfg.max_length,
    )
    valid_mask = []
    for idx in range(len(batch["tokens"])):
        word_ids = tokenized.word_ids(batch_index=idx)
        valid, previous_word = [], None
        for word_id in word_ids:
            if word_id is None: valid.append(0)
            elif word_id != previous_word: valid.append(1)
            else: valid.append(0)
            previous_word = word_id
        valid_mask.append(valid)
    tokenized["valid_mask"] = valid_mask
    return tokenized

def collate_fields(batch):
    return {k: torch.tensor([item[k] for item in batch], dtype=torch.long) for k in batch[0].keys()}

def build_data(cfg: Config, tokenizer):
    source = load_dataset(cfg.dataset_name, cfg.source_lang, trust_remote_code=True)
    
    # DYNAMIC LABEL EXTRACTION WITH BULLETPROOF FALLBACK
    try:
        label_names = source["train"].features["ner_tags"].feature.names
        num_labels = len(label_names)
    except AttributeError:
        print("Warning: Dataset stripped ClassLabel metadata. Scanning dataset to map label space...")
        unique_tags = set()
        for tags in source["train"]["ner_tags"]:
            unique_tags.update(tags)
            
        # Check if tags are text labels (e.g. "B-Prod") or string-digits (e.g. "1")
        is_text_labels = any(isinstance(t, str) and not str(t).lstrip('-').isdigit() for t in unique_tags)
        
        if is_text_labels:
            label_names = list(unique_tags)
            if "O" in label_names:
                label_names.remove("O")
                label_names = ["O"] + sorted(label_names)
            else:
                label_names = sorted(label_names)
            num_labels = len(label_names)
        else:
            max_label = max(int(t) for t in unique_tags)
            num_labels = max_label + 1
            label_names = ["O"]
            for i in range(1, num_labels):
                class_id = (i + 1) // 2
                if i % 2 != 0:
                    label_names.append(f"B-CLASS_{class_id}")
                else:
                    label_names.append(f"I-CLASS_{class_id}")

    bio_b_ids = {idx for idx, name in enumerate(label_names) if name.startswith("B-")}
    print(f"Dataset Loaded: {cfg.dataset_name}. Detected {num_labels} distinct classes.")

    src_train = source["train"].map(lambda ex: tokenize_labeled(ex, tokenizer, cfg, label_names), batched=True, remove_columns=source["train"].column_names)
    src_val = source["validation"].map(lambda ex: tokenize_labeled(ex, tokenizer, cfg, label_names), batched=True, remove_columns=source["validation"].column_names)
    src_test = source["test"].map(lambda ex: tokenize_labeled(ex, tokenizer, cfg, label_names), batched=True, remove_columns=source["test"].column_names)
    
    for ds in (src_train, src_val, src_test): ds.set_format("python")

    target_unlabeled, target_test = {}, {}
    for lang in cfg.target_langs:
        train_ds = load_dataset(cfg.dataset_name, lang, split="train", trust_remote_code=True)
        test_ds = load_dataset(cfg.dataset_name, lang, split="test", trust_remote_code=True)
        
        unlabeled_tok = train_ds.map(lambda ex: tokenize_unlabeled(ex, tokenizer, cfg), batched=True, remove_columns=train_ds.column_names)
        test_tok = test_ds.map(lambda ex: tokenize_labeled(ex, tokenizer, cfg, label_names), batched=True, remove_columns=test_ds.column_names)
        
        unlabeled_tok.set_format("python"); test_tok.set_format("python")
        target_unlabeled[lang] = {"raw": train_ds, "tok": unlabeled_tok}
        target_test[lang] = test_tok

    def make_loader(ds, shuffle): return DataLoader(ds, batch_size=cfg.train_batch_size if shuffle else cfg.eval_batch_size, shuffle=shuffle, num_workers=cfg.num_workers, collate_fn=collate_fields)

    return {
        "src_train_loader": make_loader(src_train, True),
        "src_val_loader": make_loader(src_val, False),
        "test_loaders": {**{cfg.source_lang: make_loader(src_test, False)}, **{l: make_loader(t, False) for l, t in target_test.items()}},
        "target_unlabeled": target_unlabeled,
        "unlabeled_loaders": {l: make_loader(p["tok"], True) for l, p in target_unlabeled.items()},
        "source_raw": source,
        "label_names": label_names,
        "num_labels": num_labels,
        "bio_b_ids": bio_b_ids
    }

def cycle_target_batches(unlabeled_loaders: Dict[str, DataLoader]) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    iterators = {lang: iter(loader) for lang, loader in unlabeled_loaders.items()}
    languages = list(unlabeled_loaders.keys())
    while True:
        random.shuffle(languages)
        for lang in languages:
            try: batch = next(iterators[lang])
            except StopIteration: iterators[lang] = iter(unlabeled_loaders[lang]); batch = next(iterators[lang])
            yield lang, batch

# --- LOSS FUNCTIONS ---
def token_classification_loss(logits, labels, label_smoothing):
    flat_labels = labels.view(-1)
    flat_logits = logits.view(-1, logits.size(-1))
    valid = flat_labels != -100
    if not valid.any(): return flat_logits.new_zeros(())
    return F.cross_entropy(flat_logits[valid], flat_labels[valid], label_smoothing=label_smoothing)

def symmetric_kl(logits_a, logits_b, valid_mask):
    valid = valid_mask.bool().view(-1)
    if not valid.any(): return logits_a.new_zeros(())
    log_pa = F.log_softmax(logits_a.view(-1, logits_a.size(-1))[valid], dim=-1)
    log_pb = F.log_softmax(logits_b.view(-1, logits_b.size(-1))[valid], dim=-1)
    return 0.5 * (F.kl_div(log_pa, log_pb.exp(), reduction="batchmean") + F.kl_div(log_pb, log_pa.exp(), reduction="batchmean"))

def update_ema(teacher, student, decay):
    with torch.no_grad():
        for key, value in teacher.state_dict().items():
            value.copy_(value * decay + student.state_dict()[key] * (1.0 - decay))

def evaluate(model, loader, device, label_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"], batch["attention_mask"])
            pred_ids = torch.argmax(outputs["logits"], dim=-1)
            for pred_seq, label_seq in zip(pred_ids, batch["labels"]):
                pred_tags, label_tags = [], []
                for pred, gold in zip(pred_seq.tolist(), label_seq.tolist()):
                    if gold != -100:
                        pred_tags.append(label_names[pred])
                        label_tags.append(label_names[gold])
                all_preds.append(pred_tags); all_labels.append(label_tags)
    return {
        "f1": float(f1_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds)),
        "recall": float(recall_score(all_labels, all_preds)),
    }

# --- PHASE 1: VIB-SPOT TRAINING ---
def train_vib_ot(model, data, cfg, device):
    optimizer = build_optimizer(model, cfg)
    total_steps = math.ceil(len(data["src_train_loader"]) / cfg.grad_accum_steps) * cfg.source_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(total_steps * cfg.warmup_ratio)), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")
    
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())
    target_stream = cycle_target_batches(data["unlabeled_loaders"])
    num_labels = data["num_labels"]

    for epoch in range(cfg.source_epochs):
        model.train()
        run_loss = {"total": 0.0, "ce": 0.0, "vib": 0.0, "ot": 0.0}
        optimizer.zero_grad(set_to_none=True)
        
        for step, src_batch in enumerate(data["src_train_loader"], 1):
            _, tgt_batch = next(target_stream)
            src_batch = {k: v.to(device) for k, v in src_batch.items()}
            tgt_batch = {k: v.to(device) for k, v in tgt_batch.items()}
            
            with torch.autocast("cuda" if device.type == "cuda" else "cpu", enabled=cfg.fp16):
                src_out = model(src_batch["input_ids"], src_batch["attention_mask"])
                tgt_out = model(tgt_batch["input_ids"], tgt_batch["attention_mask"])

                ce = token_classification_loss(src_out["logits"], src_batch["labels"], cfg.label_smoothing)
                vib_loss = src_out["kl_loss"] + tgt_out["kl_loss"]
                
                src_mu = src_out["mu"]
                tgt_mu = tgt_out["mu"]
                src_labels = src_batch["labels"]
                
                tgt_probs = F.softmax(tgt_out["logits"], dim=-1)
                ot_loss = torch.tensor(0.0, device=device)
                valid_classes = 0
                
                # DYNAMIC CLASS LOOP
                for c in range(num_labels):
                    src_mask = (src_labels == c) & (src_batch["valid_mask"].bool())
                    if src_mask.sum() > 0:
                        src_prototype = src_mu[src_mask].mean(dim=0)
                        
                        tgt_mask = tgt_batch["valid_mask"].bool()
                        if tgt_mask.sum() > 0:
                            tgt_valid_probs = tgt_probs[tgt_mask]
                            tgt_valid_mu = tgt_mu[tgt_mask]
                            c_probs = tgt_valid_probs[:, c].unsqueeze(-1)
                            
                            tgt_prototype = (tgt_valid_mu * c_probs).sum(dim=0) / (c_probs.sum() + 1e-8)
                            ot_loss += F.mse_loss(src_prototype, tgt_prototype)
                            valid_classes += 1
                
                if valid_classes > 0:
                    ot_loss = ot_loss / valid_classes

                loss = (ce + cfg.vib_weight * vib_loss + cfg.ot_weight * ot_loss) / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            run_loss["total"] += loss.item() * cfg.grad_accum_steps
            run_loss["ce"] += ce.item(); run_loss["vib"] += vib_loss.item(); run_loss["ot"] += float(ot_loss)

            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                scheduler.step(); update_ema(teacher, model, cfg.ema_decay)

            if step % 200 == 0:
                d = float(step)
                print(f"Ep {epoch+1} | Step {step} | CE={run_loss['ce']/d:.4f} VIB={run_loss['vib']/d:.4f} SPOT={run_loss['ot']/d:.4f}")

        val_metrics = evaluate(model, data["src_val_loader"], device, data["label_names"])
        print(f"Epoch {epoch + 1} Val F1: {val_metrics['f1']:.4f}")
        if val_metrics["f1"] > best_val:
            best_val = val_metrics["f1"]; best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    teacher.load_state_dict(best_state)
    return model, teacher

# --- PHASE 2: EMA PSEUDO-LABEL REFINEMENT ---
def span_filter(pred_ids, confs, cfg, bio_b_ids):
    final = list(pred_ids)
    idx = 0
    while idx < len(final):
        tag = final[idx]
        if tag in bio_b_ids:
            end = idx + 1
            while end < len(final) and final[end] == tag + 1: end += 1
            span_confs = confs[idx:end]
            if sum(span_confs) / len(span_confs) < cfg.pseudo_confidence:
                for pos in range(idx, end): final[pos] = 0
            idx = end
        else: idx += 1
    return final

def generate_pseudo_labels(teacher, tokenizer, raw_dataset, cfg, device, bio_b_ids):
    teacher.eval()
    accepted = []
    for start in range(0, len(raw_dataset), cfg.eval_batch_size):
        chunk = raw_dataset.select(range(start, min(start + cfg.eval_batch_size, len(raw_dataset))))
        token_lists = [list(map(str, tokens)) for tokens in chunk["tokens"]]
        encoded = tokenizer(token_lists, is_split_into_words=True, truncation=True, padding="max_length", max_length=cfg.max_length, return_tensors="pt")
        
        with torch.no_grad():
            outputs = teacher({k: v.to(device) for k, v in encoded.items()}["input_ids"], {k: v.to(device) for k, v in encoded.items()}["attention_mask"])
            probs = F.softmax(outputs["logits"], dim=-1)
            pred_ids, pred_confs = torch.argmax(probs, dim=-1), torch.max(probs, dim=-1).values

        for i, words in enumerate(token_lists):
            word_ids = encoded.word_ids(batch_index=i)
            word_pred_ids, word_confs, seen = [], [], set()
            for tok_idx, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx in seen: continue
                seen.add(word_idx)
                if word_idx < len(words):
                    word_pred_ids.append(int(pred_ids[i, tok_idx].item()))
                    word_confs.append(float(pred_confs[i, tok_idx].item()))
            if not word_pred_ids: continue
            
            filtered = span_filter(word_pred_ids, word_confs, cfg, bio_b_ids)
            
            aligned_words = words[:len(filtered)]
            if any(t > 0 for t in filtered):
                accepted.append({"tokens": aligned_words, "ner_tags": filtered})
            if len(accepted) >= cfg.max_pseudo_per_lang: return accepted
    return accepted

def train_with_pseudo(model, teacher, tokenizer, source_train_raw, target_unlabeled, src_val_loader, cfg, device, data_dict):
    pseudo_by_lang = {}
    for lang, payload in target_unlabeled.items():
        examples = generate_pseudo_labels(teacher, tokenizer, payload["raw"], cfg, device, data_dict["bio_b_ids"])
        pseudo_by_lang[lang] = examples
        print(f"Pseudo {lang}: {len(examples)} examples extracted.")

    source_dataset = source_train_raw["train"].remove_columns([c for c in source_train_raw["train"].column_names if c not in ["tokens", "ner_tags"]])
    datasets_list = [source_dataset] + [Dataset.from_list(ex, features=source_dataset.features) for ex in pseudo_by_lang.values() if ex]
    merged_tok = concatenate_datasets(datasets_list).map(lambda ex: tokenize_labeled(ex, tokenizer, cfg, data_dict["label_names"]), batched=True).remove_columns(["tokens", "ner_tags"])
    merged_tok.set_format("python")
    
    train_loader = DataLoader(merged_tok, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate_fields)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")
    best_val, best_state = -1.0, copy.deepcopy(model.state_dict())

    for epoch in range(cfg.pseudo_epochs):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda" if device.type == "cuda" else "cpu", enabled=cfg.fp16):
                out1 = model(batch["input_ids"], batch["attention_mask"])
                out2 = model(batch["input_ids"], batch["attention_mask"])
                ce = token_classification_loss(out1["logits"], batch["labels"], cfg.label_smoothing)
                cons = symmetric_kl(out1["logits"], out2["logits"], batch["valid_mask"])
                loss = (ce + cfg.consistency_weight * cons) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if step % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate(model, src_val_loader, device, data_dict["label_names"])
        print(f"Phase 2 - Epoch {epoch + 1} Val F1: {val_metrics['f1']:.4f}")
        if val_metrics["f1"] > best_val:
            best_val = val_metrics["f1"]; best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model

def main():
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()
    print(f"Using {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data = build_data(cfg, tokenizer)

    model = VIBTokenClassifier(cfg, num_labels=data["num_labels"]).to(device)
    freeze_for_transfer(model, cfg)

    print("\n--- Phase 1: VIB-SPOT Base Alignment ---")
    model, teacher = train_vib_ot(model, data, cfg, device)
    
    print("\n--- Phase 2: Refinement ---")
    if cfg.pseudo_epochs > 0:
        model = train_with_pseudo(model, teacher, tokenizer, data["source_raw"], data["target_unlabeled"], data["src_val_loader"], cfg, device, data)

    final_metrics = {lang: evaluate(model, loader, device, data["label_names"]) for lang, loader in data["test_loaders"].items()}
    print("\nFINAL RESULTS")
    for lang, m in final_metrics.items(): print(f"{lang}: F1={m['f1']:.4f}")
    
    print("\nSaving model and tokenizer to disk...")
    save_path = os.path.join(cfg.output_dir, "best_model.pt")
    torch.save({
        'state_dict': model.state_dict(),
        'config': asdict(cfg),
        'model_name': cfg.model_name
    }, save_path)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Successfully saved VIB-SPOT model to {save_path}")

if __name__ == "__main__":
    main()