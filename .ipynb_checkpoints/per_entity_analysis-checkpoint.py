"""
PER-ENTITY ANALYSIS — loads VIBTokenClassifier correctly from .pt file
"""
import os, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

LABEL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
TARGETS = ["en", "es", "hi", "fr", "de", "pt", "ar", "zh", "ta", "bn"]
MAX_LENGTH = 160
EVAL_BATCH = 32

BASELINE_PATH = "./ner_baseline_results/best_model"
VIBSPOT_PT_PATH = "./runs/vib_spot_final/best_model.pt"


# ---- Recreate the exact VIBTokenClassifier from ottrain.py ----
@dataclass
class Config:
    model_name: str = "xlm-roberta-base"
    bottleneck_dim: int = 256
    scalar_mix_top_k: int = 4
    dropout: float = 0.1
    gradient_checkpointing: bool = False  # Off for eval


class VIBTokenClassifier(nn.Module):
    def __init__(self, cfg, num_labels):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        self.encoder.config.use_cache = False
        hidden_size = self.encoder.config.hidden_size
        self.scalar_mix_top_k = min(cfg.scalar_mix_top_k, self.encoder.config.num_hidden_layers + 1)
        self.layer_weights = nn.Parameter(torch.zeros(self.scalar_mix_top_k))
        self.layer_gamma = nn.Parameter(torch.ones(1))
        self.vib_mu = nn.Linear(hidden_size, cfg.bottleneck_dim)
        self.vib_logvar = nn.Linear(hidden_size, cfg.bottleneck_dim)
        self.classifier = nn.Linear(cfg.bottleneck_dim, num_labels)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        hidden_states = outputs.hidden_states[-self.scalar_mix_top_k:]
        mix = torch.softmax(self.layer_weights, dim=0)
        mixed = self.layer_gamma * sum(w * hs for w, hs in zip(mix, hidden_states))
        mu = self.vib_mu(mixed)
        logits = self.classifier(self.dropout(mu))
        return {"logits": logits, "mu": mu, "kl_loss": torch.tensor(0.0)}


# ---- Tokenization ----
def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(
        [list(map(str, t)) for t in examples["tokens"]],
        truncation=True, max_length=MAX_LENGTH,
        is_split_into_words=True, padding="max_length",
    )
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        labels, prev = [], None
        for wid in word_ids:
            if wid is None: labels.append(-100)
            elif wid != prev: labels.append(tags[wid])
            else: labels.append(-100)
            prev = wid
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized


def collate(batch):
    return {k: torch.tensor([item[k] for item in batch], dtype=torch.long) for k in batch[0].keys()}


# ---- Evaluation with per-entity breakdown ----
def evaluate_detailed(model, loader, device, is_vibspot=False):
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"]

            if is_vibspot:
                out = model(ids, mask)
                logits = out["logits"]
            else:
                out = model(input_ids=ids, attention_mask=mask)
                logits = out.logits

            preds = torch.argmax(logits, dim=-1).cpu()
            for ps, ls in zip(preds, labs):
                pl, ll = [], []
                for p, l in zip(ps.tolist(), ls.tolist()):
                    if l != -100:
                        pl.append(LABEL_NAMES[p])
                        ll.append(LABEL_NAMES[l])
                all_p.append(pl)
                all_l.append(ll)

    return {
        "f1": f1_score(all_l, all_p),
        "precision": precision_score(all_l, all_p),
        "recall": recall_score(all_l, all_p),
        "report": classification_report(all_l, all_p, output_dict=True),
    }


def main():
    from datasets import load_dataset
    from transformers import AutoModelForTokenClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    # Load test data
    print("Loading test data...")
    test_loaders = {}
    for lang in TARGETS:
        ds = load_dataset("wikiann", lang, split="test")
        tok = ds.map(lambda ex: tokenize_and_align(ex, tokenizer), batched=True,
                     remove_columns=ds.column_names)
        tok.set_format("python")
        test_loaders[lang] = DataLoader(tok, batch_size=EVAL_BATCH, collate_fn=collate)

    # ---- Evaluate Baseline ----
    print(f"\n{'='*60}")
    print(f"EVALUATING: Baseline")
    print(f"{'='*60}")
    baseline_results = {}
    if os.path.exists(BASELINE_PATH):
        model = AutoModelForTokenClassification.from_pretrained(
            BASELINE_PATH, num_labels=len(LABEL_NAMES)
        ).to(device)
        for lang in TARGETS:
            m = evaluate_detailed(model, test_loaders[lang], device, is_vibspot=False)
            baseline_results[lang] = m
            r = m["report"]
            print(f"  {lang}: F1={m['f1']:.4f} | PER={r.get('PER',{}).get('f1-score',0):.4f} | LOC={r.get('LOC',{}).get('f1-score',0):.4f} | ORG={r.get('ORG',{}).get('f1-score',0):.4f}")
        del model; torch.cuda.empty_cache()
    else:
        print(f"  NOT FOUND: {BASELINE_PATH}")

    # ---- Evaluate VIB-SPOT ----
    print(f"\n{'='*60}")
    print(f"EVALUATING: VIB-SPOT")
    print(f"{'='*60}")
    vibspot_results = {}
    if os.path.exists(VIBSPOT_PT_PATH):
        # Load checkpoint
        checkpoint = torch.load(VIBSPOT_PT_PATH, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        saved_config = checkpoint.get("config", {})
        saved_model_name = checkpoint.get("model_name", "xlm-roberta-base")

        # Recreate model with correct config
        cfg = Config()
        cfg.model_name = saved_model_name
        cfg.bottleneck_dim = saved_config.get("bottleneck_dim", 256)
        cfg.scalar_mix_top_k = saved_config.get("scalar_mix_top_k", 4)
        cfg.dropout = saved_config.get("dropout", 0.1)

        print(f"  Model: {cfg.model_name}")
        print(f"  Bottleneck: {cfg.bottleneck_dim}")
        print(f"  Scalar mix top k: {cfg.scalar_mix_top_k}")

        model = VIBTokenClassifier(cfg, num_labels=len(LABEL_NAMES))
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"  Loaded {len(state_dict)} keys successfully")

        for lang in TARGETS:
            m = evaluate_detailed(model, test_loaders[lang], device, is_vibspot=True)
            vibspot_results[lang] = m
            r = m["report"]
            print(f"  {lang}: F1={m['f1']:.4f} | PER={r.get('PER',{}).get('f1-score',0):.4f} | LOC={r.get('LOC',{}).get('f1-score',0):.4f} | ORG={r.get('ORG',{}).get('f1-score',0):.4f}")
        del model; torch.cuda.empty_cache()
    else:
        print(f"  NOT FOUND: {VIBSPOT_PT_PATH}")

    # ---- Comparison ----
    if baseline_results and vibspot_results:
        for entity in ["Overall", "PER", "LOC", "ORG"]:
            print(f"\n  --- {entity} ---")
            print(f"  {'Lang':<6} {'Baseline':>10} {'VIB-SPOT':>10} {'Delta':>10}")
            print(f"  {'-'*40}")
            for lang in TARGETS:
                if entity == "Overall":
                    b = baseline_results[lang]["f1"]
                    v = vibspot_results[lang]["f1"]
                else:
                    b = baseline_results[lang]["report"].get(entity, {}).get("f1-score", 0)
                    v = vibspot_results[lang]["report"].get(entity, {}).get("f1-score", 0)
                print(f"  {lang:<6} {b:>10.4f} {v:>10.4f} {v-b:>+10.4f}")

    # Save
    os.makedirs("./analysis_results", exist_ok=True)
    save = {}
    for name, res in [("baseline", baseline_results), ("vibspot", vibspot_results)]:
        save[name] = {
            l: {"f1": m["f1"], "per_entity": {e: m["report"].get(e, {}) for e in ["PER","LOC","ORG"]}}
            for l, m in res.items()
        }
    with open("./analysis_results/per_entity.json", "w") as f:
        json.dump(save, f, indent=2, default=float)
    print(f"\nSaved to ./analysis_results/per_entity.json")


if __name__ == "__main__":
    main()