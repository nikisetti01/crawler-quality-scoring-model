#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    brier_score_loss, precision_recall_curve
)
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, choices=["bert", "qt5"], required=True,
                    help="Scegli quale architettura valutare: 'bert' o 'qt5'")
parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                    help="Per BERT: nome modello HF dell'encoder. Per QT5 puoi ignorarlo.")
parser.add_argument("--ckpt", type=str, required=True, help="Path ai pesi .pt")
parser.add_argument("--test_dir", type=str, required=True, help="Directory con tokenized_chunk_*")
parser.add_argument("--chunks", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

# -----------------------------
# Dataset
# -----------------------------
class HFTokenizedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        it = self.ds[i]
        numerics = it['numerics'] if it.get('numerics') is not None and all(v is not None for v in it['numerics']) else [0.0]*8
        out = {
            "input_ids":       torch.tensor(it["input_ids"], dtype=torch.long),
            "attention_mask":  torch.tensor(it["attention_mask"], dtype=torch.long),
            "anchor_out_ids":  torch.tensor(it["anchor_out_ids"], dtype=torch.long),
            "anchor_in_ids":   torch.tensor(it["anchor_in_ids"], dtype=torch.long),
            "anchor_out_mask": torch.tensor(it.get("anchor_out_mask", [1]*len(it["anchor_out_ids"])), dtype=torch.long),
            "anchor_in_mask":  torch.tensor(it.get("anchor_in_mask",  [1]*len(it["anchor_in_ids"])),  dtype=torch.long),
            "domains_out_ids": torch.tensor(it["domains_out_ids"], dtype=torch.long),
            "domains_in_ids":  torch.tensor(it["domains_in_ids"], dtype=torch.long),
            "numerics":        torch.tensor(numerics, dtype=torch.float32),
            "label":           torch.tensor(int(it["label"]), dtype=torch.long),
        }
        # decoder_labels opzionali (QT5)
        if "decoder_labels" in it and it["decoder_labels"] is not None and len(it["decoder_labels"])>0:
            out["decoder_labels"] = torch.tensor(it["decoder_labels"], dtype=torch.long)
        return out

def load_dataset_concatenated(base_dir, n_chunks):
    paths = [os.path.join(base_dir, f"tokenized_chunk_{i}") for i in range(n_chunks)]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(f"Nessun chunk trovato in {base_dir}")
    dsets = [load_from_disk(p) for p in paths]
    return concatenate_datasets(dsets)

print("🔄 Loading test dataset...")
test_hf = load_dataset_concatenated(args.test_dir, args.chunks)
test_loader = DataLoader(HFTokenizedDataset(test_hf),
                         batch_size=args.batch_size,
                         shuffle=False, pin_memory=True,
                         num_workers=args.num_workers)

# -----------------------------
# Utilities
# -----------------------------
def state_dict_has_lora_keys(sd):
    return any((".lora_A." in k) or (".lora_B." in k) for k in sd.keys())

def make_pr_curves(y_true, y_score, roc_auc, pr_auc):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(True)
    plt.savefig("roc_curve.png"); plt.close()

    precs, recs, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recs, precs, label=f"AP = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(); plt.grid(True)
    plt.savefig("pr_curve.png"); plt.close()

def print_and_save_metrics(y_true, y_score, y_pred, out_csv="evaluation_results.csv"):
    print("\n📊 Metrics (threshold 0.5)")
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = pr_auc = brier = float("nan")
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_score)
        pr_auc  = average_precision_score(y_true, y_score)
        brier   = brier_score_loss(y_true, y_score)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"PR  AUC  : {pr_auc:.4f}")
    print(f"Brier    : {brier:.6f}")

    pd.DataFrame({"label": y_true, "score": y_score, "prediction": y_pred}).to_csv(out_csv, index=False)
    print(f"💾 Saved: {out_csv}")

    if len(np.unique(y_true)) == 2:
        make_pr_curves(y_true, y_score, roc_auc, pr_auc)

# -----------------------------
# Branch: BERT
# -----------------------------
if args.arch == "bert":
    try:
        from model_bert import MultiModalWebClassifier
    except Exception:
        from model.model_bert import MultiModalWebClassifier

    print("🧠 Building BERT model...")
    encoder = AutoModel.from_pretrained(args.model_name)
    model = MultiModalWebClassifier(encoder).to(DEVICE)

    print(f"📦 Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=DEVICE)

    if state_dict_has_lora_keys(state):
        print("🔗 Detected LoRA adapters in checkpoint → applying only to encoder")
        from peft import LoraConfig, get_peft_model
        target = []
        for name, _ in model.encoder.named_modules():
            lname = name.lower()
            if ("encoder.layer" in lname) and (
                lname.endswith("attention.self.query") or
                lname.endswith("attention.self.key") or
                lname.endswith("attention.self.value")
            ):
                target.append(name)
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none")
        model.encoder = get_peft_model(model.encoder, cfg)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"⚠️ Missing keys: {missing}")
    if unexpected: print(f"⚠️ Unexpected keys: {unexpected}")
    model.eval(); print("✅ Model ready (BERT).")

    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔍 Evaluating (BERT)"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                anchor_out_ids=batch["anchor_out_ids"],
                anchor_in_ids=batch["anchor_in_ids"],
                anchor_out_mask=batch["anchor_out_mask"],
                anchor_in_mask=batch["anchor_in_mask"],
                domains_out_ids=batch["domains_out_ids"],
                domains_in_ids=batch["domains_in_ids"],
                numerics=batch["numerics"]
            )
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs  = np.concatenate(all_probs, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
    preds = (all_probs > 0.5).astype(int)

    print_and_save_metrics(all_labels, all_probs, preds)
    raise SystemExit(0)

# -----------------------------
# Branch: QT5 (decoder generativo)
# -----------------------------
# model_qualt5_decoder deve esporre:
# - domains_embedding, meta_encoder, uni_attn(attn_mask), prompt_mapper
# - backbone T5 accessibile come model/t5/backbone (usiamo helper)
try:
    from model.model_qualt5_decoder import MultiModalWebClassifier
except Exception:
    from model_qualt5_decoder import MultiModalWebClassifier

print("🧠 Building QT5 decoder model...")
model = MultiModalWebClassifier().to(DEVICE)

# helper backbone
def get_t5_backbone(m):
    for attr in ["model", "t5", "backbone", "base_model"]:
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError("Backbone T5 non trovato in MultiModalWebClassifier.")

print(f"📦 Loading checkpoint: {args.ckpt}")
state = torch.load(args.ckpt, map_location=DEVICE)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:   print(f"⚠️ Missing keys: {missing}")
if unexpected: print(f"⚠️ Unexpected keys: {unexpected}")
model.eval(); print("✅ Model ready (QT5).")

# tokenizer per ricavare gli id di 'true'/'false'
tok = AutoTokenizer.from_pretrained("pyterrier-quality/qt5-small", use_fast=True)
true_ids  = tok("true",  add_special_tokens=False).input_ids
false_ids = tok("false", add_special_tokens=False).input_ids
if len(true_ids) == 0 or len(false_ids) == 0:
    raise RuntimeError("Token 'true'/'false' non tokenizzabili—controlla il tokenizer.")
true_id, false_id = true_ids[0], false_ids[0]

# Inference: prob_true via softmax(logit_true, logit_false)
all_scores, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="🔍 Evaluating (QT5)"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        t5 = get_t5_backbone(model)

        # Encoder testo
        enc = t5.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        enc_hidden = enc.last_hidden_state

        # Metadata → fusion
        dom_out = model.domains_embedding(batch['domains_out_ids']).mean(dim=1)
        dom_in  = model.domains_embedding(batch['domains_in_ids']).mean(dim=1)
        meta_tokens = model.meta_encoder(
            batch['anchor_out_ids'], batch['anchor_in_ids'],
            dom_out, dom_in, batch['numerics']
        )
        enc_mask = batch["attention_mask"]
        if enc_mask.dtype != torch.bool:
            enc_mask = enc_mask.bool()
        uni_out = model.uni_attn(meta_tokens, enc_hidden, enc_mask)

        # Prompt (1 step decode)
        prompt = model.prompt_mapper(uni_out.mean(dim=1)).unsqueeze(1)  # (B,1,H)
        out = t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
            decoder_inputs_embeds=prompt,
            return_dict=True
        )
        logits = out.logits[:, 0, :]  # (B, V), primo step
        sel = torch.stack([logits[:, true_id], logits[:, false_id]], dim=-1)  # (B,2)
        probs = torch.softmax(sel, dim=-1)[:, 0]  # prob('true')
        scores = probs.detach().cpu().numpy()

        labels = batch["label"].detach().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)

all_scores = np.concatenate(all_scores, axis=0).reshape(-1)
all_labels = np.concatenate(all_labels, axis=0).reshape(-1)
preds = (all_scores > 0.5).astype(int)

print_and_save_metrics(all_labels, all_scores, preds)
