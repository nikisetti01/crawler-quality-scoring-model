#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                    help="HF encoder name for tokenizer/encoder")
parser.add_argument("--ckpt", type=str, required=True,
                    help="Path to .pt checkpoint")
parser.add_argument("--input_csv", type=str, required=True,
                    help="Input TSV/CSV (uses sep inferred or force with --sep)")
parser.add_argument("--sep", type=str, default="\t", help="Field separator (default: TAB)")
parser.add_argument("--output_csv", type=str, required=True,
                    help="Output CSV with predictions")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--chunksize", type=int, default=200_000)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

DEVICE = torch.device(args.device)
print(f"✅ Device: {DEVICE}")

# -----------------------------
# Column mapping (15-col header)
# -----------------------------
NUM_COLS = [
    'num_inlinks','length_inlinks','inlinks_domains_count','inlink_slashes_count',
    'num_outlinks','length_outlinks','outlinks_domains_count','outlink_slashes_count'
]
ANCHOR_IN_COL  = 'inlink_anchors'
ANCHOR_OUT_COL = 'outlink_anchors'
DOM_IN_LIST    = 'inlink_list_domains'
DOM_OUT_LIST   = 'outlink_list_domains'

# -----------------------------
# Model import
# -----------------------------
try:
    from model_bert import MultiModalWebClassifier
except Exception:
    from model.model_bert import MultiModalWebClassifier

# -----------------------------
# Helper: checkpoint remap (LoRA-safe)
# -----------------------------
def remap_state_dict_for_plain_encoder(state):
    """
    - Rimuove prefisso 'encoder.base_model.model.' -> 'encoder.'
    - Rimuove '.base_layer.' (alcuni wrapper LoRA)
    - Scarta chiavi LoRA ('.lora_A.' / '.lora_B.')
    Restituisce un nuovo state_dict caricabile con strict=False.
    """
    remapped, drop = {}, 0
    for k, v in state.items():
        k2 = k
        if k2.startswith("encoder.base_model.model."):
            k2 = "encoder." + k2[len("encoder.base_model.model."):]
        k2 = k2.replace(".base_layer.", ".")
        if ".lora_A." in k2 or ".lora_B." in k2:
            drop += 1
            continue
        remapped[k2] = v
    print(f"[INFO] remap: kept={len(remapped)} dropped_lora={drop}")
    return remapped

def load_checkpoint_into_model(model, ckpt_path):
    # torch warning consiglia weights_only=True (se disponibile)
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location="cpu")
    # Se è un dict con 'state_dict' dentro (alcuni trainer salvano così)
    if isinstance(state, dict) and "state_dict" in state and len(state) <= 2:
        state = state["state_dict"]

    # Prova a caricare diretto
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] direct load: missing={len(missing)} unexpected={len(unexpected)}")
        return
    except Exception as e:
        print(f"[WARN] direct load failed: {e}\n[INFO] trying remap LoRA/prefix...")

    # Remap LoRA/prefix
    remapped = remap_state_dict_for_plain_encoder(state)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[INFO] remap load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[DEBUG] missing (first 20): {missing[:20]}")
    if unexpected:
        print(f"[DEBUG] unexpected (first 20): {unexpected[:20]}")

# -----------------------------
# Tokenizer/Encoder/Model
# -----------------------------
print("🧠 Building encoder/tokenizer...")
tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
encoder = AutoModel.from_pretrained(args.model_name)
model = MultiModalWebClassifier(encoder).to(DEVICE)
print(f"📦 Loading checkpoint: {args.ckpt}")
load_checkpoint_into_model(model, args.ckpt)
print(f"dataset path: {args.input_csv}")
model.eval()
sigmoid = nn.Sigmoid()

# -----------------------------
# Dataset / Collate
# -----------------------------
def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if 'text' not in df.columns:
        raise ValueError("Input deve contenere la colonna 'text'.")
    # numerics
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = 0.0
    # anchors/domains
    for c in (ANCHOR_IN_COL, ANCHOR_OUT_COL, DOM_IN_LIST, DOM_OUT_LIST):
        if c not in df.columns:
            df[c] = ""
        else:
            if c.endswith("_list_domains"):
                df[c] = df[c].fillna("").astype(str).str.replace(";", " ").str.replace("|", " ")
            else:
                df[c] = df[c].fillna("").astype(str)
    # opzionali id/url/label
    for c in ("clueweb_id", "url"):
        if c not in df.columns:
            df[c] = ""
    if "label" in df.columns:
        df["label"] = df["label"].fillna(0).astype(int)
    return df

class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # tokenizzazioni (come nel train)
        te = tok(str(r["text"]), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        ao = tok(str(r[ANCHOR_OUT_COL]), truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        ai = tok(str(r[ANCHOR_IN_COL]),  truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        do = tok(str(r[DOM_OUT_LIST]),   truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        di = tok(str(r[DOM_IN_LIST]),    truncation=True, padding="max_length", max_length=64, return_tensors="pt")

        numerics = torch.tensor(self.df.loc[i, NUM_COLS].astype(np.float32).values, dtype=torch.float32)

        out = {
            "input_ids":       te["input_ids"][0],
            "attention_mask":  te["attention_mask"][0],
            "anchor_out_ids":  ao["input_ids"][0],
            "anchor_out_mask": ao["attention_mask"][0],
            "anchor_in_ids":   ai["input_ids"][0],
            "anchor_in_mask":  ai["attention_mask"][0],
            "domains_out_ids": do["input_ids"][0],
            "domains_in_ids":  di["input_ids"][0],
            "numerics":        numerics,
            "_id":   str(r.get("clueweb_id", "")),
            "_url":  str(r.get("url", "")),
        }
        if "label" in self.df.columns:
            out["_label"] = int(r["label"])
        return out

def collate_fn(batch):
    out = {
        "input_ids":       torch.stack([b["input_ids"] for b in batch]),
        "attention_mask":  torch.stack([b["attention_mask"] for b in batch]),
        "anchor_out_ids":  torch.stack([b["anchor_out_ids"] for b in batch]),
        "anchor_out_mask": torch.stack([b["anchor_out_mask"] for b in batch]),
        "anchor_in_ids":   torch.stack([b["anchor_in_ids"] for b in batch]),
        "anchor_in_mask":  torch.stack([b["anchor_in_mask"] for b in batch]),
        "domains_out_ids": torch.stack([b["domains_out_ids"] for b in batch]),
        "domains_in_ids":  torch.stack([b["domains_in_ids"] for b in batch]),
        "numerics":        torch.stack([b["numerics"] for b in batch]),
        "_ids":  [b["_id"] for b in batch],
        "_urls": [b["_url"] for b in batch],
    }
    if "_label" in batch[0]:
        out["_labels"] = torch.tensor([b["_label"] for b in batch], dtype=torch.long)
    return out

# -----------------------------
# Streaming inference
# -----------------------------
os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
first = True
written = 0
rows_seen = 0

reader = pd.read_csv(args.input_csv, sep=args.sep, chunksize=args.chunksize)

with torch.no_grad():
    for chunk in reader:
        chunk = ensure_cols(chunk)
        ds = InferenceDataset(chunk)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True, collate_fn=collate_fn)

        rows, labels_buf = [], []

        for batch in tqdm(dl, desc="🔍 Inference (BERT)"):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if not k.startswith("_")}
            logits = model(**inputs)  # (B,)
            probs  = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            preds  = (np.array(probs) >= 0.5).astype(int).tolist()

            if "_labels" in batch:
                labs = batch["_labels"].detach().cpu().numpy().tolist()
                labels_buf.extend(labs)
                for cid, url, s, p, lab in zip(batch["_ids"], batch["_urls"], probs, preds, labs):
                    rows.append((cid, url, s, p, lab))
            else:
                for cid, url, s, p in zip(batch["_ids"], batch["_urls"], probs, preds):
                    rows.append((cid, url, s, p))

        if rows:
            if labels_buf:
                out_df = pd.DataFrame(rows, columns=["clueweb_id","url","score","pred","label"])
            else:
                out_df = pd.DataFrame(rows, columns=["clueweb_id","url","score","pred"])
            out_df.to_csv(args.output_csv, mode="a", header=first, index=False)
            first = False
            written += len(out_df)
            rows_seen += len(chunk)
            print(f"[CKPT] processed~={rows_seen:,} written={written:,} → {args.output_csv}")

print(f"[DONE] Total written: {written:,} rows → {args.output_csv}")
