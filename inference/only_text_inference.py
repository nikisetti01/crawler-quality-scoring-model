#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True,
                    help="HuggingFace path o dir locale dello checkpoint QualT5")
parser.add_argument("--input_csv", type=str, required=True,
                    help="Input TSV/CSV con almeno la colonna 'text'")
parser.add_argument("--output_csv", type=str, required=True,
                    help="Output CSV con [clueweb_id,url,score,pred(,label)]")
parser.add_argument("--sep", type=str, default="\t", help="Separatore (default: TAB)")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--chunksize", type=int, default=200_000)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

DEVICE = torch.device(args.device)
print(f"✅ Device: {DEVICE}")

# -----------------------------
# Tokenizer/Model
# -----------------------------
print("🧠 Loading QualT5 tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(DEVICE)
model.eval()

# Mappa robusta dei token "true"/"false" (sentencepiece può usare l'underscore basso)
CAND_TRUE = ["▁true", "true", "Ġtrue", "<true>"]
CAND_FALSE = ["▁false", "false", "Ġfalse", "<false>"]

def find_token_id(tok, candidates):
    for c in candidates:
        tid = tok.convert_tokens_to_ids(c)
        if tid is not None and tid != tok.unk_token_id:
            return tid, c
    return None, None

true_id, true_str = find_token_id(tokenizer, CAND_TRUE)
false_id, false_str = find_token_id(tokenizer, CAND_FALSE)
if true_id is None or false_id is None:
    raise ValueError(f"Token 'true'/'false' non trovati nel vocab. Trovato true={true_str}, false={false_str}")

print(f"🔎 Using true_token={true_str}({true_id}), false_token={false_str}({false_id})")

# -----------------------------
# Dataset/Collate
# -----------------------------
REQ_COL = "text"

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if REQ_COL not in df.columns:
        raise ValueError(f"L'input deve contenere la colonna '{REQ_COL}'.")
    # opzionali
    for c in ("clueweb_id", "url"):
        if c not in df.columns:
            df[c] = ""
    if "label" in df.columns:
        df["label"] = df["label"].fillna(0).astype(int)
    df["text"] = df["text"].fillna("").astype(str)
    return df

class QTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tok: AutoTokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(
            r["text"],
            truncation=True, padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        out = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "_id": str(r.get("clueweb_id", "")),
            "_url": str(r.get("url", "")),
        }
        if "label" in self.df.columns:
            out["_label"] = int(r["label"])
        return out

def collate_fn(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "_ids": [b["_id"] for b in batch],
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
printed_preview=0
print(f"📄 Input: {args.input_csv}")
reader = pd.read_csv(args.input_csv, sep=args.sep, chunksize=args.chunksize)

with torch.no_grad():
    for chunk in reader:
        chunk = ensure_cols(chunk)
        ds = QTDataset(chunk, tokenizer, args.max_length)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=1,
                        pin_memory=True, collate_fn=collate_fn)

        rows, labels_buf = [], []

        for batch in tqdm(dl, desc="🔍 Inference (QualT5)"):
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)

            # Logits del primo step decoder (pos=0)
            dec_start = torch.full(
                (input_ids.size(0), 1),
                model.config.decoder_start_token_id,
                dtype=torch.long, device=DEVICE
            )
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=dec_start,
                return_dict=True
            )
            logits_step0 = outputs.logits[:, 0, :]  # [B, V]
            probs = torch.softmax(logits_step0, dim=-1)  # [B, V]

            # Normalizzazione su soli {true,false}
            denom = probs[:, true_id] + probs[:, false_id] + 1e-12
            scores = (probs[:, true_id] / denom).detach().cpu().numpy().tolist()
            preds = (np.array(scores) >= 0.5).astype(int).tolist()

            if "_labels" in batch:
                labs = batch["_labels"].detach().cpu().numpy().tolist()
                labels_buf.extend(labs)
                for cid, url, s, p, lab in zip(batch["_ids"], batch["_urls"], scores, preds, labs):
                    rows.append((cid, url, s, p, lab))
                    if printed_preview < 5:
                        print(f"  [P] id={cid} url={url} score={s:.4f} pred={p} label={lab}")
                        printed_preview += 1
            else:
                for cid, url, s, p in zip(batch["_ids"], batch["_urls"], scores, preds):
                    rows.append((cid, url, s, p))
                    if printed_preview < 5:
                        print(f"  [P] id={cid} url={url} score={s:.4f} pred={p}")
                        printed_preview += 1

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
