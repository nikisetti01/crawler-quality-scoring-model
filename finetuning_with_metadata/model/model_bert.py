#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import wandb

from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast
from transformers.modeling_outputs import BaseModelOutput

# =========================
# Args & init
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="pyterrier-quality/qt5-small",
                    help="bert-base-uncased | pyterrier-quality/qt5-small | qt5_decoder")
parser.add_argument("--max_steps_p1", type=int, default=25000)
parser.add_argument("--max_steps_p2", type=int, default=10000)
parser.add_argument("--qt5_train_dir", type=str,
                    default="/mnt/ssd_data/tokenized_qt5_small_generative_train",
                    help="Directory dei chunk train per qt5 generativo")
parser.add_argument("--bert_train_dir", type=str,
                    default="/mnt/ssd_data/tokenized_bert_train",
                    help="Directory dei chunk train per BERT")
parser.add_argument("--num_qt5_chunks", type=int, default=2000,
                    help="Numero max chunk da considerare (qt5)")
parser.add_argument("--num_bert_chunks", type=int, default=126,
                    help="Numero max chunk da considerare (bert)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

model_name = args.model
short_model = model_name.split("/")[-1].replace('-', '_')

# =========================
# Tokenizer & Encoder selector
# =========================
if model_name == "qt5_decoder":
    short_model = "qt5_small"
    tokenizer = AutoTokenizer.from_pretrained("pyterrier-quality/qt5-small")
    encoder = None
elif model_name.startswith("pyterrier-quality/qt5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = None
else:
    print(f"Using BERT tokenizer for {model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

print(f"Using model: {model_name} (short name: {short_model})")

# =========================
# Dataset paths
# =========================
if "bert" in model_name:
    CHUNK_TRAIN_PATHS = [
        os.path.join(args.bert_train_dir, f"tokenized_chunk_{i}")
        for i in range(args.num_bert_chunks)
    ]
else:
    # qt5 generativo: usa SOLO i chunk di TRAIN e fai split 90/10 per validation
    CHUNK_TRAIN_PATHS = [
        os.path.join(args.qt5_train_dir, f"tokenized_chunk_{i}")
        for i in range(args.num_qt5_chunks)
    ]

CHUNK_TRAIN_PATHS = [p for p in CHUNK_TRAIN_PATHS if os.path.exists(p)]
if not CHUNK_TRAIN_PATHS:
    raise FileNotFoundError("Nessun chunk trovato nei path configurati.")

# =========================
# Load datasets
# =========================
print("🔄 Loading datasets...")
if "bert" in model_name:
    raw_dataset = concatenate_datasets([load_from_disk(p) for p in CHUNK_TRAIN_PATHS])
    split = raw_dataset.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")
    train_dataset = split['train']
    val_dataset   = split['test']
else:
    # qt5 generativo: concatena SOLO train e split 90/10 stratificato
    train_datasets = [load_from_disk(p) for p in CHUNK_TRAIN_PATHS]
    raw_dataset = concatenate_datasets(train_datasets)
    split = raw_dataset.train_test_split(test_size=0.1, seed=args.seed, stratify_by_column="label")
    train_dataset = split['train']
    val_dataset   = split['test']

# =========================
# Dataset wrapper
# =========================
class HFTokenizedDataset(Dataset):
    def __init__(self, hf_dataset, for_qt5_decoder=False):
        self.dataset = hf_dataset
        self.for_qt5_decoder = for_qt5_decoder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        numerics = item['numerics'] if item.get('numerics') is not None and all(v is not None for v in item['numerics']) else [0.0]*8

        sample = {
            'input_ids':        torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask':   torch.tensor(item['attention_mask'], dtype=torch.long),
            'anchor_out_ids':   torch.tensor(item['anchor_out_ids'], dtype=torch.long),
            'anchor_out_mask':  torch.tensor(item['anchor_out_mask'], dtype=torch.long),
            'anchor_in_ids':    torch.tensor(item['anchor_in_ids'], dtype=torch.long),
            'anchor_in_mask':   torch.tensor(item['anchor_in_mask'], dtype=torch.long),
            'domains_out_ids':  torch.tensor(item['domains_out_ids'], dtype=torch.long),
            'domains_in_ids':   torch.tensor(item['domains_in_ids'], dtype=torch.long),
            'numerics':         torch.tensor(numerics, dtype=torch.float32),
            'label':            torch.tensor(int(item['label']), dtype=torch.float32),
        }

        # per qt5_decoder: usa decoder_labels dal dataset se presenti, altrimenti ricostruisci
        if self.for_qt5_decoder:
            if 'decoder_labels' in item and item['decoder_labels'] is not None and len(item['decoder_labels']) > 0:
                sample['decoder_labels'] = torch.tensor(item['decoder_labels'], dtype=torch.long)
            else:
                label_str = "true" if int(item["label"]) == 1 else "false"
                tokenized = tokenizer(label_str, return_tensors="pt").input_ids[0]
                sample['decoder_labels'] = tokenized.long()  # includiamo tutto
        else:
            # qt5 generativo “small”: se esiste decoder_labels, mantienila per training T5
            if 'decoder_labels' in item and item['decoder_labels'] is not None and len(item['decoder_labels']) > 0:
                sample['decoder_labels'] = torch.tensor(item['decoder_labels'], dtype=torch.long)

        return sample

# =========================
# Model import
# =========================
if model_name == "qt5_small":
    print("passo da qualt5 decoder")
    # adatta al tuo path/modulo
    from model.model_qualt5_decoder import MultiModalWebClassifier  # noqa
elif model_name.startswith("pyterrier-quality/qt5-small"):
    # adatta al tuo path/modulo
    from model.model_qualt5_decoder import MultiModalWebClassifier  # noqa
else:
    from model.model_bert import MultiModalWebClassifier  # noqa

# =========================
# Dataloaders
# =========================
BATCH_SIZE_P1 = 256 if "bert" in model_name else 64
BATCH_SIZE_P2 = 32

train_loader = DataLoader(
    HFTokenizedDataset(train_dataset, for_qt5_decoder=(model_name=="qt5_decoder")),
    batch_size=BATCH_SIZE_P1, shuffle=True, pin_memory=True
)
val_loader   = DataLoader(
    HFTokenizedDataset(val_dataset,   for_qt5_decoder=(model_name=="qt5_decoder")),
    batch_size=BATCH_SIZE_P1, shuffle=False, pin_memory=True
)

# =========================
# Class weight (solo per BERT)
# =========================
if "bert" in model_name:
    labels = [int(x) for x in train_dataset["label"]]
    pos = sum(labels)
    neg = len(labels) - pos
    pos_weight_val = (neg / max(pos, 1))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    print(f"🔢 Computed pos_weight: {pos_weight_val:.4f}")
else:
    pos_weight = None

# =========================
# Build model
# =========================
if "bert" in model_name:
    model = MultiModalWebClassifier(encoder)
else:
    model = MultiModalWebClassifier()

# =========================
# LoRA (solo per BERT encoder, opzionale)
# =========================
def apply_lora_to_bert_encoder(bert_encoder):
    from peft import LoraConfig, get_peft_model
    target_names = []
    for name, module in bert_encoder.named_modules():
        lname = name.lower()
        if ("encoder.layer" in lname) and (
            lname.endswith("attention.self.query") or
            lname.endswith("attention.self.key") or
            lname.endswith("attention.self.value")
        ):
            target_names.append(name)

    lconf = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=target_names,
        lora_dropout=0.05,
        bias="none",
    )
    peft_encoder = get_peft_model(bert_encoder, lconf)
    peft_encoder.print_trainable_parameters()
    return peft_encoder

# =========================
# Optim/loss helpers
# =========================
def make_optimizer(model, lr, weight_decay=0.01):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )

if "bert" in model_name:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = None  # T5 gestisce la loss con labels

# =========================
# Accelerator prepare
# =========================
optimizer = make_optimizer(model, lr=1e-3)
if "bert" in model_name:
    model, optimizer, train_loader, val_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, val_loader, criterion
    )
else:
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

# =========================
# W&B
# =========================
wandb.init(project="multimodal-web-classifier",
           name=f"{short_model}-metadata",
           config={
               "base_model": model_name,
               "batch_size_p1": BATCH_SIZE_P1,
               "batch_size_p2": BATCH_SIZE_P2,
               "pos_weight": (float(pos_weight.detach().cpu().item()) if pos_weight is not None else None)
           })

# =========================
# Eval
# =========================
def run_eval_bert(model, val_loader, criterion):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**{k: v for k, v in batch.items() if k != "label"})
            loss = criterion(outputs, batch["label"])
            val_loss += loss.item()
            preds = (outputs > 0.0).long()  # logits threshold 0.0 == sigmoid>0.5
            correct += (preds == batch["label"].long()).sum().item()
            total   += batch["label"].size(0)
    return (val_loss / max(len(val_loader), 1)), (correct / max(total, 1))

def run_eval_qt5(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            enc_out = model.model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            enc_hidden = enc_out.last_hidden_state

            dom_out = model.domains_embedding(batch['domains_out_ids']).mean(dim=1)
            dom_in  = model.domains_embedding(batch['domains_in_ids']).mean(dim=1)
            meta_tokens = model.meta_encoder(
                batch['anchor_out_ids'], batch['anchor_in_ids'],
                dom_out, dom_in, batch['numerics']
            )
            enc_mask = batch['attention_mask']
            if enc_mask.dtype != torch.bool:
                enc_mask = enc_mask.bool()
            enc_mask = batch["attention_mask"]
            if enc_mask.dtype != torch.bool:
                enc_mask = enc_mask.bool()
            uni_out = model.uni_attn(meta_tokens, enc_hidden, enc_mask)
            prompt = model.prompt_mapper(uni_out.mean(dim=1)).unsqueeze(1)  # (B,1,H)

            # Decodifica 1 token: 'true'/'false' è un singolo token nella pratica
            out = model.model(
                encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                decoder_inputs_embeds=prompt,
                return_dict=True
            )
            logits = out.logits                      # (B,1,V)
            pred_token_ids = torch.argmax(logits, dim=-1)  # (B,1)
            pred_text = tokenizer.batch_decode(pred_token_ids, skip_special_tokens=True)
            preds = torch.tensor([1 if p.strip().lower()=="true" else 0 for p in pred_text], device=device)

            correct += (preds == batch['label'].long()).sum().item()
            total   += batch['label'].size(0)

    val_acc = correct / max(total, 1)
    return 0.0, val_acc

# =========================
# Train phase
# =========================
def train_phase_bert(model, train_loader, val_loader, epochs, lr, max_steps, freeze_encoder=True, use_lora=False):
    # Freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = not freeze_encoder

    # LoRA SOLO encoder
    if use_lora:
        accelerator.wait_for_everyone()
        model.encoder = apply_lora_to_bert_encoder(model.encoder)
        model = accelerator.prepare(model)

    optimizer = make_optimizer(model, lr=lr)
    optimizer = accelerator.prepare(optimizer)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        running = 0.0
        steps_this_epoch = 0

        with tqdm(total=min(max_steps, len(train_loader)),
                  desc=f"Epoch {epoch+1}",
                  disable=not accelerator.is_local_main_process) as pbar:
            for batch in train_loader:
                if steps_this_epoch >= max_steps:
                    break
                optimizer.zero_grad(set_to_none=True)
                outputs = model(**{k: v for k, v in batch.items() if k != "label"})
                loss = criterion(outputs, batch["label"])
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running += loss.item()
                steps_this_epoch += 1
                pbar.update(1)

        val_loss, val_acc = run_eval_bert(model, val_loader, criterion)
        train_loss = running / max(1, min(max_steps, len(train_loader)))

        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"[Epoch {epoch+1}] Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {val_acc:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                accelerator.save(model.state_dict(), f"best_model_{short_model}.pt")
    return model

def train_phase_qt5(model, train_loader, val_loader, epochs, lr, max_steps):
    optimizer = make_optimizer(model, lr=lr)
    optimizer = accelerator.prepare(optimizer)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0

        loader_iter = iter(train_loader)
        with tqdm(total=min(max_steps, len(train_loader)), desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process) as pbar:
            while global_step < max_steps:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break

                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)

                # Encoder
                enc_out = model.model.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                enc_hidden = enc_out.last_hidden_state

                # Metadata fusion
                dom_out = model.domains_embedding(batch['domains_out_ids']).mean(dim=1)
                dom_in  = model.domains_embedding(batch['domains_in_ids']).mean(dim=1)
                meta_tokens = model.meta_encoder(
                    batch['anchor_out_ids'], batch['anchor_in_ids'],
                    dom_out, dom_in, batch['numerics']
                )
               # uni_out = model.uni_attn(meta_tokens, enc_hidden)
                enc_mask = batch["attention_mask"]
                if enc_mask.dtype != torch.bool:
                    enc_mask = enc_mask.bool()
                uni_out = model.uni_attn(meta_tokens, enc_hidden, enc_mask)

                # Labels del decoder
                labels = batch['decoder_labels'].view(batch['decoder_labels'].size(0), -1).long()

                # Prompt: espandi alla stessa lunghezza delle labels
                prompt = model.prompt_mapper(uni_out.mean(dim=1)).unsqueeze(1).expand(-1, labels.size(1), -1)

                out = model.model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
                    decoder_inputs_embeds=prompt,
                    labels=labels,
                    return_dict=True
                )
                loss = out.loss

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running += loss.item()
                global_step += 1
                pbar.update(1)

        _, val_acc = run_eval_qt5(model, val_loader)
        train_loss = running / max(1, min(max_steps, len(train_loader)))

        if accelerator.is_local_main_process:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_accuracy": val_acc})
            print(f"[Epoch {epoch+1}] Train {train_loss:.4f} | Val Acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                accelerator.save(model.state_dict(), f"best_model_{short_model}.pt")
    return model

# =========================
# Run
# =========================
if "bert" in model_name:
    # Phase 1: freeze encoder
    model = train_phase_bert(model, train_loader, val_loader,
                             epochs=2, lr=1e-3, max_steps=args.max_steps_p1,
                             freeze_encoder=True, use_lora=False)

    # Phase 2: unfreeze + LoRA SOLO encoder (opzionale)
    train_loader_p2 = DataLoader(HFTokenizedDataset(train_dataset), batch_size=BATCH_SIZE_P2, shuffle=True, pin_memory=True)
    val_loader_p2   = DataLoader(HFTokenizedDataset(val_dataset),   batch_size=BATCH_SIZE_P2, shuffle=False,pin_memory=True)
    train_loader_p2, val_loader_p2 = accelerator.prepare(train_loader_p2, val_loader_p2)

    model = train_phase_bert(model, train_loader_p2, val_loader_p2,
                             epochs=1, lr=1e-4, max_steps=args.max_steps_p2,
                             freeze_encoder=False, use_lora=True)
else:
    # QT5 generativo
    model = train_phase_qt5(model, train_loader, val_loader,
                            epochs=2, lr=1e-3, max_steps=args.max_steps_p1)

# Save finale
if accelerator.is_local_main_process:
    accelerator.save(model.state_dict(), f"best_model_{short_model}.pt")
    print("✅ Done. Saved:", f"best_model_{short_model}.pt")
