import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel
from datasets import load_from_disk, concatenate_datasets, ClassLabel
from tqdm import tqdm
import wandb
import os
import argparse

# === PARSE MODEL NAME ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="Base model to use (e.g. bert-base-uncased or pyterrier-quality/qt5-small or qt5_decoder)")
args = parser.parse_args()
model_name = args.model

# === CONFIG ===
short_model = model_name.split("/")[-1].replace('-', '_')
print(f"Using model: {model_name} (short name: {short_model})")
CHUNK_TRAIN_PATHS = [f"tokenized_chunks_{short_model}/tokenized_chunk_{i}_train" for i in range(220)]
CHUNK_TEST_PATHS  = [f"tokenized_chunks_{short_model}/tokenized_chunk_{i}_test"  for i in range(220)]
BATCH_SIZE = 64
EPOCHS_PHASE1 = 3
EPOCHS_PHASE2 = 2
HIDDEN_DIM = 128
LOG_EVERY = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === INIT WANDB ===
wandb.init(project="multimodal-web-classifier", name=f"{short_model}-metadata", config={
    "base_model": model_name,
    "epochs_phase1": EPOCHS_PHASE1,
    "epochs_phase2": EPOCHS_PHASE2,
    "batch_size": BATCH_SIZE,
    "hidden_dim": HIDDEN_DIM
})

# === LOAD DATASET ===
train_datasets = [load_from_disk(path) for path in CHUNK_TRAIN_PATHS if os.path.exists(path)]
test_datasets  = [load_from_disk(path) for path in CHUNK_TEST_PATHS if os.path.exists(path)]

train_dataset = concatenate_datasets(train_datasets)
test_dataset  = concatenate_datasets(test_datasets)

# ✅ Fix per label mapping
train_dataset = train_dataset.map(lambda x: {"label": int(x["label"])})
train_dataset = train_dataset.cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"]))
test_dataset = test_dataset.map(lambda x: {"label": int(x["label"])})
test_dataset = test_dataset.cast_column("label", ClassLabel(num_classes=2, names=["neg", "pos"]))

# === MODEL ===
if model_name == "qt5_decoder":
    from model.model_qualt5_decoder import MultiModalWebClassifier
    print("Using QualT5 Decoder model.")
elif model_name.startswith("pyterrier-quality/qt5-small"):
    from model.model_qualt5 import MultiModalWebClassifier
    print("Using PyTerrier Quality T5 model.")
else:
    from model.model_bert import MultiModalWebClassifier
    print("Using BERT model.")

# === WRAPPER DATASET ===
class HFTokenizedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
        self.bad_numerics_count = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        numerics = item['numerics']
        if numerics is None or any(v is None for v in numerics):
            self.bad_numerics_count += 1
            numerics = [0.0, 0.0, 0.0, 0.0]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'anchor_out_ids': torch.tensor(item['anchor_out_ids']),
            'anchor_out_mask': torch.tensor(item['anchor_out_mask']),
            'anchor_in_ids': torch.tensor(item['anchor_in_ids']),
            'anchor_in_mask': torch.tensor(item['anchor_in_mask']),
            'domains_out_ids': torch.tensor(item['domains_out_ids']),
            'domains_in_ids': torch.tensor(item['domains_in_ids']),
            'numerics': torch.tensor(numerics, dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.float32),
        }

train_loader = DataLoader(HFTokenizedDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(HFTokenizedDataset(test_dataset),  batch_size=BATCH_SIZE, shuffle=False)

# === TRAIN FUNCTION ===
def train_phase(model, train_loader, val_loader, freeze_encoder=True, use_lora=False, epochs=5, lr=1e-3):
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q", "k", "v"],
                                 lora_dropout=0.05, bias="none")
        model.encoder = get_peft_model(model.encoder, lora_config)

    if not freeze_encoder:
        for param in model.parameters():
            param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCELoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**{k: batch[k] for k in batch if k != 'label'})
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # ✅ Log ogni LOG_EVERY step
            if step % LOG_EVERY == 0:
                wandb.log({
                    "train_step_loss": loss.item(),
                    "step": epoch * len(train_loader) + step
                })

        # === VALIDATION ===
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**{k: batch[k] for k in batch if k != 'label'})
                val_loss += criterion(outputs, batch['label']).item()
                preds = (outputs > 0.5).long()
                correct += (preds == batch['label'].long()).sum().item()
                total += batch['label'].size(0)

        acc = correct / total
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": acc
        })

        print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{short_model}.pt")

    return model

# === TRAIN PHASES ===
model = MultiModalWebClassifier().to(DEVICE)
model = train_phase(model, train_loader, val_loader, freeze_encoder=True, use_lora=False, epochs=EPOCHS_PHASE1, lr=1e-3)
model = train_phase(model, train_loader, val_loader, freeze_encoder=False, use_lora=True, epochs=EPOCHS_PHASE2, lr=1e-4)
torch.save(model.state_dict(), f"final_model_{short_model}.pt")
