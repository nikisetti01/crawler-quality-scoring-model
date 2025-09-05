import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, DebertaForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import load_dataset, ClassLabel, load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score
import numpy as np
import wandb
from sklearn.utils.class_weight import compute_class_weight

# Initialize Weights & Biases logging
wandb.init(project="deberta_freeze_finetune", name="deberta-freeze-last2", mode="online")

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# Load dataset from CSV files
dataset = DatasetDict({
    "train": load_dataset("csv", data_files="dataset_generation/dataset/dataset_train.csv", split="train"),
    "test": load_dataset("csv", data_files="dataset_generation/dataset/dataset_test.csv", split="train")
})

# Ensure the "label" column is a proper ClassLabel type
num_classes = len(set(dataset["train"]["label"]))
dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))

# Split train into train/validation with stratification on label
train_val = dataset["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Filter out invalid rows (non-string or empty text)
def is_valid(example):
    return isinstance(example["text"], str) and example["text"].strip() != ""

train_dataset = train_dataset.filter(is_valid)
val_dataset = val_dataset.filter(is_valid)
test_dataset = test_dataset.filter(is_valid)

# Load tokenizer for DeBERTa
print("🔤 Loading DeBERTa tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Tokenization function: truncate/pad to 512 tokens
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print("🔤 Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Save tokenized datasets to disk (so they can be reused without recomputing)
train_dataset.save_to_disk("tokenized_deberta/train")
val_dataset.save_to_disk("tokenized_deberta/validation")
test_dataset.save_to_disk("tokenized_deberta/test")

# Reload tokenized datasets from disk
print("📂 Loading pre-tokenized datasets from disk...")
train_dataset = load_from_disk("tokenized_deberta/train")
val_dataset = load_from_disk("tokenized_deberta/validation")
test_dataset = load_from_disk("tokenized_deberta/test")

# Compute class weights to handle imbalance
train_labels = train_dataset["label"]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights_tensor.tolist()}")

# Load pre-trained DeBERTa model with a classification head
model = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=num_classes)

# Freeze all layers except the last two encoder layers and the classifier head
for name, param in model.named_parameters():
    if any(layer in name for layer in ["encoder.layer.10", "encoder.layer.11", "classifier"]):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Show which parameters remain trainable
trainable = [name for name, param in model.named_parameters() if param.requires_grad]
print(f"🧊 Trainable parameters (frozen strategy):\n{trainable}")

# Move model to device
model.to(device)

# Custom Trainer that applies class weights to CrossEntropyLoss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Compute metrics: here we only track accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Training configuration
training_args = TrainingArguments(
    output_dir="checkpoints/Deberta_freeze_last2",  # where checkpoints will be saved
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=20000,
    save_steps=20000,
    fp16=True,  # mixed precision
    save_total_limit=3,
    logging_dir="logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to=["wandb"],  # log metrics to Weights & Biases
)

# Initialize Trainer with our custom WeightedTrainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Training loop
print("🧠 Starting training with partial freezing (last 2 layers only)...")
trainer.train()

# Evaluate on validation set
print("\n📊 Evaluation on VALIDATION set:")
val_results = trainer.evaluate(eval_dataset=val_dataset)
print(val_results)

# Evaluate on test set
print("\n📊 Evaluation on TEST set:")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)
