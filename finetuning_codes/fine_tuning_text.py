import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, ClassLabel, load_from_disk
from sklearn.metrics import accuracy_score

import numpy as np
import wandb

# Initialize Weights & Biases (W&B) for experiment tracking
wandb.init(project="bert_lora_webpages", name="lora-36m-run", mode="online")  # Use "disabled" to turn off logging

# Detect and log the computation device
print("🔍 Checking device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# Load the dataset from a CSV file
print("📂 Loading dataset...")
dataset = load_dataset("csv", data_files="dataset_generation/dataset/balanced_dataset.csv", split="train")

# Automatically infer number of classes and cast 'label' to a categorical ClassLabel
num_classes = len(set(dataset["label"]))
dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))

# Stratified train/validation/test split (80% train, 10% val, 10% test)
print("✂️ Performing stratified data split...")
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
train_val = dataset["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Load BERT tokenizer
print("🔤 Loading tokenizer...")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenization function with basic validation and padding
def tokenize_function(examples):
    if "text" not in examples:
        return {}

    valid_texts = []
    valid_indices = []

    for i, text in enumerate(examples["text"]):
        if isinstance(text, str) and text.strip():  # Skip empty or invalid text
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        return {}

    # Tokenize only valid entries
    encodings = tokenizer(valid_texts, truncation=True, padding="max_length", max_length=512)

    # Map back tokenized output to original indices
    output = {k: [None] * len(examples["text"]) for k in encodings}
    for k in encodings:
        for i, idx in enumerate(valid_indices):
            output[k][idx] = encodings[k][i]

    return output

# Comment out the tokenization section and replace with loading from disk
# print("⚙️ Tokenizing datasets...")
# train_dataset = train_dataset.map(tokenize_function, batched=True)
# val_dataset = val_dataset.map(tokenize_function, batched=True)
# test_dataset = test_dataset.map(tokenize_function, batched=True)
# try:
#     train_dataset.save_to_disk("tokenized/train")
#     print("✅ Train dataset salvato correttamente.")
# except Exception as e:
#     print(f"⚠️ Errore nel salvataggio del train_dataset: {e}")
# try:
#     val_dataset.save_to_disk("tokenized/validation")
#     print("✅ Validation dataset salvato correttamente.")
# except Exception as e:
#     print(f"⚠️ Errore nel salvataggio del val_dataset: {e}")
# try:
#     test_dataset.save_to_disk("tokenized/test")
#     print("✅ Test dataset salvato correttamente.")
# except Exception as e:
#     print(f"⚠️ Errore nel salvataggio del test_dataset: {e}")

# ✅ Load pre-tokenized datasets from disk
print("💾 Caricamento dei dataset tokenizzati dal disco...")

train_dataset = load_from_disk("tokenized/train")
val_dataset = load_from_disk("tokenized/validation")
test_dataset = load_from_disk("tokenized/test")
base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],  # Target LoRA injection points
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()  # Print summary of which parameters are trainable (LoRA-specific)

# Define metric computation (Accuracy)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Define training arguments
print("🚀 Preparing training configuration...")
training_args = TrainingArguments(
    output_dir="checkpoints/lora_bert_webpages",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=30000,
    save_steps=30000,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to=["tensorboard", "wandb"],  # Enable visual logging
)

# Initialize Hugging Face Trainer with LoRA model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),  # Automatically pad to longest in batch
)

# Start the training process
print("🧠 Starting training...")
trainer.train()

# Evaluate on validation set
print("\n📊 Evaluation on VALIDATION set:")
val_results = trainer.evaluate(eval_dataset=val_dataset)
print(val_results)

# Evaluate on test set
print("\n📊 Evaluation on TEST set:")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)
