import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset, DatasetDict, ClassLabel, load_from_disk
from sklearn.metrics import accuracy_score
import numpy as np
import wandb

# Inizializzazione W&B
wandb.init(project="qualt5_freeze_finetune", name="qualt5-adapted", mode="online")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}, tiny")

# Caricamento dataset
dataset = DatasetDict({
    "train": load_dataset("csv", data_files="dataset_generation/dataset/dataset_train.csv", split="train"),
    "test": load_dataset("csv", data_files="dataset_generation/dataset/dataset_test.csv", split="train")
})

# Conversione label in classi binarie (stringhe)
num_classes = len(set(dataset["train"]["label"]))
dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))

# Stratificazione
train_val = dataset["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Filtro testi validi
def is_valid(example):
    return isinstance(example["text"], str) and example["text"].strip() != ""

train_dataset = train_dataset.filter(is_valid)
val_dataset = val_dataset.filter(is_valid)
test_dataset = test_dataset.filter(is_valid)

# Format prompt
def format_prompt(example):
    prompt = f"Document: {example['text']} Relevant:"
    target = "true" if int(example["label"]) == 1 else "false"
    return {"input_text": prompt, "target_text": target}

train_dataset = train_dataset.map(format_prompt)
val_dataset = val_dataset.map(format_prompt)
test_dataset = test_dataset.map(format_prompt)

# Tokenizer
model_name = "pyterrier-quality/qt5-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizzazione
def tokenize(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            examples["target_text"],
            max_length=5,
            padding="max_length",
            truncation=True
        )
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])



# Salva i dataset tokenizzati
train_dataset.save_to_disk("tokenized_qualt5/train")
val_dataset.save_to_disk("tokenized_qualt5/validation")
test_dataset.save_to_disk("tokenized_qualt5/test")

# Carica i dataset tokenizzati da disco
print("📂 Loading pre-tokenized datasets from disk...")
train_dataset = load_from_disk("tokenized_qualt5/train")
val_dataset = load_from_disk("tokenized_qualt5/validation")
test_dataset = load_from_disk("tokenized_qualt5/test")

# Metriche
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # per predizione generativa
        preds = logits[0]
    else:
        preds = logits
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bin_preds = [1 if p.strip().lower() == "true" else 0 for p in decoded_preds]
    bin_labels = [1 if l.strip().lower() == "true" else 0 for l in decoded_labels]

    return {"accuracy": accuracy_score(bin_labels, bin_preds)}

# Training args (stessi di DeBERTa, adattati a Seq2Seq)
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints/QualT5_finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=256,
    weight_decay=0.01,
    num_train_epochs=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=20000,
    save_steps=20000,
    save_total_limit=3,
    logging_dir="logs",
    logging_steps=500,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="wandb",
)

# Modello
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
)

# Addestramento
print("🚀 Fine-tuning QualT5...")
trainer.train()

# Salvataggio
trainer.save_model("checkpoints/QualT5_finetuned/final_model")
tokenizer.save_pretrained("checkpoints/QualT5_finetuned/final_model")

# Valutazione
print("\n📊 Evaluation on VALIDATION set:")
val_results = trainer.evaluate(eval_dataset=val_dataset)
print(val_results)

print("\n📊 Evaluation on TEST set:")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)
