import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, ClassLabel, load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import wandb

# Initialize W&B
wandb.init(project="bert_lora_finetune", name="bert-lora-classweights", mode="online")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Using device: {device}")

# Load train/test CSV and do stratified train/val split
dataset = DatasetDict({
    "train": load_dataset("csv", data_files="dataset_generation/dataset/dataset_train.csv", split="train"),
    "test": load_dataset("csv", data_files="dataset_generation/dataset/dataset_test.csv", split="train")
})

# Cast label to ClassLabel
num_classes = len(set(dataset["train"]["label"]))
dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))

# Split train into train/val
train_val = dataset["train"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
train_dataset = train_val["train"]
val_dataset = train_val["test"]
test_dataset = dataset["test"]

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

# Tokenize datasets
print("🔤 Tokenizing datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Compute class weights
labels = train_dataset["label"]
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Load BERT and apply LoRA
base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],
)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
model.to(device)

# Loss function with class weights
def compute_loss_with_weights(model, inputs):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    loss = loss_fct(logits, labels)
    return loss

# Metric
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Training args
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
    report_to=["tensorboard", "wandb"],
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_loss=compute_loss_with_weights,
)

# Train
print("🧠 Starting training (BERT + LoRA + class weights)...")
trainer.train()

# Evaluate
print("\n📊 Evaluation on VALIDATION set:")
val_results = trainer.evaluate(eval_dataset=val_dataset)
print(val_results)

print("\n📊 Evaluation on TEST set:")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(test_results)
