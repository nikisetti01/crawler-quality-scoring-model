import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# 📌 Imposta device globale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"📦 Using device: {device}")

# 🎯 W&B
wandb.init(project="QualT5-eval", name="checkpoint-380k")

# 📥 Carica le label dal CSV
labels = pd.read_csv("dataset_generation/dataset/dataset_test.csv")["label"].tolist()

# 📦 Carica modello e tokenizer
model_path = "checkpoints/QualT5_finetuned/checkpoint-380000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
model.eval()

# 📄 Carica dataset tokenizzato
dataset = load_from_disk("tokenized_qualt5/test")
assert len(dataset) == len(labels), "Mismatch tra tokenized test e CSV"

# 🔢 Inference
batch_size = 64
scores = []
dataloader = DataLoader(dataset, batch_size=batch_size)

true_id = tokenizer.convert_tokens_to_ids("▁true")
false_id = tokenizer.convert_tokens_to_ids("▁false")
if true_id is None or false_id is None:
    raise ValueError("Token '▁true' o '▁false' non presenti nel tokenizer")

for batch in tqdm(dataloader, desc="Inferenza", unit="batch"):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    with torch.no_grad():
        decoder_start = torch.full((input_ids.size(0), 1), model.config.decoder_start_token_id, device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        decoder_input_ids=decoder_start, return_dict=True)
        logits = outputs.logits[:, 0, :]
        probs = torch.softmax(logits, dim=-1)
        denom = probs[:, true_id] + probs[:, false_id]
        batch_scores = (probs[:, true_id] / denom).tolist()
    scores.extend(batch_scores)

# 🧮 Metriche
true_labels = labels[:len(scores)]
pred_labels = [1 if s >= 0.5 else 0 for s in scores]
acc = accuracy_score(true_labels, pred_labels)
prec = precision_score(true_labels, pred_labels)
rec = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
cm = confusion_matrix(true_labels, pred_labels)

# 📤 wandb log
wandb.log({
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=true_labels,
        preds=pred_labels,
        class_names=["not relevant", "relevant"]
    )
})

# 📈 Stampa metriche
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print("Confusion matrix:\n", cm)
print("\n📈 Prime 20 score true:", [round(s, 4) for s in scores[:20]])

# 📊 Bucket analysis
df = pd.DataFrame({"score": scores, "label": true_labels})
df["bucket"] = (df["score"] // 0.05) * 0.05
bucket_stats = df.groupby("bucket").agg(
    count_0=("label", lambda x: int((x == 0).sum())),
    count_1=("label", lambda x: int((x == 1).sum())),
    avg_score=("score", "mean"),
    total=("label", "count")
).reset_index()

print("\n🏷 Bucket analysis:")
for _, r in bucket_stats.iterrows():
    print(f"{r['bucket']:.2f}–{r['bucket']+0.05:.2f}: "
          f"{r['count_0']} not relevant, {r['count_1']} relevant, avg score {r['avg_score']:.4f}")

# 📈 Grafico: distribuzione etichette per bucket
plt.figure(figsize=(10, 5))
plt.bar(bucket_stats["bucket"], bucket_stats["count_0"], width=0.045, alpha=0.7, label="Label 0")
plt.bar(bucket_stats["bucket"], bucket_stats["count_1"], width=0.045,
        bottom=bucket_stats["count_0"], alpha=0.7, label="Label 1")
plt.xlabel("Score bucket")
plt.ylabel("Count")
plt.title("Label distribution per score bucket")
plt.legend()
plt.tight_layout()
plt.savefig("label_distribution.png")
wandb.log({"bucket_distribution": wandb.Image(plt)})
plt.show()
plt.savefig("label_distribution.png")

# 📈 Score medio per bucket
plt.figure(figsize=(10, 4))
plt.plot(bucket_stats["bucket"], bucket_stats["avg_score"], marker="o")
plt.xlabel("Score bucket")
plt.ylabel("Avg score")
plt.title("Average score per bucket")
plt.grid(True)
plt.tight_layout()
plt.savefig("avg_score_per_bucket.png")
wandb.log({"avg_score_per_bucket": wandb.Image(plt)})
plt.show()
plt.savefig("avg_score_per_bucket.png")
# 📈 ROC Curve
fpr, tpr, _ = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)
print(f"\n📈 AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
wandb.log({"roc_curve": wandb.Image(plt)})
plt.show()
plt.savefig("roc_curve.png")
# 💾 Salvataggio CSV con score, label, predizione e bucket
df["pred"] = pred_labels
output_path = "results_QualT5_eval.csv"
df.to_csv(output_path, index=False)
print(f"\n📁 Risultati salvati in: {output_path}")
