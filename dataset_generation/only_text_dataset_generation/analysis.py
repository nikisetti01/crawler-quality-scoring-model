import csv
from collections import Counter
import random

csv.field_size_limit(10**8)

# File paths
test_path = "dataset/test_set.csv"
link_label_path = "dataset/link_labels2.csv"
output_path = "dataset/train_set_filtered.csv"

# Step 1: Carica link test_set
print("📥 Caricamento link da test_set.csv daje...")
test_links = set()
with open(test_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        test_links.add(row["url"])
print(f"✅ Totale link in test_set.csv: {len(test_links):,}")

# Step 2: Filtra link non presenti in test
print("📥 Lettura di link_labels2.csv e filtraggio...")
label0 = []
label1 = []
with open(link_label_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["url"] in test_links:
            continue
        if row["label"] == "0":
            label0.append(row)
        elif row["label"] == "1":
            label1.append(row)

print(f"✅ Dopo il filtro, disponibili:")
print(f"   ➡️ label 0: {len(label0):,}")
print(f"   ➡️ label 1: {len(label1):,}")

# Step 3: Calcolo label1 necessari per avere 60/40 (usando tutti i label0)
target_total = int(len(label0)*2.5)
target_1 = target_total - len(label0)
label1_sampled = random.sample(label1, target_1)

# Step 4: Unione e shuffle
balanced_train = label0 + label1_sampled
random.shuffle(balanced_train)

print(f"✅ Train finale: {len(balanced_train):,} (label 0: {len(label0):,}, label 1: {len(label1_sampled):,})")

# Step 5: Scrivi su CSV
print(f"💾 Scrittura su {output_path}...")
with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=balanced_train[0].keys())
    writer.writeheader()
    writer.writerows(balanced_train)

print("🎉 Completato!")
