import os
import csv
import gzip
import json

CLUE_IDS = "/workspace/dataset_generation/dataset/crawled_docids_cw22b.txt"
LINK_CSV = "/workspace/dataset_generation/dataset/link_labels2.csv"
OUTLINK_DIR = "/data/CW22B/outlink/en/"
INLINK_DIR = "/data/CW22B/inlink/en/"
TEST_CSV = "/workspace/test_set.csv"
NEW_LINK_CSV = "/workspace/link_labels2_cleaned.csv"

# === 1. Carica etichette in dizionario url → label
print("📥 Caricamento etichette...")
url_to_label = {}
with open(LINK_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        url_to_label[row["url"]] = int(row["label"])
print(f"✅ Etichette caricate: {len(url_to_label)}")

# === 2. Carica ClueWeb22-ID target
with open(CLUE_IDS, 'r') as f:
    target_ids = set(line.strip() for line in f)
print(f"🎯 ID target da cercare: {len(target_ids)}")

# === 3. Setup per output e contatori
test_counts = {0: 0, 1: 0}
max_per_class = 4_000_000
found_urls = set()
test_writer = csv.writer(open(TEST_CSV, 'w', newline='', encoding='utf-8'))
test_writer.writerow(["url", "label"])

def match_id_to_test(path, field):
    global test_counts
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                docid = obj.get("ClueWeb22-ID")
                if docid not in target_ids:
                    continue
                url = obj.get("url")
                if url not in url_to_label:
                    continue
                label = url_to_label[url]
                if test_counts[label] >= max_per_class:
                    continue
                test_writer.writerow([url, label])
                test_counts[label] += 1
                found_urls.add(url)
                del url_to_label[url]  # rimuove dal dizionario principale
                if sum(test_counts.values()) % 100_000 == 0:
                    print(f"✏️ Scritti {sum(test_counts.values())} test (1: {test_counts[1]}, 0: {test_counts[0]})")
                if all(c >= max_per_class for c in test_counts.values()):
                    return True
            except:
                continue
    return False

# === 4. Scansione JSON
def scan_all_json(base_dir, field):
    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json.gz"):
                path = os.path.join(subdir, file)
                if match_id_to_test(path, field):
                    return True
    return False

print("🔎 Inizio scansione OUTLINK...")
if not scan_all_json(OUTLINK_DIR, "outlinks"):
    print("🔎 Inizio scansione INLINK...")
    scan_all_json(INLINK_DIR, "anchors")

print(f"🏁 Completato. Test salvato in {TEST_CSV}")
print(f"📊 1: {test_counts[1]} — 0: {test_counts[0]} — Totale: {sum(test_counts.values())}")

# === 5. Salva nuovo link_labels2.csv (senza quelli usati)
print("💾 Salvataggio nuovo link_labels2.csv senza test...")
with open(NEW_LINK_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label"])
    for url, label in url_to_label.items():
        writer.writerow([url, label])
print(f"✅ File salvato in {NEW_LINK_CSV} con {len(url_to_label)} righe.")
