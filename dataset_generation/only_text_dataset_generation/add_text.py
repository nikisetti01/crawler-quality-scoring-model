import csv
import os
import json
import gzip
import sys

# --- CONFIG ---
txt_root = "/data/CW22B/txt/en/en00"
output_train = "dataset/dataset_train.csv"
output_test = "dataset/dataset_test.csv"
train_csv = "dataset/train_set_filtered.csv"
test_csv = "dataset/test_set.csv"

# --- Carica le etichette ---
def load_labels(path):
    print(f"📥 Caricamento {path}", file=sys.stderr)
    url2label = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['url'].strip()
            label = row['label']
            url2label[url] = label
    print(f"✅ Trovate {len(url2label)} etichette", file=sys.stderr)
    return url2label

train_labels = load_labels(train_csv)
test_labels = load_labels(test_csv)

# --- Inizializza gli output ---
def init_writer(path):
    f = open(path, "w", newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=["text", "url", "label"])
    writer.writeheader()
    return f, writer

f_train, writer_train = init_writer(output_train)
f_test, writer_test = init_writer(output_test)

written_train = 0
written_test = 0

# --- Estrazione dati dai JSON ---
for subdir, dirs, files in os.walk(txt_root):
    for file in files:
        if file.endswith(".json.gz"):
            file_path = os.path.join(subdir, file)
            print(f"🔍 Scanning: {file_path}", file=sys.stderr)

            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            url = data.get("URL", "").strip()
                            text = data.get("Clean-Text", "").strip()

                            if url in train_labels and text:
                                writer_train.writerow({
                                    "text": text,
                                    "url": url,
                                    "label": train_labels[url]
                                })
                                written_train += 1
                                if written_train % 1000 == 0:
                                    print(f"📝 Train: {written_train}", file=sys.stderr)

                            elif url in test_labels and text:
                                writer_test.writerow({
                                    "text": text,
                                    "url": url,
                                    "label": test_labels[url]
                                })
                                written_test += 1
                                if written_test % 1000 == 0:
                                    print(f"📝 Test: {written_test}", file=sys.stderr)

                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"⚠️ Errore file {file_path}: {e}", file=sys.stderr)

# --- Chiusura ---
f_train.close()
f_test.close()
print(f"🏁 Completato. Train: {written_train} righe → {output_train}", file=sys.stderr)
print(f"🏁 Completato. Test:  {written_test} righe → {output_test}", file=sys.stderr)
