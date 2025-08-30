import sys
import json
import csv
import gzip
import os

sys.stdout.reconfigure(line_buffering=True)

# Percorsi delle cartelle montate dentro Docker
qrels_path = "/workspace/qrels/qrels_train.tsv"  # Cartella dei qrels
output_csv = "/data/link_labels2.csv"  # File CSV di output

# === 1. Carica set di hash nei qrels
print("🚀 Avvio script con qrels:", qrels_path)
qrels_hashes = set()
with open(qrels_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            _, h = parts
            qrels_hashes.add(h)

print(f"✅ Caricati {len(qrels_hashes)} hash da qrels")

# === 2. Setup CSV (scrittura)
csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(["url", "label"])

# === 3. Scansiona le cartelle degli outlink e processa i file .json.gz
counter = 0
root_outlink_dir = '/data/CW22B/outlink/en/en00'  # Cartella radice degli outlink

for subdir, dirs, files in os.walk(root_outlink_dir):
    for file in files:
        # Verifica se il file è un JSON .gz
        if file.endswith('.json.gz'):
            file_path = os.path.join(subdir, file)
            print(f"📂 Processando il file: {file_path}", file=sys.stderr, flush=True)

            try:
                # Apri e leggi il file JSON compresso
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            url = obj.get("url", "")
                            outlinks = obj.get("outlinks", [])

                            # Imposta il label iniziale come 0
                            label = 0
                            for out in outlinks:
                                if len(out) >= 2:
                                    out_hash = out[1]
                                    if out_hash in qrels_hashes:
                                        label = 1
                                        print(f"✅ OUTLINK rilevante per {url} -> {out_hash}", file=sys.stderr, flush=True)
                                        break  # Appena trovi un outlink valido, basta

                            # Scrivi la riga nel CSV
                            writer.writerow([url, label])
                            counter += 1

                            if counter % 50 == 0:
                                print(f"📊 Processati {counter} documenti", file=sys.stderr, flush=True)

                        except json.JSONDecodeError:
                            continue  # Ignora eventuali righe non JSON valide

            except Exception as e:
                print(f"❌ Errore durante l'elaborazione del file {file_path}: {e}", file=sys.stderr, flush=True)

csv_file.close()
print("🏁 Fine elaborazione")
print(f"✅ Scritti {counter} righe in {output_csv}")
