import sys
import json
import csv
import gzip
import os

# Ensure immediate flushing of stdout lines (useful when tailing logs)
sys.stdout.reconfigure(line_buffering=True)

# Paths mounted inside Docker
qrels_path = "/workspace/qrels/qrels_train.tsv"   # Qrels file (tab-separated: <query_id>\t<doc_hash>)
output_csv = "/data/link_labels2.csv"             # Output CSV with columns: url,label

# === 1) Load the set of relevant hashes from qrels ============================
print("🚀 Starting script with qrels:", qrels_path)
qrels_hashes = set()
with open(qrels_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Expect lines like: "<qid>\t<hash>"
        parts = line.strip().split('\t')
        if len(parts) == 2:
            _, h = parts
            qrels_hashes.add(h)

print(f"✅ Loaded {len(qrels_hashes)} hashes from qrels")

# === 2) Prepare CSV writer ====================================================
# We write a header once and then append one (url, label) row per page scanned.
csv_file = open(output_csv, 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(["url", "label"])

# === 3) Walk the outlink directory, read .json.gz files, and label pages =====
# Each JSON line is expected to be an object with:
#   { "url": <page_url>, "outlinks": [[<target_url>, <target_hash>, ...], ...] }
counter = 0
root_outlink_dir = '/data/CW22B/outlink/en/en00'  # Root directory containing outlink JSON.GZ files

for subdir, dirs, files in os.walk(root_outlink_dir):
    for file in files:
        # Process only gzipped JSON files
        if file.endswith('.json.gz'):
            file_path = os.path.join(subdir, file)
            # Log to stderr so progress messages don't mix with CSV/stdout
            print(f"📂 Processing file: {file_path}", file=sys.stderr, flush=True)

            try:
                # Stream the compressed file line by line
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            url = obj.get("url", "")
                            outlinks = obj.get("outlinks", [])

                            # Default label is 0 (no relevant outlink found)
                            label = 0
                            for out in outlinks:
                                # out is expected to be a list/tuple, where index 1 is the target hash
                                if len(out) >= 2:
                                    out_hash = out[1]
                                    if out_hash in qrels_hashes:
                                        label = 1
                                        # Log a positive match (stderr to keep CSV clean)
                                        print(f"✅ Relevant OUTLINK for {url} -> {out_hash}", file=sys.stderr, flush=True)
                                        break  # Stop early once we find a relevant outlink

                            # Write the result row to the CSV
                            writer.writerow([url, label])
                            counter += 1

                            # Periodic progress report
                            if counter % 50 == 0:
                                print(f"📊 Processed {counter} documents", file=sys.stderr, flush=True)

                        except json.JSONDecodeError:
                            # Skip any malformed JSON line and continue
                            continue

            except Exception as e:
                # Keep going even if a file fails; log the error
                print(f"❌ Error while processing {file_path}: {e}", file=sys.stderr, flush=True)

# Cleanup and final log
csv_file.close()
print("🏁 Processing complete")
print(f"✅ Wrote {counter} rows to {output_csv}")
