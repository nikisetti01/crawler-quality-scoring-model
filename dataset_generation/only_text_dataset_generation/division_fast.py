import os
import csv
import gzip
import json

# Paths / configuration --------------------------------------------------------
CLUE_IDS     = "/workspace/dataset_generation/dataset/crawled_docids_cw22b.txt"
LINK_CSV     = "/workspace/dataset_generation/dataset/link_labels2.csv"
OUTLINK_DIR  = "/data/CW22B/outlink/en/"
INLINK_DIR   = "/data/CW22B/inlink/en/"
TEST_CSV     = "/workspace/test_set.csv"
NEW_LINK_CSV = "/workspace/link_labels2_cleaned.csv"

# 1) Load URL → label map from link_labels2.csv --------------------------------
# We keep a dictionary: url_to_label[url] = {0,1}
print("📥 Loading labels...")
url_to_label = {}
with open(LINK_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        url_to_label[row["url"]] = int(row["label"])
print(f"✅ Labels loaded: {len(url_to_label)}")

# 2) Load target ClueWeb22 IDs to match against --------------------------------
# Only documents whose "ClueWeb22-ID" is in this set will be considered for test.
with open(CLUE_IDS, 'r') as f:
    target_ids = set(line.strip() for line in f)
print(f"🎯 Target IDs to find: {len(target_ids)}")

# 3) Prepare output writers and counters ---------------------------------------
# We aim to build a balanced test set with up to `max_per_class` per class.
test_counts = {0: 0, 1: 0}
max_per_class = 4_000_000
found_urls = set()

test_writer = csv.writer(open(TEST_CSV, 'w', newline='', encoding='utf-8'))
test_writer.writerow(["url", "label"])

def match_id_to_test(path, field):
    """
    Stream a gzipped JSON file and add rows to TEST_CSV for pages that:
      - have 'ClueWeb22-ID' in target_ids
      - have 'url' present in url_to_label
      - do not exceed the per-class cap (max_per_class) for label 0 and 1

    Parameters:
      path  : full path to a .json.gz file
      field : unused here (kept for compatibility / future checks)

    Returns:
      True if both classes reached max_per_class (early stop), else False.
    """
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

                # Emit test row
                test_writer.writerow([url, label])
                test_counts[label] += 1
                found_urls.add(url)

                # Remove from the global label map so we can later write a cleaned link CSV
                del url_to_label[url]

                # Periodic progress
                if sum(test_counts.values()) % 100_000 == 0:
                    print(f"✏️ Written {sum(test_counts.values())} test rows "
                          f"(1: {test_counts[1]}, 0: {test_counts[0]})")

                # Early stop if both classes are full
                if all(c >= max_per_class for c in test_counts.values()):
                    return True
            except:
                # Skip malformed lines silently
                continue
    return False

# 4) Scan all JSON files under OUTLINK first, then INLINK if needed ------------
def scan_all_json(base_dir, field):
    """
    Recursively walk a base directory and process every .json.gz file with match_id_to_test.
    Returns True if the per-class caps are met (so the caller can stop).
    """
    for subdir, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json.gz"):
                path = os.path.join(subdir, file)
                if match_id_to_test(path, field):
                    return True
    return False

print("🔎 Scanning OUTLINK JSON...")
if not scan_all_json(OUTLINK_DIR, "outlinks"):
    print("🔎 Scanning INLINK JSON...")
    scan_all_json(INLINK_DIR, "anchors")

print(f"🏁 Completed. Test saved to {TEST_CSV}")
print(f"📊 1: {test_counts[1]} — 0: {test_counts[0]} — Total: {sum(test_counts.values())}")

# 5) Write a new link_labels2.csv excluding the URLs used for test -------------
# This preserves the remaining pool for training/other splits.
print("💾 Writing cleaned link_labels2.csv (excluding test URLs)...")
with open(NEW_LINK_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label"])
    for url, label in url_to_label.items():
        writer.writerow([url, label])

print(f"✅ Saved {NEW_LINK_CSV} with {len(url_to_label)} rows.")
