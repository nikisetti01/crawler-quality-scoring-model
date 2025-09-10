#!/usr/bin/env python3
import subprocess

# 🔧Configuration 
JSON_INLINK_DIR  = "/app/data/inlink_json"
JSON_OUTLINK_DIR = "/app/data/outlink_json"
BASE_CSV         = "/app/data/results/dataset_test.csv"

TMP_INLINK = "inlink_raw.csv"
TMP_OUTLINK = "outlink_raw.csv"
SORTED_INLINK = "inlink_sorted.csv"
SORTED_OUTLINK = "outlink_sorted.csv"
JOINED = "joined.csv"

# 1) Metadata extraction
subprocess.run([
    "python3", "extract_metadata.py",
    "--mode", "inlink",
    "--input-dir", JSON_INLINK_DIR,
    "--output", TMP_INLINK
], check=True)

subprocess.run([
    "python3", "extract_metadata.py",
    "--mode", "outlink",
    "--input-dir", JSON_OUTLINK_DIR,
    "--output", TMP_OUTLINK
], check=True)

# 2) Ordinamento per url
subprocess.run([
    "python3", "sort.py",
    "--input", TMP_INLINK,
    "--output", SORTED_INLINK
], check=True)

subprocess.run([
    "python3", "sort.py",
    "--input", TMP_OUTLINK,
    "--output", SORTED_OUTLINK
], check=True)

# 3) Final Join
subprocess.run([
    "python3", "join.py",
    "--base", BASE_CSV,
    "--inlink", SORTED_INLINK,
    "--outlink", SORTED_OUTLINK,
    "--output", JOINED,
    "--intersect"   
], check=True)

print("✅ Pipeline completed, output:", JOINED)
