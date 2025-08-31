import pandas as pd
import argparse
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from transformers import AutoTokenizer, BertTokenizerFast
from sklearn.preprocessing import StandardScaler
import os
from datasets import disable_caching
disable_caching()

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="Tokenizer model name (default: bert-base-uncased)")
args = parser.parse_args()

# === CONFIG ===
input_files = {
    "train": "/app/data/results/dataset_train_metadata.csv",
    "test": "/app/data/results/dataset_test_metadata.csv"
}
output_dir = "/app/data/results/"
chunk_size = 250_000
num_cols = [
    'inlink_num_inlinks', 'inlink_length_inlinks', 'inlink_inlinks_domains_count', 'inlink_inlink_slashes_count',
    'outlink_num_outlinks', 'outlink_length_outlinks', 'outlink_outlinks_domains_count', 'outlink_outlink_slashes_count'
]

# === GENERATE TOKENIZER ===
print(f"🧠 Loading tokenizer: {args.model}")
if "bert" in args.model:
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

# === Fit StandardScaler on TRAIN only
print("📏 Fitting StandardScaler on numeric features (train only)...")
sample = pd.read_csv(input_files["train"], nrows=500_000)
scaler = StandardScaler()
scaler.fit(sample[num_cols])

# === PROCESS TRAIN + TEST ===
for split_name, input_csv in input_files.items():
    print(f"\n🚀 Processing {split_name.upper()} file: {input_csv}")
    split_dir = os.path.join(output_dir, f"tokenized_bert_{split_name}")
    os.makedirs(split_dir, exist_ok=True)

    # Check existing chunks
    existing_chunks = {
        int(f.split('_')[-1])
        for f in os.listdir(split_dir)
        if f.startswith("tokenized_chunk_") and f.split('_')[-1].isdigit()
    }
    start_chunk = max(existing_chunks) + 1 if existing_chunks else 0
    print(f"🧩 Existing chunks detected: {sorted(existing_chunks)}")
    print(f"➡️  Will resume from chunk {start_chunk}")

    chunk_id = 0
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        if chunk_id < start_chunk:
            print(f"⏩ Skipping chunk {chunk_id} (already processed)")
            chunk_id += 1
            continue

        print(f"📦 Processing chunk {chunk_id} of {split_name} ({len(chunk)} rows)...")

        # Preprocessing
        for col in ['inlink_inlink_list_domains', 'outlink_outlink_list_domains']:
            chunk[col] = chunk[col].fillna('').apply(lambda x: x.split('|') if isinstance(x, str) else [])
        for col in ['inlink_inlink_anchors', 'outlink_outlink_anchors']:
            chunk[col] = chunk[col].fillna('').astype(str)

        chunk[num_cols] = scaler.transform(chunk[num_cols])
        chunk['numerics'] = chunk[num_cols].values.tolist()
        chunk["domains_out"] = chunk["outlink_outlink_list_domains"].apply(lambda lst: " ".join(lst))
        chunk["domains_in"] = chunk["inlink_inlink_list_domains"].apply(lambda lst: " ".join(lst))

        def tokenize_fn(batch):
            def to_str_list(x):
                return [str(e) if e is not None else "" for e in x]

            text_enc = tokenizer(to_str_list(batch['text']), truncation=True, padding="max_length", max_length=512)
            anchor_out_enc = tokenizer(to_str_list(batch['outlink_outlink_anchors']), truncation=True, padding="max_length", max_length=32)
            anchor_in_enc = tokenizer(to_str_list(batch['inlink_inlink_anchors']), truncation=True, padding="max_length", max_length=32)
            domain_out_enc = tokenizer(to_str_list(batch['domains_out']), truncation=True, padding="max_length", max_length=64)
            domain_in_enc = tokenizer(to_str_list(batch['domains_in']), truncation=True, padding="max_length", max_length=64)

            return {
                'input_ids': text_enc['input_ids'],
                'attention_mask': text_enc['attention_mask'],
                'anchor_out_ids': anchor_out_enc['input_ids'],
                'anchor_out_mask': anchor_out_enc['attention_mask'],
                'anchor_in_ids': anchor_in_enc['input_ids'],
                'anchor_in_mask': anchor_in_enc['attention_mask'],
                'domains_out_ids': domain_out_enc['input_ids'],
                'domains_out_mask': domain_out_enc['attention_mask'],
                'domains_in_ids': domain_in_enc['input_ids'],
                'domains_in_mask': domain_in_enc['attention_mask'],
            }

        # Subset + HuggingFace map
        chunk = chunk[['text', 'outlink_outlink_anchors', 'inlink_inlink_anchors',
                       'domains_out', 'domains_in', 'numerics', 'label']]
        ds = Dataset.from_pandas(chunk)
        ds = ds.map(tokenize_fn, batched=True, batch_size=64, num_proc=1)

        label_classes = ClassLabel(names=["neg", "pos"])
        features = Features({
            'text': Value("string"),
            'outlink_outlink_anchors': Value("string"),
            'inlink_inlink_anchors': Value("string"),
            'domains_out': Value("string"),
            'domains_in': Value("string"),
            'numerics': Sequence(Value("float32")),
            'label': label_classes,
            'input_ids': Sequence(Value("int32")),
            'attention_mask': Sequence(Value("int8")),
            'anchor_out_ids': Sequence(Value("int64")),
            'anchor_out_mask': Sequence(Value("int64")),
            'anchor_in_ids': Sequence(Value("int64")),
            'anchor_in_mask': Sequence(Value("int64")),
            'domains_out_ids': Sequence(Value("int64")),
            'domains_out_mask': Sequence(Value("int64")),
            'domains_in_ids': Sequence(Value("int64")),
            'domains_in_mask': Sequence(Value("int64")),
        })

        ds = ds.cast(features)
        out_path = os.path.join(split_dir, f"tokenized_chunk_{chunk_id}")
        ds.save_to_disk(out_path)
        print(f"💾 Saved ={split_name} chunk {chunk_id} to {out_path}")
        chunk_id += 1

print("\n✅ All TRAIN and TEST chunks processed and saved.")
