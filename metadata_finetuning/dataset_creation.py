import pandas as pd
import argparse
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
import os

from datasets import disable_caching
disable_caching()

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="bert-base-uncased",
                    help="Tokenizer model name (e.g. bert-base-uncased or pyterrier-quality/qt5-small)")
args = parser.parse_args()

# === GENERATE OUTPUT DIR BASED ON MODEL ===
def model_to_dirname(model_name):
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    return f"tokenized_chunks_{short_name.replace('-', '_')}"

output_dir = model_to_dirname(args.model)
os.makedirs(output_dir, exist_ok=True)

# === CONFIG ===
input_csv = "third_dataset_cleaned.csv"
chunk_size = 10000
chunk_id = 0

print(f"🧠 Loading tokenizer: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)

# === Fit StandardScaler
print("📏 Fitting StandardScaler on numeric features...")
sample = pd.read_csv(input_csv, nrows=100_000)
num_cols = ['num_outlinks', 'length_outlinks', 'outlink_domains_count', 'outlink_slashes_count']
scaler = StandardScaler()
scaler.fit(sample[num_cols])

# === Start processing chunks
print("🚀 Starting tokenization into:", output_dir)

for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
    print(f"\n📦 Processing chunk {chunk_id}...")

    for col in ['outlink_list_domains', 'inlink_list_domains']:
        chunk[col] = chunk[col].fillna('').apply(lambda x: x.split('|') if isinstance(x, str) else [])
    for col in ['outlink_anchors', 'inlink_anchors']:
        chunk[col] = chunk[col].fillna('').astype(str)

    chunk[num_cols] = scaler.transform(chunk[num_cols])
    chunk['numerics'] = chunk[num_cols].values.tolist()
    chunk["domains_out"] = chunk["outlink_list_domains"].apply(lambda lst: " ".join(lst))
    chunk["domains_in"] = chunk["inlink_list_domains"].apply(lambda lst: " ".join(lst))

    def tokenize_fn(batch):
        text_enc = tokenizer(batch['text'], truncation=True, padding="max_length", max_length=256)
        anchor_out_enc = tokenizer(batch['outlink_anchors'], truncation=True, padding="max_length", max_length=32)
        anchor_in_enc = tokenizer(batch['inlink_anchors'], truncation=True, padding="max_length", max_length=32)
        domain_out_enc = tokenizer(batch['domains_out'], truncation=True, padding="max_length", max_length=64)
        domain_in_enc = tokenizer(batch['domains_in'], truncation=True, padding="max_length", max_length=64)

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

    chunk = chunk[['text', 'outlink_anchors', 'inlink_anchors',
                   'domains_out', 'domains_in', 'numerics', 'label']]

    ds = Dataset.from_pandas(chunk)
    ds = ds.map(tokenize_fn, batched=True, batch_size=32, num_proc=1)

    label_classes = ClassLabel(names=["neg", "pos"])
    features = Features({
        'text': Value("string"),
        'outlink_anchors': Value("string"),
        'inlink_anchors': Value("string"),
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

    ds = Dataset.from_dict(ds[:], features=features)

    # Split e salvataggio
    split = ds.train_test_split(test_size=0.1, seed=42, stratify_by_column="label")

    train_path = os.path.join(output_dir, f"tokenized_chunk_{chunk_id}_train")
    test_path = os.path.join(output_dir, f"tokenized_chunk_{chunk_id}_test")
    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(test_path)

    print(f"💾 Saved train to {train_path}")
    print(f"💾 Saved test  to {test_path}")

    chunk_id += 1

print("✅ All chunks processed and split into train/test correctly.")
