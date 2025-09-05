import pandas as pd
import csv
import os

input_path = "dataset_complete.csv"
label1_output = "tmp_label1.csv"
label0_output = "tmp_label0.csv"
final_output = "dataset_bilanciato.csv"

label1_target = 20_000_000
label0_target = 16_000_000

label1_count = 0
label0_count = 0
chunk_number = 0
chunksize = 1_000_000  # Read 1M rows at a time

print("🚀 Inizio elaborazione...")

# Initialize temporary output files
with open(label1_output, 'w', newline='') as f1, open(label0_output, 'w', newline='') as f0:
    writer1, writer0 = None, None

    # Stream the big CSV in chunks to keep memory usage under control
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        chunk_number += 1
        print(f"\n📦 Lettura chunk #{chunk_number} (righe lette finora: {chunk_number * chunksize})")

        # Collect rows with label=1 up to the desired target, writing them incrementally
        if label1_count < label1_target:
            label1_chunk = chunk[chunk['label'] == 1]
            needed = label1_target - label1_count
            if not label1_chunk.empty:
                to_take = label1_chunk.head(needed)
                print(f"  ➕ Aggiunte {len(to_take)} righe con label=1 (totale: {label1_count + len(to_take)})")
                if writer1 is None:
                    # Initialize the CSV writer the first time we have data
                    writer1 = csv.DictWriter(f1, fieldnames=to_take.columns)
                    writer1.writeheader()
                # Write the selected rows (converted to list of dicts)
                writer1.writerows(to_take.to_dict(orient='records'))
                label1_count += len(to_take)

        # Collect rows with label=0 up to the desired target, writing them incrementally
        if label0_count < label0_target:
            label0_chunk = chunk[chunk['label'] == 0]
            needed = label0_target - label0_count
            if not label0_chunk.empty:
                to_take = label0_chunk.head(needed)
                print(f"  ➕ Aggiunte {len(to_take)} righe con label=0 (totale: {label0_count + len(to_take)})")
                if writer0 is None:
                    # Initialize the CSV writer the first time we have data
                    writer0 = csv.DictWriter(f0, fieldnames=to_take.columns)
                    writer0.writeheader()
                # Write the selected rows (converted to list of dicts)
                writer0.writerows(to_take.to_dict(orient='records'))
                label0_count += len(to_take)

        # Early stop if both class quotas are met
        if label1_count >= label1_target and label0_count >= label0_target:
            print("\n✅ Raggiunto il numero desiderato di righe per entrambe le classi.")
            break

print(f"\n📊 Totali raccolti: label=1 → {label1_count} / {label1_target}, label=0 → {label0_count} / {label0_target}")

# Merge the two temporary files into the final output, preserving headers correctly
print("\n🔄 Unione dei file temporanei...")

with open(final_output, 'w', newline='') as fout:
    with open(label1_output, 'r') as f1, open(label0_output, 'r') as f0:
        print("  ✍️ Scrittura label=1 nel file finale...")
        fout.writelines(f1.readlines())

        print("  ✍️ Scrittura label=0 nel file finale...")
        next(f0)  # skip header of the second file to avoid duplicating headers
        fout.writelines(f0.readlines())

print(f"\n🎉 Dataset bilanciato salvato come: {final_output}")

# Cleanup temporary files to free disk space
print("🧹 Pulizia file temporanei...")
os.remove(label1_output)
os.remove(label0_output)

print("✅ Completato.")
