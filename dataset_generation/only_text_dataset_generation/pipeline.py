#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

# ==== CONFIG: metti qui i percorsi degli script se non sono nella stessa cartella ====
SCRIPTS_DIR = Path(".")  # cartella in cui si trovano gli script .py
PROCESS_LINKS   = SCRIPTS_DIR / "process_links.py"    # step 1
DIVISION_FAST   = SCRIPTS_DIR / "division_fast.py"    # step 2
ADD_TEXT        = SCRIPTS_DIR / "add_text.py"         # step 3
BALANCER        = SCRIPTS_DIR / "balancer.py"         # step 4

# (Opz.) log file separati per ogni step
LOG_PROCESS_LINKS = "logs/01_process_links.log"
LOG_DIVISION_FAST = "logs/02_division_fast.log"
LOG_ADD_TEXT      = "logs/03_add_text.log"
LOG_BALANCER      = "logs/04_balancer.log"

def run_step(cmd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n▶️  RUN: {' '.join(map(str, cmd))}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        # Stampa anche a schermo e salva su file
        proc = subprocess.run(
            ["python3", *map(str, cmd)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False
        )
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"❌ Step FAILED in {dt:.1f}s → vedi log: {log_path}")
        sys.exit(proc.returncode)
    print(f"✅ Step OK in {dt:.1f}s → log: {log_path}")

def main():
    print("🚀 Starting pipeline: process_links → division_fast → add_text → balancer")

    # 1) process_links.py  (genera/aggiorna link_labels2.csv, ecc.)
    #    NB: il tuo script usa i path hardcoded al suo interno.
    run_step([PROCESS_LINKS], LOG_PROCESS_LINKS)

    # 2) division_fast.py  (seleziona test_set bilanciato da crawled_docids, produce test_set.csv e link_labels2_cleaned.csv)
    run_step([DIVISION_FAST], LOG_DIVISION_FAST)

    # 3) add_text.py       (costruisce dataset_train.csv e dataset_test.csv unendo testi e label)
    run_step([ADD_TEXT], LOG_ADD_TEXT)

    # 4) balancer.py       (crea dataset_bilanciato.csv a partire dal dataset completo o da quello che hai scelto)
    run_step([BALANCER], LOG_BALANCER)

    print("\n🎉 Pipeline COMPLETED")

if __name__ == "__main__":
    main()
