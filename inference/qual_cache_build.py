#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from pyterrier_quality import QualCache  # re-esporta la classe QualCache
# Se serve: from qual_cache import QualCache

def iter_rows(csv_path, docno_col="clueweb_id", score_col="score", pred_col=None, chunksize=1_000_000):
    """
    Genera dict {'docno': <str>, 'quality': <float>} a stream, senza caricare tutto in RAM.
    Se pred_col è indicato, puoi filtrare/trasformare i punteggi.
    """
    counter = 0
    for chunk_id, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize), start=1):
        # Normalizza tipi
        chunk[docno_col] = chunk[docno_col].astype(str)

        # (Opzionale) se vuoi usare solo i positivi:
        if pred_col is not None:
            # esempio: tieni tutto ma azzera i negativi (oppure filtra via i pred=0)
            # chunk = chunk[chunk[pred_col] == 1]  # per FILTRARE
            chunk.loc[chunk[pred_col] == 0, score_col] = 0.0  # per TENERE e mettere 0

        # (Opzionale) gestisci duplicati sullo stesso docno
        chunk = chunk.groupby(docno_col, as_index=False)[score_col].max()

        for row in chunk.itertuples(index=False):
            counter += 1
            if counter % 1_000_000 == 0:
                print(f"[INFO] Convertiti {counter:,} score finora...")
            yield {"docno": getattr(row, docno_col), "quality": float(getattr(row, score_col))}

    print(f"[INFO] Conversione terminata, totale score processati = {counter:,}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path al CSV con colonne clueweb_id,url,score,pred")
    ap.add_argument("--outdir", required=True, help="Cartella di output del QualCache (verrà creata e deve NON esistere)")
    ap.add_argument("--docno_col", default="clueweb_id")
    ap.add_argument("--score_col", default="score")
    ap.add_argument("--pred_col", default="pred") 
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    cache = QualCache(args.outdir)

    stream = iter_rows(
        args.csv,
        docno_col=args.docno_col,
        score_col=args.score_col,
        pred_col=(args.pred_col if args.pred_col.lower() != "none" else None),
        chunksize=args.chunksize,
    )

    cache.index(stream)  # scrive quality.f4 + docno.npids

    print(f"✅ Creato QualCache in {args.outdir}")

if __name__ == "__main__":
    main()
