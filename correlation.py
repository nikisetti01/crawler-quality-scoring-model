from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time

MINE_CSV = "/app/data/results/inference_outputs.csv"
OUT_DIR = "corr_outputs_offline"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    t0 = time.time()
    print("[1] Carico il CSV del mio modello…")
    df_mine = pd.read_csv(MINE_CSV, usecols=["clueweb_id","score"], dtype={"clueweb_id":"string"})
    df_mine["score"] = pd.to_numeric(df_mine["score"], errors="coerce").astype(np.float32)
    df_mine = df_mine.dropna()
    print(f"    nostro modello: {len(df_mine):,} righe")

    print("[2] Carico QualT5 da HuggingFace (offline)…")
    ds_qualt5 = load_dataset("pyterrier-quality/qt5-base.cw22b-en.cache", split="train")
    df_qt5 = ds_qualt5.to_pandas()[["docno","score"]]
    df_qt5 = df_qt5.rename(columns={"docno":"clueweb_id","score":"score_qualt5"})
    df_qt5["score_qualt5"] = df_qt5["score_qualt5"].astype(np.float32)
    print(f"    QualT5: {len(df_qt5):,} righe")

    print("[3] Join su clueweb_id…")
    df_join = pd.merge(df_mine, df_qt5, on="clueweb_id", how="inner")
    print(f"    match trovati: {len(df_join):,}")

    if len(df_join) == 0:
        print("[WARN] Nessun match, esco.")
        return

    # Pearson, Spearman, Kendall
    pearson = df_join["score"].corr(df_join["score_qualt5"], method="pearson")
    spearman = df_join["score"].corr(df_join["score_qualt5"], method="spearman")
    kendall = df_join["score"].corr(df_join["score_qualt5"], method="kendall")

    print(f"[4] Pearson:  {pearson:.6f}")
    print(f"[4] Spearman: {spearman:.6f}")
    print(f"[4] Kendall:  {kendall:.6f}")

    # Scatter plot
    plt.figure(figsize=(8,6))
    plt.scatter(df_join["score"], df_join["score_qualt5"], s=1, alpha=0.25)
    plt.title("Nostro modello vs QualT5-base")
    plt.xlabel("Score nostro modello")
    plt.ylabel("Score QualT5-base")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"scatter.png"), dpi=150)
    plt.close()

    # Heatmap correlazione
    corr_mat = df_join[["score","score_qualt5"]].corr(method="pearson")
    plt.figure(figsize=(4,3))
    plt.imshow(corr_mat, aspect="auto")
    plt.colorbar()
    plt.xticks([0,1], ["score_mine","score_qualt5"])
    plt.yticks([0,1], ["score_mine","score_qualt5"])
    for i in range(2):
        for j in range(2):
            plt.text(j,i,f"{corr_mat.values[i,j]:.3f}", ha="center", va="center")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"heatmap.png"), dpi=150)
    plt.close()

    # Salva CSV con metriche
    pd.DataFrame([{
        "pearson":pearson,
        "spearman":spearman,
        "kendall":kendall,
        "n_pairs":len(df_join)
    }]).to_csv(os.path.join(OUT_DIR,"correlations.csv"), index=False)

    print(f"[DONE] Tempo totale: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
