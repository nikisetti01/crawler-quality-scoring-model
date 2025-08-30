#!/usr/bin/env python3
import argparse, json, os, gzip, csv, sys, re
from urllib.parse import urlparse
from typing import Iterable, Dict, List, Tuple

def open_any(path: str, mode="rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=None if "b" in mode else "utf-8")
    return open(path, mode=mode, encoding=None if "b" in mode else "utf-8")

def iter_json_records(path: str) -> Iterable[Dict]:
    """
    Supporta:
      - file JSON singolo (dizionario) con campi {url, inlinks/outlinks:[...]}
      - JSON lines (una riga = un record)
      - file con lista JSON di record
    """
    with open_any(path, "rt") as f:
        head = f.read(2048)
        f.seek(0)
        head_strip = head.lstrip()
        if head_strip.startswith("{") or head_strip.startswith("["):
            # prova JSON completo
            try:
                obj = json.load(f)
                if isinstance(obj, dict):
                    yield obj
                elif isinstance(obj, list):
                    for rec in obj:
                        if isinstance(rec, dict):
                            yield rec
                return
            except Exception:
                f.seek(0)
        # fallback: JSON lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec
            except Exception:
                continue

def clean_anchor(text: str) -> str:
    if not text:
        return ""
    # semplice pulizia (quella che usavamo): lowercase, spazi normalizzati, tolta punteggiatura “pesante”
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    # rimuovi simboli non alfanumerici tranne pochi separatori
    text = re.sub(r"[^a-z0-9àèéìòóùäöüßçñ\-_\.\s/]", "", text)
    return text.strip()

def safe_domain(u: str) -> str:
    try:
        return (urlparse(u).netloc or "").lower()
    except Exception:
        return ""

def count_slashes(u: str) -> int:
    try:
        return (urlparse(u).path or "").count("/")
    except Exception:
        return 0

def extract_links(rec: Dict, mode: str) -> Tuple[str, List[Dict]]:
    """
    mode = 'inlink'  -> lista di link che PUNTANO alla pagina (sorgenti)
    mode = 'outlink' -> lista di link USCENTI dalla pagina (destinazioni)
    Strutture possibili nel record:
      { "url": ..., "inlinks": [ {"url":..., "anchor":...}, ... ] }
      { "url": ..., "outlinks": [ {"url":..., "anchor":...}, ... ] }
      { "url": ..., "links": [ {...} ] }  # fallback
    """
    page_url = rec.get("url") or rec.get("page_url") or rec.get("doc_url") or ""
    key = "inlinks" if mode == "inlink" else "outlinks"
    links = rec.get(key)
    if links is None:
        links = rec.get("links", [])
    # normalizza elementi: vogliamo almeno {url, anchor}
    norm = []
    for x in links or []:
        if not isinstance(x, dict):
            continue
        u = x.get("url") or x.get("target") or x.get("source") or x.get("href") or ""
        a = x.get("anchor") or x.get("text") or ""
        norm.append({"url": u, "anchor": a})
    return page_url, norm

def aggregate(page_url: str, links: List[Dict], mode: str) -> Dict[str, str]:
    # domini
    domains = [safe_domain(x["url"]) for x in links if x.get("url")]
    domain_set = sorted({d for d in domains if d})
    # anchor
    anchors = [clean_anchor(x.get("anchor","")) for x in links if x.get("anchor")]
    anchors = [a for a in anchors if a]
    anchors_concat = " ".join(anchors)
    # misure
    num_links = len(links)
    length_anchors = sum(len(a) for a in anchors)
    slashes_avg = (sum(count_slashes(x["url"]) for x in links if x.get("url")) / num_links) if num_links else 0.0

    # colonne condivise che già usavamo
    # NB: per i tokenizer in training: domains_out/domains_in + outlink_outlink_anchors / inlink_inlink_anchors
    row = {"url": page_url}
    if mode == "outlink":
        row.update({
            "domains_out": " ".join(domain_set),
            "outlink_outlink_anchors": anchors_concat[:8000],  # taglio prudenziale
            "num_outlinks": str(num_links),
            "outlinks_domains_count": str(len(domain_set)),
            "outlink_list_domains": ";".join(domain_set[:200]),
            "outlink_slashes_count": f"{slashes_avg:.3f}",
            "length_outlinks": str(length_anchors),
        })
    else:
        row.update({
            "domains_in": " ".join(domain_set),
            "inlink_inlink_anchors": anchors_concat[:8000],
            "num_inlinks": str(num_links),
            "inlinks_domains_count": str(len(domain_set)),
            "inlink_list_domains": ";".join(domain_set[:200]),
            "inlink_slashes_count": f"{slashes_avg:.3f}",
            "length_inlinks": str(length_anchors),
        })
    return row

def write_header_once(writer: csv.DictWriter, wrote_header: List[bool]):
    if not wrote_header[0]:
        writer.writeheader()
        wrote_header[0] = True

def process_dir(input_dir: str, mode: str, output_path: str):
    # header unificato
    base_cols = ["url"]
    if mode == "outlink":
        cols = base_cols + [
            "domains_out","outlink_outlink_anchors","num_outlinks","outlinks_domains_count",
            "outlink_list_domains","outlink_slashes_count","length_outlinks"
        ]
    else:
        cols = base_cols + [
            "domains_in","inlink_inlink_anchors","num_inlinks","inlinks_domains_count",
            "inlink_list_domains","inlink_slashes_count","length_inlinks"
        ]

    # apertura output (auto .gz se richiesto)
    gz = output_path.endswith(".gz")
    out_f = open_any(output_path, "wt")
    writer = csv.DictWriter(out_f, fieldnames=cols, extrasaction="ignore")
    wrote_header = [False]

    files = []
    for root, _, fnames in os.walk(input_dir):
        for fn in fnames:
            if fn.endswith(".json") or fn.endswith(".json.gz"):
                files.append(os.path.join(root, fn))
    files.sort()

    total = 0
    for i, path in enumerate(files, 1):
        try:
            for rec in iter_json_records(path):
                page_url, links = extract_links(rec, mode=mode)
                if not page_url:
                    continue
                row = aggregate(page_url, links, mode=mode)
                write_header_once(writer, wrote_header)
                writer.writerow(row)
                total += 1
        except Exception as e:
            print(f"[WARN] errore in {path}: {e}", file=sys.stderr)
        if i % 500 == 0:
            print(f"…file letti: {i:,} | righe scritte: {total:,}")
    out_f.close()
    print(f"✅ Finito: {output_path} (righe: {total:,})")

def main():
    ap = argparse.ArgumentParser(description="Estrai metadati da JSON inlink/outlink → CSV")
    ap.add_argument("--mode", choices=["inlink","outlink"], required=True)
    ap.add_argument("--input-dir", required=True, help="Directory con JSON (.json o .json.gz)")
    ap.add_argument("--output", required=True, help="CSV di output (usa .gz per compressione)")
    args = ap.parse_args()
    process_dir(args.input_dir, args.mode, args.output)

if __name__ == "__main__":
    main()
