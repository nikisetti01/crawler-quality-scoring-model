#!/usr/bin/env python3
import argparse, csv, gzip

def open_any(path, mode="rt"):
    return gzip.open(path, mode=mode) if path.endswith(".gz") else open(path, mode=mode, encoding=None if "b" in mode else "utf-8")

def dict_reader_sorted(path):
    f = open_any(path, "rt")
    r = csv.DictReader(f)
    return f, r

def advance_to_url(curr_url, reader):
    # avanza il reader finché url >= curr_url (ritorna (url,row) o (None,None) a fine file)
    for row in reader:
        u = row.get("url","")
        if u >= curr_url:
            return u, row
    return None, None

def next_row(reader):
    try:
        row = next(reader)
        return row.get("url",""), row
    except StopIteration:
        return None, None

def main():
    ap = argparse.ArgumentParser(description="Join base + inlink + outlink (tutti ORDINATI per url)")
    ap.add_argument("--base", required=True, help="CSV(.gz) con almeno url,text,label,…")
    ap.add_argument("--inlink", required=True, help="CSV(.gz) metadati inlink ordinato per url")
    ap.add_argument("--outlink", required=True, help="CSV(.gz) metadati outlink ordinato per url")
    ap.add_argument("--output", required=True, help="CSV output unificato")
    ap.add_argument("--intersect", action="store_true", help="Scrivi solo righe con url presente in TUTTI e 3")
    args = ap.parse_args()

    f_base,  r_base  = dict_reader_sorted(args.base)
    f_in,    r_in    = dict_reader_sorted(args.inlink)
    f_out,   r_out   = dict_reader_sorted(args.outlink)

    # leggi prime righe di in/out
    u_in,  row_in  = next_row(r_in)
    u_out, row_out = next_row(r_out)

    # header unione
    base_fields  = r_base.fieldnames or []
    in_fields    = [c for c in (r_in.fieldnames or []) if c!="url"]
    out_fields   = [c for c in (r_out.fieldnames or []) if c!="url"]

    union_fields = ["url"] + base_fields[1:] + in_fields + out_fields  # url prima
    with open_any(args.output, "wt") as fout:
        w = csv.DictWriter(fout, fieldnames=union_fields, extrasaction="ignore")
        w.writeheader()

        count = 0
        for base_row in r_base:
            u = base_row.get("url","")
            if not u:
                continue

            # allinea inlink
            while u_in is not None and u_in < u:
                u_in, row_in = next_row(r_in)
            in_dict = row_in if (u_in == u) else {}

            # allinea outlink
            while u_out is not None and u_out < u:
                u_out, row_out = next_row(r_out)
            out_dict = row_out if (u_out == u) else {}

            if args.intersect and (not in_dict or not out_dict):
                continue

            merged = dict(base_row)
            # in caso di conflitto, base vince; metadati sono extra
            for k, v in (in_dict or {}).items():
                if k == "url": continue
                merged[k] = v
            for k, v in (out_dict or {}).items():
                if k == "url": continue
                merged[k] = v

            w.writerow({c: merged.get(c, "") for c in union_fields})
            count += 1
            if count % 100000 == 0:
                print(f"…righe scritte: {count:,}")

    f_base.close(); f_in.close(); f_out.close()
    print(f"✅ Done: {args.output} (righe: {count:,})")

if __name__ == "__main__":
    main()
