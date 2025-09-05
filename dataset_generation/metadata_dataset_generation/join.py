#!/usr/bin/env python3
import argparse, csv, gzip

def open_any(path, mode="rt"):
    """
    Open a plain text file or a gzip-compressed file depending on the extension.
    Works for both text ("rt") and binary ("rb") modes.
    """
    return gzip.open(path, mode=mode) if path.endswith(".gz") else open(path, mode=mode, encoding=None if "b" in mode else "utf-8")

def dict_reader_sorted(path):
    """
    Open a CSV(.gz) and return (file_handle, DictReader).

    IMPORTANT: This function assumes the CSV is already sorted by the 'url' column.
    The join logic relies on monotonic advancement of 'url' across readers.
    """
    f = open_any(path, "rt")
    r = csv.DictReader(f)
    return f, r

def advance_to_url(curr_url, reader):
    """
    Advance a DictReader until we reach the first row whose 'url' is >= curr_url.
    Returns (url, row) or (None, None) if the reader is exhausted.

    NOTE: This function is not used in the main loop anymore, but kept for clarity/optionality.
    """
    for row in reader:
        u = row.get("url", "")
        if u >= curr_url:
            return u, row
    return None, None

def next_row(reader):
    """
    Safely fetch the next row from a DictReader.
    Returns (url, row) or (None, None) when exhausted.
    """
    try:
        row = next(reader)
        return row.get("url", ""), row
    except StopIteration:
        return None, None

def main():
    ap = argparse.ArgumentParser(description="Merge base + inlink + outlink (all PRE-SORTED by url)")
    ap.add_argument("--base", required=True, help="CSV(.gz) with at least url,text,label,… (MUST be sorted by url)")
    ap.add_argument("--inlink", required=True, help="CSV(.gz) inlink metadata sorted by url")
    ap.add_argument("--outlink", required=True, help="CSV(.gz) outlink metadata sorted by url")
    ap.add_argument("--output", required=True, help="Unified CSV output")
    ap.add_argument("--intersect", action="store_true", help="Write only rows whose url exists in ALL THREE sources")
    args = ap.parse_args()

    # Open readers (all must be sorted by 'url')
    f_base,  r_base  = dict_reader_sorted(args.base)
    f_in,    r_in    = dict_reader_sorted(args.inlink)
    f_out,   r_out   = dict_reader_sorted(args.outlink)

    # Prime the inlink/outlink streams (hold current row and its url for each)
    u_in,  row_in  = next_row(r_in)
    u_out, row_out = next_row(r_out)

    # Build the unified header:
    # - keep base fields as-is (with 'url' first)
    # - append inlink fields (excluding 'url')
    # - append outlink fields (excluding 'url')
    base_fields  = r_base.fieldnames or []
    in_fields    = [c for c in (r_in.fieldnames or []) if c != "url"]
    out_fields   = [c for c in (r_out.fieldnames or []) if c != "url"]

    union_fields = ["url"] + base_fields[1:] + in_fields + out_fields  # ensure 'url' is first
    with open_any(args.output, "wt") as fout:
        w = csv.DictWriter(fout, fieldnames=union_fields, extrasaction="ignore")
        w.writeheader()

        count = 0
        for base_row in r_base:
            u = base_row.get("url", "")
            if not u:
                continue

            # Align inlink stream to current base url:
            # As all are sorted, we can move forward in inlink until its url >= current base url.
            while u_in is not None and u_in < u:
                u_in, row_in = next_row(r_in)
            in_dict = row_in if (u_in == u) else {}

            # Align outlink stream the same way.
            while u_out is not None and u_out < u:
                u_out, row_out = next_row(r_out)
            out_dict = row_out if (u_out == u) else {}

            # If --intersect is set, keep only rows that appear in ALL three inputs.
            if args.intersect and (not in_dict or not out_dict):
                continue

            # Merge dictionaries:
            # - Start from base_row (base wins on conflicts)
            # - Overlay inlink fields, then outlink fields (excluding 'url')
            merged = dict(base_row)
            for k, v in (in_dict or {}).items():
                if k == "url":
                    continue
                merged[k] = v
            for k, v in (out_dict or {}).items():
                if k == "url":
                    continue
                merged[k] = v

            # Write the merged row following the unified header order
            w.writerow({c: merged.get(c, "") for c in union_fields})
            count += 1
            if count % 100000 == 0:
                print(f"...rows written: {count:,}")

    # Close all files
    f_base.close(); f_in.close(); f_out.close()
    print(f"✅ Done: {args.output} (rows: {count:,})")

if __name__ == "__main__":
    main()
