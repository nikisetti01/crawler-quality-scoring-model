#!/usr/bin/env python3
import argparse, csv, gzip, heapq, os, tempfile, uuid

def open_any(path, mode="rt"):
    return gzip.open(path, mode=mode) if path.endswith(".gz") else open(path, mode=mode, encoding=None if "b" in mode else "utf-8")

def chunk_sort(input_path, chunk_size=2_000_000):
    tmp_files = []
    with open_any(input_path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        url_idx = header.index("url")
        buf = []
        for i, row in enumerate(reader, 1):
            buf.append(row)
            if len(buf) >= chunk_size:
                buf.sort(key=lambda r: r[url_idx])
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
                w = csv.writer(tmp)
                w.writerow(header)
                w.writerows(buf)
                tmp.close()
                tmp_files.append(tmp.name)
                buf.clear()
        if buf:
            buf.sort(key=lambda r: r[url_idx])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
            w = csv.writer(tmp)
            w.writerow(header)
            w.writerows(buf)
            tmp.close()
            tmp_files.append(tmp.name)
    return tmp_files, header

def merge_sorted(files, header, out_path):
    def readers():
        for fp in files:
            f = open(fp, "r", encoding="utf-8", newline="")
            r = csv.reader(f)
            next(r)  # skip header
            yield f, r
    streams = list(readers())
    iters = [r for _, r in streams]

    # build initial heap (row, idx)
    heap = []
    for idx, r in enumerate(iters):
        try:
            row = next(r)
            heap.append((row[0], idx, row))  # assume url is col 0
        except StopIteration:
            pass
    heapq.heapify(heap)

    with open_any(out_path, "wt") as fout:
        w = csv.writer(fout)
        w.writerow(header)
        while heap:
            _, idx, row = heapq.heappop(heap)
            w.writerow(row)
            try:
                nxt = next(iters[idx])
                heapq.heappush(heap, (nxt[0], idx, nxt))
            except StopIteration:
                pass

    for f, _ in streams:
        f.close()
    for fp in files:
        os.unlink(fp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-size", type=int, default=2_000_000)
    args = ap.parse_args()

    tmps, header = chunk_sort(args.input, args.chunk_size)
    merge_sorted(tmps, header, args.output)
    print(f"✅ ordinato → {args.output}")

if __name__ == "__main__":
    main()
