#!/usr/bin/env python3
import argparse, csv, gzip, heapq, os, tempfile, uuid

def open_any(path, mode="rt"):
    """
    Open a plain text file or a gzip-compressed file based on its extension.
    Supports both text ("rt") and binary ("rb") modes.
    """
    return gzip.open(path, mode=mode) if path.endswith(".gz") else open(path, mode=mode, encoding=None if "b" in mode else "utf-8")

def chunk_sort(input_path, chunk_size=2_000_000):
    """
    External sort (by 'url' column) using a chunking strategy.

    Steps:
      1) Read the input CSV in memory-limited chunks of `chunk_size` rows.
      2) For each chunk:
         - Sort rows by the 'url' column
         - Write the sorted chunk to a temporary CSV file
      3) Return the list of temporary file paths and the CSV header.

    Notes:
      - The function expects a header row and a column named 'url'.
      - `chunk_size` controls memory usage: smaller values use less RAM but create more temp files.
    """
    tmp_files = []
    with open_any(input_path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        url_idx = header.index("url")  # raises if 'url' is missing
        buf = []
        for i, row in enumerate(reader, 1):
            buf.append(row)
            if len(buf) >= chunk_size:
                # Sort the current chunk and spill it to disk
                buf.sort(key=lambda r: r[url_idx])
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
                w = csv.writer(tmp)
                w.writerow(header)
                w.writerows(buf)
                tmp.close()
                tmp_files.append(tmp.name)
                buf.clear()
        # Flush the last partial chunk (if any)
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
    """
    N-way merge of multiple chunk-sorted CSV files into a single globally sorted CSV.

    Implementation details:
      - Each temp file is assumed to be sorted by 'url' and to have the same header.
      - We open all temp files and maintain a min-heap of (url_value, file_index, row).
      - At each step, we pop the smallest url from the heap, write it to the output,
        then push the next row from that same file (if any).
      - Finally, we close all file handles and delete temp files.

    Complexity:
      - Time ~ O(N log K), where N is total rows and K is number of chunks.
      - Memory ~ O(K) for the heap and one row per file.
    """
    def readers():
        for fp in files:
            f = open(fp, "r", encoding="utf-8", newline="")
            r = csv.reader(f)
            next(r)  # skip header
            yield f, r

    streams = list(readers())
    iters = [r for _, r in streams]

    # Build initial heap: (url_value, iterator_index, row)
    heap = []
    for idx, r in enumerate(iters):
        try:
            row = next(r)
            heap.append((row[0], idx, row))  # assumes 'url' is the first column in header
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

    # Cleanup: close all streams and remove temp files
    for f, _ in streams:
        f.close()
    for fp in files:
        os.unlink(fp)

def main():
    ap = argparse.ArgumentParser(description="External sort of a CSV by the 'url' column using chunked merge sort.")
    ap.add_argument("--input", required=True, help="Input CSV(.gz) to sort (must contain a 'url' column).")
    ap.add_argument("--output", required=True, help="Output CSV(.gz) path for the sorted result.")
    ap.add_argument("--chunk-size", type=int, default=2_000_000, help="Rows per in-memory chunk before spilling to disk.")
    args = ap.parse_args()

    tmps, header = chunk_sort(args.input, args.chunk_size)
    merge_sorted(tmps, header, args.output)
    print(f"✅ sorted → {args.output}")

if __name__ == "__main__":
    main()
