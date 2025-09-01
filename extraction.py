#!/usr/bin/env python3
import io, json, os, tarfile
import lz4.frame

TAR_LZ4 = "/app/hf/qt5_cache/artifact.tar.lz4"
MAX_MEMBERS = 120
PREVIEW_LINES = 5
CAND_EXT = (".jsonl", ".json", ".csv", ".parquet", ".arrow")

def main():
    if not os.path.exists(TAR_LZ4):
        print("[ERR] Non trovo", TAR_LZ4)
        return

    print("[INFO] Apro", TAR_LZ4)
    with lz4.frame.open(TAR_LZ4, mode="rb") as lz:
        with tarfile.open(fileobj=lz, mode="r|*") as tf:
            members = []
            first_preview = None
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name
                members.append(name)
                if first_preview is None and name.lower().endswith((".jsonl",".json",".csv",".parquet",".arrow")):
                    first_preview = (m, name)
                if len(members) >= MAX_MEMBERS and first_preview:
                    break

    print("\n[INFO] Primi membri (max %d):" % MAX_MEMBERS)
    for i, n in enumerate(members, 1):
        print(f"  {i:3d}. {n}")

    if not first_preview:
        print("\n[WARN] Nessun file candidato tra:", CAND_EXT)
        return

    m, name = first_preview
    print(f"\n[INFO] Provo a fare preview di: {name}")

    with lz4.frame.open(TAR_LZ4, mode="rb") as lz:
        with tarfile.open(fileobj=lz, mode="r|*") as tf:
            for m2 in tf:
                if m2.name == name:
                    f = tf.extractfile(m2)
                    if f is None:
                        print("[WARN] extractfile(None)")
                        return
                    if name.lower().endswith(".csv"):
                        data = f.read()
                        head = data.splitlines()[:PREVIEW_LINES]
                        print("\n--- CSV HEAD ---")
                        for ln in head:
                            try:
                                print(ln.decode("utf-8", "ignore"))
                            except Exception:
                                print(ln)
                    elif name.lower().endswith(".jsonl"):
                        print("\n--- JSONL HEAD ---")
                        for _ in range(PREVIEW_LINES):
                            ln = f.readline()
                            if not ln: break
                            try:
                                obj = json.loads(ln)
                                keys = list(obj.keys())
                                print("keys:", keys)
                                print(obj if len(json.dumps(obj)) < 400 else str(obj)[:400]+" ...")
                            except Exception as e:
                                print("raw:", ln[:200], "err:", e)
                    elif name.lower().endswith(".json"):
                        print("\n--- JSON HEAD ---")
                        data = f.read()
                        try:
                            obj = json.loads(data)
                            if isinstance(obj, dict):
                                print("dict keys:", list(obj.keys()))
                            elif isinstance(obj, list):
                                print("list[0] keys:", list(obj[0].keys()) if obj else [])
                            print(str(obj)[:800], "...")
                        except Exception as e:
                            print("err:", e)
                    elif name.lower().endswith((".parquet",".arrow")):
                        # ATTENZIONE: qui leggiamo tutto il file in RAM (ok se piccolo)
                        import pandas as pd
                        import io as iolib
                        print("\n--- PARQUET/ARROW HEAD ---")
                        data = f.read()
                        try:
                            if name.lower().endswith(".parquet"):
                                df = pd.read_parquet(iolib.BytesIO(data))
                            else:
                                # alcuni .arrow sono un table ipc; proviamo a leggerlo con pyarrow
                                import pyarrow.ipc as ipc
                                import pyarrow as pa
                                table = ipc.open_stream(iolib.BytesIO(data)).read_all()
                                df = table.to_pandas()
                            print("columns:", list(df.columns))
                            print(df.head(5))
                        except Exception as e:
                            print("err:", e)
                    else:
                        print("[INFO] Estensione non gestita per preview:", name)
                    break

if __name__ == "__main__":
    main()
