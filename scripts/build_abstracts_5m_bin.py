#!/usr/bin/env python3
"""Build abstracts_5m.bin: zstd-dictionary compressed abstract store.

Layout:
  magic "YPA1" (4B) | ver u8 | block_count u64 | paper_count u64
  | dict_len u32 | block_target u32
  dictionary bytes (zstd)
  index: block_count x [u64 first_id][u64 offset][u32 n_records][u32 comp_len]
  blocks: zstd frames; decompressed stream per block:
      varint first_id (LEB128 u64)
      n_records x [varint id_delta][varint byte_len][abstract bytes]
  Every paper id 0..N-1 has exactly one slot (empty string if no abstract),
  so id -> block is a binary search on first_id.

Records are abstract-ONLY: titles/authors live in papers_5m.db.
"""
import os, json, sqlite3, time, random
import numpy as np
import zstandard as zstd

DATA = os.path.expanduser("~/yellow_phoenix/data")
PHOENIX_DB = os.path.join(DATA, "phoenix_arxiv_1m.db")
OUT = os.path.join(DATA, "abstracts_5m.bin")
WORK = os.path.join(DATA, "_build_papers5m")
BLOCK_TARGET = 64 * 1024
DICT_SIZE = 112 * 1024
LEVEL = 17
MAGIC = b"YPA1"


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def load_registry(path):
    rows = [json.loads(l) for l in open(path)]
    return {r["new_idx"]: r["src_rowid"] for r in rows}


def main():
    t0 = time.time()
    ph = sqlite3.connect(PHOENIX_DB)
    dr = sqlite3.connect(os.path.join(DATA, "droplet_papers_106298.db"))
    t3 = sqlite3.connect(os.path.join(DATA, "titles_3m.db"))
    pids = [r[0] for r in t3.execute("SELECT pid FROM papers ORDER BY id")]
    t3.close()

    kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
    reg_oai = load_registry(os.path.join(DATA, "new_registry.jsonl"))
    reg_pm = load_registry(os.path.join(DATA, "new_registry_pubmed.jsonl"))
    reg_pm2 = load_registry(os.path.join(DATA, "new_registry_pubmed2.jsonl"))

    # ---- dictionary from a random sample of the corpus ----
    print("[1/3] training dictionary...", flush=True)
    random.seed(3)
    rowids = random.sample(range(1, 5_617_581), 12000)
    samples = [r[0] or "" for r in ph.execute(
        f"SELECT abstract FROM papers WHERE rowid IN ({','.join('?'*len(rowids))})",
        rowids)]
    drowids = random.sample(range(1, 106_299), 2000)
    samples += [r[0] or "" for r in dr.execute(
        f"SELECT abstract FROM papers WHERE rowid IN ({','.join('?'*len(drowids))})",
        drowids)]
    samples = [s.encode("utf-8", "replace") for s in samples if s]
    dict_data = zstd.train_dictionary(DICT_SIZE, samples)
    print(f"    dict trained on {len(samples)} abstracts "
          f"({sum(map(len,samples))/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)

    cctx = zstd.ZstdCompressor(dict_data=dict_data, level=LEVEL)

    # ---- stream abstracts in ISM id order, compress blocks ----
    print("[2/3] building blocks...", flush=True)
    f = open(OUT, "wb")
    header_len = 4 + 1 + 8 + 8 + 4 + 4
    f.write(b"\x00" * header_len)
    f.write(dict_data.as_bytes())
    index = []          # (first_id, offset, n_records, comp_len)
    block_ids = []      # ids in current block
    block_buf = bytearray()
    total = 0
    abs_bytes = 0

    def flush_block():
        nonlocal block_buf, block_ids, abs_bytes
        if not block_ids:
            return
        payload = cctx.compress(bytes(block_buf))
        offset = f.tell()
        f.write(payload)
        index.append((block_ids[0], offset, len(block_ids), len(payload)))
        block_ids, block_buf = [], bytearray()

    def emit(ism_id, text):
        nonlocal block_buf, abs_bytes
        b = (text or "").encode("utf-8", "replace")
        if not block_ids:
            block_buf += varint(ism_id)
        else:
            block_buf += varint(ism_id - block_ids[-1])
        block_buf += varint(len(b)) + b
        block_ids.append(ism_id)
        abs_bytes += len(b)
        if len(block_buf) >= BLOCK_TARGET:
            flush_block()

    ism_base = 0
    for si, kept_arr in enumerate(kept):
        n = len(kept_arr)
        by_pid = False
        if si == 0:
            kv = [(ism_base + j, pids[int(r)]) for j, r in enumerate(kept_arr)]
            src = ph
            by_pid = True
        elif si == 1:
            kv = [(ism_base + j, reg_oai[int(r)]) for j, r in enumerate(kept_arr)]
            src = ph
        elif si == 2:
            kv = [(ism_base + j, reg_pm[int(r)]) for j, r in enumerate(kept_arr)]
            src = ph
        elif si == 3:
            kv = [(ism_base + j, reg_pm2[int(r)]) for j, r in enumerate(kept_arr)]
            src = ph
        else:
            kv = [(ism_base + j, int(r) + 1) for j, r in enumerate(kept_arr)]
            src = dr
        B = 4000
        for lo in range(0, n, B):
            chunk = kv[lo:lo + B]
            vals = [v for _, v in chunk]
            if by_pid:
                q = f"SELECT id, abstract FROM papers WHERE id IN ({','.join('?'*len(vals))})"
            else:
                q = f"SELECT rowid, abstract FROM papers WHERE rowid IN ({','.join('?'*len(vals))})"
            m = {k: a for k, a in src.execute(q, vals)}
            for i, v in chunk:
                emit(i, m.get(v))
        ism_base += n
        print(f"    source {si}: total {ism_base:,} | "
              f"{time.time()-t0:.0f}s", flush=True)
    flush_block()
    total = ism_base

    # ---- write index, patch header ----
    print("[3/3] writing index...", flush=True)
    for first_id, offset, nrec, clen in index:
        f.write(np.uint64(first_id).tobytes())
        f.write(np.uint64(offset).tobytes())
        f.write(np.uint32(nrec).tobytes())
        f.write(np.uint32(clen).tobytes())
    index_offset = header_len + len(dict_data.as_bytes())
    f.seek(0)
    f.write(MAGIC + b"\x01" + np.uint64(len(index)).tobytes()
            + np.uint64(total).tobytes()
            + np.uint32(len(dict_data.as_bytes())).tobytes()
            + np.uint32(BLOCK_TARGET).tobytes())
    # index entries carry absolute offsets already; record index_offset nowhere
    # (readers locate the index region right after the dictionary).
    f.close()

    size = os.path.getsize(OUT)
    print(f"DONE {total:,} papers, {len(index):,} blocks, "
          f"{abs_bytes/1e9:.2f}GB raw -> {size/1e9:.3f}GB "
          f"(ratio {abs_bytes/size:.1f}x) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
