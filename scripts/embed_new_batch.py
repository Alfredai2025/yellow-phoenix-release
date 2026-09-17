#!/usr/bin/env python3
"""Embed a batch of real papers with MiniLM-L6-v2 (MPS) into a fresh .npy.

Usage: embed_new_batch.py <db_path> <where_clause> <out_npy> <registry_jsonl> <source_tag>

- Selects rows (rowid, title, abstract) from the papers table.
- Encodes f"{title}. {abstract}"[:512], L2-normalized by ST.
- Writes a proper .npy (N x 384 float32) and appends one registry line per row:
  {"new_idx": i, "source": <source_tag>, "src_rowid": rowid}
- Resumable: skips rows already present in the output npy (via checkpoint file).
"""
import os, sys, json, sqlite3, numpy as np, torch
from sentence_transformers import SentenceTransformer

DB_PATH, WHERE, OUT_NPY, REGISTRY, TAG = sys.argv[1:6]
BATCH = 256
DIM = 384
CKPT = OUT_NPY + ".ckpt"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
rows = cur.execute(f"SELECT rowid, title, abstract FROM papers WHERE {WHERE} ORDER BY rowid").fetchall()
conn.close()
print(f"selected {len(rows):,} rows")

done = 0
if os.path.exists(CKPT):
    with open(CKPT) as f:
        done = int(f.read().strip())
print(f"resuming at {done:,}")

if os.path.exists(OUT_NPY) and done > 0:
    out = np.load(OUT_NPY, mmap_mode='r+')
else:
    out = np.lib.format.open_memmap(OUT_NPY, mode='w+', dtype=np.float32, shape=(len(rows), DIM))

model = SentenceTransformer('all-MiniLM-L6-v2', device="mps" if torch.backends.mps.is_available() else "cpu")
reg = open(REGISTRY, 'a', buffering=1)

i = done
t0 = __import__('time').time()
while i < len(rows):
    chunk = rows[i:i+BATCH]
    texts = [f"{t or ''}. {a or ''}"[:512] for _, t, a in chunk]
    vecs = model.encode(texts, batch_size=BATCH, convert_to_numpy=True, show_progress_bar=False)
    out[i:i+len(chunk)] = vecs
    out.flush()
    for k, (rowid, _, _) in enumerate(chunk):
        reg.write(json.dumps({"new_idx": i + k, "source": TAG, "src_rowid": rowid}) + "\n")
    i += len(chunk)
    if i % (BATCH * 20) == 0:
        with open(CKPT, 'w') as f:
            f.write(str(i))
        rate = (i - done) / (__import__('time').time() - t0)
        print(f"  [{i:,}/{len(rows):,}] {rate:.0f}/s", flush=True)

with open(CKPT, 'w') as f:
    f.write(str(i))
reg.close()
print(f"DONE {i:,} rows -> {OUT_NPY}")
