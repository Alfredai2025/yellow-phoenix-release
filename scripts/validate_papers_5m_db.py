#!/usr/bin/env python3
"""Validate papers_5m.db: re-embed N random papers and check the ITQ hash
of the fresh embedding lands at the same ISM id.

Ground truth: meta.title/meta.authors in papers_5m.db came from a join;
if the join is wrong, the re-embedded hash will mismatch.
Text recipe must match embed_new_batch.py: f"{title}. {abstract}"[:512].
"""
import os, sys, sqlite3, random
import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")
WORK = os.path.join(DATA, "_build_papers5m")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

meta = sqlite3.connect(os.path.join(DATA, "papers_5m.db"))
ph = sqlite3.connect(os.path.join(DATA, "phoenix_arxiv_1m.db"))
dr = sqlite3.connect(os.path.join(DATA, "droplet_papers_106298.db"))

# source boundaries from replay
kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
bounds = []
acc = 0
for a in kept:
    bounds.append((acc, acc + len(a)))
    acc += len(a)

# kept npy-row -> lookup into source db: need pid for src0 (via titles_3m)
t3 = sqlite3.connect(os.path.join(DATA, "titles_3m.db"))
pids = [r[0] for r in t3.execute("SELECT pid FROM papers ORDER BY id")]
t3.close()
reg = {}
for si, path in [(1, "new_registry.jsonl"), (2, "new_registry_pubmed.jsonl"), (3, "new_registry_pubmed2.jsonl")]:
    rows = [__import__("json").loads(l) for l in open(os.path.join(DATA, path))]
    reg[si] = {r["new_idx"]: r["src_rowid"] for r in rows}

def source_text(ism_id):
    for si, (lo, hi) in enumerate(bounds):
        if lo <= ism_id < hi:
            row = int(kept[si][ism_id - lo])
            if si == 0:
                pid = pids[row]
                t, a = ph.execute("SELECT title, abstract FROM papers WHERE id=?", (pid,)).fetchone()
            elif si in (1, 2, 3):
                t, a = ph.execute("SELECT title, abstract FROM papers WHERE rowid=?", (reg[si][row],)).fetchone()
            else:
                t, a = dr.execute("SELECT title, abstract FROM papers WHERE rowid=?", (row + 1,)).fetchone()
            return si, (t or ""), (a or "")
    raise ValueError(ism_id)

random.seed(7)
sample = sorted(random.sample(range(acc), N))

import torch
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device="mps" if torch.backends.mps.is_available() else "cpu")

m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
mean, proj = m["mean"].astype(np.float32), m["proj"].astype(np.float32)

with open(os.path.join(DATA, "real_5m.ism"), "rb") as f:
    cnt = np.frombuffer(f.read(8), dtype="<u8")[0]
    ism = np.frombuffer(f.read(cnt * 64), dtype=np.uint8).reshape(-1, 64)

texts, srcs = [], []
for i in sample:
    si, t, a = source_text(i)
    texts.append(f"{t}. {a}"[:512])
    srcs.append(si)

vecs = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
codes = np.packbits(((vecs.astype(np.float32) - mean) @ proj) > 0, axis=1)

ok = 0
for k, i in enumerate(sample):
    match = bool(np.array_equal(codes[k], ism[i]))
    ok += match
    db_title = meta.execute("SELECT title FROM meta WHERE id=?", (i,)).fetchone()[0]
    title_ok = db_title is not None and texts[k].split(". ")[0][:30] in (db_title or "")
    print(f"id {i:>8} src{srcs[k]} hash={'OK ' if match else 'BAD'} "
          f"title={'OK' if title_ok else 'DIFF'} | {db_title[:60] if db_title else 'NULL'}")
print(f"\n{ok}/{N} hashes match")
