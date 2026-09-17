#!/usr/bin/env python3
"""SQL cross-validation of papers_5m.db — no model needed.

For every ISM id we know (source, npy_row) from the replay. Independent checks:
  src0 (3m bulk): titles_3m.db row npy_row has its OWN title (written by the
                  3M pipeline, independent of phoenix db) -> must equal meta.title
  src4 (droplet): droplet db rowid npy_row+1 -> title must equal meta.title
  src1-3:         registry npy_row -> phoenix rowid -> title must equal meta.title
Sampled (N per source) for src0/src4 via SQL join; full-count for src2/3 spot.
"""
import os, sqlite3, json
import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")
WORK = os.path.join(DATA, "_build_papers5m")
N = 2000  # samples per source

meta = sqlite3.connect(os.path.join(DATA, "papers_5m.db"))
ph = sqlite3.connect(os.path.join(DATA, "phoenix_arxiv_1m.db"))
dr = sqlite3.connect(os.path.join(DATA, "droplet_papers_106298.db"))
t3 = sqlite3.connect(os.path.join(DATA, "titles_3m.db"))

kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
reg = {}
for si, path in [(1, "new_registry.jsonl"), (2, "new_registry_pubmed.jsonl"), (3, "new_registry_pubmed2.jsonl")]:
    rows = [json.loads(l) for l in open(os.path.join(DATA, path))]
    reg[si] = {r["new_idx"]: r["src_rowid"] for r in rows}

rng = np.random.default_rng(7)
total_ok = total = 0
bounds = []
acc = 0
for a in kept:
    bounds.append(acc)
    acc += len(a)

def check(name, pairs):
    """pairs: list of (ism_id, expected_title). Returns (ok, n)."""
    ok = 0
    for ism_id, exp in pairs:
        got = meta.execute("SELECT title FROM meta WHERE id=?", (int(ism_id),)).fetchone()[0]
        if (got or "") == (exp or ""):
            ok += 1
    print(f"  {name}: {ok}/{len(pairs)} titles match")
    return ok, len(pairs)

# src0: sample npy rows, expected title from titles_3m by id == npy_row
rows = rng.choice(kept[0], size=min(N, len(kept[0])), replace=False)
pairs = []
for r in rows:
    exp = t3.execute("SELECT title FROM papers WHERE id=?", (int(r),)).fetchone()
    if exp:
        ism_id = int(np.searchsorted(kept[0], r))
        if ism_id < len(kept[0]) and kept[0][ism_id] == r:
            pairs.append((bounds[0] + ism_id, exp[0]))
ok, n = check("src0 3m-bulk (titles_3m independent)", pairs); total_ok += ok; total += n

# src1-3: sample, registry -> phoenix title
for si, name in [(1, "src1 oai"), (2, "src2 pubmed"), (3, "src3 pubmed2")]:
    src_kept = kept[si]
    rows = rng.choice(src_kept, size=min(N, len(src_kept)), replace=False)
    pairs = []
    for r in rows:
        rowid = reg[si].get(int(r))
        if rowid is None:
            continue
        exp = ph.execute("SELECT title FROM papers WHERE rowid=?", (rowid,)).fetchone()
        if exp:
            ism_id = int(np.searchsorted(src_kept, r))
            if ism_id < len(src_kept) and src_kept[ism_id] == r:
                pairs.append((bounds[si] + ism_id, exp[0]))
    ok, n = check(f"{name} (registry->phoenix)", pairs); total_ok += ok; total += n

# src4 droplet: rowid = npy_row+1
rows = rng.choice(kept[4], size=min(N, len(kept[4])), replace=False)
pairs = []
for r in rows:
    exp = dr.execute("SELECT title FROM papers WHERE rowid=?", (int(r) + 1,)).fetchone()
    if exp:
        ism_id = int(np.searchsorted(kept[4], r))
        if ism_id < len(kept[4]) and kept[4][ism_id] == r:
            pairs.append((bounds[4] + ism_id, exp[0]))
ok, n = check("src4 droplet (rowid+1)", pairs); total_ok += ok; total += n

print(f"\nTOTAL: {total_ok}/{total} titles match")
print("GATE:", "PASS" if total_ok == total and total >= 6000 else "FAIL")
