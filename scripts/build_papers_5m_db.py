#!/usr/bin/env python3
"""Build papers_5m.db: metadata + FTS for every row of real_5m.ism.

The ISM row order comes from build_5m_real_dedup.py: five embedding sources
streamed in fixed order, first-occurrence dedup by 512-bit ITQ hash.
This script replays that dedup to learn, for every kept ISM id, WHICH source
row it came from, then resolves title/authors/year/categories via:
  - 3m bulk : titles_3m.db (id == embeddings_3m row) -> phoenix db by pid
  - oai     : new_registry.jsonl        -> phoenix db by rowid
  - pubmed  : new_registry_pubmed*.jsonl-> phoenix db by rowid
  - droplet : droplet db, rowid == npy_row + 1 (aligned file)
Output: data/papers_5m.db  (meta + FTS5 detail='column': fts(title,authors)
+ authors_fts(authors) — bm25 ranking without re-tokenizing docs)
"""
import os, sys, json, sqlite3, time
import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")
PHOENIX_DB = os.path.join(DATA, "phoenix_arxiv_1m.db")
OUT_DB = os.path.join(DATA, "papers_5m.db")
WORK = os.path.join(DATA, "_build_papers5m")
os.makedirs(WORK, exist_ok=True)

SOURCES = [
    ("embeddings_3m.npy", "raw"),
    ("embeddings_oai439k.npy", "npy"),
    ("embeddings_pubmed_new.npy", "npy"),
    ("embeddings_pubmed_new2.npy", "npy"),
    ("droplet_embeddings_aligned.npy", "npy"),
]
EXPECTED_KEPT = 5_107_508
CHUNK = 50_000


def open_source(rel, kind):
    path = os.path.join(os.path.expanduser("~/yellow_phoenix"), "data", rel)
    if kind == "raw":
        mm = np.memmap(path, dtype=np.float32, mode="r")
        assert mm.size % 384 == 0
        return mm.reshape(-1, 384)
    return np.load(path)


def stage_a_replay_dedup():
    """Replay the dedup; return per-source uint32 arrays of kept npy rows."""
    saved = [os.path.join(WORK, f"kept_src{i}.npy") for i in range(len(SOURCES))]
    if all(os.path.exists(p) for p in saved):
        return [np.load(p) for p in saved]

    m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
    mean = m["mean"].astype(np.float32)
    proj = m["proj"].astype(np.float32)

    seen = set()
    kept = [[] for _ in SOURCES]
    t0 = time.time()
    for si, (rel, kind) in enumerate(SOURCES):
        arr = open_source(rel, kind)
        n = arr.shape[0]
        print(f"[A] source {si} {rel}: {n:,} rows", flush=True)
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            x = np.asarray(arr[lo:hi], dtype=np.float32)
            bits = ((x - mean) @ proj) > 0
            codes = np.ascontiguousarray(np.packbits(bits, axis=1))
            cv = codes.view(np.dtype((np.void, 64))).ravel()
            for j in range(len(cv)):
                h = cv[j].tobytes()
                if h not in seen:
                    seen.add(h)
                    kept[si].append(lo + j)
        del arr
        print(f"    kept so far: {sum(len(k) for k in kept):,} "
              f"({time.time()-t0:.0f}s)", flush=True)

    out = []
    for si, k in enumerate(kept):
        a = np.array(k, dtype=np.uint32)
        np.save(saved[si], a)
        out.append(a)
    total = sum(len(a) for a in out)
    assert total == EXPECTED_KEPT, f"kept {total} != {EXPECTED_KEPT}"
    print(f"[A] replay done: {total:,} kept", flush=True)
    return out


def stage_b_build(kept):
    if os.path.exists(OUT_DB):
        os.remove(OUT_DB)
    db = sqlite3.connect(OUT_DB)
    db.execute("PRAGMA page_size=16384")
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("""CREATE TABLE meta(
        id INTEGER PRIMARY KEY, ext_id TEXT, title TEXT, authors TEXT,
        year INTEGER, categories TEXT, source TEXT)""")
    db.execute("""CREATE VIRTUAL TABLE fts USING fts5(
        title, authors, tokenize='unicode61', detail='column')""")
    db.execute("""CREATE VIRTUAL TABLE authors_fts USING fts5(
        authors, tokenize='unicode61', detail='column')""")

    ph = sqlite3.connect(PHOENIX_DB)
    dr = sqlite3.connect(os.path.join(DATA, "droplet_papers_106298.db"))

    # registries: npy row -> phoenix rowid
    def load_registry(path):
        rows = [json.loads(l) for l in open(path)]
        rows.sort(key=lambda r: r["new_idx"])
        assert [r["new_idx"] for r in rows] == list(range(len(rows)))
        return np.array([r["src_rowid"] for r in rows], dtype=np.int64)

    reg_oai = load_registry(os.path.join(DATA, "new_registry.jsonl"))
    reg_pm = load_registry(os.path.join(DATA, "new_registry_pubmed.jsonl"))
    reg_pm2 = load_registry(os.path.join(DATA, "new_registry_pubmed2.jsonl"))

    # titles_3m: embeddings_3m row -> pid
    t3 = sqlite3.connect(os.path.join(DATA, "titles_3m.db"))
    pids = [r[0] for r in t3.execute("SELECT pid FROM papers ORDER BY id")]
    assert len(pids) == 3_025_752
    t3.close()

    ism_id = 0
    stats = {"joined": 0, "null_title": 0}
    t0 = time.time()

    def batch_insert(recs):
        db.executemany(
            "INSERT INTO meta VALUES(?,?,?,?,?,?,?)", recs)
        db.executemany(
            "INSERT INTO fts(rowid, title, authors) VALUES(?,?,?)",
            [(r[0], r[2] or "", r[3] or "") for r in recs])
        db.executemany(
            "INSERT INTO authors_fts(rowid, authors) VALUES(?,?)",
            [(r[0], r[3] or "") for r in recs])

    def resolve_rows(conn, key_rows, by_pid=False):
        """key_rows: list of (ism_id, lookup_value). Returns dict value->row."""
        out = {}
        B = 4000
        for lo in range(0, len(key_rows), B):
            chunk = key_rows[lo:lo + B]
            vals = [v for _, v in chunk]
            if by_pid:
                q = f"SELECT id, id, title, authors, year, categories, source FROM papers WHERE id IN ({','.join('?'*len(vals))})"
            else:
                q = f"SELECT rowid, id, title, authors, year, categories, source FROM papers WHERE rowid IN ({','.join('?'*len(vals))})"
            for r in conn.execute(q, vals):
                out[r[0]] = r[1:]
        return out

    B = 20_000
    for si, kept_arr in enumerate(kept):
        recs = []
        key_rows = [(ism_id + j, int(v)) for j, v in enumerate(kept_arr)]
        if si == 0:      # 3m bulk via pid
            kv = [(i, pids[v]) for i, v in key_rows]
            m = resolve_rows(ph, kv, by_pid=True)
            for i, v in kv:
                r = m.get(v)
                recs.append((i,) + r if r else (i, None, None, None, None, None, None))
        elif si in (1, 2, 3):
            reg = (reg_oai, reg_pm, reg_pm2)[si - 1]
            kv = [(i, int(reg[v])) for i, v in key_rows]
            m = resolve_rows(ph, kv)
            for i, v in kv:
                r = m.get(v)
                recs.append((i,) + r if r else (i, None, None, None, None, None, None))
        else:            # droplet via rowid
            kv = [(i, v + 1) for i, v in key_rows]
            m = resolve_rows(dr, kv)
            for i, v in kv:
                r = m.get(v)
                if r:
                    recs.append((i, r[0], r[1], r[2], r[3], r[4], "droplet"))
                else:
                    recs.append((i, None, None, None, None, None, None))
        for lo in range(0, len(recs), B):
            batch_insert(recs[lo:lo + B])
        db.commit()
        joined = sum(1 for r in recs if r[2] is not None)
        stats["joined"] += joined
        stats["null_title"] += len(recs) - joined
        ism_id += len(recs)
        print(f"[B] source {si}: +{len(recs):,} (joined {joined:,}) "
              f"| total {ism_id:,} | {time.time()-t0:.0f}s", flush=True)

    db.execute("CREATE INDEX idx_year ON meta(year)")
    db.execute("INSERT INTO fts(fts) VALUES('optimize')")
    db.execute("INSERT INTO authors_fts(authors_fts) VALUES('optimize')")
    db.commit()
    db.execute("PRAGMA optimize")
    db.close()
    print(f"[B] DONE {ism_id:,} rows, joined {stats['joined']:,}, "
          f"missing title {stats['null_title']:,}", flush=True)


if __name__ == "__main__":
    kept = stage_a_replay_dedup()
    stage_b_build(kept)
    print("papers_5m.db:", os.path.getsize(OUT_DB) / 1e9, "GB")
