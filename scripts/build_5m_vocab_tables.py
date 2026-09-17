#!/usr/bin/env python3
"""Build the autocomplete term table for papers_5m.db.

fts5vocab has no whole-table mode (only per-doc 'row'/'col'), so this script
tokenizes meta once and accumulates document frequency itself:

    autocomplete_terms(field TEXT, term TEXT, df INTEGER, PRIMARY KEY(field,term))

field: 't' = title tokens, 'a' = author tokens. Terms with df < min_df are
dropped (typo junk never surfaces as a suggestion). Consumed by
YPMetadataStore.autocomplete(_:mode:limit:). Idempotent; --check reports.

Build time ~2-4 min (streams meta once).
"""
import os, re, sqlite3, sys, time

DB = os.path.expanduser("~/yellow_phoenix/data/papers_5m.db")
MIN_DF = 2
CHUNK = 200_000
TOKEN = re.compile(r"[a-z0-9]+")


def main():
    if "--check" in sys.argv:
        db = sqlite3.connect(DB)
        have = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='autocomplete_terms'").fetchone()
        print("autocomplete_terms:", bool(have))
        if have:
            for f, n in db.execute(
                    "SELECT field, COUNT(*) FROM autocomplete_terms GROUP BY field"):
                print(f"  field={f}: {n:,} terms")
        return

    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("DROP TABLE IF EXISTS autocomplete_terms")
    db.execute("""CREATE TABLE autocomplete_terms(
        field TEXT, term TEXT, df INTEGER, PRIMARY KEY(field, term)) WITHOUT ROWID""")

    df_t: dict[str, int] = {}
    df_a: dict[str, int] = {}
    n = db.execute("SELECT MAX(id)+1 FROM meta").fetchone()[0]
    t0 = time.time()
    lo = 0
    while lo < n:
        rows = db.execute(
            "SELECT COALESCE(title,''), COALESCE(authors,'') FROM meta "
            "WHERE id >= ? AND id < ?", (lo, min(lo + CHUNK, n)))
        for title, authors in rows:
            for tok in set(TOKEN.findall(title.lower())):
                df_t[tok] = df_t.get(tok, 0) + 1
            for tok in set(TOKEN.findall(authors.lower())):
                df_a[tok] = df_a.get(tok, 0) + 1
        lo += CHUNK
        print(f"  {min(lo,n):,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)

    for field, d in (("t", df_t), ("a", df_a)):
        recs = [(field, t, c) for t, c in d.items() if c >= MIN_DF]
        db.executemany("INSERT INTO autocomplete_terms VALUES(?,?,?)", recs)
        print(f"field {field}: {len(d):,} raw terms -> {len(recs):,} kept (df>={MIN_DF})",
              flush=True)
    db.commit()
    db.execute("PRAGMA optimize")
    db.close()
    print(f"DONE ({time.time()-t0:.0f}s), db {os.path.getsize(DB)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
