#!/usr/bin/env python3
"""In-place FTS upgrade for papers_5m.db.

Old schema: single `fts(title, authors)` FTS5, detail='none' — doclists store
docids only, so bm25() re-tokenizes every matching document (title search:
~90ms for 17K matches on Mac, ~70ms on device). Author search additionally
JOINed meta + LIKE over the candidate set (~370ms Mac / ~307ms device).

New schema:
  - `fts(title, authors)`       recreated with detail='column' (tf per column
                                in the doclist -> bm25 is pure arithmetic)
  - `authors_fts(authors)`      new, detail='column', authors-only — author
                                search is a direct MATCH + bm25, no JOIN, no LIKE

Rebuildable from scripts/build_papers_5m_db.py (kept arrays in
data/_build_papers5m/). Backup: papers_5m.db.bak_prefts (APFS clone).
"""
import os, sqlite3, time

DB = os.path.expanduser("~/yellow_phoenix/data/papers_5m.db")
CHUNK = 200_000


def main():
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    t0 = time.time()

    n = db.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    print(f"meta rows: {n:,}", flush=True)

    db.execute("""CREATE VIRTUAL TABLE fts_new USING fts5(
        title, authors, tokenize='unicode61', detail='column')""")
    db.execute("""CREATE VIRTUAL TABLE authors_fts USING fts5(
        authors, tokenize='unicode61', detail='column')""")

    lo = 0
    while lo < n:
        hi = min(lo + CHUNK, n)
        rows = db.execute(
            "SELECT id, COALESCE(title,''), COALESCE(authors,'') "
            "FROM meta WHERE id >= ? AND id < ? ORDER BY id",
            (lo, hi)).fetchall()
        db.executemany(
            "INSERT INTO fts_new(rowid, title, authors) VALUES(?,?,?)", rows)
        db.executemany(
            "INSERT INTO authors_fts(rowid, authors) VALUES(?,?)",
            [(r[0], r[2]) for r in rows])
        lo = hi
        print(f"  {lo:,}/{n:,}  ({time.time()-t0:.0f}s)", flush=True)
        db.commit()

    # Only swap once the new tables are fully built — old fts stays live
    # for readers until this point.
    db.execute("DROP TABLE fts")
    db.execute("ALTER TABLE fts_new RENAME TO fts")
    db.execute("INSERT INTO fts(fts) VALUES('optimize')")
    db.execute("INSERT INTO authors_fts(authors_fts) VALUES('optimize')")
    db.execute("PRAGMA optimize")
    db.commit()
    db.close()
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
