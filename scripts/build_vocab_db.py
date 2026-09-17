#!/usr/bin/env python3
"""Build vocab_5m.db: token/frequency vocabulary from papers_5m.db titles + authors.

Consumed by YPSpellSuggest (iOS) for typo-tolerant "Did you mean?" suggestions.
iOS SQLite lacks spellfix1, so we ship a small sidecar DB and do edit-distance
lookup in Swift. Schema:

  vocab(token TEXT, freq INTEGER, field INTEGER, fc TEXT, PRIMARY KEY(token, field))
    field: 0 = title token, 1 = author token
    fc: first character (index support for candidate filtering)
  stats(k TEXT PRIMARY KEY, v INTEGER)  -- row counts for diagnostics
"""
import re
import sqlite3
import sys
import time

SRC = "data/papers_5m.db"
DST = "data/vocab_5m.db"

TOKEN_RE = re.compile(r"[a-z0-9]+")

def batched(it, n=50_000):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    t0 = time.time()
    src = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
    dst = sqlite3.connect(DST)
    dst.execute("PRAGMA journal_mode=OFF")
    dst.execute("PRAGMA synchronous=OFF")
    dst.execute("DROP TABLE IF EXISTS vocab")
    dst.execute("DROP TABLE IF EXISTS stats")
    dst.execute(
        "CREATE TABLE vocab(token TEXT, freq INTEGER, field INTEGER, fc TEXT,"
        " PRIMARY KEY(token, field))"
    )
    dst.execute("CREATE INDEX idx_vocab_fc ON vocab(fc, field, freq)")

    # Pass 1: title tokens (field 0), Pass 2: author tokens (field 1)
    for field, column in ((0, "title"), (1, "authors")):
        counts = {}
        for (text,) in src.execute(f"SELECT {column} FROM meta"):
            if not text:
                continue
            for tok in set(TOKEN_RE.findall(text.lower())):
                counts[tok] = counts.get(tok, 0) + 1
        rows = ((tok, freq, field, tok[0]) for tok, freq in counts.items())
        for chunk in batched(rows):
            dst.executemany(
                "INSERT OR REPLACE INTO vocab VALUES (?,?,?,?)", chunk
        )
        print(f"field {field} ({column}): {len(counts):,} distinct tokens "
              f"({time.time()-t0:.0f}s)", flush=True)
        del counts

    dst.execute("CREATE TABLE stats(k TEXT PRIMARY KEY, v INTEGER)")
    for field, name in ((0, "title"), (1, "author")):
        n = dst.execute("SELECT count(*) FROM vocab WHERE field=?",
                        (field,)).fetchone()[0]
        dst.execute("INSERT INTO stats VALUES (?,?)", (f"tokens_{name}", n))
    dst.execute("INSERT INTO stats VALUES ('papers', ?)",
                (src.execute("SELECT count(*) FROM meta").fetchone()[0],))
    dst.commit()
    dst.execute("PRAGMA optimize")
    total = dst.execute("SELECT count(*) FROM vocab").fetchone()[0]
    print(f"total {total:,} tokens in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    sys.exit(main())
