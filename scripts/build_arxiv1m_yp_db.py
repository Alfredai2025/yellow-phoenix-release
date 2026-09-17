#!/usr/bin/env python3
"""Build data/phoenix_arxiv_1m.db from papers_arxiv_1m.db + paper_hashes_arxiv_1m.pkl.

Adds a yp_hash512 TEXT column (128-char lowercase hex) so YPEngine's static
mesh loader can consume the 1M arXiv dataset.
"""
import json
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

SRC_DB = ROOT / "data" / "papers_arxiv_1m.db"
HASH_PKL = ROOT / "data" / "paper_hashes_arxiv_1m.pkl"
DST_DB = ROOT / "data" / "phoenix_arxiv_1m.db"


def main():
    if not SRC_DB.exists():
        raise SystemExit(f"Source DB not found: {SRC_DB}")
    if not HASH_PKL.exists():
        raise SystemExit(f"Hash pkl not found: {HASH_PKL}")

    print("Loading 1.27M hashes...")
    t0 = time.time()
    with open(HASH_PKL, "rb") as f:
        hash_dict = pickle.load(f)
    print(f"  loaded {len(hash_dict):,} hashes in {time.time()-t0:.1f}s")

    # Validate a sample hash format
    sample_key = next(iter(hash_dict))
    sample_hash = hash_dict[sample_key]
    if not isinstance(sample_hash, bytes) or len(sample_hash) != 64:
        raise SystemExit(f"Unexpected hash format for {sample_key}: {type(sample_hash)} len={len(sample_hash)}")

    if DST_DB.exists():
        print(f"Removing old {DST_DB}")
        DST_DB.unlink()

    print("Creating destination DB...")
    dst = sqlite3.connect(DST_DB)
    dst.execute("PRAGMA journal_mode=WAL;")
    dst.execute("PRAGMA synchronous=NORMAL;")
    dst.execute("""
        CREATE TABLE papers (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            authors TEXT,
            year INTEGER,
            categories TEXT,
            abstract TEXT,
            content_hash TEXT,
            downloaded_at TEXT,
            yp_hash512 TEXT
        )
    """)
    dst.execute("CREATE INDEX idx_hash ON papers(yp_hash512);")
    dst.execute("CREATE INDEX idx_title ON papers(title);")

    print("Streaming papers from source DB...")
    src = sqlite3.connect(SRC_DB)
    src.row_factory = sqlite3.Row
    cur = src.cursor()
    cur.execute("SELECT id, source, title, authors, year, categories, abstract, content_hash, downloaded_at FROM papers")

    batch = []
    inserted = 0
    missing_hash = 0
    batch_size = 5000

    t0 = time.time()
    for row in cur:
        pid = row["id"]
        h = hash_dict.get(pid)
        if h is None:
            missing_hash += 1
            continue
        # Convert 64 packed bytes to 128-char lowercase hex
        h512 = h.hex()
        batch.append((pid, row["source"], row["title"], row["authors"], row["year"],
                      row["categories"], row["abstract"], row["content_hash"],
                      row["downloaded_at"], h512))
        if len(batch) >= batch_size:
            dst.executemany("""
                INSERT INTO papers (id, source, title, authors, year, categories, abstract, content_hash, downloaded_at, yp_hash512)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            inserted += len(batch)
            batch.clear()
            if inserted % 100000 == 0:
                print(f"  inserted {inserted:,} ... {time.time()-t0:.1f}s")

    if batch:
        dst.executemany("""
            INSERT INTO papers (id, source, title, authors, year, categories, abstract, content_hash, downloaded_at, yp_hash512)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        inserted += len(batch)

    dst.commit()
    src.close()
    dst.close()

    print(f"\nDone. Inserted {inserted:,} papers into {DST_DB}")
    print(f"Papers without hash: {missing_hash:,}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
