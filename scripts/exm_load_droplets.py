#!/usr/bin/env python3
"""
exm_load_droplets.py — Load droplet metadata for intelligence layers

Reads data/droplets_papers_106298.db (SQLite)
Writes data/droplets_metadata.jsonl (titles + abstracts)

Note: Autopoiesis mesh mutation is NOT implemented here because
the Rust FFI (yp_autopoiesis_step) does not exist in libpams.dylib.
This is a Python-side loader only.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/Users/mac/yellow_phoenix/data/droplet_papers_106298.db")
META_OUT = Path("/Users/mac/yellow_phoenix/data/droplets_metadata.jsonl")
BATCH_SIZE = 1000


def load_droplets():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Droplet DB not found: {DB_PATH}")

    print(f"[EXM] Loading from {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Inspect schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"[EXM] Tables: {tables}")

    # Try to find the papers table
    table_name = None
    for t in tables:
        cursor.execute(f"PRAGMA table_info({t})")
        cols = [c[1] for c in cursor.fetchall()]
        if "title" in cols or "abstract" in cols or "text" in cols:
            table_name = t
            print(f"[EXM] Using table '{t}' with columns: {cols}")
            break

    if not table_name:
        # Fallback: just dump first table
        table_name = tables[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [c[1] for c in cursor.fetchall()]
        print(f"[EXM] Fallback table '{table_name}' columns: {cols}")

    # Write metadata JSONL
    META_OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(META_OUT, 'w') as f:
        cursor.execute(f"SELECT * FROM {table_name}")
        # Get column names from cursor description
        col_names = [desc[0] for desc in cursor.description]

        for row in cursor:
            data = dict(zip(col_names, row))

            # Extract title/abstract with flexible column names
            title = data.get("title", data.get("paper_title", data.get("name", "")))
            abstract = data.get("abstract", data.get("summary", data.get("text", "")))

            if not title:
                continue

            record = {
                "pid": data.get("id", data.get("paper_id", written)),
                "title": title,
                "abstract": str(abstract)[:500] if abstract else "",
            }
            f.write(json.dumps(record) + '\n')
            written += 1

            if written % BATCH_SIZE == 0:
                print(f"  Written {written} droplets...")

    conn.close()
    print(f"[EXM] Done. Wrote {written} droplets to {META_OUT}")
    return written


if __name__ == "__main__":
    load_droplets()
