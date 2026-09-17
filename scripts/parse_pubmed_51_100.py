#!/usr/bin/env python3
"""Parse PubMed baseline files 0051-0100 into phoenix_arxiv_1m.db.

Idempotent: INSERT OR IGNORE on pmid dedups. Source tag: pubmed_medline.
"""
import glob
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_parse_hourly import parse_pubmed_file, DB_PATH

BASELINE = os.path.expanduser("~/yellow_phoenix/data/pubmed_baseline")


def main():
    files = sorted(glob.glob(os.path.join(BASELINE, "pubmed26n0[05]*.xml.gz")))
    files = [f for f in files if os.path.basename(f)[9:13].isdigit()
             and 51 <= int(os.path.basename(f)[9:13]) <= 100]
    print(f"[+] {len(files)} target files")

    conn = sqlite3.connect(DB_PATH)
    before = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    total_new = 0
    for i, f in enumerate(files, 1):
        name = os.path.basename(f)
        try:
            new = parse_pubmed_file(f, conn)
        except Exception as e:
            print(f"    [!] {name}: parse error {e}")
            conn.rollback()
            continue
        conn.commit()
        total_new += new
        print(f"    [{i:2d}/{len(files)}] {name}: +{new:,} (cumulative +{total_new:,})")
    after = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    maxrow = conn.execute("SELECT MAX(rowid) FROM papers").fetchone()[0]
    print(f"[+] done: db {before:,} -> {after:,} (max rowid {maxrow:,})")
    conn.close()


if __name__ == "__main__":
    main()
