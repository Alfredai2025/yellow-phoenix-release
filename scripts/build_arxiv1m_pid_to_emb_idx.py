#!/usr/bin/env python3
"""Create a pid -> embedding index mapping for the 1M arXiv dataset.

Assumes data/paper_embeddings_arxiv_1m.npy was generated in the same row order
as papers_arxiv_1m.db (rowid order).
"""
import pickle
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC_DB = ROOT / "data" / "papers_arxiv_1m.db"
OUT_PKL = ROOT / "data" / "arxiv1m_pid_to_emb_idx.pkl"


def main():
    print("Building pid -> embedding index mapping...")
    conn = sqlite3.connect(SRC_DB)
    cur = conn.cursor()
    cur.execute("SELECT id FROM papers ORDER BY rowid")
    mapping = {}
    for idx, (pid,) in enumerate(cur):
        mapping[pid] = idx
    conn.close()
    print(f"  mapped {len(mapping):,} papers")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(mapping, f)
    print(f"  saved {OUT_PKL}")


if __name__ == "__main__":
    main()
