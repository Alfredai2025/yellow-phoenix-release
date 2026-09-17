#!/usr/bin/env python3
"""Build an inverted index over paper titles for fast constrained search.

Maps each lowercase word (>=3 chars, not stopword) to the list of paper IDs
whose title contains that word. Used for keyword-candidate generation followed
by geometric re-rank.
"""
import json
import os
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from yp_bridge import YP_ROOT

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
    'can', 'need', 'dare', 'ought', 'used', 'pdf', 'hal', 'arxiv', 'et', 'al', 'using',
    'based', 'via', 'we', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    'they', 'them', 'their', 'our', 'its', 'as', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'about', 'up', 'out', 'if', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'between', 'among'
}

DB_PATH = Path(YP_ROOT) / "data" / "phoenix_arxiv_1m.db"
OUT_PATH = Path(YP_ROOT) / "data" / "title_inverted_index.pkl"


def tokenize(title):
    return [w for w in title.lower().split() if len(w) >= 3 and w not in STOPWORDS]


def main():
    print(f"Building inverted index from {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM papers")

    index = defaultdict(list)
    total = 0
    t0 = time.time()
    for pid, title in cursor:
        words = set(tokenize(title))
        for w in words:
            index[w].append(pid)
        total += 1
        if total % 100000 == 0:
            print(f"  indexed {total:,} papers in {time.time()-t0:.1f}s")

    conn.close()
    print(f"Indexed {total:,} papers, {len(index):,} unique words")

    # Convert to regular dict and sort pid lists for determinism
    index = {w: sorted(pids) for w, pids in index.items()}

    print(f"Saving to {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {OUT_PATH.stat().st_size:,} bytes in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
