#!/usr/bin/env python3
"""
Validate 512-bit ITQ end-to-end: hash-only vs engine search R@1/R@5/R@10.
"""
import numpy as np
import sqlite3
import random
import sys
import os

DB_PATH = "data/phoenix_arxiv_1m.db"
HASHES_PATH = "data/paper_hashes_512.npy"
N_TEST = 500
TOP_K = [1, 5, 10]

# Try engine
engine = None
try:
    sys.path.insert(0, ".")
    from yp_bridge import YPEngine
    engine = YPEngine()
    print("[val] Engine loaded")
except Exception as e:
    print(f"[val] Engine not available: {e}")


def load_papers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT rowid, id, title FROM papers WHERE title IS NOT NULL ORDER BY rowid")
    papers = cur.fetchall()
    conn.close()
    return papers


def hash_recall_test(hashes, n_test=N_TEST):
    n = len(hashes)
    indices = random.sample(range(n), min(n_test, n))
    hits = {k: 0 for k in TOP_K}

    for idx in indices:
        query_hash = hashes[idx]
        dists = np.sum(hashes != query_hash, axis=1)
        # DO NOT exclude self — ground truth is the same paper
        ranked = np.argsort(dists)

        for k in TOP_K:
            if idx in ranked[:k]:
                hits[k] += 1

    print("\n=== HASH-ONLY RECALL (incl. self) ===")
    for k in TOP_K:
        print(f"  R@{k}: {hits[k]/len(indices):.3%} ({hits[k]}/{len(indices)})")
    return hits


def engine_recall_test(papers, hashes, n_test=100):
    if engine is None:
        print("[val] Skipping engine test")
        return

    # Build rowid -> index map for checking
    rowid_to_idx = {row[0]: i for i, row in enumerate(papers)}

    random.shuffle(papers)
    test_papers = papers[:n_test]
    hits = {k: 0 for k in TOP_K}

    for rowid, pid, title in test_papers:
        try:
            # YPEngine.search returns [(score, (pid, title)), ...]
            results = engine.search(title, top_k=max(TOP_K))
            result_pids = [r[1][0] for r in results]  # extract pid from tuple

            for k in TOP_K:
                if pid in result_pids[:k]:
                    hits[k] += 1
        except Exception as e:
            print(f"[val] Search failed for '{title[:40]}...': {e}")

    print("\n=== END-TO-END SEARCH RECALL ===")
    for k in TOP_K:
        print(f"  R@{k}: {hits[k]/len(test_papers):.3%} ({hits[k]}/{len(test_papers)})")
    return hits


def main():
    random.seed(42)
    papers = load_papers()
    print(f"[val] Loaded {len(papers)} papers")

    hashes = np.load(HASHES_PATH)
    print(f"[val] Loaded hashes: {hashes.shape}")

    hash_recall_test(hashes, n_test=N_TEST)
    engine_recall_test(papers, hashes, n_test=min(100, len(papers)))


if __name__ == "__main__":
    main()
