#!/usr/bin/env python3
"""Smoke test YPEngine wired to 1M arXiv DB + BinaryHNSW."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from yp_bridge import YPEngine

QUERIES = [
    "machine learning",
    "neural network",
    "computer vision",
    "natural language processing",
    "deep learning",
    "reinforcement learning",
    "transformer architecture",
    "graph neural network",
    "attention mechanism",
    "large language model",
]


def main():
    print("Booting YPEngine with 1M arXiv DB...")
    t0 = time.time()
    engine = YPEngine(db_path="data/phoenix_arxiv_1m.db")
    print(f"Booted in {time.time()-t0:.1f}s")
    print(f"Cache size: {len(engine.cache):,}")
    print(f"HNSW size: {len(getattr(engine, '_hnsw', None) or [])}")

    print("\nSample queries:")
    for q in QUERIES:
        t0 = time.perf_counter()
        res = engine.search(q, top_k=5)
        lat = (time.perf_counter() - t0) * 1000
        if res:
            score, (pid, title) = res[0]
            print(f"  '{q[:40]:<40}' {lat:7.2f}ms | {title[:70]}")
        else:
            print(f"  '{q[:40]:<40}' {lat:7.2f}ms | NO RESULTS")


if __name__ == "__main__":
    main()
