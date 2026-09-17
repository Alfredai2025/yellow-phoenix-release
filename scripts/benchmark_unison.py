#!/usr/bin/env python3
"""Benchmark the dual-model unison engine (TF-IDF + MiniLM-L3-v2).

This script avoids the heavy YPEngine initialisation and exercises the
UnisonEngine class directly on titles from data/phoenix_arxiv_1m.db.  If
sentence_transformers / MiniLM-L3-v2 is unavailable, the engine falls back to
TF-IDF-only mode and reports zero MiniLM runs.
"""

import sqlite3
import time
import random
import numpy as np
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yp_bridge import UnisonEngine


def load_titles(db_path: str, limit: int = 5000):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM papers WHERE title IS NOT NULL AND LENGTH(title) > 3 LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def main(n_queries: int = 100):
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/phoenix_arxiv_1m.db")
    print(f"Loading up to 5000 titles from {db_path}...")
    rows = load_titles(db_path, 5000)
    if len(rows) < n_queries:
        print(f"Only {len(rows)} titles available.")
        return

    print(f"Building UnisonEngine from {len(rows)} titles...")
    engine = UnisonEngine(target_p50_ms=10.0)
    for pid, title in rows:
        engine.add(pid, title)
    engine.finalize()

    titles = [t for _, t in rows]
    queries = random.sample(titles, n_queries)

    # Warm-up
    for q in queries[:5]:
        engine.search(q, confidence=0.95, top_k=5)

    times = []
    for q in queries:
        t0 = time.perf_counter()
        engine.search(q, confidence=0.95, top_k=5)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    stats = engine.get_stats()

    report = {
        "titles_indexed": len(rows),
        "queries": n_queries,
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "tfidf_avg_ms": float(stats['tfidf_avg_ms']),
        "minilm_avg_ms": float(stats['minilm_avg_ms']),
        "tfidf_runs": int(stats['tfidf_runs']),
        "minilm_runs": int(stats['minilm_runs']),
        "agreement_rate": float(stats['agreement_rate']),
        "rectifications": int(stats['rectifications']),
        "minilm_name": getattr(engine, 'minilm_name', 'unknown'),
    }

    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "unison_benchmark_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n=== Unison Engine Benchmark ===")
    print(f"Titles indexed: {len(rows)}")
    print(f"Queries: {n_queries}")
    print(f"P50 latency: {report['p50_ms']:.2f} ms")
    print(f"P95 latency: {report['p95_ms']:.2f} ms")
    print(f"Mean latency: {report['mean_ms']:.2f} ms")
    print(f"TF-IDF avg: {report['tfidf_avg_ms']:.2f} ms")
    print(f"MiniLM avg: {report['minilm_avg_ms']:.2f} ms")
    print(f"TF-IDF runs: {report['tfidf_runs']}")
    print(f"MiniLM runs: {report['minilm_runs']}")
    print(f"Agreement rate: {report['agreement_rate']:.2%}")
    print(f"Rectifications: {report['rectifications']}")
    print(f"MiniLM name: {report['minilm_name']}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main(100)
