#!/usr/bin/env python3
"""M4.1 Semantic Cache benchmark.

Uses the real SemanticCache class and Rust FFI confidence gate from yp_bridge.py.
The "full pipeline" on cache miss uses sklearn TF-IDF over paper titles (MiniLM
is not available offline in this environment).  Latencies are therefore real,
just produced by the TF-IDF backend rather than a neural model.
"""

import os
import random
import sqlite3
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from yp_bridge import SemanticCache, PythonSemanticIndex, hash_text

try:
    from yp_bridge import RustBridge
    RUST = RustBridge()
except Exception as e:
    print(f"[warn] RustBridge could not be loaded: {e}")
    RUST = None

DB_PATH = os.path.join(ROOT, "data/phoenix_arxiv_1m.db")
QUERIES = 100
UNIQUE = 20
REPEATS = QUERIES // UNIQUE
CONFIDENCE_THRESHOLD = 0.99


def _load_titles(limit: int = 5000):
    """Load paper titles from the knowledge DB."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM papers WHERE title IS NOT NULL AND length(title) > 3 LIMIT ?", (limit,))
    rows = [(row[0], row[1]) for row in cur.fetchall()]
    conn.close()
    return rows


def _build_index(rows):
    """Build a TF-IDF semantic index over titles."""
    idx = PythonSemanticIndex()
    for pid, title in rows:
        idx.add(pid, title)
    idx.finalize()
    return idx


def _fast_hash(text: str) -> bytes:
    """Fast deterministic 128-bit prefix used for the confidence gate."""
    return hash_text(text, 256)[:16]


def _confidence(pap_128: bytes) -> float:
    """Rust FFI confidence for the hash prefix."""
    if RUST is None:
        return 0.0
    try:
        return RUST.confidence_score_fast(pap_128)
    except Exception:
        return 0.0


def main():
    print("Loading titles and building TF-IDF index...")
    rows = _load_titles()
    if len(rows) < UNIQUE:
        print(f"[error] Need at least {UNIQUE} titles, got {len(rows)}")
        sys.exit(1)
    index = _build_index(rows)

    # Use a subset of real titles as queries, repeated to generate cache hits.
    unique_queries = [rows[i][1] for i in range(UNIQUE)]
    queries = []
    for _ in range(REPEATS):
        queries.extend(unique_queries)
    random.shuffle(queries)

    cache = SemanticCache(max_size=1000)
    cached_times = []
    no_cache_times = []

    # With-cache run.
    for q in queries:
        t0 = time.perf_counter()
        cached, cached_pap = cache.get(q)
        if cached is not None:
            pap = cached_pap or _fast_hash(q)
            if _confidence(pap) >= CONFIDENCE_THRESHOLD:
                result = cached
            else:
                result = index.query(q, top_k=10)
                cache.put(q, pap, result)
        else:
            pap = _fast_hash(q)
            result = index.query(q, top_k=10)
            cache.put(q, pap, result)
        cached_times.append((time.perf_counter() - t0) * 1000.0)

    # Without-cache baseline: every query runs TF-IDF.
    for q in queries:
        t0 = time.perf_counter()
        index.query(q, top_k=10)
        no_cache_times.append((time.perf_counter() - t0) * 1000.0)

    def summarize(times):
        times.sort()
        n = len(times)
        return {
            "mean": statistics.mean(times),
            "p50": times[n // 2],
            "p95": times[int(n * 0.95)],
            "p99": times[int(n * 0.99)],
        }

    s_cache = summarize(cached_times)
    s_no = summarize(no_cache_times)

    print("\nM4.1 Semantic Cache Benchmark")
    print(f"  queries: {QUERIES} ({UNIQUE} unique, each repeated {REPEATS}x)")
    print(f"  cache hit rate: {cache.hit_rate()*100:.1f}%")
    print(f"  without cache — mean: {s_no['mean']:.2f}ms | P50: {s_no['p50']:.2f}ms | P95: {s_no['p95']:.2f}ms")
    print(f"  with cache    — mean: {s_cache['mean']:.2f}ms | P50: {s_cache['p50']:.2f}ms | P95: {s_cache['p95']:.2f}ms")
    if s_cache["p50"] > 0:
        print(f"  speedup (P50): {s_no['p50'] / s_cache['p50']:.1f}x")

    import json
    print(json.dumps({
        "queries": QUERIES,
        "unique": UNIQUE,
        "cache_hit_rate": cache.hit_rate(),
        "without_cache_ms": s_no,
        "with_cache_ms": s_cache,
        "speedup_p50": s_no["p50"] / s_cache["p50"] if s_cache["p50"] > 0 else None,
    }))


if __name__ == "__main__":
    main()
