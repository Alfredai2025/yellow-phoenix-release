#!/usr/bin/env python3
"""10K Paraphrased Query Benchmark — tests diversity + beacon routing"""
import os, sys, time, random, sqlite3, statistics, re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault("YP_BASE_DIR", str(BASE))

from yp_bridge import YPEngine

DB_PATH = BASE / "data" / "phoenix_arxiv_1m.db"
SAMPLE = 10000
TOP_K = 5

def paraphrase(title: str) -> str:
    """Simple title paraphrase: drop last 2-4 words, shuffle order, or truncate."""
    words = title.split()
    if len(words) <= 3:
        return title
    # Strategy: drop last 30-50% of words (simulates partial recall)
    keep = max(2, int(len(words) * random.uniform(0.4, 0.7)))
    return " ".join(words[:keep])

def main():
    engine = YPEngine(db_path=str(DB_PATH))
    print("Pre-warming 1M HNSW fallback...")
    engine.build_hnsw()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title FROM papers WHERE title IS NOT NULL AND title != '' LIMIT 500000")
    rows = cur.fetchall()
    conn.close()

    sample = min(SAMPLE, len(rows))
    queries = [(pid, paraphrase(title), title) for pid, title in random.sample(rows, sample)]

    latencies, empty, beacon_hits, total_res = [], 0, 0, 0
    exact_fallback = 0

    print(f"Running {sample:,} PARAPHRASED queries against 1M arXiv...")
    t0 = time.time()
    for i, (pid, query, original) in enumerate(queries):
        t1 = time.perf_counter()
        try:
            res = engine.search_with_sah(query, k=TOP_K)
        except Exception as e:
            print(f"  FAIL query {i}: {e}")
            continue
        ms = (time.perf_counter() - t1) * 1000
        latencies.append(ms)

        results = res.get("results", []) if isinstance(res, dict) else res
        if not results:
            empty += 1
        else:
            total_res += len(results)

        source = res.get("source", "") if isinstance(res, dict) else ""
        if source == "beacon_fast":
            beacon_hits += 1

        # Check if exact title still matched despite paraphrase
        if any(r.get("pid") == pid for r in results if isinstance(r, dict)):
            exact_fallback += 1

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1:,} / {sample:,}  ({elapsed:.0f}s)")

    latencies.sort()
    n = len(latencies)
    if n == 0:
        print("No successful queries.")
        return

    print(f"\n{'='*60}")
    print(f"10K PARAPHRASED QUERY RESULTS")
    print(f"{'='*60}")
    print(f"Queries executed : {n:,}")
    print(f"Empty results    : {empty} ({empty/n*100:.2f}%)")
    print(f"Beacon hits      : {beacon_hits} ({beacon_hits/n*100:.2f}%)")
    print(f"Avg results/q    : {total_res/n:.2f}")
    print(f"Exact match rec. : {exact_fallback} ({exact_fallback/n*100:.2f}%)")
    print(f"Latency P50      : {latencies[n//2]:.2f} ms")
    print(f"Latency P95      : {latencies[int(n*0.95)]:.2f} ms")
    print(f"Latency P99      : {latencies[int(n*0.99)]:.2f} ms")
    print(f"Latency Avg      : {statistics.mean(latencies):.2f} ms")
    print(f"Latency Min/Max  : {latencies[0]:.2f} / {latencies[-1]:.2f} ms")
    print(f"Total wall time  : {time.time()-t0:.1f} s")
    print(f"Throughput       : {n/(time.time()-t0):.0f} qps")

if __name__ == "__main__":
    main()
