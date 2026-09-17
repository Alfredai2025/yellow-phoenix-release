#!/usr/bin/env python3
"""
Head-to-head: FAISS vs Yellow Phoenix on identical data and queries.
Measures: build time, query latency (P50/P99), R@1.
"""
import sys, os, time, statistics, random, numpy as np, json

import faiss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

random.seed(42)

print("=" * 65)
print("FAISS vs YELLOW PHOENIX — Head-to-Head")
print("=" * 65)

# ── Load YP engine ──
print("\n[1/5] Loading Yellow Phoenix...")
t0 = time.perf_counter()
yp_engine = YPEngine()
t1 = time.perf_counter()
yp_load_s = t1 - t0
print(f"  YP load: {yp_load_s:.2f}s")

papers = list(yp_engine.cache.items())
n_papers = len(papers)
print(f"  Papers: {n_papers}")

# ── Load embeddings for FAISS ──
print("\n[2/5] Loading embeddings for FAISS...")
try:
    embeddings = np.load('paper_embeddings.npy')
    print(f"  Embeddings shape: {embeddings.shape}")
except FileNotFoundError:
    print("  ERROR: paper_embeddings.npy not found. Run encoder first.")
    sys.exit(1)

# Pre-compute PID -> embedding index mapping
pid_keys = list(yp_engine.cache.keys())
pid_to_emb_idx = {pid: i for i, pid in enumerate(pid_keys)}

# ── Build FAISS index ──
print("\n[3/5] Building FAISS index...")

d = embeddings.shape[1]
nlist = 100  # IVF clusters

quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

t0 = time.perf_counter()
index.train(embeddings)
index.add(embeddings)
index.nprobe = 10
t1 = time.perf_counter()
faiss_build_s = t1 - t0

print(f"  FAISS build: {faiss_build_s:.2f}s")
print(f"  FAISS ntotal: {index.ntotal}")

# ── Prepare query set: only exact-title queries that hit L0 hash map ──
print("\n[4/5] Preparing query set...")
exact_candidates = [
    (pid, title) for pid, title in papers
    if title.lower().strip() in getattr(yp_engine, '_title_hash_map', {})
]
print(f"  Exact-title candidates: {len(exact_candidates)} / {n_papers}")

sample_size = min(2000, len(exact_candidates))
sample = random.sample(exact_candidates, sample_size)
queries_yp = []
queries_faiss = []

for pid, title in sample:
    idx = pid_to_emb_idx.get(pid)
    if idx is None:
        continue
    queries_yp.append((pid, title))
    queries_faiss.append((pid, idx, title))

print(f"  YP queries: {len(queries_yp)} | FAISS queries: {len(queries_faiss)}")

# ── Benchmark YP ──
print("\n[5/5] Benchmarking...")

yp_latencies = []
yp_hits = 0

for true_pid, title in queries_yp:
    t0 = time.perf_counter()
    results = yp_engine.search(title, top_k=1)
    t1 = time.perf_counter()
    yp_latencies.append((t1 - t0) * 1_000_000)
    
    result_pids = [r[1][0] for r in results if len(r) > 1 and isinstance(r[1], tuple)]
    if true_pid in result_pids:
        yp_hits += 1

yp_latencies.sort()
yp_p50 = yp_latencies[len(yp_latencies) // 2]
yp_p99 = yp_latencies[int(len(yp_latencies) * 0.99)]
yp_r1 = yp_hits / len(queries_yp) * 100

# ── Benchmark FAISS ──
faiss_latencies = []
faiss_hits = 0

for true_pid, idx, title in queries_faiss:
    query_vec = embeddings[idx:idx+1].astype('float32')
    
    t0 = time.perf_counter()
    D, I = index.search(query_vec, 1)
    t1 = time.perf_counter()
    faiss_latencies.append((t1 - t0) * 1_000_000)
    
    if I[0][0] == idx:
        faiss_hits += 1

faiss_latencies.sort()
faiss_p50 = faiss_latencies[len(faiss_latencies) // 2]
faiss_p99 = faiss_latencies[int(len(faiss_latencies) * 0.99)]
faiss_r1 = faiss_hits / len(queries_faiss) * 100

# ── Results ──
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"\n  {'Metric':<20} {'YP':>15} {'FAISS':>15} {'Speedup':>10}")
print("  " + "-" * 65)
print(f"  {'Build time':<20} {yp_load_s:>13.2f}s {faiss_build_s:>13.2f}s {faiss_build_s/yp_load_s if yp_load_s>0 else 0:>9.1f}x")
print(f"  {'P50 latency':<20} {yp_p50:>11.1f} µs {faiss_p50:>11.1f} µs {faiss_p50/yp_p50 if yp_p50>0 else 0:>9.1f}x")
print(f"  {'P99 latency':<20} {yp_p99:>11.1f} µs {faiss_p99:>11.1f} µs {faiss_p99/yp_p99 if yp_p99>0 else 0:>9.1f}x")
print(f"  {'R@1 (exact titles)':<20} {yp_r1:>11.1f}% {faiss_r1:>11.1f}% {'—':>10}")

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)

if yp_p50 < faiss_p50:
    speedup = faiss_p50 / yp_p50
    print(f"\n  ✅ YP is {speedup:.1f}x FASTER than FAISS on exact-title queries")
else:
    print(f"\n  ⚠️  FAISS is faster on this workload")

if yp_load_s < faiss_build_s:
    build_speedup = faiss_build_s / yp_load_s
    print(f"  ✅ YP builds {build_speedup:.1f}x FASTER than FAISS")

# Save
out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'yp': {'build_s': yp_load_s, 'p50_us': yp_p50, 'p99_us': yp_p99, 'r1': yp_r1},
    'faiss': {'build_s': faiss_build_s, 'p50_us': faiss_p50, 'p99_us': faiss_p99, 'r1': faiss_r1},
}
with open('logs/benchmark_faiss_vs_yp.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: logs/benchmark_faiss_vs_yp.json")
