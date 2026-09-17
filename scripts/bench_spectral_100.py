# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""100-query benchmark: HNSW-only vs HNSW+Spectral512."""
import sys, os, time, random, statistics, json, sqlite3
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yp_engine
from yp_engine import YPEngine
from yp_bridge import RustBridge

N_QUERIES = 100
TOP_K = 10
SPECTRAL_N = 50000

# ── Align engine to the 1M arxiv corpus ──
yp_engine.META_PATH = Path("data/paper_meta_arxiv_1m.pkl")
_orig_np_load = np.load
def _patched_np_load(path, *a, **kw):
    if str(path) == "paper_embeddings.npy":
        path = "data/paper_embeddings_arxiv_1m.npy"
    return _orig_np_load(path, *a, **kw)
np.load = _patched_np_load

_orig_load_hashes = yp_engine._load_hashes
def _patched_load_hashes(path):
    name = Path(path).name
    if name in ("paper_hashes.pkl", "paper_hashes_itq_model_512.pkl"):
        return _orig_load_hashes("data/paper_hashes_arxiv_1m.pkl")
    return _orig_load_hashes(path)
yp_engine._load_hashes = _patched_load_hashes

def _build_1m_source():
    model_data = np.load("data/itq_model_512.npz")
    m = model_data.get("m", model_data.get("mean")).astype(np.float32)
    R = None
    for key in ("proj", "W", "R"):
        if key in model_data and model_data[key].shape[0] == m.shape[0]:
            R = model_data[key].astype(np.float32)
            break
    if R is None:
        raise ValueError("data/itq_model_512.npz has no usable projection")
    hashes = _orig_load_hashes("data/paper_hashes_arxiv_1m.pkl")

    def encode_from_emb(emb):
        x = emb - m
        x = x @ R
        bits = (x > 0).astype(np.uint8)
        return np.packbits(bits).tobytes()

    def encode_text(text):
        from yp_engine import get_model
        return encode_from_emb(get_model().encode(text, convert_to_numpy=True))

    return yp_engine.HashSource("active_itq", hashes, encode_text, encode_from_emb)

# ── Stub heavy/experimental init so startup is fast ──
YPEngine._build_spectral_512_index = lambda self: None
YPEngine._load_unified_engine = lambda self: None
RustBridge.flat_init = lambda self, *args: 0
RustBridge.ism_init = lambda self, *args: 0
RustBridge.dynamic_mesh_new = lambda self, *args: 0
RustBridge.shard_new = lambda self, *args: 0
RustBridge.cascade_new = lambda self, *args: 0
RustBridge.temporal_new = lambda self: 0
RustBridge.energy_new = lambda self: 0
RustBridge.mesh_snapshot_new = lambda self: 0
RustBridge.flat_insert = lambda self, *args: 0
RustBridge.dynamic_mesh_insert = lambda self, *args: 0
RustBridge.shard_insert = lambda self, *args: 0
RustBridge.cascade_insert = lambda self, *args: 0
RustBridge.ism_insert = lambda self, *args: 0

engine = YPEngine(sources=[_build_1m_source()], verbose=False)
print(f"Engine loaded: {len(engine.pids):,} papers")

# ── Build / inject 50k spectral index ──
hash_source = engine.sources[0].hashes
items = list(hash_source.items())
if len(items) > SPECTRAL_N:
    items = random.sample(items, SPECTRAL_N)
ids = []
hash_bytes = bytearray()
internal_to_pid = {}
for internal_id, (pid, hbytes) in enumerate(items):
    ids.append(internal_id)
    internal_to_pid[internal_id] = pid
    hash_bytes.extend(bytes(hbytes)[:64].ljust(64, b'\x00'))

b = RustBridge()
print(f"Building spectral index (k=64, n={len(ids):,})...")
t0 = time.perf_counter()
rc = b.spectral_512_build(bytes(hash_bytes), ids)
if rc != 0:
    raise RuntimeError(f"spectral_512_build failed with code {rc}")
print(f"Built in {(time.perf_counter() - t0) * 1000.0:,.1f} ms")
engine.rust.spectral_512_handle = b.spectral_512_handle
engine._spectral_512_id_to_pid = internal_to_pid

# ── Query loader ──
def load_queries(engine, n=100):
    conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title FROM papers WHERE title IS NOT NULL AND length(title) > 10 ORDER BY random() LIMIT ?",
        (n,)
    )
    rows = cursor.fetchall()
    conn.close()
    # Keep only papers actually in the engine
    valid = [(pid, title) for pid, title in rows if pid in engine.papers]
    return valid[:n]

def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p / 100.0)
    return s[min(idx, len(s) - 1)]

def benchmark_mode(engine, queries, mode="with_spectral"):
    latencies, candidates, spectral_times, hnsw_times, rerank_times = [], [], [], [], []
    r1_hits = 0

    orig_handle = None
    if mode == "without_spectral":
        orig_handle = engine.rust.spectral_512_handle
        engine.rust.spectral_512_handle = None

    try:
        for pid, title in queries:
            t0 = time.perf_counter()
            results = engine.search(title, top_k=TOP_K)
            latency_ms = (time.perf_counter() - t0) * 1000

            latencies.append(latency_ms)
            stage_times = getattr(engine, '_last_stage_times', {})
            candidates.append(len(getattr(engine, '_last_candidates', [])))
            spectral_times.append(stage_times.get('spectral', 0.0))
            hnsw_times.append(stage_times.get('hamming', 0.0))
            rerank_times.append(stage_times.get('rerank', 0.0))

            if results and results[0][1][0] == pid:
                r1_hits += 1
    finally:
        if mode == "without_spectral" and orig_handle is not None:
            engine.rust.spectral_512_handle = orig_handle

    return {
        "mode": mode,
        "n_queries": len(queries),
        "r1": r1_hits / len(queries) if queries else 0,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0,
        "cand_mean": statistics.mean(candidates) if candidates else 0,
        "spectral_time_mean_ms": statistics.mean(spectral_times) if spectral_times else 0,
        "hnsw_time_mean_ms": statistics.mean(hnsw_times) if hnsw_times else 0,
        "rerank_time_mean_ms": statistics.mean(rerank_times) if rerank_times else 0,
    }

print("=" * 60)
print("SPECTRAL 512 — 100 QUERY BENCHMARK")
print("=" * 60)

spectral_active = engine.rust.spectral_512_handle is not None
print(f"Spectral index active: {spectral_active}")
if not spectral_active:
    print("WARNING: Spectral not built. Check search_config.json.")
    sys.exit(1)

print(f"Loading {N_QUERIES} random paper titles...")
queries = load_queries(engine, N_QUERIES)
print(f"Using {len(queries)} valid queries")

print(f"Running {len(queries)} queries WITH spectral...")
with_r = benchmark_mode(engine, queries, "with_spectral")
print(f"Running {len(queries)} queries WITHOUT spectral...")
without_r = benchmark_mode(engine, queries, "without_spectral")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
for key in ["r1", "latency_p50_ms", "latency_p95_ms", "latency_mean_ms",
            "cand_mean", "spectral_time_mean_ms", "hnsw_time_mean_ms", "rerank_time_mean_ms"]:
    w, wo = with_r[key], without_r[key]
    delta = w - wo
    pct_c = (delta / wo * 100) if wo != 0 else 0
    print(f"  {key:30s}: with={w:8.2f}  without={wo:8.2f}  delta={delta:+8.2f} ({pct_c:+5.1f}%)")

report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "n_queries": len(queries),
    "with_spectral": with_r,
    "without_spectral": without_r,
}
out_path = f"logs/bench_spectral_100_{time.strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved: {out_path}")

r1_gain = (with_r["r1"] - without_r["r1"]) * 100
latency_cost = with_r["latency_p50_ms"] - without_r["latency_p50_ms"]
print("\n" + "=" * 60)
if r1_gain > 0:
    print(f"VERDICT: Spectral improves R@1 by {r1_gain:.1f}pp at +{latency_cost:.1f}ms latency")
    print("KEEP ACTIVE" if latency_cost < 5 else "MARGINAL" if latency_cost < 20 else "TOO SLOW")
else:
    print(f"VERDICT: No R@1 gain ({r1_gain:+.1f}pp). MUSEUM IT.")
print("=" * 60)
