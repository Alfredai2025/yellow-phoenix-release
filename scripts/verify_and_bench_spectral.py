# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Build spectral index, verify search hook, benchmark 100 queries."""
import sys, os, time, random, statistics, json, argparse
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from yp_bridge import RustBridge
import yp_engine
from yp_engine import YPEngine


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p / 100.0)
    return s[min(idx, len(s) - 1)]


def build_spectral_index(bridge, hash_source, max_n=0):
    """Build a spectral 512 index over hash_source {pid: 64-byte hash}.

    Uses internal numeric ids and returns (handle_pid_map, build_ms).
    If max_n > 0 and len(hash_source) > max_n, a random subset is used.
    """
    items = list(hash_source.items())
    if max_n > 0 and max_n < len(items):
        items = random.sample(items, max_n)

    ids = []
    hash_bytes = bytearray()
    internal_to_pid = {}
    for internal_id, (pid, hbytes) in enumerate(items):
        ids.append(internal_id)
        internal_to_pid[internal_id] = pid
        if isinstance(hbytes, (bytes, bytearray)):
            hash_bytes.extend(bytes(hbytes)[:64].ljust(64, b'\x00'))
        else:
            hash_bytes.extend(bytes.fromhex(hbytes)[:64].ljust(64, b'\x00'))

    print(f"Building spectral index with k=64, n={len(ids):,}...")
    t0 = time.perf_counter()
    rc = bridge.spectral_512_build(bytes(hash_bytes), ids)
    build_ms = (time.perf_counter() - t0) * 1000.0

    if rc != 0:
        raise RuntimeError(f"spectral_512_build failed with code {rc}")

    return internal_to_pid, build_ms, len(ids)


def main():
    parser = argparse.ArgumentParser(description="Verify and benchmark Spectral 512")
    parser.add_argument("--spectral-n", type=int, default=0,
                        help="Number of papers to include in spectral index (0 = all)")
    parser.add_argument("--queries", type=int, default=100,
                        help="Number of benchmark queries")
    parser.add_argument("--top-k", type=int, default=10,
                        help="top_k passed to engine.search")
    parser.add_argument("--disable-exact-title", action="store_true", default=True,
                        help="Disable exact-title shortcut so the benchmark stresses hash/spectral retrieval")
    parser.add_argument("--output-dir", type=str, default="logs",
                        help="Directory for JSON report")
    parser.add_argument("--query-mode", type=str, default="prefix",
                        choices=["title", "prefix"],
                        help="'title' = full title as query; 'prefix' = first 3 words")
    parser.add_argument("--use-1m-corpus", action="store_true", default=True,
                        help="Point engine at data/paper_meta_arxiv_1m.pkl + data/paper_embeddings_arxiv_1m.npy")
    parser.add_argument("--candidate-topk", type=int, default=5000,
                        help="Number of hash candidates to keep before re-rank")
    args = parser.parse_args()

    # ── PHASE 1: Load YPEngine (with spectral auto-build disabled) ──
    print("=" * 60)
    print("PHASE 1: Loading YPEngine")
    print("=" * 60)

    # Optionally point the engine at the full 1M arxiv corpus instead of the
    # root 13k demo files.
    if args.use_1m_corpus:
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

        # Build a hash source aligned with the 1M corpus hashes/model so
        # query-time hashes match the indexed papers.
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

        custom_sources = [_build_1m_source()]
    else:
        custom_sources = None

    # Stub out some heavy/experimental new-module helpers so engine init finishes
    # in a reasonable time; spectral is built externally and injected afterwards.
    orig_build = YPEngine._build_spectral_512_index
    orig_load_unified = YPEngine._load_unified_engine
    YPEngine._build_spectral_512_index = lambda self: None
    YPEngine._load_unified_engine = lambda self: None

    RustBridge.ism_init = lambda self, *args: 0
    RustBridge.flat_insert = lambda self, *args: 0
    RustBridge.dynamic_mesh_insert = lambda self, *args: 0
    RustBridge.shard_insert = lambda self, *args: 0
    RustBridge.cascade_insert = lambda self, *args: 0

    try:
        engine = YPEngine(sources=custom_sources, verbose=False)
    finally:
        YPEngine._build_spectral_512_index = orig_build
        YPEngine._load_unified_engine = orig_load_unified
        if args.use_1m_corpus:
            np.load = _orig_np_load
            yp_engine._load_hashes = _orig_load_hashes

    print(f"Engine loaded: {len(engine.pids):,} papers, {len(engine.sources)} hash source(s)")

    # ── PHASE 2: Build Spectral Index from the active hash source ──
    print("\n" + "=" * 60)
    print("PHASE 2: Building Spectral 512 Index")
    print("=" * 60)

    b = RustBridge()
    hash_source = engine.sources[0].hashes if engine.sources else engine.paper_hashes
    print(f"Active hash source: {len(hash_source):,} papers")

    internal_to_pid, build_ms, n_indexed = build_spectral_index(
        b, hash_source, max_n=args.spectral_n
    )
    print(f"BUILD SUCCESS: {n_indexed:,} papers in {build_ms:,.1f} ms ({build_ms/1000.0:,.1f}s)")
    print(f"Handle active: {b.spectral_512_handle is not None}")

    # Inject the externally-built handle and id map into the engine.
    engine.rust.spectral_512_handle = b.spectral_512_handle
    engine._spectral_512_id_to_pid = internal_to_pid
    print(f"Injected spectral handle: {engine.rust.spectral_512_handle is not None}")

    # ── PHASE 3: Quick Query Test ──
    print("\n" + "=" * 60)
    print("PHASE 3: Quick Query Test")
    print("=" * 60)

    first_hash = next(iter(hash_source.values()))
    if isinstance(first_hash, str):
        first_hash = bytes.fromhex(first_hash)
    query_hash = bytes(first_hash)[:64].hex()
    t0 = time.perf_counter()
    results = b.spectral_512_query(query_hash, top_k=10)
    q_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Query returned {len(results)} results in {q_ms:.2f} ms")
    if results:
        top_pid = internal_to_pid.get(results[0][0], results[0][0])
        print(f"Top result: internal_id={results[0][0]}, paper_id={top_pid}, score={results[0][1]:.4f}")

    # ── PHASE 4: YPEngine Search Integration ──
    print("\n" + "=" * 60)
    print("PHASE 4: YPEngine Search Integration")
    print("=" * 60)

    engine.search_cfg["candidate_topk"] = args.candidate_topk
    if args.disable_exact_title:
        engine._exact_title_index = {}
        print("Disabled exact-title shortcut for benchmark")

    search_result = engine.search("neural network architecture", top_k=args.top_k)
    stage_times = getattr(engine, '_last_stage_times', {})

    print(f"Search returned {len(search_result)} results")
    print(f"Stage times: { {k: f'{v:.2f}ms' for k, v in stage_times.items()} }")
    spectral_ms = stage_times.get('spectral', 0.0)
    print(f"Spectral stage fired: {spectral_ms > 0.0}")

    # ── PHASE 5: 100-Query Benchmark ──
    print("\n" + "=" * 60)
    print("PHASE 5: Benchmark")
    print("=" * 60)

    eligible = [
        (pid, title) for pid, title in engine.papers.items()
        if len(title) > 10 and pid in hash_source
    ]
    sample = random.sample(eligible, min(args.queries, len(eligible)))

    queries = []
    for pid, title in sample:
        if args.query_mode == "prefix":
            words = title.split()
            q = " ".join(words[:3]) if len(words) >= 3 else title
        else:
            q = title
        queries.append((pid, q))

    print(f"Benchmarking {len(queries)} queries (mode={args.query_mode}, top_k={args.top_k})...")

    def bench_mode(engine, queries, mode):
        latencies = []
        cand_counts = []
        r1_hits = 0
        r5_hits = 0

        orig_handle = engine.rust.spectral_512_handle
        if mode == "without":
            engine.rust.spectral_512_handle = None

        # Ensure candidate gate is wide enough for a fair comparison.
        engine.search_cfg["candidate_topk"] = args.candidate_topk

        # Clear per-query embedding cache so both modes pay the encode cost.
        engine._query_emb_cache.clear()

        try:
            for pid, query in queries:
                t0 = time.perf_counter()
                results = engine.search(query, top_k=args.top_k)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                cand_counts.append(len(getattr(engine, '_last_candidates', [])))

                if results:
                    pids = [r[1][0] for r in results]
                    if pids[0] == pid:
                        r1_hits += 1
                    if pid in pids[:5]:
                        r5_hits += 1
        finally:
            if mode == "without":
                engine.rust.spectral_512_handle = orig_handle

        n = len(queries)
        return {
            "r1": r1_hits / n,
            "r5": r5_hits / n,
            "latency_p50": percentile(latencies, 50),
            "latency_p95": percentile(latencies, 95),
            "latency_mean": statistics.mean(latencies),
            "cand_mean": statistics.mean(cand_counts) if cand_counts else 0.0,
        }

    print("Running WITH spectral...")
    with_r = bench_mode(engine, queries, "with")
    print("Running WITHOUT spectral...")
    without_r = bench_mode(engine, queries, "without")

    print("\n" + "-" * 70)
    print(f"{'Metric':20s} {'With Spectral':>15s} {'Without Spectral':>17s} {'Delta':>12s}")
    print("-" * 70)
    for k in ["r1", "r5", "latency_p50", "latency_p95", "latency_mean", "cand_mean"]:
        w, wo = with_r[k], without_r[k]
        print(f"{k:20s} {w:15.2f} {wo:17.2f} {w-wo:+12.2f}")

    r1_gain_pp = (with_r["r1"] - without_r["r1"]) * 100.0
    lat_cost = with_r["latency_p50"] - without_r["latency_p50"]

    print("-" * 70)
    if r1_gain_pp > 0:
        print(f"VERDICT: R@1 improves by {r1_gain_pp:.1f} pp at +{lat_cost:.1f} ms median cost")
        if lat_cost < 5:
            print("         → KEEP ACTIVE")
        elif lat_cost < 20:
            print("         → MARGINAL")
        else:
            print("         → TOO SLOW")
    else:
        print(f"VERDICT: No R@1 gain ({r1_gain_pp:+.1f} pp). MUSEUM IT.")

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spectral_n": n_indexed,
        "n_queries": len(queries),
        "top_k": args.top_k,
        "query_mode": args.query_mode,
        "build_ms": build_ms,
        "with_spectral": with_r,
        "without_spectral": without_r,
        "r1_gain_pp": r1_gain_pp,
        "latency_cost_median_ms": lat_cost,
    }
    out = os.path.join(
        args.output_dir,
        f"bench_spectral_{n_indexed}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
