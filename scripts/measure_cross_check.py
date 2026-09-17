#!/usr/bin/env python3
"""
YP-CRT-CROSS-CHECK-v1.1 measurement protocol.

One script, one before/after table, computed verdict.

Steps:
  1. Load mesh_12758.bin, build UnifiedEngine (fast path).
  2. Measure baseline fast-path R@1/5/10 and latency.
  3. Build CRT cross-check bank at insert/build time.
  4. Measure bank quality: agreement rate, banked-entry precision.
  5. Re-measure fast-path metrics with identical queries/seeds.
  6. Print verdict table and verdict (KEEP / MUSEUM / REVERT).

Output: logs/cross_check_YYYYMMDD_HHMMSS.txt
"""
import argparse
import ctypes
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from yp_bridge import RustBridge


# ---------------------------------------------------------------------------
# Spec constants (v1.1)
# ---------------------------------------------------------------------------
R1_GAIN_THRESHOLD = 0.030  # +3.0 percentage points
LATENCY_REGRESSION_FRACTION = 0.05  # +5%
BANK_PRECISION_THRESHOLD = 0.50  # 50%
AGREEMENT_RATE_MIN = 0.001  # 0.1%
AGREEMENT_RATE_MAX = 0.50  # 50%
N_LATENCY_QUERIES = 1000
BANK_PRECISION_SAMPLE = 500
SEED = 43


# ---------------------------------------------------------------------------
# Mesh load
# ---------------------------------------------------------------------------
def parse_ypms_mesh(path):
    """Parse mesh file in YPMS format (ffi.rs global mesh save format)."""
    with open(path, "rb") as f:
        header = f.read(8)
        if header[:4] != b"YPMS":
            raise ValueError(f"{path} is not a YPMS mesh file")
        version = struct.unpack("<I", header[4:8])[0]
        coarse_count = struct.unpack("<Q", f.read(8))[0]
        coarse = []
        for _ in range(coarse_count):
            pap = f.read(16)
            id_ = struct.unpack("<Q", f.read(8))[0]
            coarse.append((id_, pap))
        fine_count = struct.unpack("<Q", f.read(8))[0]
        fine = []
        for _ in range(fine_count):
            pap = f.read(32)
            id_ = struct.unpack("<Q", f.read(8))[0]
            fine.append((id_, pap))
    return coarse, fine


# ---------------------------------------------------------------------------
# Ground truth: Hamming nearest neighbours on 32-byte hashes
# ---------------------------------------------------------------------------
def build_ground_truth(hashes, top_ks=(1, 5, 10)):
    """For each slot, ground-truth top-k by Hamming distance on 32-byte hashes."""
    n = len(hashes)
    popcount = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    H = np.frombuffer(b''.join(hashes), dtype=np.uint8).reshape(n, 32)
    D = np.zeros((n, n), dtype=np.uint16)
    for i in range(n):
        xor_row = H[i] ^ H
        D[i] = popcount[xor_row].sum(axis=1)
    np.fill_diagonal(D, 0xFFFF)

    truth = {}
    for k in top_ks:
        top = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        top_d = D[rows, top]
        order = np.argsort(top_d, axis=1)
        truth[k] = top[rows, order]
    return truth


# ---------------------------------------------------------------------------
# Unified engine helpers
# ---------------------------------------------------------------------------
def build_unified_engine(bridge, coarse, fine):
    handle = bridge.unified_engine_new(shard_size=1000)
    if handle is None:
        raise RuntimeError("Failed to create UnifiedEngine")

    ids_128 = [id_ for (id_, pap) in coarse]
    paps_128 = [pap for (id_, pap) in coarse]
    ids_512 = [id_ for (id_, pap) in fine]
    paps_512 = [pap for (id_, pap) in fine]

    rc = bridge.unified_engine_build(handle, ids_128, paps_128, ids_512, paps_512)
    if rc != 0:
        raise RuntimeError(f"unified_engine_build returned {rc}")
    return handle


def measure_fast_path(bridge, handle, hashes, truth, n_queries=1000):
    """Measure R@k and latency via UnifiedEngine fast path."""
    n = len(hashes)
    rng = np.random.default_rng(SEED)
    q_indices = rng.choice(n, size=min(n_queries, n), replace=False)

    recalls = {1: [], 5: [], 10: []}
    latencies = []
    hits = 0  # non-empty result

    for idx in q_indices:
        q_hash = hashes[idx]
        q_128 = q_hash[:16]
        q_512 = q_hash

        t0 = time.perf_counter()
        result = bridge.unified_engine_query(handle, q_128, q_512, top_k=10)
        dt = time.perf_counter() - t0
        latencies.append(dt)

        if result:
            hits += 1

        # Exclude self from results (fast hash returns exact self match)
        result_ids = [int(r[1]) for r in result if int(r[1]) != idx]

        for k in [1, 5, 10]:
            true_set = set(truth[k][idx].tolist())
            hit = len(true_set.intersection(result_ids[:k]))
            recalls[k].append(hit / k)

    return {
        "recall@1": float(np.mean(recalls[1])),
        "recall@5": float(np.mean(recalls[5])),
        "recall@10": float(np.mean(recalls[10])),
        "hit_rate": hits / len(q_indices),
        "latency_p50_s": float(np.percentile(latencies, 50)),
        "latency_p99_s": float(np.percentile(latencies, 99)),
    }


def measure_bank_precision(bridge, handle, hashes, truth_top10, sample_size=500):
    """Sample banked links and check if target is in source's true Hamming top-10."""
    stats = bridge.cross_check_stats(handle)
    links = bridge.cross_check_links(handle)
    if not links:
        return 0.0, 0

    n = len(hashes)
    if len(links) > sample_size:
        rng = np.random.default_rng(SEED + 1)
        idxs = rng.choice(len(links), size=sample_size, replace=False)
        sample = [links[i] for i in idxs]
    else:
        sample = links

    true_by_src = {src: set(tops.tolist()) for src, tops in enumerate(truth_top10)}
    hits = 0
    for src, tgt in sample:
        src = int(src)
        tgt = int(tgt)
        if 0 <= src < n and 0 <= tgt < n and tgt in true_by_src[src]:
            hits += 1

    return float(hits / len(sample)), len(sample)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def compute_verdict(baseline, overlay, stats, precision):
    r1_delta = overlay["recall@1"] - baseline["recall@1"]
    latency_rel = (overlay["latency_p50_s"] - baseline["latency_p50_s"]) / baseline["latency_p50_s"] if baseline["latency_p50_s"] > 0 else 0.0
    agreement_rate = stats.get("agreement_rate", 0.0)

    # Dead-hypothesis sanity traps (pre-committed)
    if agreement_rate < AGREEMENT_RATE_MIN:
        return "MUSEUM", f"agreement rate {agreement_rate:.4%} < {AGREEMENT_RATE_MIN:.1%}: voters never agree"
    if agreement_rate > AGREEMENT_RATE_MAX:
        return "MUSEUM", f"agreement rate {agreement_rate:.4%} > {AGREEMENT_RATE_MAX:.0%}: blocks not independent"

    # K1: banked-entry precision is a required leg of KEEP
    if precision is None:
        return "MUSEUM", "banked-entry precision not measured"
    if precision < BANK_PRECISION_THRESHOLD:
        return "MUSEUM", f"banked-entry precision {precision:.1%} < {BANK_PRECISION_THRESHOLD:.0%}"

    # K2: KEEP also requires recall gain within latency budget
    if r1_delta >= R1_GAIN_THRESHOLD and latency_rel <= LATENCY_REGRESSION_FRACTION:
        return "KEEP", f"R@1 Δ {r1_delta:+.4f}, precision {precision:.1%}, latency Δ {latency_rel:+.2%}"

    # K3: partial gain
    if r1_delta > 0 and r1_delta < R1_GAIN_THRESHOLD:
        return "MUSEUM", f"R@1 improved {r1_delta:+.4f} but below +{R1_GAIN_THRESHOLD} threshold"

    # K4: flat/negative or other failure
    if r1_delta <= 0:
        return "MUSEUM", f"R@1 Δ {r1_delta:+.4f}: no recall gain from banked links"

    if latency_rel > LATENCY_REGRESSION_FRACTION:
        return "MUSEUM", f"P50 latency Δ {latency_rel:+.2%} exceeds {LATENCY_REGRESSION_FRACTION:.0%} budget"

    return "MUSEUM", "composite failure"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YP-CRT-CROSS-CHECK-v1.1 measurement")
    parser.add_argument("--mesh", default="mesh_12758.bin")
    parser.add_argument("--n-queries", type=int, default=N_LATENCY_QUERIES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else ROOT / "logs" / f"cross_check_{timestamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = []

    def emit(line):
        print(line)
        log.append(line)

    emit(f"YP-CRT-CROSS-CHECK-v1.1 measurement run")
    emit(f"Timestamp: {timestamp}")
    emit(f"Mesh: {args.mesh}")
    emit("")

    bridge = RustBridge()
    if "yp_cross_check_build" not in bridge.available:
        raise RuntimeError("Rust lib was not built with --features cross-check")

    # ------------------------------------------------------------------
    # Load mesh
    # ------------------------------------------------------------------
    emit("[load] Loading mesh...")
    coarse, fine = parse_ypms_mesh(ROOT / args.mesh)
    n_fine = len(fine)
    emit(f"[load] Coarse slots: {len(coarse)}, fine slots: {n_fine}")

    hashes = [pap for (_, pap) in fine]
    emit("[truth] Building Hamming ground truth...")
    t0 = time.time()
    truth = build_ground_truth(hashes, top_ks=(1, 5, 10))
    emit(f"[truth] done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Build UnifiedEngine and measure baseline
    # ------------------------------------------------------------------
    emit("")
    emit("=" * 60)
    emit("BASELINE (flag OFF — UnifiedEngine fast path)")
    emit("=" * 60)

    t0 = time.time()
    handle = build_unified_engine(bridge, coarse, fine)
    emit(f"[baseline] UnifiedEngine build took {time.time()-t0:.2f}s")

    baseline = measure_fast_path(bridge, handle, hashes, truth, args.n_queries)
    emit(f"[baseline] R@1 = {baseline['recall@1']:.4f}")
    emit(f"[baseline] R@5 = {baseline['recall@5']:.4f}")
    emit(f"[baseline] R@10 = {baseline['recall@10']:.4f}")
    emit(f"[baseline] hit rate = {baseline['hit_rate']:.4f}")
    emit(f"[baseline] P50 latency = {baseline['latency_p50_s']*1e6:.2f} µs")
    emit(f"[baseline] P99 latency = {baseline['latency_p99_s']*1e6:.2f} µs")

    # ------------------------------------------------------------------
    # Build cross-check bank
    # ------------------------------------------------------------------
    emit("")
    emit("=" * 60)
    emit("CROSS-CHECK (flag ON — build-time voters only)")
    emit("=" * 60)

    t0 = time.time()
    banked = bridge.cross_check_build(handle)
    build_time = time.time() - t0
    emit(f"[cross-check] build took {build_time:.2f}s ({build_time / (n_fine / 1000):.2f}s per 1k papers)")
    emit(f"[cross-check] banked links = {banked}")

    stats = bridge.cross_check_stats(handle)
    emit(f"[cross-check] prefixes = {stats.get('prefixes', 0)}")
    emit(f"[cross-check] theta_a = {stats.get('theta_a', 0)}, theta_b = {stats.get('theta_b', 0)}")
    emit(f"[cross-check] examined pairs = {stats.get('examined', 0)}")
    emit(f"[cross-check] agreed pairs = {stats.get('agreed', 0)}")
    emit(f"[cross-check] agreement rate = {stats.get('agreement_rate', 0.0):.4%}")

    precision, precision_n = measure_bank_precision(bridge, handle, hashes, truth[10], BANK_PRECISION_SAMPLE)
    emit(f"[cross-check] banked-entry precision = {precision:.1%} (n={precision_n})")

    # ------------------------------------------------------------------
    # Re-measure with cross-check
    # ------------------------------------------------------------------
    overlay = measure_fast_path(bridge, handle, hashes, truth, args.n_queries)
    emit(f"[cross-check] R@1 = {overlay['recall@1']:.4f}")
    emit(f"[cross-check] R@5 = {overlay['recall@5']:.4f}")
    emit(f"[cross-check] R@10 = {overlay['recall@10']:.4f}")
    emit(f"[cross-check] hit rate = {overlay['hit_rate']:.4f}")
    emit(f"[cross-check] P50 latency = {overlay['latency_p50_s']*1e6:.2f} µs")
    emit(f"[cross-check] P99 latency = {overlay['latency_p99_s']*1e6:.2f} µs")

    # ------------------------------------------------------------------
    # Verdict table
    # ------------------------------------------------------------------
    emit("")
    emit("=" * 60)
    emit("BEFORE/AFTER TABLE")
    emit("=" * 60)

    r1_delta = overlay["recall@1"] - baseline["recall@1"]
    latency_rel = (overlay["latency_p50_s"] - baseline["latency_p50_s"]) / baseline["latency_p50_s"] if baseline["latency_p50_s"] > 0 else 0.0
    agreement_rate = stats.get("agreement_rate", 0.0)

    def row(metric, base, over, delta, threshold):
        emit(f"{metric:<28} {base:>18} {over:>18} {delta:>18} {threshold:>18}")

    row("Metric", "Baseline", "Cross-Check", "Δ", "Threshold")
    emit("-" * 103)
    row("R@1", f"{baseline['recall@1']:.4f}", f"{overlay['recall@1']:.4f}",
        f"{r1_delta:+.4f}", f"≥ +{R1_GAIN_THRESHOLD}")
    row("R@5", f"{baseline['recall@5']:.4f}", f"{overlay['recall@5']:.4f}",
        f"{overlay['recall@5']-baseline['recall@5']:+.4f}", "report only")
    row("R@10", f"{baseline['recall@10']:.4f}", f"{overlay['recall@10']:.4f}",
        f"{overlay['recall@10']-baseline['recall@10']:+.4f}", "report only")
    row("P50 latency", f"{baseline['latency_p50_s']*1e6:.2f} µs", f"{overlay['latency_p50_s']*1e6:.2f} µs",
        f"{latency_rel:+.2%}", f"≤ +{LATENCY_REGRESSION_FRACTION:.0%}")
    row("P99 latency", f"{baseline['latency_p99_s']*1e6:.2f} µs", f"{overlay['latency_p99_s']*1e6:.2f} µs",
        "report only", "report only")
    row("Cheat sheet hit rate", f"{baseline['hit_rate']:.4f}", f"{overlay['hit_rate']:.4f}",
        f"{overlay['hit_rate']-baseline['hit_rate']:+.4f}", "report only")
    row("Cross entries banked", "—", f"{banked}", "—", "report only")
    row("Agreement rate", "—", f"{agreement_rate:.4%}", "—", f"{AGREEMENT_RATE_MIN:.1%}–{AGREEMENT_RATE_MAX:.0%}")
    row("Build time / 1k papers", "—", f"{build_time/(n_fine/1000):.2f}s", "—", "report only")
    row("Banked precision", "—", f"{precision:.1%} (n={precision_n})", "—", f"≥ {BANK_PRECISION_THRESHOLD:.0%}")

    if baseline["recall@1"] == 0.0:
        emit("")
        emit("DEGENERATE BASELINE WARNING: R@1 = 0.0000 because corpus papers are used")
        emit("as queries. The direct hash returns the paper itself; self is excluded, so")
        emit("the fast path returns no neighbours. R@1 Δ is therefore not meaningful.")
        emit("The banked-entry precision line is the honest signal test.")

    verdict, reason = compute_verdict(baseline, overlay, stats, precision)
    emit("")
    emit(f"FINAL VERDICT: {verdict}")
    emit(f"Reason: {reason}")

    # Tuning attempt if latency blew budget
    if verdict == "MUSEUM" and latency_rel > LATENCY_REGRESSION_FRACTION:
        emit("")
        emit("[tuning] Latency regression exceeded budget; one tuning attempt allowed.")
        emit("[tuning] Spec v1.1 does not implement threshold tuning in script; stopping.")

    # Cleanup
    bridge.cross_check_drop(handle)
    bridge.unified_engine_free(handle)

    emit("")
    emit(f"Raw log: {out_path}")
    with open(out_path, "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
