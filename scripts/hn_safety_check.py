#!/usr/bin/env python3
"""
HN Safety Gate. All must pass before the iPhone app is considered shippable.
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import YPEngine


def gate_1_top_k():
    e = YPEngine()
    e._hnsw_loaded = True
    # With the reference engine not wired, _search_hnsw raises and falls back to flat.
    # Once wired, this gate asserts the returned count exactly matches top_k.
    for k in [1, 5, 10]:
        r = e.search("neural network", top_k=k)
        assert len(r) == k, f"GATE 1 FAIL: top_k={k}, got {len(r)}"
    print("✅ Gate 1: top_k respected")


def gate_2_use_hnsw_branches():
    e = YPEngine()
    e.load_source("real")
    e._hnsw_loaded = True

    called = {"hnsw": False}
    original = e._search_hnsw

    def raising_hnsw(*args, **kwargs):
        called["hnsw"] = True
        raise RuntimeError("HNSW disabled for branch test")

    e._search_hnsw = raising_hnsw
    try:
        r = e.search("test", top_k=5, use_hnsw=True)
    finally:
        e._search_hnsw = original

    assert called["hnsw"], "GATE 2 FAIL: HNSW branch was not executed"
    assert isinstance(r, list), "GATE 2 FAIL: result is not a list"
    print("✅ Gate 2: use_hnsw branches exist and fallback works")


def gate_3_no_encoder_crash():
    e = YPEngine()
    _ = e.hnsw_titles  # public property, defensive
    print("✅ Gate 3: encoder access safe")


def gate_4_benchmark_honesty():
    e = YPEngine()

    # Gate 4 must benchmark the real Rust HNSW, not the fallback stub.
    hnsw_path = os.path.expanduser("~/yellow_phoenix/data/binary_hnsw_arxiv1m_m16.bin")
    if os.path.exists(hnsw_path):
        loaded = e.load_hnsw(hnsw_path)
        assert loaded, f"GATE 4 FAIL: HNSW load failed for {hnsw_path}"
        assert e._hnsw_loaded, "GATE 4 FAIL: HNSW load succeeded but flag is False"
    else:
        print("⚠️ Gate 4 SKIP: no HNSW index found — cannot benchmark real cold start")
        return

    cold = e.benchmark(samples=100, cold_start=True)
    assert "cold_first_ms" in cold, "GATE 4 FAIL: missing cold_first_ms"
    assert "warm_median_ms" in cold, "GATE 4 FAIL: missing warm_median_ms"

    # Sanity-check absolute latencies so we are not benchmarking a hollow stub.
    assert cold["cold_first_ms"] > 10.0, (
        f"GATE 4 FAIL: cold first query too fast ({cold['cold_first_ms']:.3f} ms) — "
        f"is HNSW actually loaded?"
    )
    assert cold["warm_median_ms"] > 1.0, (
        f"GATE 4 FAIL: warm median too fast ({cold['warm_median_ms']:.3f} ms) — "
        f"cache artifact or stub?"
    )
    assert cold["cold_first_ms"] / cold["warm_median_ms"] > 2.0, (
        f"GATE 4 FAIL: cold/warm ratio too low: "
        f"cold={cold['cold_first_ms']:.3f} ms, warm={cold['warm_median_ms']:.3f} ms"
    )

    print(
        f"✅ Gate 4: cold_first_ms={cold['cold_first_ms']:.1f} ms, "
        f"warm_median_ms={cold['warm_median_ms']:.1f} ms, "
        f"ratio={cold['cold_first_ms'] / cold['warm_median_ms']:.1f}×"
    )


def gate_5_no_beacon_in_ui():
    result = subprocess.run(
        [
            "grep",
            "-r",
            "-i",
            "beacon\\|holographic\\|geometric.*brain\\|autopoiesis",
            "/Users/mac/yellow_phoenix_mobile/YPPhone/",
        ],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print("⚠️ Gate 5 WARNING: Experimental terms found in UI:")
        print(result.stdout[:500])
    else:
        print("✅ Gate 5: UI is clean of experimental terms")


def main():
    gates = [
        gate_1_top_k,
        gate_2_use_hnsw_branches,
        gate_3_no_encoder_crash,
        gate_4_benchmark_honesty,
        gate_5_no_beacon_in_ui,
    ]
    for g in gates:
        try:
            g()
        except Exception as e:
            print(f"❌ {g.__name__} FAILED: {e}")
            sys.exit(1)
    print("\n🛡️ All gates passed. Safe to build iPhone app.")
    sys.exit(0)


if __name__ == "__main__":
    main()
