#!/usr/bin/env python3
"""audit_thresholds.py — M3.4 CGT-style threshold audit.

Reads the M3 benchmark report and derives recommended thresholds for the
confidence gate and lazy activation. This is a lightweight audit helper;
full CGT proof generation would require a large query log and the CGT engine.
"""

import json
import math
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "benchmark_m3_report.json"


def recommend_confidence_gate(base_p50: float, m3_p50: float, r1: float) -> dict:
    """Recommend a confidence gate threshold given observed speedup and accuracy."""
    speedup = base_p50 / m3_p50 if m3_p50 > 0 else 1.0
    # If R@1 is still high, we can be aggressive (low threshold = more fast paths).
    # If R@1 dropped, we need a conservative (high) threshold.
    if r1 >= 0.99:
        recommended = 0.85
    elif r1 >= 0.97:
        recommended = 0.95
    elif r1 >= 0.95:
        recommended = 0.98
    else:
        recommended = 0.999

    return {
        "observed_speedup": round(speedup, 2),
        "recommended_confidence_threshold": recommended,
        "r1_at_recommendation": r1,
        "notes": (
            f"Speedup {speedup:.1f}x with R@1 {r1:.4f}. "
            f"Use confidence >= {recommended} to keep accuracy while maximizing fast-path hits."
        ),
    }


def recommend_lazy_activation(r1: float) -> dict:
    """Recommend per-feature skip thresholds for lazy activation."""
    # Conservative defaults: only skip a feature if historical data says it
    # almost never changes the ranking.
    return {
        "skip_spectral_if_useless_rate": 0.80,
        "skip_wedge_if_useless_rate": 0.85,
        "skip_hologram_if_useless_rate": 0.90,
        "notes": (
            "These are conservative starting points. As the self-learning table "
            "accumulates query history, CGT can tighten them with proven bounds."
        ),
    }


def main():
    if not BENCHMARK.exists():
        print(f"Benchmark report not found: {BENCHMARK}")
        print("Run: python scripts/benchmark_m3.py")
        return

    report = json.loads(BENCHMARK.read_text())
    base = report["baseline"]
    m3 = report["m3"]

    print("=" * 60)
    print("M3.4 THRESHOLD AUDIT")
    print("=" * 60)

    conf_gate = recommend_confidence_gate(base["p50_us"], m3["p50_us"], m3["r1"])
    lazy = recommend_lazy_activation(m3["r1"])

    print("\nConfidence Gate (M3.1):")
    for k, v in conf_gate.items():
        print(f"  {k}: {v}")

    print("\nLazy Activation (M3.2):")
    for k, v in lazy.items():
        print(f"  {k}: {v}")

    print("\nRecommended config:")
    config = {
        "fast_path_confidence_threshold": conf_gate["recommended_confidence_threshold"],
        "lazy_activation": lazy,
    }
    print(json.dumps(config, indent=2))

    # Save audit output.
    audit_path = BENCHMARK.parent / "threshold_audit.json"
    audit_path.write_text(json.dumps({"confidence_gate": conf_gate, "lazy_activation": lazy}, indent=2))
    print(f"\nAudit saved to: {audit_path}")


if __name__ == "__main__":
    main()
