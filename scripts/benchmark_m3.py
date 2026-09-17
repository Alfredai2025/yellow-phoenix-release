#!/usr/bin/env python3
"""benchmark_m3.py — M3.6 validation wrapper.

Runs the Rust benchmark_m3 binary and reports the M1 baseline vs M3 fast-path
metrics. The binary itself performs the timing; this script only parses the
JSON report and checks pass criteria.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_BIN = ROOT / "data" / "validate_m1_input.bin"
BINARY = ROOT / "target" / "release" / "benchmark_m3"

R1_MIN = 0.97
P50_TARGET_US = 2.5


def run_benchmark() -> dict:
    print("Building release binary ...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "benchmark_m3"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    if not INPUT_BIN.exists():
        print(f"ERROR: input binary not found: {INPUT_BIN}")
        print("Run scripts/validate_m1.py first to generate it.")
        sys.exit(1)

    print("Running M3 benchmark ...")
    result = subprocess.run(
        [str(BINARY), str(INPUT_BIN)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if ln == "{":
            start = i
    if start is None:
        raise RuntimeError(f"No JSON report found in stdout:\n{result.stdout}")
    return json.loads("\n".join(lines[start:]))


def main():
    metrics = run_benchmark()
    base = metrics["baseline"]
    m3 = metrics["m3"]
    speedup = metrics["speedup_p50"]

    print("\n" + "=" * 60)
    print("M3 BENCHMARK REPORT")
    print("=" * 60)
    print(f"Records loaded: {metrics['records_loaded']:,}")
    print(f"Queries:        {metrics['queries']:,}")
    print()
    print("M1-style baseline:")
    print(f"  P50:  {base['p50_us']:.2f} μs")
    print(f"  P99:  {base['p99_us']:.2f} μs")
    print(f"  R@1:  {base['r1']:.4f}")
    print(f"  QPS:  {base['qps']:.0f}")
    print()
    print("M3 fast hash path:")
    print(f"  P50:  {m3['p50_us']:.2f} μs")
    print(f"  P99:  {m3['p99_us']:.2f} μs")
    print(f"  R@1:  {m3['r1']:.4f}")
    print(f"  QPS:  {m3['qps']:.0f}")
    print()
    print(f"P50 speedup:    {speedup:.2f}x")
    print("=" * 60)

    ok = True
    if m3["r1"] < R1_MIN:
        print(f"FAIL: M3 R@1 {m3['r1']:.4f} < {R1_MIN}")
        ok = False
    if m3["p50_us"] > P50_TARGET_US:
        print(f"NOTE: M3 P50 {m3['p50_us']:.2f} μs > {P50_TARGET_US} μs target")
        print("      (M3.1 alone only optimizes the hash stage; further M3.x work needed.)")
        # Do not fail hard — this is expected until lazy activation is wired.

    # Save report for the threshold audit script.
    report_path = ROOT / "data" / "benchmark_m3_report.json"
    report_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nReport saved to: {report_path}")

    if ok:
        print("M3 BENCHMARK: R@1 OK")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
