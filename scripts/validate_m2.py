#!/usr/bin/env python3
"""validate_m2.py — M2 throughput + zero-downtime validation.

Runs the Rust validate_m2 binary against the same 15K-record input as M1,
reports baseline vs orchestrator QPS, latency percentiles, and swap-test
results.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_BIN = ROOT / "data" / "validate_m1_input.bin"
BINARY = ROOT / "target" / "release" / "validate_m2"

# Pass criteria.
SPEEDUP_MIN = 2.0
SWAPS_MIN = 10


def run_validation() -> dict:
    print("Building release binary ...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "validate_m2"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    if not INPUT_BIN.exists():
        print(f"ERROR: input binary not found: {INPUT_BIN}")
        print("Run scripts/validate_m1.py first to generate it.")
        sys.exit(1)

    print("Running M2 throughput validation ...")
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
    metrics = run_validation()
    base = metrics["baseline"]
    orch = metrics["orchestrator"]
    swap = metrics["swap_test"]
    speedup = metrics["speedup"]

    print("\n" + "=" * 60)
    print("M2 VALIDATION REPORT")
    print("=" * 60)
    print(f"Records loaded:     {metrics['records_loaded']:,}")
    print()
    print("M1-style synchronous baseline:")
    print(f"  Queries:          {base['queries']:,}")
    print(f"  QPS:              {base['qps']:.1f}")
    print(f"  P50 latency:      {base['p50_us']:.1f} μs")
    print(f"  P99 latency:      {base['p99_us']:.1f} μs")
    print()
    print("M2 temporal orchestrator:")
    print(f"  Queries:          {orch['queries']:,}")
    print(f"  QPS:              {orch['qps']:.1f}")
    print(f"  P50 latency:      {orch['p50_us']:.1f} μs")
    print(f"  P99 latency:      {orch['p99_us']:.1f} μs")
    print()
    print(f"Speedup:            {speedup:.2f}x")
    print()
    print("Zero-downtime swap test:")
    print(f"  Swaps performed:  {swap['swaps']}")
    print(f"  Reads during:     {swap['reads_during_swaps']:,}")
    print(f"  Final version:    {swap['final_version']}")
    print("=" * 60)

    ok = True
    if speedup < SPEEDUP_MIN:
        print(f"FAIL: speedup {speedup:.2f}x < {SPEEDUP_MIN}x")
        ok = False
    if swap["swaps"] < SWAPS_MIN:
        print(f"FAIL: swaps {swap['swaps']} < {SWAPS_MIN}")
        ok = False

    if ok:
        print("M2 VALIDATION PASSED")
    else:
        print("M2 VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
