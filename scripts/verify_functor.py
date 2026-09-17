#!/usr/bin/env python3
"""
YP Functor Verification — Phase B (API wired)
Runs Rust math/unit tests and reports scaffold status.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "logs" / "functor_audit_results.json"

RESULTS = {
    "phase": "B",
    "status": "api_wired",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "measurements": {
        "f1": {"k1": None, "r2": None, "status": "ready_for_real_data"},
        "f2": {"mean_rank": None, "p95_rank": None, "status": "ready_for_real_data"},
        "f3": {"kendall_tau": None, "status": "ready_for_real_data"},
        "f4": {"delta": None, "status": "ready_for_real_data"},
    },
    "next_step": "Run measurements on a loaded mesh/HNSW/spectral index; see docs/functor_audit.md",
}


def run_rust_tests() -> bool:
    print("[*] Running Rust functor tests...")
    result = subprocess.run(
        ["cargo", "test", "--lib", "functor_bounds::tests", "--", "--nocapture"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    passed = result.returncode == 0
    print(result.stdout[-1200:] if len(result.stdout) > 1200 else result.stdout)
    if not passed:
        print("[!] Rust tests FAILED — fix before measuring real data.")
        print(result.stderr[-500:])
    return passed


def save_report():
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"[+] Report saved to {RESULTS_PATH}")


if __name__ == "__main__":
    print("=" * 60)
    print("YP FUNCTOR AUDIT — PHASE B (API WIRED)")
    print("=" * 60)

    if not run_rust_tests():
        sys.exit(1)

    print("\n[+] Math helpers and API wiring compile + pass tests.")
    print("    Next: load a real mesh/HNSW/spectral index and call the")
    print("    measurement functions from a binary or Python harness.")

    save_report()

    print("\n" + "=" * 60)
    print("STATUS BOARD")
    print("=" * 60)
    for k, v in RESULTS["measurements"].items():
        print(f"  {k.upper()}: {v['status']}")
