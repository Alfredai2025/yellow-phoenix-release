#!/usr/bin/env python3
"""
head_to_head.py — Run the existing YP and FAISS benchmark scripts on this machine
and print a side-by-side summary.

NOTE: The scripts operate on different datasets today:
  - YP uses real paper hashes from data/phoenix_arxiv_1m.db (~12.7k records)
  - FAISS uses synthetic random vectors (default 1M)
This wrapper is honest about that mismatch. For a true apples-to-apples comparison,
both systems need to query the same vectors.
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_yp():
    print("=== Running YP benchmark: scripts/benchmark_m3_4_sharded.py ===")
    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "benchmark_m3_4_sharded.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("YP benchmark FAILED:", proc.stderr[-500:])
        return None

    report_path = ROOT / "data" / "benchmark_m3_4_sharded.json"
    report = json.loads(report_path.read_text())
    report["wall_time_s"] = round(time.time() - start, 1)
    return report


def run_faiss(scale=1_000_000, nq=1000):
    print(f"=== Running FAISS benchmark: benchmark_faiss_flat_mmap.py --scale {scale} --nq {nq} ===")
    start = time.time()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmark_faiss_flat_mmap.py"), "--scale", str(scale), "--nq", str(nq)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("FAISS benchmark FAILED:", proc.stderr[-500:])
        return None

    files = glob.glob(str(ROOT / "benchmark_results" / "faiss_flat_mmap_*.json"))
    if not files:
        print("No FAISS result JSON found")
        return None
    latest = max(files, key=os.path.getmtime)
    report = json.loads(Path(latest).read_text())
    report["wall_time_s"] = round(time.time() - start, 1)
    return report


def main():
    yp = run_yp()
    faiss = run_faiss()
    if yp is None or faiss is None:
        sys.exit(1)

    speedup = None
    if yp.get("p50_us") and faiss.get("p50_us"):
        speedup = round(faiss["p50_us"] / yp["p50_us"], 2)

    summary = {
        "machine_note": "Same MacBook Pro M3 Pro, same Python process",
        "yp": {
            "script": "scripts/benchmark_m3_4_sharded.py",
            "dataset": "data/phoenix_arxiv_1m.db real papers (~12.7k)",
            "p50_us": round(yp.get("p50_us", 0), 2),
            "p99_us": round(yp.get("p99_us", 0), 2) if "p99_us" in yp else None,
            "r@1": round(yp.get("r@1", 0), 4),
            "queries": yp.get("queries_run"),
            "wall_time_s": yp.get("wall_time_s"),
        },
        "faiss": {
            "script": "benchmark_faiss_flat_mmap.py",
            "dataset": f"synthetic random vectors ({faiss['n_vectors']:,})",
            "p50_us": round(faiss.get("p50_us", 0), 2),
            "qps": round(faiss.get("qps", 0), 1),
            "r@1_self": round(faiss.get("recall", 0), 4),
            "build_s": round(faiss.get("build_s", 0), 2),
            "wall_time_s": faiss.get("wall_time_s"),
        },
        "speedup_p50": speedup,
        "caveat": "Datasets differ; this compares script behavior, not identical data.",
    }

    out_path = ROOT / "data" / "head_to_head_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70)
    print(f"Full summary saved: {out_path}")


if __name__ == "__main__":
    main()
