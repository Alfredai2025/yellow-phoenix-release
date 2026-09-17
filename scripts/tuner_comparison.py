#!/usr/bin/env python3
"""Run Python tuner and Rust tuner side-by-side for shadow-mode comparison."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nightly_autopoiesis import AutopoiesisLoop
from yp_bridge import SelfTuner


def run_both():
    py = AutopoiesisLoop(scale="1m")
    rust = SelfTuner(threshold_us=600.0)

    # Run observation
    obs = py.observe(n_queries=100)
    print(f"[OBSERVE] P50={obs['p50_us']:.1f} µs  degraded={obs['degraded']}")

    # Feed same data to Rust
    rust_diag = rust.submit_observation(
        {
            "p50_us": obs["p50_us"],
            "p95_us": obs["p95_us"],
            "p99_us": obs["p99_us"],
            "avg_us": obs["avg_us"],
            "degraded": obs["degraded"],
            "ef": obs["ef"],
        }
    )
    print(f"[RUST]    phase={rust_diag['phase']}  status={rust_diag.get('status')}  action={rust_diag.get('action')}")

    # Python diagnosis
    py_diag = py.diagnose(obs)
    print(f"[PYTHON]  status={py_diag['status']}  action={py_diag.get('action')}")

    # Compare
    agreement = rust_diag.get("action") == py_diag.get("action")
    print(f"[COMPARE] {'AGREEMENT' if agreement else 'DISAGREEMENT'}")

    # Log
    entry = {
        "timestamp": time.time(),
        "observation": obs,
        "rust_diagnosis": rust_diag,
        "python_diagnosis": py_diag,
        "agreement": agreement,
    }
    with open("logs/tuner_comparison.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    # If degraded, run Rust proposal through Python evaluation
    if rust_diag.get("action"):
        prop = rust.propose()
        print(f"[RUST]    proposes ef={prop.get('proposed_ef')}")
        eval_result = py.propose_and_evaluate(py_diag)
        rust.submit_evaluation(eval_result["applied"], eval_result.get("new_ef", obs["ef"]))
        print(f"[APPLIED] {eval_result}")


if __name__ == "__main__":
    run_both()
