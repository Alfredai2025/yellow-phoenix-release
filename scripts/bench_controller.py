#!/usr/bin/env python3
"""Benchmark: Static vs Bang-Bang vs PID controller.

Two benchmarks:
  1. Simulated drift on 100K (fast, for quick regression checks).
  2. Real 1M corpus benchmark (slow, the production-relevant test).

Usage:
    source .venv/bin/activate
    python scripts/bench_controller.py              # runs both
    python scripts/bench_controller.py --sim-only    # simulated only
    python scripts/bench_controller.py --real-only   # real 1M only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.phoenix_bench_api import ScaleEngine

STEPS = 100
N_QUERIES_PER_STEP = 20
SIM_SCALE = "100k"
REAL_SCALE = "1m"
SETPOINT_US = 600.0


# ============================================================
# Simulated 100K benchmark (legacy)
# ============================================================
def simulate_workload(engine, embs, ef, nq, scale):
    idx = np.random.choice(len(embs), nq, replace=False)
    times = []
    index = engine._engines[scale][1]
    index.set_ef(ef)
    for i in idx:
        q = embs[i]
        t0 = time.perf_counter()
        engine.search_hybrid_by_embedding(q, k=5, scale=scale)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return float(np.percentile(times, 50))


def strategy_static(engine, embs, ef_history, p50_history):
    return 25


def strategy_bangbang(engine, embs, ef_history, p50_history):
    if not p50_history:
        return 25
    last_p50 = p50_history[-1]
    if last_p50 > 600:
        return 50
    elif last_p50 < 400:
        return 25
    return ef_history[-1]


def strategy_pid(engine, embs, ef_history, p50_history, Kp=0.5, Ki=0.1, Kd=0.05):
    if len(p50_history) < 2:
        return ef_history[-1] if ef_history else 25

    setpoint = 500.0
    error = p50_history[-1] - setpoint
    integral = sum(p50_history) - setpoint * len(p50_history)
    derivative = p50_history[-1] - p50_history[-2]
    delta = Kp * error + Ki * integral + Kd * derivative
    return int(np.clip(ef_history[-1] + delta, 10, 100))


def run_simulated_benchmark():
    print("Loading 100K engine for simulated benchmark...")
    engine = ScaleEngine()
    engine._get_engine(SIM_SCALE)
    embs = engine._engines[SIM_SCALE][0]

    results = []
    for name, fn in [
        ("static", strategy_static),
        ("bangbang", strategy_bangbang),
        ("pid", strategy_pid),
    ]:
        print(f"\nRunning {name}...")
        ef = 25
        ef_history = []
        p50_history = []
        violations = 0

        np.random.seed(42)

        for step in range(STEPS):
            p50 = simulate_workload(engine, embs, ef, N_QUERIES_PER_STEP, SIM_SCALE)
            p50 += step * 2.0  # simulated drift

            ef_history.append(ef)
            p50_history.append(p50)

            if p50 > 600:
                violations += 1

            ef = int(fn(engine, embs, ef_history, p50_history))

        results.append({
            "name": name,
            "cumulative_latency": float(sum(p50_history)),
            "violations": violations,
            "ef_changes": len([i for i in range(1, len(ef_history)) if ef_history[i] != ef_history[i - 1]]),
            "final_ef": ef,
            "mean_p50": float(np.mean(p50_history)),
        })
        print(f"  cumulative={results[-1]['cumulative_latency']:.0f}  "
              f"violations={violations}  ef_changes={results[-1]['ef_changes']}")

    print("\n" + "=" * 70)
    print(f"{'Simulated 100K':<12} {'Cum.latency':>12} {'Violations':>10} {'EF changes':>10} {'Mean P50':>10}")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:<12} {r['cumulative_latency']:>12.0f} {r['violations']:>10} "
              f"{r['ef_changes']:>10} {r['mean_p50']:>10.1f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_controller_sim.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved logs/bench_controller_sim.json")
    return results


# ============================================================
# Real 1M corpus benchmark
# ============================================================
class PIDController:
    def __init__(self, Kp=0.5, Ki=0.1, Kd=0.05, setpoint=600.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, measured_p50, dt=1.0):
        error = measured_p50 - self.setpoint
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -1000.0, 1000.0))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        control = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return float(np.clip(control, -50.0, 50.0))


def compute_exact_tops(embs, query_idx, k=10):
    """Exact top-k for each query index using full cosine."""
    exact = {}
    for idx in query_idx:
        q = embs[idx]
        sims = embs @ q
        sims[idx] = -np.inf
        exact[idx] = set(np.argsort(-sims)[:k])
    return exact


def measure_recall(engine, embs, probe_idx, exact_tops, k=10, scale=REAL_SCALE):
    """R@k of HNSW + re-rank vs exact cosine."""
    hits = 0
    total = 0
    for idx in probe_idx:
        results = engine.search_hybrid_by_embedding(embs[idx], k=k, scale=scale)
        hnsw_top = {int(pid) for score, (pid, _) in results[:k] if int(pid) != idx}
        if len(hnsw_top) == 0:
            continue
        hits += len(exact_tops[idx] & hnsw_top)
        total += k
    return hits / total if total > 0 else 0.0


def run_real_strategy(name, engine, embs, query_idx_pool, exact_tops,
                      probe_idx, steps=50, nq=20):
    """Run one control strategy on the real 1M index."""
    index = engine._engines[REAL_SCALE][1]
    initial_ef = 150  # Start well above the budget to create a tuning challenge
    ef = initial_ef
    index.set_ef(ef)

    ef_history = []
    p50_history = []
    violations = 0
    recall_history = []

    pid = PIDController(Kp=0.15, Ki=0.01, Kd=0.1, setpoint=SETPOINT_US)

    t_start = time.time()

    for step in range(steps):
        index.set_ef(ef)

        # Run a fresh sample of queries
        sample = np.random.choice(query_idx_pool, nq, replace=False)
        times = []
        for idx in sample:
            t0 = time.perf_counter()
            engine.search_hybrid_by_embedding(embs[idx], k=10, scale=REAL_SCALE)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        p50 = float(np.percentile(times, 50))

        ef_history.append(ef)
        p50_history.append(p50)
        if p50 > SETPOINT_US:
            violations += 1

        # Strategy-specific update
        if name == "static":
            new_ef = initial_ef
        elif name == "current_rule":
            # Mirrors the old nightly_autopoiesis.py logic
            if p50 > SETPOINT_US:
                new_ef = min(ef + 25, 400)
            else:
                new_ef = ef
        elif name == "pid":
            delta = pid.update(p50, dt=1.0)
            # Negative feedback: too slow -> lower ef; too fast -> raise ef.
            # Deadband: ignore tiny deltas from measurement noise.
            if abs(delta) < 5.0:
                delta = 0.0
            new_ef = int(np.clip(ef - delta, 10, 400))
        else:
            raise ValueError(name)

        ef = new_ef

        # Measure recall every 10 steps
        if step % 10 == 0 or step == steps - 1:
            recall_history.append({
                "step": step,
                "ef": ef,
                "r_at_10": measure_recall(engine, embs, probe_idx, exact_tops, k=10),
            })

    elapsed = time.time() - t_start

    final_recall = recall_history[-1]["r_at_10"] if recall_history else 0.0

    return {
        "name": name,
        "scale": REAL_SCALE,
        "steps": steps,
        "nq_per_step": nq,

        "initial_ef": initial_ef,
        "final_ef": ef,
        "cumulative_latency": float(sum(p50_history)),
        "mean_p50": float(np.mean(p50_history)),
        "violations": violations,
        "ef_changes": len([i for i in range(1, len(ef_history)) if ef_history[i] != ef_history[i - 1]]),
        "final_r_at_10": final_recall,
        "recall_history": recall_history,
        "elapsed_s": elapsed,
    }


def run_real_1m_benchmark(steps=50, nq=20):
    print("\n" + "=" * 70)
    print("REAL 1M CORPUS BENCHMARK")
    print("=" * 70)
    print("Loading 1M engine...")
    engine = ScaleEngine()
    engine._get_engine(REAL_SCALE)
    embs = engine._engines[REAL_SCALE][0]
    n = len(embs)
    print(f"Loaded {n:,} vectors.")

    np.random.seed(42)

    # Fixed pool of queries for latency measurement
    query_idx_pool = np.random.choice(n, 1000, replace=False)

    # Separate probe set for recall (smaller, exact top-10 computed once)
    probe_idx = np.random.choice(n, 100, replace=False)
    print("Computing exact top-10 for recall probe set...")
    exact_tops = compute_exact_tops(embs, probe_idx, k=10)

    # Baseline recall at initial ef
    index = engine._engines[REAL_SCALE][1]
    initial_ef = int(index.ef)
    baseline_recall = measure_recall(engine, embs, probe_idx, exact_tops, k=10)
    print(f"Baseline recall at ef={initial_ef}: R@10 = {baseline_recall:.3f}")

    results = []
    for name in ["static", "current_rule", "pid"]:
        print(f"\nRunning {name}...")
        r = run_real_strategy(name, engine, embs, query_idx_pool, exact_tops,
                              probe_idx, steps=steps, nq=nq)
        results.append(r)
        print(f"  cum_latency={r['cumulative_latency']:.0f}  "
              f"violations={r['violations']}  ef_changes={r['ef_changes']}  "
              f"final_ef={r['final_ef']}  final_R@10={r['final_r_at_10']:.3f}  "
              f"time={r['elapsed_s']:.1f}s")

    print("\n" + "=" * 70)
    print("REAL 1M RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<14} {'Cum.latency':>12} {'Violations':>10} {'EF changes':>10} "
          f"{'Final EF':>9} {'R@10':>7} {'Mean P50':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<14} {r['cumulative_latency']:>12.0f} {r['violations']:>10} "
              f"{r['ef_changes']:>10} {r['final_ef']:>9} {r['final_r_at_10']:>7.3f} "
              f"{r['mean_p50']:>10.1f}")

    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_controller_1m.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved logs/bench_controller_1m.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ANN controllers.")
    parser.add_argument("--sim-only", action="store_true", help="Run only simulated 100K benchmark")
    parser.add_argument("--real-only", action="store_true", help="Run only real 1M benchmark")
    parser.add_argument("--steps", type=int, default=50, help="Steps for 1M benchmark")
    parser.add_argument("--nq", type=int, default=50, help="Queries per step for 1M benchmark")
    args = parser.parse_args()

    if not args.real_only:
        run_simulated_benchmark()
    if not args.sim_only:
        run_real_1m_benchmark(steps=args.steps, nq=args.nq)


if __name__ == "__main__":
    main()
