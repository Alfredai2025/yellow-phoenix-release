#!/usr/bin/env python3
"""
M3.7: Nightly autopoiesis loop wired to the real 1M embedding-HNSW index.

Run:
    source .venv/bin/activate
    python scripts/nightly_autopoiesis.py

The loop uses self-query embeddings (random samples from the indexed corpus) so
it does not require a live text encoder or network access. It still exercises
the full retrieval path: HNSW candidate fetch + exact cosine re-rank.

This version uses a PID controller to tune HNSW ef:
  - observes P50 latency
  - computes a smooth ef delta
  - applies it and logs proof-of-life
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from phoenix_bench_api import ScaleEngine

HISTORY_FILE = "logs/autopoiesis_history.json"
LOG_FILE = "logs/proof/autopoiesis_log.jsonl"
PID_STATE_FILE = "logs/autopoiesis_pid_state.json"
THRESHOLD_US = 600.0


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


class PIDController:
    """Discrete PID controller for HNSW ef tuning.

    Error sign convention:
        error = measured_p50 - setpoint
        positive error  -> latency is too high -> decrease ef
        negative error  -> latency is below budget -> increase ef
    """

    def __init__(self, Kp=0.5, Ki=0.1, Kd=0.05, setpoint=600.0,
                 integral=0.0, prev_error=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = integral
        self.prev_error = prev_error

    def update(self, measured_p50: float, dt: float = 1.0) -> float:
        error = measured_p50 - self.setpoint

        P = self.Kp * error

        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -1000.0, 1000.0))
        I = self.Ki * self.integral

        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        control = P + I + D
        return float(np.clip(control, -50.0, 50.0))

    def state(self):
        return {
            "integral": float(self.integral),
            "prev_error": float(self.prev_error),
            "setpoint": float(self.setpoint),
            "Kp": self.Kp,
            "Ki": self.Ki,
            "Kd": self.Kd,
        }


class AutopoiesisLoop:
    def __init__(self, scale: str = "1m", threshold_us: float = THRESHOLD_US,
                 Kp=0.15, Ki=0.01, Kd=0.1):
        self.scale = scale
        self.threshold_us = threshold_us
        self.engine = ScaleEngine()
        print(f"[Autopoiesis] Pre-building {scale} index...")
        self.engine._get_engine(scale)
        self.embs = self.engine._engines[scale][0]
        self.n = len(self.embs)
        print(f"[Autopoiesis] Index ready: {self.n:,} vectors.")

        self.history = self._load_history()
        self.pid_state = self._load_pid_state()
        self.pid = PIDController(
            Kp=Kp, Ki=Ki, Kd=Kd, setpoint=threshold_us,
            integral=self.pid_state.get("integral", 0.0),
            prev_error=self.pid_state.get("prev_error", 0.0),
        )

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE) as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Autopoiesis] Could not load history: {e}")
        return []

    def _save_history(self):
        os.makedirs(os.path.dirname(HISTORY_FILE) or ".", exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

    def _load_pid_state(self):
        if os.path.exists(PID_STATE_FILE):
            try:
                with open(PID_STATE_FILE) as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Autopoiesis] Could not load PID state: {e}")
        return {}

    def _save_pid_state(self):
        os.makedirs(os.path.dirname(PID_STATE_FILE) or ".", exist_ok=True)
        with open(PID_STATE_FILE, "w") as f:
            json.dump(self.pid.state(), f, indent=2)

    def observe(self, n_queries: int = 100):
        """Run self-query embeddings through the index and record metrics."""
        query_idx = random.sample(range(self.n), min(n_queries, self.n))
        latencies = []

        for idx in query_idx:
            q = self.embs[idx]
            t0 = time.perf_counter()
            self.engine.search_hybrid_by_embedding(q, k=10, scale=self.scale)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)

        if not latencies:
            raise RuntimeError("No queries succeeded")

        observation = {
            "timestamp": time.time(),
            "scale": self.scale,
            "n_queries": len(latencies),
            "p50_us": round(percentile(latencies, 0.50), 1),
            "p95_us": round(percentile(latencies, 0.95), 1),
            "p99_us": round(percentile(latencies, 0.99), 1),
            "avg_us": round(sum(latencies) / len(latencies), 1),
            "degraded": percentile(latencies, 0.50) > self.threshold_us,
            "ef": int(self.engine._engines[self.scale][1].ef),
        }
        self.history.append(observation)
        return observation

    def measure_recall(self, n_probe: int = 50, k: int = 10):
        """Measure R@k against exact cosine on a random probe set."""
        probe_idx = random.sample(range(self.n), min(n_probe, self.n))
        recalls = []

        for idx in probe_idx:
            q = self.embs[idx]

            # Exact top-k (excluding self)
            exact_sims = self.embs @ q
            exact_sims[idx] = -np.inf
            exact_top = set(np.argsort(-exact_sims)[:k])

            # HNSW top-k (excluding self)
            results = self.engine.search_hybrid_by_embedding(q, k=k, scale=self.scale)
            hnsw_top = set()
            for score, (pid, _) in results[:k]:
                pid_int = int(pid)
                if pid_int != idx:
                    hnsw_top.add(pid_int)

            if len(hnsw_top) == 0:
                recalls.append(0.0)
            else:
                recalls.append(len(exact_top & hnsw_top) / k)

        return {
            "n_probe": len(probe_idx),
            "r_at_k": round(sum(recalls) / len(recalls), 4),
            "r_min": round(min(recalls), 4),
            "r_max": round(max(recalls), 4),
        }

    def step(self, observation):
        """PID control step: compute ef delta, apply, return evaluation."""
        old_ef = int(self.engine._engines[self.scale][1].ef)

        # Each invocation of the nightly script is one control step.
        # Use dt=1 so the integral accumulates per-night error, not wall-clock hours.
        delta_ef = self.pid.update(observation["p50_us"], dt=1.0)
        # Negative feedback: positive error (too slow) -> lower ef;
        # negative error (too fast) -> raise ef to use the budget.
        # Deadband: ignore tiny deltas from measurement noise.
        if abs(delta_ef) < 5.0:
            delta_ef = 0.0
        new_ef = int(np.clip(old_ef - delta_ef, 10, 400))

        self.engine._engines[self.scale][1].set_ef(new_ef)

        # Measure recall after applying the new ef
        recall = self.measure_recall(n_probe=20, k=10)

        evaluation = {
            "applied": True,
            "old_ef": old_ef,
            "new_ef": new_ef,
            "delta_ef": round(delta_ef, 2),
            "p50_us": observation["p50_us"],
            "error": round(observation["p50_us"] - self.threshold_us, 2),
            "integral": round(self.pid.integral, 2),
            "dt_hours": round(dt, 3),
            "recall": recall,
        }

        self._save_pid_state()
        return evaluation

    def consolidate(self, evaluation):
        """Write proof-of-life entry."""
        if not evaluation.get("applied"):
            return

        entry = {
            "timestamp": time.time(),
            "event": "autopoiesis_tuned_ef",
            "old_ef": evaluation["old_ef"],
            "new_ef": evaluation["new_ef"],
            "delta_ef": evaluation["delta_ef"],
            "p50_us": evaluation["p50_us"],
            "error_us": evaluation["error"],
            "recall": evaluation["recall"],
            "hash": os.popen("git rev-parse HEAD").read().strip(),
        }
        os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(
            f"[Autopoiesis] Tuned ef -> {evaluation['new_ef']} "
            f"(P50: {evaluation['p50_us']} µs, "
            f"error: {evaluation['error']:+} µs, "
            f"R@10: {evaluation['recall']['r_at_k']})"
        )


def main():
    loop = AutopoiesisLoop(scale="1m", threshold_us=THRESHOLD_US)
    print("[Autopoiesis] Starting observation...")

    obs = loop.observe(n_queries=100)
    print(
        f"  P50: {obs['p50_us']} µs | "
        f"P95: {obs['p95_us']} µs | "
        f"P99: {obs['p99_us']} µs | "
        f"Degraded: {obs['degraded']} | "
        f"ef: {obs['ef']}"
    )

    eval_result = loop.step(obs)
    print(
        f"  PID: ef {eval_result['old_ef']} -> {eval_result['new_ef']} "
        f"(delta={eval_result['delta_ef']:+.1f}, "
        f"error={eval_result['error']:+.1f} µs)"
    )

    loop.consolidate(eval_result)
    loop._save_history()
    print("[Autopoiesis] Done.")


if __name__ == "__main__":
    main()
