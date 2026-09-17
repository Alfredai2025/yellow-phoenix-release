#!/usr/bin/env python3
"""
ANNController: PID control of HNSW ef parameter.
Maintains query latency at a setpoint by adjusting ef online.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.phoenix_bench_api import ScaleEngine

HISTORY_FILE = "logs/controller_history.json"
TELEMETRY_FILE = "logs/controller_telemetry.jsonl"
THRESHOLD_US = 500.0  # setpoint


class PIDController:
    """Discrete PID controller for HNSW ef tuning."""

    def __init__(self, Kp=0.15, Ki=0.01, Kd=0.1, setpoint=500.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, measured_p50: float, dt: float = 1.0) -> float:
        """
        Compute control signal (delta ef) given measured latency.
        dt = time since last update (hours).
        """
        error = measured_p50 - self.setpoint

        # Proportional
        P = self.Kp * error

        # Integral (accumulate over time)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1000.0, 1000.0)  # anti-windup
        I = self.Ki * self.integral

        # Derivative
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        control = P + I + D
        # Clamp to safe range
        return float(np.clip(control, -50.0, 50.0))


class ANNController:
    """Online controller for HNSW index parameters."""

    def __init__(self, scale: str = "1m", Kp=0.5, Ki=0.1, Kd=0.05):
        self.engine = ScaleEngine()
        self.engine._get_engine(scale)
        self.embs = self.engine._engines[scale][0]
        self.n = len(self.embs)
        self.scale = scale

        self.pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=THRESHOLD_US)
        self.ef = int(self.engine._engines[scale][1].ef)

        self.history = self._load_history()

    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                return json.load(f)
        return []

    def observe(self, n_queries: int = 100):
        """Measure P50 latency on self-queries."""
        idx = np.random.choice(self.n, n_queries, replace=False)
        times = []
        for i in idx:
            q = self.embs[i]
            t0 = time.perf_counter()
            self.engine.search_hybrid_by_embedding(q, k=5, scale=self.scale)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        p50 = float(np.percentile(times, 50))
        p95 = float(np.percentile(times, 95))
        return {"p50_us": p50, "p95_us": p95, "n": n_queries}

    def step(self, dt_hours: float = 6.0):
        """One control step: observe, compute PID, apply."""
        obs = self.observe()
        p50 = obs["p50_us"]

        # Compute control signal
        delta_ef = self.pid.update(p50, dt=dt_hours)

        # Apply
        # Negative feedback: positive error (too slow) -> lower ef;
        # negative error (too fast) -> raise ef to use the latency budget.
        new_ef = int(np.clip(self.ef - delta_ef, 10, 400))
        old_ef = self.ef
        self.ef = new_ef
        self.engine._engines[self.scale][1].set_ef(new_ef)

        # Log telemetry
        entry = {
            "timestamp": time.time(),
            "p50_us": p50,
            "p95_us": obs["p95_us"],
            "error": p50 - THRESHOLD_US,
            "integral": self.pid.integral,
            "derivative": self.pid.prev_error,
            "delta_ef": delta_ef,
            "old_ef": old_ef,
            "new_ef": new_ef,
        }
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self.history.append(entry)
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"[PID] P50={p50:.1f}µs  error={entry['error']:.1f}  "
              f"delta_ef={delta_ef:+.1f}  ef={old_ef}→{new_ef}")
        return entry


def main():
    ctrl = ANNController(scale="100k", Kp=0.5, Ki=0.1, Kd=0.05)
    print("[ANNController] Starting control loop...")
    for step in range(5):
        print(f"\n--- Step {step+1}/5 ---")
        ctrl.step(dt_hours=6.0)
    print("[ANNController] Done.")


if __name__ == "__main__":
    main()
