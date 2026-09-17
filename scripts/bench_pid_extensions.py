#!/usr/bin/env python3
"""
Benchmark: PID Cascade + Geometric Router + Cache TTL
Simulates 100 steps of workload drift, measures convergence.
"""

import json
import os
import time
import numpy as np
from pid_extensions import CascadePID, GeometricRouterPID, CacheTTLPID


# ============================================================
# Simulated workload generators
# ============================================================
class SimulatedCascade:
    """Simulates a cascade where threshold affects hit rate."""
    def __init__(self):
        self.base_thresholds = {"L1": 8, "L2": 4, "L3": 2}
        # True optimal thresholds drift over time
        self.drift = 0.0

    def query(self, thresholds, n=100):
        self.drift += 0.02  # slow drift per step
        true_l1 = 8 + self.drift
        true_l2 = 4 + self.drift * 0.5
        true_l3 = 2 + self.drift * 0.2

        # Hit rate decreases as threshold moves away from optimal
        def hit_prob(th, true_th):
            diff = abs(th - true_th)
            return max(0.05, min(0.95, 0.75 - diff * 0.15 + np.random.normal(0, 0.03)))

        hr1 = hit_prob(thresholds["L1"], true_l1)
        hr2 = hit_prob(thresholds["L2"], true_l2) * (1 - hr1)  # conditional
        hr3 = hit_prob(thresholds["L3"], true_l3) * (1 - hr1 - hr2)

        l1_hits = int(n * hr1 + np.random.normal(0, 2))
        l2_hits = int(n * hr2 + np.random.normal(0, 2))
        l3_hits = int(n * hr3 + np.random.normal(0, 2))
        return max(0, l1_hits), max(0, l2_hits), max(0, l3_hits)


class SimulatedGeometricRouter:
    """Simulates geometric structure latency vs route %."""
    def __init__(self):
        self.base_latency = 400  # µs at 50% route
        self.load_factor = 1.0

    def query(self, route_percent, n=50):
        # Latency increases with route % (more geometric work)
        # Also has random noise
        latency = self.base_latency + route_percent * 4.0 + np.random.normal(0, 40)

        # Precision of fast path drops as we route MORE to geometric
        # (the easy queries stay on fast path, hard ones go to geometric)
        fast_precision = 0.92 - (route_percent / 100.0) * 0.15 + np.random.normal(0, 0.02)

        self.base_latency += np.random.normal(0, 5)  # slow drift
        return max(200, latency), max(0.5, min(0.99, fast_precision))


class SimulatedCache:
    """Simulates cache hit rate vs TTL."""
    def __init__(self):
        self.workload_diversity = 0.5  # higher = more unique queries

    def query(self, ttl_seconds, n=200):
        # Hit rate increases with TTL but plateaus
        # Also affected by workload diversity
        base_hr = 0.3 + 0.5 * (1 - np.exp(-ttl_seconds / 600.0))
        hr = base_hr * (1 - self.workload_diversity * 0.3) + np.random.normal(0, 0.03)

        self.workload_diversity += np.random.normal(0, 0.01)
        self.workload_diversity = np.clip(self.workload_diversity, 0.1, 0.9)

        hits = int(n * max(0.05, min(0.95, hr)))
        return hits, n - hits


# ============================================================
# Benchmark runner
# ============================================================
def run_benchmark(steps=100):
    print("=" * 70)
    print("PID EXTENSIONS BENCHMARK")
    print("=" * 70)

    cascade_sim = SimulatedCascade()
    router_sim = SimulatedGeometricRouter()
    cache_sim = SimulatedCache()

    cascade_pid = CascadePID(l1_target=0.50, l2_target=0.25, l3_target=0.05)
    router_pid = GeometricRouterPID(latency_setpoint_us=600.0, precision_setpoint=0.88)
    cache_pid = CacheTTLPID(hit_rate_target=0.70)

    # History tracking
    history = {
        "cascade": [],
        "router": [],
        "cache": [],
    }

    for step in range(steps):
        # --- CASCADE ---
        l1_h, l2_h, l3_h = cascade_sim.query(cascade_pid.thresholds, n=100)
        cascade_result = cascade_pid.step(l1_h, l2_h, l3_h, 100)
        history["cascade"].append({
            "step": step,
            **cascade_result,
        })

        # --- ROUTER ---
        geo_p50, fast_prec = router_sim.query(router_pid.route_percent, n=50)
        router_result = router_pid.step(geo_p50, fast_prec, 50)
        history["router"].append({
            "step": step,
            **router_result,
        })

        # --- CACHE ---
        hits, misses = cache_sim.query(cache_pid.ttl_seconds, n=200)
        cache_result = cache_pid.step(hits, misses)
        history["cache"].append({
            "step": step,
            **cache_result,
        })

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)

    # Cascade
    c_final = history["cascade"][-1]
    print(f"\n[CASCADE] Thresholds: L1={c_final['thresholds']['L1']} "
          f"L2={c_final['thresholds']['L2']} L3={c_final['thresholds']['L3']}")
    print(f"          Hit rates:  L1={c_final['hit_rates']['L1']} "
          f"L2={c_final['hit_rates']['L2']} L3={c_final['hit_rates']['L3']}")

    # Router
    r_final = history["router"][-1]
    print(f"\n[ROUTER]  Route % to geometric: {r_final['route_percent']}%")
    print(f"          Geometric P50: {r_final['geometric_p50_us']}µs")
    print(f"          Fast-path precision: {r_final['fast_path_precision']}")

    # Cache
    cache_final = history["cache"][-1]
    print(f"\n[CACHE]   TTL: {cache_final['ttl_seconds']}s")
    print(f"          Hit rate: {cache_final['hit_rate']}")

    # Convergence analysis
    print("\n" + "=" * 70)
    print("CONVERGENCE (last 20 steps vs first 20 steps)")
    print("=" * 70)

    def analyze_scalar(series, key, target=None):
        first = [s[key] for s in series[:20]]
        last = [s[key] for s in series[-20:]]
        print(f"  {key:20s}  first-20 mean: {np.mean(first):8.2f}  "
              f"last-20 mean: {np.mean(last):8.2f}  "
              f"std(last-20): {np.std(last):6.2f}", end="")
        if target:
            print(f"  target: {target}")
        else:
            print()

    def analyze_dict(series, key, targets):
        for subkey, target in targets.items():
            first = [s[key][subkey] for s in series[:20]]
            last = [s[key][subkey] for s in series[-20:]]
            print(f"  {key}.{subkey:16s}  first-20 mean: {np.mean(first):8.3f}  "
                  f"last-20 mean: {np.mean(last):8.3f}  "
                  f"std(last-20): {np.std(last):6.3f}  target: {target}")

    analyze_dict(history["cascade"], "hit_rates", {"L1": 0.50, "L2": 0.25, "L3": 0.05})
    analyze_scalar(history["router"], "route_percent")
    analyze_scalar(history["cache"], "hit_rate", "0.70")

    # Save
    os.makedirs("logs", exist_ok=True)
    with open("logs/bench_pid_extensions.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nSaved logs/bench_pid_extensions.json")

    return history


if __name__ == "__main__":
    run_benchmark(steps=100)
