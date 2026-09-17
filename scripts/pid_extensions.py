#!/usr/bin/env python3
"""
PID Extensions for Yellow Phoenix v3.6
- Cascade threshold PID
- Geometric structure routing PID
- Result cache TTL PID
"""

import json
import os
import time
from pathlib import Path
import numpy as np

# ============================================================
# Base PID (same logic as nightly_autopoiesis.py)
# ============================================================
class PIDController:
    def __init__(self, Kp=0.5, Ki=0.05, Kd=0.05, setpoint=0.0,
                 integral=0.0, prev_error=0.0,
                 output_min=-50.0, output_max=50.0,
                 integral_min=-1000.0, integral_max=1000.0,
                 deadband=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = integral
        self.prev_error = prev_error
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.deadband = deadband  # ignore errors within ±deadband

    def update(self, measured, dt=1.0):
        error = measured - self.setpoint
        if abs(error) < self.deadband:
            error = 0.0

        P = self.Kp * error
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, self.integral_min, self.integral_max))
        I = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        D = self.Kd * derivative
        self.prev_error = error

        control = P + I + D
        return float(np.clip(control, self.output_min, self.output_max))

    def state(self):
        return {
            "Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd,
            "setpoint": self.setpoint,
            "integral": float(self.integral),
            "prev_error": float(self.prev_error),
        }


# ============================================================
# 1. CASCADE THRESHOLD PID
# ============================================================
class CascadePID:
    """
    Tunes L1/L2/L3 prefix-match thresholds to maintain target hit rates.

    Direction: if hit_rate < target -> LOWER threshold (easier to match)
               if hit_rate > target -> RAISE threshold (stricter, push to next level)
    """
    STATE_FILE = "logs/pid_cascade_state.json"

    def __init__(self, l1_target=0.50, l2_target=0.25, l3_target=0.05):
        self.l1_pid = PIDController(Kp=5.0, Ki=1.0, Kd=0.5,
                                     setpoint=l1_target,
                                     output_min=-5, output_max=5,
                                     deadband=0.05)
        self.l2_pid = PIDController(Kp=5.0, Ki=1.0, Kd=0.5,
                                     setpoint=l2_target,
                                     output_min=-5, output_max=5,
                                     deadband=0.05)
        self.l3_pid = PIDController(Kp=5.0, Ki=1.0, Kd=0.5,
                                     setpoint=l3_target,
                                     output_min=-5, output_max=5,
                                     deadband=0.05)
        self.thresholds = {"L1": 8, "L2": 4, "L3": 2}  # prefix bytes
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE) as f:
                    state = json.load(f)
                for k, v in state.get("pids", {}).items():
                    getattr(self, f"{k.lower()}_pid").integral = v.get("integral", 0.0)
                    getattr(self, f"{k.lower()}_pid").prev_error = v.get("prev_error", 0.0)
                self.thresholds = state.get("thresholds", self.thresholds)
            except Exception as e:
                print(f"[CascadePID] Could not load state: {e}")

    def _save_state(self):
        os.makedirs(os.path.dirname(self.STATE_FILE) or ".", exist_ok=True)
        state = {
            "pids": {
                "L1": self.l1_pid.state(),
                "L2": self.l2_pid.state(),
                "L3": self.l3_pid.state(),
            },
            "thresholds": self.thresholds,
            "timestamp": time.time(),
        }
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def step(self, l1_hits, l2_hits, l3_hits, total_queries):
        if total_queries == 0:
            return {"thresholds": self.thresholds.copy()}

        hr1 = l1_hits / total_queries
        hr2 = l2_hits / total_queries
        hr3 = l3_hits / total_queries

        # --- INTELLIGENT STEP: big error = big jump, small error = fine-tune ---
        def smart_step(error):
            abs_err = abs(error)
            if abs_err > 0.20:
                return 3      # far away — sprint
            elif abs_err > 0.10:
                return 2      # getting close — jog
            elif abs_err > 0.05:
                return 1      # nearly there — creep
            else:
                return 0      # deadband — hold perfectly still

        # PID gives direction (+ or -), smart_step gives magnitude
        ctrl1 = self.l1_pid.update(hr1, dt=1.0)
        ctrl2 = self.l2_pid.update(hr2, dt=1.0)
        ctrl3 = self.l3_pid.update(hr3, dt=1.0)

        step1 = smart_step(hr1 - self.l1_pid.setpoint)
        step2 = smart_step(hr2 - self.l2_pid.setpoint)
        step3 = smart_step(hr3 - self.l3_pid.setpoint)

        # Apply: direction from PID, size from smart_step
        self.thresholds["L1"] = int(np.clip(
            self.thresholds["L1"] + int(np.sign(ctrl1) * step1),
            SafetyEnvelope.CASCADE_MIN["L1"],
            SafetyEnvelope.CASCADE_MAX["L1"]
        ))
        self.thresholds["L2"] = int(np.clip(
            self.thresholds["L2"] + int(np.sign(ctrl2) * step2),
            SafetyEnvelope.CASCADE_MIN["L2"],
            SafetyEnvelope.CASCADE_MAX["L2"]
        ))
        self.thresholds["L3"] = int(np.clip(
            self.thresholds["L3"] + int(np.sign(ctrl3) * step3),
            SafetyEnvelope.CASCADE_MIN["L3"],
            SafetyEnvelope.CASCADE_MAX["L3"]
        ))

        # Anti-jitter: if L3 hits floor, reset windup so it doesn't buzz
        if self.thresholds["L3"] <= SafetyEnvelope.CASCADE_MIN["L3"]:
            self.l3_pid.integral = 0.0

        self._save_state()
        return {
            "thresholds": self.thresholds.copy(),
            "hit_rates": {"L1": round(hr1, 3), "L2": round(hr2, 3), "L3": round(hr3, 3)},
            "steps_taken": {"L1": step1, "L2": step2, "L3": step3},
            "errors": {
                "L1": round(hr1 - self.l1_pid.setpoint, 3),
                "L2": round(hr2 - self.l2_pid.setpoint, 3),
                "L3": round(hr3 - self.l3_pid.setpoint, 3),
            },
        }


# ============================================================
# 2. GEOMETRIC STRUCTURE ROUTING PID
# ============================================================
class GeometricRouterPID:
    """
    Tunes what % of queries get routed through the geometric structure.

    Direction: if geometric latency > budget -> route FEWER to geometric
               if fast-path precision too low -> route MORE to geometric
    """
    STATE_FILE = "logs/pid_geometric_router_state.json"

    def __init__(self, latency_setpoint_us=600.0, precision_setpoint=0.90):
        # Primary controller: latency budget
        self.latency_pid = PIDController(
            Kp=0.3, Ki=0.03, Kd=0.02,
            setpoint=latency_setpoint_us,
            output_min=-20, output_max=20,
            deadband=30.0  # ±30µs deadband
        )
        # Secondary controller: precision floor
        self.precision_pid = PIDController(
            Kp=50.0, Ki=5.0, Kd=2.0,
            setpoint=precision_setpoint,
            output_min=-10, output_max=10,
            deadband=0.02
        )
        self.route_percent = 50.0  # % of queries -> geometric structure
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE) as f:
                    state = json.load(f)
                self.latency_pid.integral = state.get("latency_integral", 0.0)
                self.latency_pid.prev_error = state.get("latency_prev_error", 0.0)
                self.precision_pid.integral = state.get("precision_integral", 0.0)
                self.precision_pid.prev_error = state.get("precision_prev_error", 0.0)
                self.route_percent = state.get("route_percent", 50.0)
            except Exception as e:
                print(f"[GeometricRouterPID] Could not load state: {e}")

    def _save_state(self):
        os.makedirs(os.path.dirname(self.STATE_FILE) or ".", exist_ok=True)
        state = {
            "route_percent": self.route_percent,
            "latency_integral": self.latency_pid.integral,
            "latency_prev_error": self.latency_pid.prev_error,
            "precision_integral": self.precision_pid.integral,
            "precision_prev_error": self.precision_pid.prev_error,
            "timestamp": time.time(),
        }
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def step(self, geometric_p50_us, fast_path_precision, total_queries):
        """
        geometric_p50_us: P50 latency of queries that went through geometric structure
        fast_path_precision: precision@10 of queries that stayed on fast path
        total_queries: for logging only
        """
        # Latency: if too slow, NEGATIVE delta -> lower route %
        lat_delta = -self.latency_pid.update(geometric_p50_us, dt=1.0)

        # Precision: if fast path precision too LOW, NEGATIVE delta -> lower route %
        # (keep more queries on the fast path so its precision rises)
        prec_delta = self.precision_pid.update(fast_path_precision, dt=1.0)

        # Combine: latency is primary, precision is guardrail
        combined_delta = lat_delta + (prec_delta * 0.3)

        self.route_percent = float(np.clip(self.route_percent + combined_delta, 5.0, 95.0))

        # Hard floor: if fast-path precision drops below 0.80, force at least 30% geometric
        if fast_path_precision < 0.80:
            self.route_percent = max(self.route_percent, 30.0)
            guard_active = True
        else:
            guard_active = False

        self._save_state()
        return {
            "route_percent": round(self.route_percent, 1),
            "geometric_p50_us": round(geometric_p50_us, 1),
            "fast_path_precision": round(fast_path_precision, 3),
            "lat_delta": round(lat_delta, 2),
            "prec_delta": round(prec_delta, 2),
            "combined_delta": round(combined_delta, 2),
            "guard_active": guard_active,
        }


# ============================================================
# 3. RESULT CACHE TTL PID
# ============================================================
class CacheTTLPID:
    """
    Tunes cache TTL to maintain target hit rate.

    Direction: if hit_rate < target -> INCREASE TTL (keep entries longer)
               if hit_rate > target -> DECREASE TTL (fresher results, save memory)
    """
    STATE_FILE = "logs/pid_cache_ttl_state.json"

    def __init__(self, hit_rate_target=0.70, min_ttl=60, max_ttl=86400):
        self.pid = PIDController(
            Kp=100.0, Ki=10.0, Kd=5.0,
            setpoint=hit_rate_target,
            output_min=-300, output_max=300,
            deadband=0.05
        )
        self.ttl_seconds = 300  # start at 5 minutes
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.STATE_FILE):
            try:
                with open(self.STATE_FILE) as f:
                    state = json.load(f)
                self.pid.integral = state.get("integral", 0.0)
                self.pid.prev_error = state.get("prev_error", 0.0)
                self.ttl_seconds = state.get("ttl_seconds", 300)
            except Exception as e:
                print(f"[CacheTTLPID] Could not load state: {e}")

    def _save_state(self):
        os.makedirs(os.path.dirname(self.STATE_FILE) or ".", exist_ok=True)
        state = {
            "ttl_seconds": self.ttl_seconds,
            "integral": self.pid.integral,
            "prev_error": self.pid.prev_error,
            "timestamp": time.time(),
        }
        with open(self.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def step(self, cache_hits, cache_misses):
        total = cache_hits + cache_misses
        if total == 0:
            return {"ttl_seconds": self.ttl_seconds, "hit_rate": 0.0}

        hit_rate = cache_hits / total

        # Decay TTL when hit rate is comfortably above target (saves memory)
        if hit_rate > self.pid.setpoint + 0.10:
            self.ttl_seconds = int(self.ttl_seconds * 0.95)
            self.ttl_seconds = max(self.min_ttl, self.ttl_seconds)
            self._save_state()
            return {
                "ttl_seconds": self.ttl_seconds,
                "hit_rate": round(hit_rate, 3),
                "delta": "decay",
            }

        # Negative feedback: if hit rate too LOW, INCREASE TTL
        delta = -self.pid.update(hit_rate, dt=1.0)

        self.ttl_seconds = int(np.clip(self.ttl_seconds + delta, self.min_ttl, self.max_ttl))

        self._save_state()
        return {
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": round(hit_rate, 3),
            "delta": round(delta, 1),
        }


# ============================================================
# Unified manager (optional convenience wrapper)
# ============================================================
class PIDManager:
    def __init__(self):
        self.cascade = CascadePID()
        self.router = GeometricRouterPID()
        self.cache = CacheTTLPID()

    def step_all(self, metrics):
        """
        metrics = {
            "cascade": {"l1_hits": int, "l2_hits": int, "l3_hits": int, "total": int},
            "router": {"geometric_p50_us": float, "fast_path_precision": float, "total_queries": int},
            "cache": {"hits": int, "misses": int},
        }
        """
        results = {}
        if "cascade" in metrics:
            results["cascade"] = self.cascade.step(**metrics["cascade"])
        if "router" in metrics:
            results["router"] = self.router.step(**metrics["router"])
        if "cache" in metrics:
            results["cache"] = self.cache.step(**metrics["cache"])
        return results


# ============================================================
# 4. SAFETY ENVELOPE
# ============================================================
class SafetyEnvelope:
    """
    Hard limits that override PID output.
    Never let any controller put the engine in a broken state.
    """
    CASCADE_MIN = {"L1": 2, "L2": 1, "L3": 1}
    CASCADE_MAX = {"L1": 14, "L2": 8, "L3": 8}
    ROUTER_MIN_PERCENT = 5.0
    ROUTER_MAX_PERCENT = 95.0
    CACHE_MIN_TTL = 30
    CACHE_MAX_TTL = 86400

    @classmethod
    def clamp_cascade(cls, thresholds):
        return {k: int(np.clip(v, cls.CASCADE_MIN[k], cls.CASCADE_MAX[k]))
                for k, v in thresholds.items()}

    @classmethod
    def clamp_router(cls, percent):
        return float(np.clip(percent, cls.ROUTER_MIN_PERCENT, cls.ROUTER_MAX_PERCENT))

    @classmethod
    def clamp_cache(cls, ttl):
        return int(np.clip(ttl, cls.CACHE_MIN_TTL, cls.CACHE_MAX_TTL))
