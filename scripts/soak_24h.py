#!/usr/bin/env python3
# Ensure project root is on Python path for yp_autonomic / yp_bridge imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
soak_24h.py — Constitutional Soak Test with intelligence layers

Safe for Mac thermal constraints:
  • Checks thermal every tick
  • Sleeps between ticks (no spin)
  • Saves state to disk every minute
  • Survives interruption (Ctrl+C) and resumes
  • NO caffeinate, NO unattended thermal risk

Adds Step 1–4 intelligence layers:
  • Shadow query drift detection (every 1000 ticks)
  • Knowledge seed (loaded once at start)
  • Query-pattern cascade priorities (loaded once at start)
  • Adversarial stress test (every 5000 ticks)

Usage:
    python3 scripts/soak_24h.py
"""

import ctypes
import json
import numpy as np
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOAK_DURATION_HOURS = 24
TICK_INTERVAL_SECONDS = 1
THERMAL_PAUSE_THRESHOLD = 72.0
LOG_FILE = Path("/Users/mac/yellow_phoenix/soak_log.jsonl")
STATE_FILE = Path("/Users/mac/yellow_phoenix/soak_state.json")
YP_ROOT = Path("/Users/mac/yellow_phoenix")

# -----------------------------------------------------------------------------
# Load raw YP clockwork FFI
# -----------------------------------------------------------------------------
LIB_PATH = YP_ROOT / "target" / "release" / "libpams.dylib"
if not LIB_PATH.exists():
    print("ERROR: Build YP first: YP_GENESIS_PHRASE='Be here now' cargo build --release")
    sys.exit(1)

lib = ctypes.CDLL(str(LIB_PATH))
lib.yp_clockwork_init.restype = ctypes.c_int
lib.yp_golden_hash_tick.restype = ctypes.c_ulonglong
lib.yp_golden_hash_adjudicate.argtypes = [
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
lib.yp_golden_hash_adjudicate.restype = ctypes.c_int
lib.yp_memory_distressed.restype = ctypes.c_int

# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------
shutdown_requested = False


def handle_sigint(signum, frame):
    global shutdown_requested
    print("\n🛑 Soak interrupted. Saving state...")
    shutdown_requested = True


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_cpu_temp():
    """Read CPU temperature (macOS)."""
    try:
        import subprocess

        result = subprocess.run(["osx-cpu-temp"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return float(result.stdout.strip().replace("°C", ""))
    except Exception:
        pass
    return 50.0


# --- DISK PREEMPTION WIRING (Block 2) ---
def get_disk_pressure():
    """Read disk pressure and return throttle advice."""
    try:
        import sys
        sys.path.insert(0, "/Users/mac/yellow_phoenix")
        from yp_autonomic.sensors.disk_pressure import DiskPressureSensor
        sensor = DiskPressureSensor(path="/Users/mac", threshold_pct=80.0, critical_pct=90.0)
        return sensor.throttle_advice()
    except Exception as e:
        return {"sensor": "disk_pressure", "throttle": False, "pressure": "unknown", "error": str(e), "action": "normal"}


def should_ingest(autopoiesis, disk_advice):
    """Decide if autopoiesis should ingest this tick."""
    if disk_advice.get("action") == "skip_ingest":
        print(f"[DISK] CRITICAL: {disk_advice.get('pct', 0):.1f}% full — skipping ingest")
        return False, 0
    if disk_advice.get("action") == "reduce_batch":
        reduced = max(10, 50 // 2)  # half the default batch
        print(f"[DISK] HIGH: {disk_advice.get('pct', 0):.1f}% full — throttling batch to {reduced}")
        return True, reduced
    return True, 50


def get_memory_pct():
    """Read memory usage percentage."""
    try:
        import psutil

        return psutil.virtual_memory().percent
    except Exception:
        return 50.0


def _sphere_synthetic_query(bridge):
    """Feed the sphere shadow a synthetic labeled query from the 1M exam set."""
    try:
        from yp.crystal.soak_sphere_integration import sphere_synthetic_query
        sphere_synthetic_query(bridge)
    except Exception as e:
        print(f"[SphereShadow] query failed: {e}")


def load_state():
    """Resume from saved state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"tick": 0, "start_time": time.time(), "autopoiesis_score": 1.0}


def save_state(state):
    """Save current state for resume."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def log_event(event):
    """Append event to log file."""
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(event) + '\n')


# -----------------------------------------------------------------------------
# Intelligence layers
# -----------------------------------------------------------------------------
def init_intelligence(bridge):
    """Wire shadow drift and adversarial stress detectors."""
    from yp_autonomic.shadow_drift import wire_shadow_drift, run_shadow_check
    from yp_autonomic.adversarial_stress import (
        wire_adversarial_stress,
        run_adversarial_check,
    )

    return {
        "shadow": wire_shadow_drift(bridge),
        "stress": wire_adversarial_stress(bridge),
        "run_shadow": run_shadow_check,
        "run_stress": run_adversarial_check,
    }


# -----------------------------------------------------------------------------
# Main soak loop
# -----------------------------------------------------------------------------
def run_soak():
    print("=" * 60)
    print("YELLOW PHOENIX — 24-HOUR CONSTITUTIONAL SOAK")
    print("=" * 60)
    print(f"Duration: {SOAK_DURATION_HOURS} hours")
    print(f"Tick interval: {TICK_INTERVAL_SECONDS}s")
    print(f"Thermal pause: >{THERMAL_PAUSE_THRESHOLD}°C")
    print(f"Log: {LOG_FILE}")
    print(f"State: {STATE_FILE}")
    print("=" * 60)

    # Initialize clockwork
    rc = lib.yp_clockwork_init()
    if rc != 0:
        print(f"ERROR: clockwork_init failed: {rc}")
        sys.exit(1)

    # Load seed / cascade priorities once
    print("\n[SETUP] Seeding knowledge graph...")
    from yp_autonomic.knowledge_seed import seed_knowledge_graph

    seed_knowledge_graph()

    print("[SETUP] Tuning cascade router...")
    from yp_autonomic.query_patterns import tune_cascade_router

    tune_cascade_router()

    # Log janitor — thermal/disk-aware retention
    print("[SETUP] Wiring intelligent log janitor...")
    from yp_autonomic.log_janitor import wire_log_janitor

    janitor = wire_log_janitor(
        LOG_FILE,
        YP_ROOT,
        thermal_sensor=get_cpu_temp,
        disk_sensor=lambda: shutil.disk_usage(YP_ROOT).free / (1024 ** 3),
    )

    # Bridge for intelligence-layer search
    print("[SETUP] Initializing RustBridge for shadow/adversarial layers...")
    try:
        from yp_bridge import RustBridge

        bridge = RustBridge()
        sys.modules['__main__'].bridge = bridge  # expose for sphere shadow hook
        intelligence = init_intelligence(bridge)
        import yp.crystal.soak_sphere_integration  # auto-wraps on import
    except Exception as e:
        print(f"[WARN] Could not initialize intelligence layers: {e}")
        intelligence = None

    # Autopoiesis — feed droplets into the geometric mesh
    print("[SETUP] Wiring autopoiesis engine...")
    try:
        from yp_autonomic.autopoiesis_engine import wire_autopoiesis

        autopoiesis = wire_autopoiesis(
            bridge,
            thermal_sensor=get_cpu_temp,
            memory_sensor=get_memory_pct,
        )
        print("[SETUP] Ingesting droplet seed into mesh...")
        # Disk-preemptive throttle
        disk_advice = get_disk_pressure()
        can_ingest, batch_size = should_ingest(autopoiesis, disk_advice)
        if can_ingest:
            auto_result = autopoiesis.ingest_from_file(max_droplets=1000, batch_size=batch_size)
        else:
            auto_result = {"skipped": True, "reason": "disk_pressure", "advice": disk_advice}
        print(f"[SETUP] Autopoiesis seed: {auto_result}")
        print(f"[SETUP] Mesh count after seed: {autopoiesis.mesh_count()}")
    except Exception as e:
        print(f"[WARN] Could not initialize autopoiesis: {e}")
        autopoiesis = None

    state = load_state()
    start_tick = state["tick"]
    start_time = state.get("start_time", time.time())
    autopoiesis_score = state.get("autopoiesis_score", 1.0)
    if autopoiesis is not None:
        autopoiesis_score = autopoiesis.compute_score()
        print(f"[SETUP] Initial autopoiesis score: {autopoiesis_score}")
    max_ticks = SOAK_DURATION_HOURS * 3600

    print(f"\nResuming from tick {start_tick}")
    print("Press Ctrl+C to pause gracefully.\n")

    tick = start_tick
    for tick in range(start_tick, max_ticks):
        if shutdown_requested:
            break

        # Thermal check
        temp = get_cpu_temp()
        if temp > THERMAL_PAUSE_THRESHOLD:
            print(f"[{tick}] 🔥 Thermal {temp:.1f}°C > {THERMAL_PAUSE_THRESHOLD}°C — PAUSING")
            while temp > THERMAL_PAUSE_THRESHOLD - 3.0:
                time.sleep(5)
                temp = get_cpu_temp()
                if shutdown_requested:
                    break
            print(f"[{tick}] 🌡️  Thermal cooled to {temp:.1f}°C — RESUMING")
            if shutdown_requested:
                break

        # Constitutional tick
        tick_id = lib.yp_golden_hash_tick()

        # Intelligence layers
        drift = {"status": "no_layers"}
        adv = {"status": "no_layers"}
        if intelligence is not None:
            drift = intelligence["run_shadow"](intelligence["shadow"], tick)
            if tick % 10 == 0:
                _sphere_synthetic_query(bridge)
            if drift.get("drift", {}).get("degraded"):
                autopoiesis_score = max(0.1, autopoiesis_score * 0.95)

            adv = intelligence["run_stress"](intelligence["stress"], tick)
            if adv.get("scar_triggered"):
                # Robustness scar placeholder: log it. Real scar FFI can be added.
                print(f"[{tick}] 🩹 Robustness scar signal logged")

        # Real autopoiesis score from mesh health (will be damped by drift)
        if autopoiesis is not None:
            autopoiesis_score = autopoiesis.compute_score()

        # Simulated telemetry (clamped to valid ranges)
        memory_pct = get_memory_pct()
        p50_ms = max(0.1, 3.5 + random.gauss(0, 0.5))
        p99_ms = max(0.1, p50_ms * 1.8)
        r1 = max(0.0, min(1.0, 0.87 + random.gauss(0, 0.02)))
        r5 = max(0.0, min(1.0, 0.96 + random.gauss(0, 0.01)))

        # Adjudicate
        level = ctypes.c_int()
        gear = ctypes.c_int()
        lib.yp_golden_hash_adjudicate(
            p50_ms,
            p99_ms,
            r1,
            r5,
            temp,
            memory_pct,
            autopoiesis_score,
            0,
            0,
            0,
            ctypes.byref(level),
            ctypes.byref(gear),
        )

        mem_distressed = lib.yp_memory_distressed()

        # Feed constitutional policy back to autopoiesis
        if autopoiesis is not None:
            autopoiesis.set_policy(level.value, gear.value)

        event = {
            "tick": tick_id,
            "timestamp": datetime.now().isoformat(),
            "p50_ms": round(p50_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "r1": round(r1, 4),
            "r5": round(r5, 4),
            "cpu_temp": round(temp, 1),
            "memory_pct": round(memory_pct, 1),
            "autopoiesis_score": round(autopoiesis_score, 4),
            "override_level": level.value,
            "target_gear": gear.value,
            "memory_distressed": mem_distressed,
            "mesh_count": autopoiesis.mesh_count() if autopoiesis else 0,
            "drift": drift,
            "adversarial": adv,
        }

        # Log through janitor (compact on quiet ticks, full on anomalies)
        janitor.ingest(event)

        if tick % 60 == 0:
            janitor.run_maintenance(tick)
            state["tick"] = tick
            state["start_time"] = start_time
            state["autopoiesis_score"] = autopoiesis_score
            save_state(state)

            elapsed = time.time() - start_time
            progress = tick / max_ticks * 100
            mesh_count = autopoiesis.mesh_count() if autopoiesis else 0
            print(
                f"[{tick}] {progress:.1f}% | {elapsed/3600:.1f}h | "
                f"p50={p50_ms:.1f}ms temp={temp:.1f}°C mem={memory_pct:.0f}% "
                f"override={level.value} gear={gear.value} "
                f"autopoiesis={autopoiesis_score:.3f} mesh={mesh_count}"
            )

        # Continuous autopoiesis ingestion (per-tick)
        disk_advice = get_disk_pressure()
        can_ingest, batch_size = should_ingest(autopoiesis, disk_advice)
        if can_ingest:
            if hasattr(autopoiesis, 'ingest_next_batch'):
                auto_result = autopoiesis.ingest_next_batch(batch_size=batch_size)
            elif hasattr(autopoiesis, 'process_queue'):
                auto_result = autopoiesis.process_queue(batch_size=batch_size)
            elif hasattr(autopoiesis, 'tick'):
                auto_result = autopoiesis.tick(batch_size=batch_size)
            else:
                auto_result = {"skipped": True, "reason": "no_ingest_method"}
        else:
            auto_result = {"skipped": True, "reason": "disk_pressure", "advice": disk_advice}

        time.sleep(TICK_INTERVAL_SECONDS)

    # Final save
    state["tick"] = tick
    state["start_time"] = start_time
    state["autopoiesis_score"] = autopoiesis_score
    save_state(state)

    print("\n" + "=" * 60)
    print("SOAK COMPLETE")
    print("=" * 60)
    print(f"Total ticks: {tick}")
    print(f"Duration: {(time.time() - start_time)/3600:.1f} hours")
    print(f"Log: {LOG_FILE}")
    print("\nTo analyze: python3 analyze_soak.py")


if __name__ == "__main__":
    run_soak()
