"""Benchmark quarantine filter.
Call filter_quarantined(candidate_pids) before selecting query papers for any benchmark.
"""
import json, logging
from pathlib import Path
from scripts.enforce_quarantine import is_quarantined

logger = logging.getLogger("quarantine")

def filter_quarantined(candidate_pids: list) -> list:
    safe = []
    for pid in candidate_pids:
        if is_quarantined(pid):
            logger.warning(f"BENCH FILTER: removed quarantined PID {pid}")
        else:
            safe.append(pid)
    return safe

def assert_bench_clean(candidate_pids: list):
    bad = [p for p in candidate_pids if is_quarantined(p)]
    if bad:
        raise RuntimeError(f"BENCH QUARANTINE VIOLATION: {len(bad)} quarantined PIDs in bench pool: {bad[:5]}")
