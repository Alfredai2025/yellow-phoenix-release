#!/usr/bin/env python3
"""verify_wiring.py — M1 + M2 integration + wiring scan.

Performs checks across all Milestone 1 and Milestone 2 components and reports
a pass/fail summary.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SRC = ROOT / "src"
TESTS = ROOT / "tests"

CHECKS = []


def check(name, condition, detail=""):
    ok = bool(condition)
    CHECKS.append((name, ok, detail))
    return ok


def file_size(path):
    return path.stat().st_size if path.exists() else -1


def source_contains(substr, path=SRC):
    for p in path.glob("*.rs"):
        text = p.read_text(errors="ignore")
        if substr in text:
            return True
    return False


def cargo_test(filter_name):
    args = ["cargo", "test"] + filter_name.split() + ["--", "--quiet"]
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def main():
    print(f"WIRING_MAP.md validation [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]:\n")

    # ------------------------------------------------------------------
    # M1 Component 1: Result Cache (2 checks)
    # ------------------------------------------------------------------
    cache_file = DATA / "cache_entries_v1.bin"
    cache_size = file_size(cache_file)
    check("result_cache::file_exists", cache_file.exists(), "data/cache_entries_v1.bin")
    check("result_cache::size <= 10MB", cache_file.exists() and cache_size <= 10 * 1024 * 1024, f"size={cache_size}B")

    # ------------------------------------------------------------------
    # M1 Component 2: Learned Router (2 checks)
    # ------------------------------------------------------------------
    router_file = DATA / "router_model_v1.bin"
    router_size = file_size(router_file)
    check("learned_router::file_exists", router_file.exists(), "data/router_model_v1.bin")
    check("learned_router::51 params (204B)", router_file.exists() and router_size == 204, f"size={router_size}B")

    # ------------------------------------------------------------------
    # M1 Component 3: Self-Learning Table (3 checks)
    # ------------------------------------------------------------------
    table_file = DATA / "learning_table_v1.bin"
    table_size = file_size(table_file)
    backups = [table_file.with_suffix(f".bak{i}") for i in (1, 2, 3)]
    backups_ok = all(b.exists() for b in backups)
    check("self_learning::file_exists", table_file.exists(), "data/learning_table_v1.bin")
    check("self_learning::256 entries (4104B)", table_file.exists() and table_size == 4104, f"size={table_size}B")
    check("self_learning::3 backups", backups_ok, f"backups={sum(b.exists() for b in backups)}/3")

    # ------------------------------------------------------------------
    # M1 Component 4: Engine Feeder (2 checks)
    # ------------------------------------------------------------------
    feeds = ["cascade_miss", "faiss_disagree", "domain"]
    feeds_wired = all(source_contains(f'"{feed}"') for feed in feeds)
    pipes = [Path("/tmp/yp_feed_cascade"), Path("/tmp/yp_feed_faiss"), Path("/tmp/yp_feed_domain")]
    pipes_ok = all(p.exists() for p in pipes)
    check("engine_feeder::3 feeds active", feeds_wired, ", ".join(feeds))
    check("engine_feeder::pipes exist", pipes_ok, ", ".join(str(p) for p in pipes))

    # ------------------------------------------------------------------
    # M1 Component 5: Collaborative Engine (2 checks)
    # ------------------------------------------------------------------
    stages_wired = all(
        source_contains(f"mod {stage}") and source_contains(f"{stage}::")
        for stage in ["hash_stage", "spectral_stage", "wedge_stage", "hologram_stage"]
    )
    integration_targets = ["ResultCache", "LearnedRouter", "SelfLearningTable", "EngineFeeder"]
    integrations_wired = all(source_contains(name) for name in integration_targets)
    check("collaborative_engine::4 stages wired", stages_wired, "hash + spectral + wedge + hologram")
    check("collaborative_engine::4 integrations wired", integrations_wired, ", ".join(integration_targets))

    # ------------------------------------------------------------------
    # M1 Component 6: Fuzz Tests (1 check)
    # ------------------------------------------------------------------
    fuzz_file = TESTS / "fuzz_tests.rs"
    fuzz_exists = fuzz_file.exists()
    check("fuzz_tests::2/2 pass", fuzz_exists and cargo_test("--test fuzz_tests"), "2/2 pass")

    # ------------------------------------------------------------------
    # M2 Component 1: Async FFI (2 checks)
    # ------------------------------------------------------------------
    check("async_ffi::module_wired", source_contains("pub mod async_ffi"), "lib.rs declares module")
    check(
        "async_ffi::exports_present",
        source_contains("yp_temporal_submit") and source_contains("yp_temporal_poll"),
        "yp_temporal_submit + yp_temporal_poll",
    )

    # ------------------------------------------------------------------
    # M2 Component 2: Temporal Orchestrator (2 checks)
    # ------------------------------------------------------------------
    check("temporal_orchestrator::module_wired", source_contains("pub mod temporal_orchestrator"), "lib.rs declares module")
    check(
        "temporal_orchestrator::pipeline_present",
        source_contains("TemporalOrchestrator") and source_contains("fn tick"),
        "TemporalOrchestrator + tick()",
    )

    # ------------------------------------------------------------------
    # M2 Component 3: Versioned Tables (2 checks)
    # ------------------------------------------------------------------
    check("versioned_tables::module_wired", source_contains("pub mod versioned_tables"), "lib.rs declares module")
    check(
        "versioned_tables::swap_present",
        source_contains("VersionedTable") and source_contains("fn swap"),
        "VersionedTable + swap()",
    )

    # ------------------------------------------------------------------
    # M2 Component 4: Shadow Build (2 checks)
    # ------------------------------------------------------------------
    check("shadow_build::module_wired", source_contains("pub mod shadow_build"), "lib.rs declares module")
    check(
        "shadow_build::lifecycle_present",
        source_contains("ShadowBuild") and source_contains("fn start") and source_contains("fn try_swap"),
        "ShadowBuild + start() + try_swap()",
    )

    # ------------------------------------------------------------------
    # M2 Component Tests (4 checks)
    # ------------------------------------------------------------------
    check("async_ffi::unit_tests", cargo_test("async_ffi"), "cargo test async_ffi")
    check("temporal_orchestrator::unit_tests", cargo_test("temporal_orchestrator"), "cargo test temporal_orchestrator")
    check("versioned_tables::unit_tests", cargo_test("versioned_tables"), "cargo test versioned_tables")
    check("shadow_build::unit_tests", cargo_test("shadow_build"), "cargo test shadow_build")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    for name, ok, detail in CHECKS:
        status = "✅" if ok else "❌"
        print(f"├── {name:<45} {status} ({detail})")
    print(f"\n└── ALL WIRING CHECKS PASS         {'✅' if passed == total else '❌'} {passed}/{total}")

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
