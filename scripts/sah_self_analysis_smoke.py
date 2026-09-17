#!/usr/bin/env python3
"""SAH Stage 8 smoke: self-analysis triggers shard requests when gaps detected."""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from scripts.sah_registry import SAHRegistry
from scripts.sah_auto_downloader import SAHAutoDownloader
from scripts.sah_source_finder import SAHSourceFinder
from scripts.sah_self_analysis import SAHSelfAnalysis


def main():
    tmpdir = tempfile.mkdtemp()
    SAHRegistry.REGISTRY_PATH = Path(tmpdir) / ".sah_registry.json"

    registry = SAHRegistry()
    finder = SAHSourceFinder()
    auto = SAHAutoDownloader(finder, registry, targets={0x01: 500, 0x02: 500, 0x03: 200, 0x04: 100})

    query_count = [0]

    def counter():
        return query_count[0]

    analysis = SAHSelfAnalysis(registry, auto, query_counter=counter)

    # Patch download to avoid pulling multi-GB models during smoke test
    def fake_download_and_register(model_id: str, registry):
        fake = Path(tmpdir) / "downloads" / (model_id.replace("/", "--") + ".safetensors")
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_text("fake")
        registry.register_shard(fake, model_id)
        return fake.parent

    finder.download_and_register = fake_download_and_register

    # 1. Empty state → big gaps
    metrics = analysis.collect_metrics()
    assert metrics["total_beacons"] == 0
    print(f"[1] Initial metrics: {metrics['total_beacons']} beacons")

    # 2. Detect gaps
    gaps = analysis.detect_gaps(metrics)
    assert len(gaps) > 0
    print(f"[2] Gaps detected: {gaps}")

    # 3. should_run triggers on first call
    assert analysis.should_run(min_queries=1, min_interval_sec=0) is True
    print("[3] should_run = True (first run)")

    # 4. Analyze — will attempt download (may fail offline, but logic runs)
    report = analysis.analyze()
    assert report["action"] == "requested"
    print(f"[4] Analysis action: {report['action']}")
    print(f"[4] Gaps: {report['gaps']}")
    print(f"[4] Models requested: {report['models_requested']}")

    # 5. Simulate harvesting to reduce gaps
    fake1 = Path(tmpdir) / "m1" / "a.safetensors"
    fake1.parent.mkdir(parents=True, exist_ok=True)
    fake1.write_text("x")
    registry.register_shard(fake1, "test/model", arch="transformer")
    registry.mark_harvested(fake1, 600, {0x01: 300, 0x02: 200, 0x03: 100})

    fake2 = Path(tmpdir) / "m2" / "b.safetensors"
    fake2.parent.mkdir(parents=True, exist_ok=True)
    fake2.write_text("y")
    registry.register_shard(fake2, "test/model2")
    registry.mark_harvested(fake2, 600, {0x01: 300, 0x02: 400, 0x03: 150, 0x04: 50})

    metrics2 = analysis.collect_metrics()
    gaps2 = analysis.detect_gaps(metrics2)
    # Note: hit_rate is 0, so targets are boosted by 50% (e.g. 0x01 -> 750)
    # 0x01: 600 < 750 (gap due to boost), 0x02: 600 < 750, 0x03: 250 < 300, 0x04: 50 < 150
    assert 0x04 in gaps2
    assert metrics2["by_tag"].get(0x01, 0) == 600
    print(f"[5] After harvest: gaps = {gaps2}")

    # 6. tick respects interval
    query_count[0] = 50
    result = analysis.tick(min_queries=100, min_interval_sec=3600)
    assert result is None
    print("[6] tick returned None (too early)")

    # 7. Force run with zero interval
    result = analysis.tick(min_queries=1, min_interval_sec=0)
    assert result is not None
    print(f"[7] Forced tick: action={result['action']}")

    # 8. Request log
    assert len(analysis.request_log) >= 1
    print(f"[8] Request log entries: {len(analysis.request_log)}")

    shutil.rmtree(tmpdir)
    print("\nSAH SELF-ANALYSIS SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
