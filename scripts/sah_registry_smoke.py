#!/usr/bin/env python3
"""SAH Stage 7 smoke: registry + auto-downloader + coverage tracking."""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from scripts.sah_registry import SAHRegistry
from scripts.sah_auto_downloader import SAHAutoDownloader
from scripts.sah_source_finder import SAHSourceFinder


def main():
    tmpdir = tempfile.mkdtemp()
    SAHRegistry.REGISTRY_PATH = Path(tmpdir) / ".sah_registry.json"

    registry = SAHRegistry()
    finder = SAHSourceFinder()
    auto = SAHAutoDownloader(finder, registry, targets={0x01: 100, 0x02: 100, 0x03: 50, 0x04: 20})

    # 1. Empty registry
    assert registry.coverage_summary()["total_beacons"] == 0
    print("[1] Empty registry OK")

    # 2. Register fake shards
    fake1 = Path(tmpdir) / "m1" / "a.safetensors"
    fake1.parent.mkdir(parents=True, exist_ok=True)
    fake1.write_text("x")
    registry.register_shard(fake1, "qwen/Qwen2.5-7B-Instruct", arch="transformer", params="7B")
    registry.mark_harvested(fake1, 80, {0x01: 50, 0x02: 20, 0x03: 10})
    print("[2] Fake shard harvested")

    # 3. Coverage check
    summary = registry.coverage_summary()
    assert summary["total_beacons"] == 80
    assert summary["by_tag"]["0x1"] == 50
    print(f"[3] Coverage: {summary}")

    # 4. Missing tags
    missing = registry.missing_tags({0x01: 100, 0x02: 100, 0x03: 50, 0x04: 20})
    assert 0x01 in missing
    assert 0x04 in missing
    print(f"[4] Missing: {[hex(t) for t in missing]}")

    # 5. Auto-downloader seed scoring
    seed = auto.next_seed_model()
    assert seed is not None
    assert seed[0] != "qwen/Qwen2.5-7B-Instruct"
    print(f"[5] Next seed model: {seed}")

    # 6. Auto-downloader status
    status = auto.status()
    assert "next_model" in status
    assert status["next_model"] is not None
    print(f"[6] Status next: {status['next_model']}")

    # 7. Unharvested list
    fake2 = Path(tmpdir) / "m2" / "b.safetensors"
    fake2.parent.mkdir(parents=True, exist_ok=True)
    fake2.write_text("y")
    registry.register_shard(fake2, "test/model2")
    unharvested = registry.get_unharvested()
    assert len(unharvested) == 1
    assert unharvested[0].name == "b.safetensors"
    print("[7] Unharvested tracking OK")

    # 8. Registry dedup
    registry.register_shard(fake1, "qwen/Qwen2.5-7B-Instruct")
    assert len(registry._data["shards"]) == 2
    print("[8] Deduplication OK")

    shutil.rmtree(tmpdir)
    print("\nSAH REGISTRY / AUTO-DOWNLOADER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
