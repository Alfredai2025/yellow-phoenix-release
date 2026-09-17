#!/usr/bin/env python3
"""SAH Stage 5 smoke test: adaptive threshold tuning."""

import sys
import tempfile
import os

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from scripts.sah_adaptive_loop import SAHAdaptiveLoop


def main():
    # 1. Create loop with default thresholds
    loop = SAHAdaptiveLoop()
    assert loop.thresholds[0x01] == 50
    print("[1] Adaptive loop created, default threshold tag 0x01 = 50")

    # 2. Simulate low hit rate → threshold should increase
    for _ in range(100):
        loop.record(0x01, "fallback")
    for _ in range(10):
        loop.record(0x01, "beacon", best_distance=45)

    rate_before = loop.hit_rate(0x01)
    print(f"[2] Hit rate before tune: {rate_before:.2%}")

    new_thresholds = loop.tune()
    assert new_thresholds[0x01] > 50, f"Expected increase, got {new_thresholds[0x01]}"
    print(f"[3] Threshold relaxed: 50 → {new_thresholds[0x01]}")

    # 3. Simulate high hit rate → threshold should decrease
    loop2 = SAHAdaptiveLoop(thresholds={0x01: 150, 0x02: 150})
    for _ in range(100):
        loop2.record(0x01, "beacon", best_distance=30)
    for _ in range(5):
        loop2.record(0x01, "fallback")

    new2 = loop2.tune()
    assert new2[0x01] < 150, f"Expected decrease, got {new2[0x01]}"
    print(f"[4] Threshold tightened: 150 → {new2[0x01]}")

    # 4. Bounds enforcement
    loop3 = SAHAdaptiveLoop(thresholds={0x01: 5})
    for _ in range(100):
        loop3.record(0x01, "fallback")
    new3 = loop3.tune()
    assert new3[0x01] >= loop3.MIN_THRESHOLD, f"Below min: {new3[0x01]}"
    print(f"[5] Min bound enforced: {new3[0x01]} >= {loop3.MIN_THRESHOLD}")

    loop4 = SAHAdaptiveLoop(thresholds={0x01: 250})
    for _ in range(100):
        loop4.record(0x01, "beacon", best_distance=10)
    new4 = loop4.tune()
    assert new4[0x01] <= loop4.MAX_THRESHOLD, f"Above max: {new4[0x01]}"
    print(f"[6] Max bound enforced: {new4[0x01]} <= {loop4.MAX_THRESHOLD}")

    # 5. Save / load round-trip
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "sah_adaptive.json")
    loop.save(path)
    restored = SAHAdaptiveLoop.load(path)
    assert restored.thresholds[0x01] == loop.thresholds[0x01]
    print(f"[7] Save/load round-trip OK: threshold={restored.thresholds[0x01]}")

    # 6. Apply to cascade mock
    class MockCascade:
        THRESHOLDS = {}
        DEFAULT_THRESHOLD = 0
    loop.apply_to_cascade(MockCascade)
    assert MockCascade.THRESHOLDS[0x01] == loop.thresholds[0x01]
    print("[8] apply_to_cascade works")

    os.remove(path)
    os.rmdir(tmpdir)

    print("\nSAH Stage 5 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
