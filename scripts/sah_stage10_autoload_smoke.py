#!/usr/bin/env python3
"""SAH Stage 10 autoload smoke: YPEngine auto-loads persisted beacon index on startup."""

import sys
import os

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import YPEngine


def main() -> int:
    print("[1] Starting YPEngine (this triggers beacon auto-load)...")
    engine = YPEngine()

    count = engine.rust.beacon_index_count()
    print(f"[2] Beacon index count after init: {count}")

    beacon_path = getattr(engine, "_sah_beacon_path", os.path.join(os.path.expanduser("~/yellow_phoenix"), "data", "sah_beacon_index.bin"))
    if os.path.exists(beacon_path):
        assert count > 0, f"Expected beacons to be auto-loaded from {beacon_path}, got {count}"
        print(f"[OK] Beacon index auto-loaded from {beacon_path}")
    else:
        print(f"[SKIP] No persisted beacon index at {beacon_path}; auto-load not applicable")

    print("[3] Running search_with_sah on a natural-language query...")
    result = engine.search_with_sah("machine learning", k=5, use_cascade=True)
    assert "results" in result, "search_with_sah must return 'results'"
    assert "source" in result, "search_with_sah must return 'source'"
    assert "latency_ms" in result, "search_with_sah must return 'latency_ms'"
    print(f"[4] source={result['source']}, latency_ms={result['latency_ms']:.3f}, results={len(result['results'])}")

    print("[5] Verifying production bypass path...")
    prod = engine.search_with_sah("machine learning", k=5, use_cascade=False)
    assert prod["source"] == "production"
    print(f"[OK] production bypass works ({len(prod['results'])} results)")

    print("\nSAH Stage 10 autoload smoke: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
