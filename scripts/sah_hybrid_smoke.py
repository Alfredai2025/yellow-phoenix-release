#!/usr/bin/env python3
"""SAH Hybrid smoke: YPEngine.search_hybrid returns papers + substructure patterns."""

import sys
import os

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import YPEngine


def main() -> int:
    print("[1] Starting YPEngine (beacon index auto-loads)...")
    engine = YPEngine()

    beacon_count = engine.rust.beacon_index_count()
    print(f"[2] Beacon count: {beacon_count}")

    print("[3] Running search_hybrid...")
    results = engine.search_hybrid("neural hashing comparison", top_k=10)

    species = {}
    for r in results:
        species[r.get("species", "?")] = species.get(r.get("species", "?"), 0) + 1

    print(f"[4] Top-{len(results)} blended results: {species}")
    for r in results[:5]:
        label = r.get("title") or r.get("tag_name") or r.get("tensor") or r.get("id", "?")
        print(f"    {r['species']:6} | score={r['score']:.4f} | {str(label)[:60]}")

    # Sanity checks
    assert results, "search_hybrid must return results"
    assert any(r.get("species") == "paper" for r in results), "Expected at least one paper"
    if beacon_count > 0:
        assert any(r.get("species") == "mind" for r in results), "Expected at least one mind pattern when beacons exist"

    print("\nSAH HYBRID SMOKE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
