#!/usr/bin/env python3
"""SAH Stage 4: Cascade Prime — beacon-first query routing."""

import sys
from typing import List, Tuple, Optional

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge


class SAHCascade:
    """Query router: beacon index first, production fallback."""

    # Distance thresholds by tag (Hamming distance, lower = better)
    # Tuned for mixed concept + document beacons:
    #   - concept anchors (attention/embed) allow loose partial matches
    #   - document beacons (paper hashes, tag 2/5) require tight matches
    THRESHOLDS = {
        0x01: 120,  # ATTENTION — concept beacons, partial-match tolerant
        0x02: 20,   # FFN / document beacons — tight exact-title matches
        0x03: 120,  # EMBED — concept beacons, partial-match tolerant
        0x04: 80,   # OTHER — generic fallback
        0x05: 20,   # DOCUMENT beacons — tight exact-title matches
    }

    DEFAULT_THRESHOLD = 120
    DEFAULT_BEACON_K = 5

    def __init__(self, bridge: RustBridge, production_search_fn=None):
        """
        bridge: RustBridge with beacon_index_search available
        production_search_fn: callable(query_hash_bytes, k) -> List[(node_id, distance)]
                              or None to skip fallback
        """
        self.bridge = bridge
        self.production_search = production_search_fn

    def beacon_search(self, query_hash: bytes, k: int = 5) -> List[Tuple[int, int, int]]:
        """Search beacon index. Returns [(node_id, distance, tag)]."""
        return self.bridge.beacon_index_search(query_hash, k=k)

    def _best_beacon(self, results: List[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
        """Pick best beacon result if it passes threshold."""
        if not results:
            return None
        # Sort by distance ascending
        best = min(results, key=lambda x: x[1])
        node_id, dist, tag = best
        threshold = self.THRESHOLDS.get(tag, self.DEFAULT_THRESHOLD)
        if dist <= threshold:
            return best
        return None

    def query(self, query_hash: bytes, k: int = 5) -> dict:
        """
        Full cascade query.
        Returns dict with keys:
          - 'source': 'beacon' or 'fallback' or 'empty'
          - 'results': list of (node_id, distance, tag)
          - 'latency_hint': 'fast' or 'slow'
        """
        # 1. Beacon probe
        beacon_results = self.beacon_search(query_hash, k=self.DEFAULT_BEACON_K)
        best = self._best_beacon(beacon_results)

        if best is not None:
            # Beacon hit — return beacon results filtered by threshold
            filtered = [
                r for r in beacon_results
                if r[1] <= self.THRESHOLDS.get(r[2], self.DEFAULT_THRESHOLD)
            ]
            return {
                "source": "beacon",
                "results": filtered[:k],
                "latency_hint": "fast",
            }

        # 2. Fallback to production
        if self.production_search is not None:
            fallback = self.production_search(query_hash, k)
            return {
                "source": "fallback",
                "results": fallback,
                "latency_hint": "slow",
            }

        # 3. Nothing available
        return {
            "source": "empty",
            "results": [],
            "latency_hint": "fast",
        }

    def query_batch(self, query_hashes: List[bytes], k: int = 5) -> List[dict]:
        """Batch cascade query."""
        return [self.query(h, k=k) for h in query_hashes]
