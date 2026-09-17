#!/usr/bin/env python3
"""SAH Stage 5: Adaptive Loop — auto-tune beacon thresholds from query logs."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional


class SAHAdaptiveLoop:
    """Monitors query outcomes and adjusts per-tag Hamming thresholds."""

    # Target: 40% of queries should hit beacon fast-path
    TARGET_HIT_RATE = 0.40

    # Adjustment step size
    STEP = 5

    # Bounds (Hamming distance 0-512)
    MIN_THRESHOLD = 10
    MAX_THRESHOLD = 200

    def __init__(self, thresholds: Optional[Dict[int, int]] = None):
        """
        thresholds: initial dict {tag: distance}. If None, uses SAHCascade defaults.
        """
        self.thresholds = thresholds or {
            0x01: 50,
            0x02: 60,
            0x03: 70,
            0x04: 80,
        }
        self.stats = {
            tag: {"hits": 0, "fallbacks": 0, "empties": 0, "distances": []}
            for tag in self.thresholds.keys()
        }
        self.log_buffer: List[dict] = []

    def record(self, tag: int, source: str, best_distance: Optional[int] = None):
        """Record one query outcome."""
        if tag not in self.stats:
            self.stats[tag] = {"hits": 0, "fallbacks": 0, "empties": 0, "distances": []}

        if source == "beacon":
            self.stats[tag]["hits"] += 1
            if best_distance is not None:
                self.stats[tag]["distances"].append(best_distance)
        elif source == "fallback":
            self.stats[tag]["fallbacks"] += 1
        elif source == "empty":
            self.stats[tag]["empties"] += 1

        self.log_buffer.append({
            "ts": time.time(),
            "tag": tag,
            "source": source,
            "dist": best_distance,
        })

    def hit_rate(self, tag: int) -> float:
        """Return beacon hit rate for a tag."""
        s = self.stats.get(tag, {})
        total = s.get("hits", 0) + s.get("fallbacks", 0) + s.get("empties", 0)
        if total == 0:
            return 0.0
        return s.get("hits", 0) / total

    def avg_distance(self, tag: int) -> Optional[float]:
        """Return average beacon distance for hits."""
        d = self.stats.get(tag, {}).get("distances", [])
        if not d:
            return None
        return sum(d) / len(d)

    def tune(self) -> Dict[int, int]:
        """
        Adjust thresholds based on hit rates.
        Returns new thresholds dict.
        """
        for tag in list(self.thresholds.keys()):
            rate = self.hit_rate(tag)
            current = self.thresholds[tag]

            if rate < self.TARGET_HIT_RATE:
                # Too few hits — relax threshold
                new = current + self.STEP
            elif rate > self.TARGET_HIT_RATE + 0.15:
                # Too many hits (may be noisy) — tighten slightly
                new = current - self.STEP
            else:
                new = current

            # Clamp to bounds
            new = max(self.MIN_THRESHOLD, min(new, self.MAX_THRESHOLD))
            self.thresholds[tag] = new

        return dict(self.thresholds)

    def apply_to_cascade(self, cascade):
        """Push current thresholds into a SAHCascade instance."""
        cascade.THRESHOLDS.update(self.thresholds)
        cascade.DEFAULT_THRESHOLD = max(self.thresholds.values())

    def save(self, path: str):
        """Persist thresholds and stats to JSON."""
        data = {
            "thresholds": {hex(k): v for k, v in self.thresholds.items()},
            "stats": self.stats,
            "log_count": len(self.log_buffer),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SAHAdaptiveLoop":
        """Restore from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        thresholds = {int(k, 16): v for k, v in data["thresholds"].items()}
        loop = cls(thresholds=thresholds)
        loop.stats = data.get("stats", {})
        return loop
