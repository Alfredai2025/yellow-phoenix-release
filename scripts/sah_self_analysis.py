#!/usr/bin/env python3
"""SAH Stage 8: Self-Analysis — measures performance, detects gaps, requests new shards."""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from scripts.sah_registry import SAHRegistry
from scripts.sah_auto_downloader import SAHAutoDownloader


class SAHSelfAnalysis:
    """Periodic performance monitor that triggers shard acquisition."""

    MIN_HIT_RATE = 0.25
    MIN_RECALL_BY_TAG = 0.30

    TAG_NAMES = {
        0x01: "attention",
        0x02: "ffn",
        0x03: "embed",
        0x04: "other",
    }

    def __init__(
        self,
        registry: SAHRegistry,
        auto_downloader: SAHAutoDownloader,
        query_counter: Optional[Callable[[], int]] = None,
    ):
        self.registry = registry
        self.auto = auto_downloader
        self.query_counter = query_counter
        self.last_query_count = 0
        self.last_run = 0.0
        self.request_log: List[dict] = []

    def collect_metrics(self) -> dict:
        """Read current state from registry and cascade."""
        coverage = self.registry.coverage_summary()
        by_tag = self.registry.count_beacons_by_tag()

        hit_rate = 0.0
        try:
            from scripts.sah_adaptive_loop import SAHAdaptiveLoop
            loop_path = Path("/Users/mac/yellow_phoenix/data/sah_adaptive.json")
            if loop_path.exists():
                loop = SAHAdaptiveLoop.load(str(loop_path))
                rates = [loop.hit_rate(t) for t in self.auto.targets.keys()]
                hit_rate = sum(rates) / len(rates) if rates else 0.0
        except Exception:
            pass

        return {
            "timestamp": time.time(),
            "total_beacons": coverage["total_beacons"],
            "total_shards": coverage["total_shards"],
            "harvested_shards": coverage["harvested_shards"],
            "by_tag": by_tag,
            "beacon_hit_rate": hit_rate,
            "query_count": self.query_counter() if self.query_counter else 0,
        }

    def detect_gaps(self, metrics: dict) -> Dict[int, int]:
        """Return {tag: additional_beacons_needed}."""
        gaps = {}
        by_tag = metrics.get("by_tag", {})

        for tag, target in self.auto.targets.items():
            current = by_tag.get(tag, 0)
            if metrics.get("beacon_hit_rate", 1.0) < self.MIN_HIT_RATE:
                boost = int(target * 0.5)
                target += boost

            needed = max(0, target - current)
            if needed > 0:
                gaps[tag] = needed

        return gaps

    def analyze(self) -> dict:
        """Full analysis cycle. Returns report dict."""
        metrics = self.collect_metrics()
        gaps = self.detect_gaps(metrics)

        report = {
            "metrics": metrics,
            "gaps": {hex(k): v for k, v in gaps.items()},
            "action": "idle",
            "models_requested": [],
        }

        if gaps:
            old_targets = dict(self.auto.targets)
            for tag, needed in gaps.items():
                self.auto.targets[tag] = old_targets.get(tag, 0) + needed

            requested = self.auto.auto_fill(max_models=2)

            self.auto.targets = old_targets

            report["action"] = "requested"
            report["models_requested"] = requested

            for model_id in requested:
                self.request_log.append({
                    "ts": time.time(),
                    "model_id": model_id,
                    "gaps": dict(gaps),
                    "reason": "performance_gap",
                })

        return report

    def should_run(self, min_queries: int = 100, min_interval_sec: float = 3600) -> bool:
        """Return True if enough queries or time has passed."""
        current_queries = self.query_counter() if self.query_counter else 0
        queries_since = current_queries - self.last_query_count

        time_since = time.time() - self.last_run

        if queries_since >= min_queries or time_since >= min_interval_sec:
            self.last_query_count = current_queries
            self.last_run = time.time()
            return True
        return False

    def tick(self, min_queries: int = 100, min_interval_sec: float = 3600) -> Optional[dict]:
        """Check if analysis should run, and run if so."""
        if self.should_run(min_queries, min_interval_sec):
            return self.analyze()
        return None
