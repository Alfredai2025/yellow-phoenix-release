#!/usr/bin/env python3
"""SAH Stage 7: Shard Registry — tracks harvested shards, coverage counts, avoids re-work."""

import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional


class SAHRegistry:
    """Atomic JSON registry for all downloaded and harvested shards."""

    REGISTRY_PATH = Path("/Users/mac/yellow_phoenix/data/sah_models/.sah_registry.json")

    def __init__(self):
        self.path = self.REGISTRY_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "shards": [], "beacon_index_count": 0}
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    def register_shard(self, shard_path: Path, model_id: str, arch: str = "unknown", params: str = "unknown"):
        rel_path = str(shard_path.relative_to(self.path.parent))
        for s in self._data["shards"]:
            if s["path"] == rel_path:
                return
        self._data["shards"].append({
            "path": rel_path,
            "model_id": model_id,
            "arch": arch,
            "params": params,
            "harvested": False,
            "beacons": 0,
            "tags": {},
            "downloaded_at": time.time(),
            "last_harvested": None,
        })
        self._save()

    def mark_harvested(self, shard_path: Path, beacons: int, tags: Dict[int, int]):
        rel_path = str(shard_path.relative_to(self.path.parent))
        for s in self._data["shards"]:
            if s["path"] == rel_path:
                s["harvested"] = True
                s["beacons"] = beacons
                s["tags"] = {hex(k): v for k, v in tags.items()}
                s["last_harvested"] = time.time()
                break
        self._recalc_total()
        self._save()

    def is_harvested(self, shard_path: Path) -> bool:
        rel_path = str(shard_path.relative_to(self.path.parent))
        for s in self._data["shards"]:
            if s["path"] == rel_path:
                return s["harvested"]
        return False

    def get_unharvested(self) -> List[Path]:
        out = []
        for s in self._data["shards"]:
            if not s["harvested"]:
                out.append(self.path.parent / s["path"])
        return out

    def _recalc_total(self):
        self._data["beacon_index_count"] = sum(
            s["beacons"] for s in self._data["shards"] if s["harvested"]
        )

    def count_beacons_by_tag(self) -> Dict[int, int]:
        totals = {}
        for s in self._data["shards"]:
            if not s["harvested"]:
                continue
            for hex_tag, count in s.get("tags", {}).items():
                tag = int(hex_tag, 16)
                totals[tag] = totals.get(tag, 0) + count
        return totals

    def coverage_summary(self) -> dict:
        by_tag = self.count_beacons_by_tag()
        return {
            "total_beacons": self._data["beacon_index_count"],
            "total_shards": len(self._data["shards"]),
            "harvested_shards": sum(1 for s in self._data["shards"] if s["harvested"]),
            "by_tag": {hex(k): v for k, v in by_tag.items()},
        }

    def missing_tags(self, targets: Dict[int, int]) -> List[int]:
        current = self.count_beacons_by_tag()
        missing = []
        for tag, target in targets.items():
            if current.get(tag, 0) < target:
                missing.append(tag)
        return missing

    def list_model_ids(self) -> set:
        return set(s["model_id"] for s in self._data["shards"])
