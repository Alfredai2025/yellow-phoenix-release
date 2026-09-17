#!/usr/bin/env python3
"""SAH Stage 7: Coverage-Driven Auto-Downloader — self-requests new models to fill gaps."""

import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from scripts.sah_registry import SAHRegistry
from scripts.sah_source_finder import SAHSourceFinder


class SAHAutoDownloader:
    """Downloads models until beacon coverage targets are met. Can search for new models."""

    SEED_CATALOG: List[Tuple[str, str, Dict[int, int]]] = [
        ("qwen/Qwen2.5-7B-Instruct", "transformer_attention", {0x01: 300, 0x02: 400, 0x03: 100}),
        ("deepseek-ai/DeepSeek-V2-Lite-Chat", "moe_routing", {0x01: 200, 0x02: 500, 0x04: 100}),
        ("ZhipuAI/chatglm3-6b", "chinese_embed", {0x01: 150, 0x02: 200, 0x03: 300}),
        ("baichuan-inc/Baichuan2-7B-Chat", "chinese_reasoning", {0x01: 250, 0x02: 350, 0x03: 50}),
        ("01-ai/Yi-1.5-6B-Chat", "general_baseline", {0x01: 200, 0x02: 250, 0x03: 100}),
    ]

    TAG_KEYWORDS = {
        0x01: ["attention", "transformer", "llama", "qwen", "gpt", "mamba"],
        0x02: ["mlp", "ffn", "feedforward", "deepseek", "mixtral", "moe"],
        0x03: ["embedding", "embed", "bge", "e5", "gte", "sentence-transformer"],
        0x04: ["vision", "audio", "multimodal", "clip", "whisper", "diffusion"],
    }

    DEFAULT_TARGETS = {
        0x01: 500,
        0x02: 500,
        0x03: 200,
        0x04: 100,
    }

    def __init__(self, finder: SAHSourceFinder, registry: SAHRegistry, targets: Optional[Dict[int, int]] = None):
        self.finder = finder
        self.registry = registry
        self.targets = targets or dict(self.DEFAULT_TARGETS)

    def _score_seed(self, model_id: str, contrib: Dict[int, int]) -> float:
        if model_id in self.registry.list_model_ids():
            return -1.0
        missing = self.registry.missing_tags(self.targets)
        if not missing:
            return 0.0
        score = sum(contrib.get(tag, 0) for tag in missing if tag in contrib)
        return score

    def _score_search_result(self, model_id: str, keywords_hit: List[int]) -> float:
        if model_id in self.registry.list_model_ids():
            return -1.0
        missing = self.registry.missing_tags(self.targets)
        if not missing:
            return 0.0
        score = sum(50 for tag in keywords_hit if tag in missing)
        return score

    def next_seed_model(self) -> Optional[Tuple[str, str]]:
        best_score = 0.0
        best = None
        for model_id, desc, contrib in self.SEED_CATALOG:
            score = self._score_seed(model_id, contrib)
            if score > best_score:
                best_score = score
                best = (model_id, desc)
        return best

    def discover_new_model(self) -> Optional[Tuple[str, str]]:
        missing = self.registry.missing_tags(self.targets)
        if not missing:
            return None

        target_tag = missing[0]
        keywords = self.TAG_KEYWORDS.get(target_tag, ["transformer"])

        for kw in keywords:
            try:
                results = self.finder.search_models(kw, limit=5)
            except Exception:
                continue
            for model_id, source in results:
                if source != "modelscope":
                    continue
                hit_tags = [t for t, kws in self.TAG_KEYWORDS.items() if any(w in model_id.lower() for w in kws)]
                score = self._score_search_result(model_id, hit_tags)
                if score > 0:
                    return (model_id, f"search_fill_{hex(target_tag)}")
        return None

    def next_model(self) -> Optional[Tuple[str, str]]:
        seed = self.next_seed_model()
        if seed is not None:
            return seed
        return self.discover_new_model()

    def auto_fill(self, max_models: int = 5) -> List[str]:
        downloaded = []
        for _ in range(max_models):
            nxt = self.next_model()
            if nxt is None:
                print("[SAH Auto] Coverage targets met. Stopping.")
                break

            model_id, reason = nxt
            print(f"\n[SAH Auto] Downloading {model_id} ({reason})...")
            try:
                self.finder.download_and_register(model_id, self.registry)
                downloaded.append(model_id)
            except Exception as e:
                print(f"[SAH Auto] FAILED {model_id}: {e}")
                continue

        return downloaded

    def status(self) -> dict:
        current = self.registry.coverage_summary()
        missing = self.registry.missing_tags(self.targets)
        nxt = self.next_model()
        return {
            "current": current,
            "targets": {hex(k): v for k, v in self.targets.items()},
            "missing_tags": [hex(t) for t in missing],
            "next_seed": self.next_seed_model(),
            "next_discovered": self.discover_new_model(),
            "next_model": nxt,
        }
