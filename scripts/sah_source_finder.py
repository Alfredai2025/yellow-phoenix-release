#!/usr/bin/env python3
"""SAH Source Finder — geo-aware, no-login LLM shard discovery."""

import os
import sys
import time
import socket
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, "/Users/mac/yellow_phoenix")

# Optional imports — gracefully degrade if not installed
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_download
    from modelscope.hub.api import HubApi
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False
    ms_download = None
    HubApi = None

try:
    from huggingface_hub import snapshot_download as hf_download
    from huggingface_hub import list_models as hf_list
    HAS_HF = True
except ImportError:
    HAS_HF = False
    hf_download = None
    hf_list = None


class SAHSourceFinder:
    """Auto-discovers network region and picks the best source."""

    CACHE_DIR = Path("/Users/mac/yellow_phoenix/data/sah_models")
    TIMEOUT_SEC = 3

    # Curated seed list — same IDs across sources where possible
    SEED_MODELS = [
        ("qwen/Qwen2.5-7B-Instruct", "Alibaba Qwen 7B — attention + FFN + embed"),
        ("deepseek-ai/DeepSeek-V2-Lite-Chat", "DeepSeek MoE — unique routing patterns"),
        ("ZhipuAI/chatglm3-6b", "ChatGLM3 — Chinese-optimized embeddings"),
        ("baichuan-inc/Baichuan2-7B-Chat", "Baichuan 7B — Chinese reasoning"),
        ("01-ai/Yi-1.5-6B-Chat", "Yi 6B — general baseline"),
    ]

    def __init__(self):
        self.cache_dir = self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.region = self._detect_region()
        print(f"[SAH Source] Region detected: {self.region}")

    # ── Network Detection ─────────────────────────────────────────

    def _can_reach(self, host: str, port: int = 443, timeout: float = 3.0) -> bool:
        """TCP connectivity probe. No HTTP needed."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    def _detect_region(self) -> str:
        """Detect if we're on mainland China (HF blocked) or global."""
        hf_ok = self._can_reach("huggingface.co", timeout=1.5)
        if hf_ok:
            return "global"
        ms_ok = self._can_reach("www.modelscope.cn", timeout=1.5)
        if ms_ok:
            return "china_mainland"
        return "unknown"

    # ── Source Selection ─────────────────────────────────────────

    def _pick_source(self, model_id: str) -> Tuple[str, callable]:
        """Return (source_name, download_function) for this model."""
        if self.region == "global" and HAS_HF:
            return ("huggingface", lambda m, p: self._hf_download(m, p))
        if HAS_MODELSCOPE:
            return ("modelscope", lambda m, p: self._ms_download(m, p))
        if HAS_HF:
            return ("hf_mirror", lambda m, p: self._hf_mirror_download(m, p))
        raise RuntimeError(
            "No download backend available. Install: pip install modelscope huggingface-hub"
        )

    # ── Download Backends ────────────────────────────────────────

    def _hf_download(self, model_id: str, local_dir: Path) -> Path:
        """Standard Hugging Face download. No login for public models."""
        if not HAS_HF:
            raise RuntimeError("huggingface-hub not installed")
        return hf_download(
            model_id,
            cache_dir=str(local_dir),
            allow_patterns=["*.safetensors", "*.json", "config.json"],
            resume_download=True,
        )

    def _hf_mirror_download(self, model_id: str, local_dir: Path) -> Path:
        """HF via hf-mirror.com proxy. No login."""
        if not HAS_HF:
            raise RuntimeError("huggingface-hub not installed")
        old_endpoint = os.environ.get("HF_ENDPOINT", "")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            return hf_download(
                model_id,
                cache_dir=str(local_dir),
                allow_patterns=["*.safetensors", "*.json", "config.json"],
                resume_download=True,
            )
        finally:
            if old_endpoint:
                os.environ["HF_ENDPOINT"] = old_endpoint
            else:
                os.environ.pop("HF_ENDPOINT", None)

    def _ms_download(self, model_id: str, local_dir: Path) -> Path:
        """ModelScope download. No login for public models."""
        if not HAS_MODELSCOPE:
            raise RuntimeError("modelscope not installed")
        return ms_download(
            model_id,
            cache_dir=str(local_dir),
            allow_patterns=["*.safetensors", "*.json", "config.json"],
        )

    # ── Public API ───────────────────────────────────────────────

    def download_model(self, model_id: str) -> Path:
        """Download one model using the best available source."""
        source_name, downloader = self._pick_source(model_id)
        local_path = self.cache_dir / model_id.replace("/", "--")
        local_path.mkdir(parents=True, exist_ok=True)

        # Skip if already cached
        existing = list(local_path.rglob("*.safetensors"))
        if existing:
            print(f"[SAH Source] Cached: {model_id} ({len(existing)} shards)")
            return local_path

        print(f"[SAH Source] Downloading from {source_name}: {model_id}")
        start = time.time()
        try:
            downloader(model_id, local_path)
            elapsed = time.time() - start
            shards = list(local_path.rglob("*.safetensors"))
            print(f"[SAH Source] Done: {len(shards)} shards in {elapsed:.1f}s")
            return local_path
        except Exception as e:
            print(f"[SAH Source] FAILED ({source_name}): {e}")
            # Try next source if available
            if source_name == "huggingface" and HAS_MODELSCOPE:
                print("[SAH Source] Retrying via ModelScope...")
                return self._ms_download(model_id, local_path)
            raise

    def download_seed_list(self, max_models: int = 3) -> List[Path]:
        """Download first N seed models, skipping failures."""
        downloaded = []
        for model_id, desc in self.SEED_MODELS[:max_models]:
            print(f"\n[SAH Source] {desc}")
            try:
                path = self.download_model(model_id)
                downloaded.append(path)
            except Exception as e:
                print(f"  SKIP: {e}")
        return downloaded

    def find_local_shards(self, model_path: Path) -> List[Path]:
        """Find all .safetensors files in a downloaded model dir."""
        return sorted(model_path.rglob("*.safetensors"))

    def search_models(self, keyword: str, limit: int = 10) -> List[Tuple[str, str]]:
        """Search ModelScope public catalog. No login required."""
        if not HAS_MODELSCOPE:
            return []
        try:
            api = HubApi()
            # ModelScope LegacyHubApi.list_models takes owner_or_group as positional arg.
            # If keyword looks like 'owner/model', search under the owner namespace.
            owner = keyword.split("/")[0] if "/" in keyword else keyword
            results = api.list_models(owner)
            matches = []
            for m in results:
                name = getattr(m, "Name", "") or getattr(m, "ModelId", "")
                if keyword.lower() in name.lower():
                    matches.append((name, "modelscope"))
                    if len(matches) >= limit:
                        break
            return matches
        except Exception as e:
            print(f"[SAH Source] Search failed: {e}")
            return []

    def download_and_register(self, model_id: str, registry) -> Path:
        """Download and auto-register all discovered shards in registry."""
        path = self.download_model(model_id)
        shards = self.find_local_shards(path)
        for shard in shards:
            registry.register_shard(shard, model_id)
        return path
