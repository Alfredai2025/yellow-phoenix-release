#!/usr/bin/env python3
"""
SAH Stage 3: Harvester — reads safetensor shards, extracts eigenvectors, feeds beacon index.

SAF HARVESTER — CRITICAL NOTE: BFLOAT16 HANDLING
================================================
Modern LLM safetensors (Qwen, DeepSeek, Llama-3) use bfloat16 weights.
NumPy cannot read bfloat16. This module uses torch as a converter:
    tensor.to(torch.float32).detach().cpu().numpy()
If torch is unavailable, bfloat16 tensors are skipped.

DO NOT revert to framework="np" alone — it will crash.
This is a production requirement, not optional.
"""

import json
import os
import sys
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from scripts.sah_itq_loader import load_rotation_matrix, has_itq_model, get_fallback_rotation

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import RustBridge


class SAHHarvester:
    """Reads .safetensors shards, SVD-decomposes 2D weight matrices,
    converts top-k singular vectors to beacon hashes via Stage 2 bridge."""

    TAG_ATTENTION = 0x01   # concept beacon
    TAG_FFN       = 0x02   # reasoning beacon
    TAG_EMBED     = 0x03   # topic beacon
    TAG_OTHER     = 0x04   # generic beacon

    def __init__(self, bridge: RustBridge, rotation_matrix: Optional[List[List[float]]] = None, max_dim: int = 512):
        self.bridge = bridge
        if rotation_matrix is None:
            if has_itq_model():
                self.rotation = load_rotation_matrix()
                print("[SAH Harvester] Loaded real ITQ rotation")
            else:
                self.rotation = get_fallback_rotation()
                print("[SAH Harvester] WARNING: ITQ model not found, using fallback identity")
        else:
            self.rotation = rotation_matrix
        self.max_dim = max_dim
        self.stats = {"shards": 0, "tensors": 0, "beacons": 0, "skipped": 0}
        self.beacon_meta = {}

    def discover_shards(self, directory: str) -> List[str]:
        """Find all .safetensors files recursively."""
        return sorted(str(p) for p in Path(directory).rglob("*.safetensors"))

    @classmethod
    def tag_name(cls, tag: int) -> str:
        return {
            cls.TAG_ATTENTION: "attention",
            cls.TAG_FFN: "ffn",
            cls.TAG_EMBED: "embed",
            cls.TAG_OTHER: "other",
        }.get(tag, "unknown")

    def classify_tag(self, tensor_name: str) -> int:
        """Map tensor name to beacon tag."""
        name = tensor_name.lower()
        if any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "attn", "attention"]):
            return self.TAG_ATTENTION
        if any(x in name for x in ["gate_proj", "up_proj", "down_proj", "mlp", "ffn", "feed_forward"]):
            return self.TAG_FFN
        if any(x in name for x in ["embed", "lm_head", "token"]):
            return self.TAG_EMBED
        return self.TAG_OTHER

    def compute_eigenvectors(self, tensor: np.ndarray, k: int = 3) -> List[tuple]:
        """Compute top-k right singular vectors from a 2D matrix.
        Returns list of (normalized float32 vector length=max_dim, singular value)."""
        if tensor.ndim != 2:
            return []

        # Work with float32, transpose so rows >= cols for stable SVD
        matrix = tensor.astype(np.float32)
        if matrix.shape[0] < matrix.shape[1]:
            matrix = matrix.T

        # Guard: skip enormous matrices to avoid OOM
        max_svd = 4096
        if matrix.shape[0] > max_svd or matrix.shape[1] > max_svd:
            # Randomized subsample
            if matrix.shape[0] > max_svd:
                idx = np.random.choice(matrix.shape[0], max_svd, replace=False)
                matrix = matrix[idx, :]
            if matrix.shape[1] > max_svd:
                idx = np.random.choice(matrix.shape[1], max_svd, replace=False)
                matrix = matrix[:, idx]

        try:
            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        except Exception as e:
            print(f"  SVD failed: {e}")
            return []

        top_k = min(k, vh.shape[0])
        out = []
        for i in range(top_k):
            vec = vh[i].astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)

            # Pad or truncate to max_dim
            if len(vec) >= self.max_dim:
                vec = vec[:self.max_dim]
            else:
                vec = np.pad(vec, (0, self.max_dim - len(vec)), mode="constant")

            eigenvalue = float(s[i]) if i < len(s) else 0.0
            out.append((vec, eigenvalue))
        return out

    def harvest_shard(self, shard_path: str, beacons_per_tensor: int = 3) -> int:
        """Read one shard file and insert all beacons. Returns count inserted."""
        if safe_open is None:
            raise RuntimeError("safetensors library not installed")

        inserted = 0
        framework = "pt" if HAS_TORCH else "np"
        with safe_open(shard_path, framework=framework, device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                tensor = f.get_tensor(key)
                if HAS_TORCH and hasattr(tensor, "to"):
                    tensor = tensor.to(torch.float32).detach().cpu().numpy()
                self.stats["tensors"] += 1

                if tensor.ndim != 2:
                    self.stats["skipped"] += 1
                    continue

                tag = self.classify_tag(key)
                eigenvectors = self.compute_eigenvectors(tensor, k=beacons_per_tensor)
                shard_name = Path(shard_path).name
                model_id = Path(shard_path).parent.name

                for idx, (vec, eigenvalue) in enumerate(eigenvectors):
                    hash_bytes = self.bridge.sah_eigenvector_to_hash(vec.tolist(), self.rotation)
                    if len(hash_bytes) != 64:
                        continue

                    # Deterministic uint32 node_id from (path, tensor, idx)
                    node_id = int(hashlib.md5(f"{shard_path}:{key}:{idx}".encode()).hexdigest()[:8], 16)
                    rc = self.bridge.beacon_index_insert(hash_bytes, node_id=node_id, tag=tag)
                    if rc == 0:
                        inserted += 1
                        self.stats["beacons"] += 1
                        self.beacon_meta[str(node_id)] = {
                            "shard": shard_name,
                            "tensor": key,
                            "dim": idx,
                            "tag": tag,
                            "tag_name": self.tag_name(tag),
                            "eigenvalue": eigenvalue,
                            "model_id": model_id,
                        }

        self.stats["shards"] += 1
        return inserted

    def harvest_directory(self, directory: str, beacons_per_tensor: int = 3) -> Dict:
        """Harvest all shards in a directory. Returns stats dict."""
        shards = self.discover_shards(directory)
        total = 0
        for shard in shards:
            print(f"Harvesting {os.path.basename(shard)} ...")
            n = self.harvest_shard(shard, beacons_per_tensor)
            total += n
            print(f"  -> {n} beacons")
        self.stats["total_inserted"] = total
        return self.stats

    def save_meta(self, path: str = "data/sah_beacon_meta.json") -> int:
        """Persist beacon metadata sidecar mapping node_id -> source tensor info."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.beacon_meta, f, indent=2)
        print(f"[SAH Harvester] Saved beacon metadata: {path} ({len(self.beacon_meta)} entries)")
        return len(self.beacon_meta)
