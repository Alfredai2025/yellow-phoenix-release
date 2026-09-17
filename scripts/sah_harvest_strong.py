#!/usr/bin/env python3
"""
SAH Harvest Strong — download a stronger model and rebuild the beacon index.

Self-contained. Respects HF_ENDPOINT for mirrors. Skips surface layers (0-5).
Stores eigenvalue + model_id in metadata. Mechanical naming only.
"""

import sys
import os
import re
import hashlib
import json
import time
from pathlib import Path
import numpy as np

YP_ROOT = "/Users/mac/yellow_phoenix"
sys.path.insert(0, YP_ROOT)

# ------------------------------------------------------------------
# 0. Dependencies
# ------------------------------------------------------------------
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: pip install huggingface_hub")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("ERROR: pip install safetensors")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
    print("WARNING: torch not found; bfloat16 safetensors will fail.")

from yp_bridge import RustBridge
from scripts.sah_registry import SAHRegistry

# ------------------------------------------------------------------
# 1. Config
# ------------------------------------------------------------------
MODEL_ID = "qwen/Qwen2.5-1.5B-Instruct"
CACHE_DIR = Path(YP_ROOT) / "data" / "sah_models" / MODEL_ID.replace("/", "--")
BEACONS_PER_TENSOR = 8
MAX_DIM = 512
LAYER_FILTER_MIN = 6  # skip layers 0-5 (surface noise)

# ------------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------------
def classify_tag(tensor_name: str) -> int:
    name = tensor_name.lower()
    if "embed_tokens" in name or "lm_head" in name:
        return 0x03
    if "self_attn" in name:
        return 0x01
    if "mlp" in name:
        return 0x02
    return 0x04

def is_deep_tensor(tensor_name: str) -> bool:
    """Skip shallow layers 0-5. Keep embeddings."""
    match = re.search(r'layers\.(\d+)', tensor_name)
    if not match:
        return "embed" in tensor_name.lower()
    return int(match.group(1)) >= LAYER_FILTER_MIN

def make_node_id(shard_name: str, tensor_name: str, idx: int) -> int:
    """Deterministic 32-bit node ID."""
    h = hashlib.sha256(f"{shard_name}:{tensor_name}:{idx}".encode()).hexdigest()
    return int(h, 16) % (2 ** 32)

def vec_to_hash512(vec: np.ndarray) -> bytes:
    """Sign-based binarization to 512-bit (64 bytes)."""
    bits = (vec >= 0).astype(np.uint8)
    out = bytearray(64)
    for i in range(0, 512, 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits) and bits[i + j]:
                byte |= 1 << (7 - j)
        out[i // 8] = byte
    return bytes(out)

# ------------------------------------------------------------------
# 3. Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SAH HARVEST STRONG")
    print(f"Model: {MODEL_ID}")
    print(f"Cache:  {CACHE_DIR}")
    print(f"Filter: layers >= {LAYER_FILTER_MIN}")
    print(f"Beacons per tensor: {BEACONS_PER_TENSOR}")
    print("=" * 60)

    bridge = RustBridge()
    print("\n[1] Rust bridge loaded.")

    bridge.beacon_index_new(100000)
    print("[2] Old beacon index cleared (100K capacity).")

    print(f"\n[3] Downloading {MODEL_ID}...")
    print("    (respects HF_ENDPOINT environment variable)")
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(CACHE_DIR),
            allow_patterns=["*.safetensors", "*.json"],
        )
    except KeyboardInterrupt:
        print("\n    Cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n    Download failed: {e}")
        sys.exit(1)
    print(f"    Done in {(time.time() - t0) / 60:.1f} min.")

    shards = sorted(CACHE_DIR.glob("*.safetensors"))
    print(f"\n[4] Found {len(shards)} shard(s).")

    stats = {"shards": 0, "tensors": 0, "beacons": 0, "skipped": 0}
    beacon_meta = {}
    framework = "pt" if HAS_TORCH else "np"

    for shard in shards:
        shard_name = shard.name
        print(f"\n  Harvesting {shard_name}...")
        shard_beacons = 0

        try:
            with safe_open(str(shard), framework=framework, device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    if HAS_TORCH and hasattr(tensor, "to"):
                        tensor = tensor.to(torch.float32).detach().cpu().numpy()

                    if tensor.ndim != 2:
                        continue
                    r, c = tensor.shape
                    if r < 64 or c < 64:
                        continue

                    if not is_deep_tensor(key):
                        stats["skipped"] += 1
                        continue

                    tag = classify_tag(key)

                    try:
                        U, s, Vt = np.linalg.svd(tensor.astype(np.float32), full_matrices=False)
                    except Exception as e:
                        print(f"    SVD fail {key}: {e}")
                        continue

                    stats["tensors"] += 1

                    for idx in range(min(BEACONS_PER_TENSOR, len(s))):
                        v = Vt[idx]
                        v512 = np.zeros(MAX_DIM, dtype=np.float32)
                        v512[:min(len(v), MAX_DIM)] = v[:min(len(v), MAX_DIM)]

                        hash_bytes = vec_to_hash512(v512)
                        node_id = make_node_id(shard_name, key, idx)

                        rc = bridge.beacon_index_insert(hash_bytes, node_id, tag)
                        if rc == 0:
                            shard_beacons += 1
                            stats["beacons"] += 1

                            layer_match = re.search(r'layers\.(\d+)', key)
                            beacon_meta[str(node_id)] = {
                                "shard": shard_name,
                                "tensor": key,
                                "dim": idx,
                                "tag": tag,
                                "tag_name": {0x01: "attention", 0x02: "ffn", 0x03: "embed", 0x04: "other"}.get(tag, "unknown"),
                                "eigenvalue": float(s[idx]),
                                "model_id": MODEL_ID,
                                "layer": int(layer_match.group(1)) if layer_match else -1,
                            }

        except Exception as e:
            print(f"    SKIP shard {shard_name}: {e}")
            continue

        stats["shards"] += 1
        print(f"    -> {shard_beacons} beacons (shallow skipped: {stats['skipped']})")

    print(f"\n[5] Harvest complete.")
    print(f"    Shards:  {stats['shards']}")
    print(f"    Tensors: {stats['tensors']}")
    print(f"    Beacons: {stats['beacons']}")
    print(f"    Skipped (surface): {stats['skipped']}")

    index_path = Path(YP_ROOT) / "data" / "sah_beacon_index.bin"
    meta_path = Path(YP_ROOT) / "data" / "sah_beacon_meta.json"

    bridge.beacon_index_save(str(index_path))
    with open(meta_path, "w") as f:
        json.dump(beacon_meta, f, indent=2)

    registry = SAHRegistry()
    for shard in shards:
        registry.register_shard(shard, MODEL_ID)
        registry.mark_harvested(shard, 0, {})

    print(f"\n[6] Saved.")
    print(f"    Index: {index_path}")
    print(f"    Meta:  {meta_path}")
    print(f"    Count: {bridge.beacon_index_count()}")

    tags = {}
    layers = {}
    for m in beacon_meta.values():
        tags[m["tag_name"]] = tags.get(m["tag_name"], 0) + 1
        if m["layer"] >= 0:
            layers[m["layer"]] = layers.get(m["layer"], 0) + 1

    print(f"\n[7] Tag distribution:")
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"    {tag:12} : {count}")

    print(f"\n[8] Layer distribution:")
    for layer, count in sorted(layers.items()):
        print(f"    layer {layer:2} : {count}")

    print("\n" + "=" * 60)
    print("HARVEST STRONG COMPLETE")
    print("Next: python scripts/sah_hybrid_smoke.py")
    print("=" * 60)
