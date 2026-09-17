#!/usr/bin/env python3
"""Generate lightweight synthetic papers for 10M and 20M shootout datasets.

ISM format matches hnsw_extract_ism.rs:
    uint64 count
    count * 64 bytes hashes
    count * uint64 ids

No abstracts, no embeddings kept in memory — just random 512-bit hashes.
This keeps ISM files small enough for iPhone storage (~72 bytes / paper).
"""

import os
import struct
from pathlib import Path
import shutil
import numpy as np

DATA_DIR = Path("/Users/mac/yellow_phoenix/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Toggle to also write titles metadata JSONL. Not required by the iOS app,
# and it adds ~2 GB for 20M, so it defaults to False.
WRITE_METADATA = False

SEED = 42
np.random.seed(SEED)


def sizeof_fmt(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def generate_dataset(name: str, count: int):
    print(f"\n{'=' * 60}")
    print(f"Generating {name}: {count:,} papers")

    ism_path = DATA_DIR / f"papers_{name}.ism"
    hnsw_path = DATA_DIR / f"binary_hnsw_{name}.bin"

    # Write ISM in one pass: count + hashes + ids
    print("  Writing 512-bit hashes + ids...")
    with open(ism_path, "wb") as f:
        f.write(struct.pack("<Q", count))

        # Stream in 1M-row chunks to keep memory modest
        chunk = 1_000_000
        for start in range(0, count, chunk):
            rows = min(chunk, count - start)
            hashes = np.random.randint(0, 256, size=(rows, 64), dtype=np.uint8)
            f.write(hashes.tobytes())

        ids = np.arange(count, dtype=np.uint64)
        f.write(ids.tobytes())

    print(f"  ✅ ISM: {ism_path} ({sizeof_fmt(ism_path.stat().st_size)})")

    if WRITE_METADATA:
        meta_path = DATA_DIR / f"synthetic_{name}_metadata.jsonl"
        topics = ["Neural", "Quantum", "Crypto", "Bio", "AI"]
        with open(meta_path, "w") as f:
            for i in range(count):
                topic = topics[i % len(topics)]
                title = f"Synthetic paper {i:08d} on topic {topic}"
                f.write(f'{{"id":{i},"title":"{title}"}}\n')
        print(f"  ✅ Meta: {meta_path} ({sizeof_fmt(meta_path.stat().st_size)})")

    # HNSW placeholder / real index
    src_1m = DATA_DIR / "binary_hnsw_arxiv1m_m16.bin"
    src_10m_real = DATA_DIR / "binary_hnsw_10m_verify.bin"

    if src_1m.exists():
        shutil.copy(src_1m, hnsw_path)
        print(f"  ⚠️  HNSW: copied 1M index as placeholder ({sizeof_fmt(hnsw_path.stat().st_size)})")
        print("     (Replace with a real 10M/20M HNSW for accurate shootout numbers)")
    else:
        print(f"  ❌ No 1M HNSW found to copy. Build real HNSW for {name}.")

    total = sum(
        p.stat().st_size for p in [ism_path, hnsw_path] if p.exists()
    )
    print(f"  Total dataset size: {sizeof_fmt(total)}")
    return ism_path, hnsw_path


if __name__ == "__main__":
    generate_dataset("10m", 10_000_000)
    generate_dataset("20m", 20_000_000)
    print(f"\n{'=' * 60}")
    print("Done. Files in:", DATA_DIR)
    print("\nPush to iPhone with:")
    print("  cd /Users/mac/yellow_phoenix_mobile && ./scripts/push_now.sh")
