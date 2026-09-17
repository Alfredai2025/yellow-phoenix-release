#!/usr/bin/env python3
"""Generate lightweight synthetic benchmarks for all scales.

No real papers — fake titles only. Produces ISM flat indexes and small
HNSW placeholders (copied from the 1M HNSW) so the shootout UI works
without multi-gigabyte real indexes.

ISM format matches the Rust loader:
    uint64 count (little-endian)
    count * 64 bytes hashes
    count * uint64 ids
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np

DATA_DIR = Path("/Users/mac/yellow_phoenix/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

TOPICS = ["Neural", "Quantum", "Crypto", "Bio", "AI", "Math", "Physics", "Chem"]

# Metadata JSONL is generated only when True. It is not used by the iOS app,
# and the 20M file would be ~1.5 GB, so it defaults to False.
WRITE_METADATA = False


def sizeof_fmt(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"


def generate(scale: str, count: int) -> None:
    print(f"\n{'='*60}")
    print(f"Generating {scale}: {count:,} synthetic papers")

    ism_path = DATA_DIR / f"synth_{scale}.ism"

    # Write ISM in one pass: count + hashes + ids
    print("  Writing 512-bit hashes + ids...")
    with open(ism_path, "wb") as f:
        f.write(struct.pack("<Q", count))

        chunk = 1_000_000
        for start in range(0, count, chunk):
            rows = min(chunk, count - start)
            hashes = np.random.randint(0, 256, size=(rows, 64), dtype=np.uint8)
            f.write(hashes.tobytes())

        ids = np.arange(count, dtype=np.uint64)
        f.write(ids.tobytes())

    print(f"  ✅ ISM: {ism_path.name} ({sizeof_fmt(ism_path.stat().st_size)})")

    if WRITE_METADATA:
        meta_path = DATA_DIR / f"synth_{scale}_meta.jsonl"
        with open(meta_path, "w") as f:
            for i in range(count):
                topic = TOPICS[i % len(TOPICS)]
                title = f"Synthetic paper {i:09d} on {topic} networks"
                f.write(json.dumps({"id": i, "title": title, "topic": topic}) + "\n")
        print(f"  ✅ Meta: {meta_path.name} ({sizeof_fmt(meta_path.stat().st_size)})")

    # HNSW placeholder: copy the 1M HNSW so the shootout buttons work
    hnsw_path = DATA_DIR / f"synth_{scale}_hnsw.bin"
    src_1m = DATA_DIR / "binary_hnsw_arxiv1m_m16.bin"
    if src_1m.exists():
        shutil.copy(src_1m, hnsw_path)
        print(f"  ⚠️  HNSW: copied from 1M placeholder ({sizeof_fmt(hnsw_path.stat().st_size)})")
    else:
        print(f"  ❌ No 1M HNSW found. Build real HNSW for accurate numbers.")


if __name__ == "__main__":
    generate("106k", 106_000)
    generate("1m", 1_000_000)
    generate("10m", 10_000_000)
    generate("20m", 20_000_000)
    print(f"\n{'='*60}")
    print("Done. Push to iPhone with:")
    print("  cd /Users/mac/yellow_phoenix_mobile && ./scripts/push_now.sh")
