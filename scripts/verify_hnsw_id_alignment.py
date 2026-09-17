#!/usr/bin/env python3
"""Verify that rebuilt HNSW files use the same IDs as their source ISM files.

Prints PASS/FAIL per scale. A failed HNSW must not be copied to the iPhone.
"""
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SCALES = [
    ("106K", "synth_106k.ism", "synth_106k_hnsw.bin"),
    ("1M",   "synth_1m.ism",   "synth_1m_hnsw.bin"),
    ("5M",   "synth_5m.ism",   "synth_5m_hnsw.bin"),
    ("10M",  "synth_10m.ism",  "synth_10m_hnsw.bin"),
    ("20M",  "synth_20m.ism",  "synth_20m_hnsw.bin"),
]


def load_ism_ids(path: Path):
    with open(path, "rb") as f:
        count = struct.unpack("<Q", f.read(8))[0]
        # Skip hashes, then read ids.
        f.seek(count * 64, os.SEEK_CUR)
        ids = struct.unpack(f"<{count}Q", f.read(count * 8))
    return ids


def load_hnsw_ids(path: Path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"YPH5":
            raise ValueError(f"bad magic in {path}: {magic!r}")
        version = f.read(1)[0]
        if version != 4:
            raise ValueError(f"unsupported version {version} in {path}")
        m, ef_construction, ef_search, max_layers = struct.unpack("<4Q", f.read(32))
        ep_raw = struct.unpack("<I", f.read(4))[0]
        node_count = struct.unpack("<Q", f.read(8))[0]

        ids = []
        for i in range(node_count):
            node_id = struct.unpack("<Q", f.read(8))[0]
            ids.append(node_id)
            # Skip hash, tag, num_layers, layer_counts, alt_layer_counts
            f.seek(64 + 1 + 1 + 16 + 16, os.SEEK_CUR)
            num_layers = 0  # already skipped; read from earlier would need buffering
            # Actually we skipped num_layers above, so we cannot know it.  Fix below.
            _ = i
        return ids


def load_hnsw_ids_fixed(path: Path):
    """Stream-read only the IDs from a YPH5v4 HNSW file."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"YPH5":
            raise ValueError(f"bad magic in {path}: {magic!r}")
        version = f.read(1)[0]
        if version != 4:
            raise ValueError(f"unsupported version {version} in {path}")
        m, ef_construction, ef_search, max_layers = struct.unpack("<4Q", f.read(32))
        ep_raw = struct.unpack("<I", f.read(4))[0]
        node_count = struct.unpack("<Q", f.read(8))[0]

        ids = []
        for i in range(node_count):
            node_id = struct.unpack("<Q", f.read(8))[0]
            ids.append(node_id)
            # Skip 64-byte hash
            f.seek(64, os.SEEK_CUR)
            # tag + num_layers
            tag = f.read(1)[0]
            num_layers = f.read(1)[0]
            # layer_counts[16] + alt_layer_counts[16]
            f.seek(16 + 16, os.SEEK_CUR)
            # Skip primary/alternative neighbor lists for each layer
            for _layer in range(num_layers):
                cnt = struct.unpack("<I", f.read(4))[0]
                f.seek(cnt * 4, os.SEEK_CUR)
                alt_cnt = struct.unpack("<I", f.read(4))[0]
                f.seek(alt_cnt * 4, os.SEEK_CUR)
            if (i + 1) % 1_000_000 == 0:
                print(f"  HNSW {i + 1:,} / {node_count:,}", file=sys.stderr)
        return ids


def check_scale(name: str, ism_name: str, hnsw_name: str) -> bool:
    ism_path = DATA / ism_name
    hnsw_path = DATA / hnsw_name
    print(f"\n[{name}] {ism_name} <-> {hnsw_name}")
    if not ism_path.exists():
        print(f"  FAIL: ISM missing {ism_path}")
        return False
    if not hnsw_path.exists():
        print(f"  FAIL: HNSW missing {hnsw_path}")
        return False

    print("  Loading ISM IDs...", file=sys.stderr)
    ism_ids = load_ism_ids(ism_path)
    print(f"  ISM records: {len(ism_ids):,}", file=sys.stderr)

    print("  Loading HNSW IDs...", file=sys.stderr)
    hnsw_ids = load_hnsw_ids_fixed(hnsw_path)
    print(f"  HNSW nodes:  {len(hnsw_ids):,}", file=sys.stderr)

    ism_set = set(ism_ids)
    hnsw_set = set(hnsw_ids)

    ok = True
    if len(ism_ids) != len(ism_set):
        print(f"  FAIL: ISM has {len(ism_ids) - len(ism_set)} duplicate IDs")
        ok = False
    if len(hnsw_ids) != len(hnsw_set):
        print(f"  FAIL: HNSW has {len(hnsw_ids) - len(hnsw_set)} duplicate IDs")
        ok = False
    if len(hnsw_ids) != len(ism_ids):
        print(f"  FAIL: count mismatch ISM={len(ism_ids):,} HNSW={len(hnsw_ids):,}")
        ok = False

    missing = ism_set - hnsw_set
    extra = hnsw_set - ism_set
    if missing:
        print(f"  FAIL: {len(missing):,} ISM IDs missing from HNSW")
        ok = False
    if extra:
        print(f"  FAIL: {len(extra):,} HNSW IDs not in ISM")
        ok = False

    if ok:
        print(f"  PASS: {len(hnsw_ids):,} IDs aligned")
    else:
        print(f"  FAIL: IDs do not align")
    return ok


def main() -> int:
    all_ok = True
    for name, ism, hnsw in SCALES:
        if not check_scale(name, ism, hnsw):
            all_ok = False

    print("\n" + "=" * 40)
    if all_ok:
        print("ALL SCALES PASS")
        return 0
    else:
        print("SOME SCALES FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
