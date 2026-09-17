#!/usr/bin/env python3
"""Convert a N x 64 uint8 .npy hash file to the functor_audit .bin format.

The .bin format is:
    u32 LE      : record count N
    N * 88 bytes: [id u64 LE][pap128 16 bytes][pap512 64 bytes]

pap128 is taken as the first 16 bytes of the 64-byte hash (128-bit coarse PAP).
pap512 is the full 64-byte hash.
"""

import argparse
import struct
from pathlib import Path

import numpy as np


def convert(input_path: Path, output_path: Path) -> None:
    hashes = np.load(input_path)
    if hashes.dtype != np.uint8:
        raise ValueError(f"expected uint8 hashes, got {hashes.dtype}")
    if hashes.ndim != 2 or hashes.shape[1] != 64:
        raise ValueError(f"expected shape (N, 64), got {hashes.shape}")

    n = hashes.shape[0]
    print(f"Converting {n:,} hashes from {input_path} -> {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<Q", i))            # id
            f.write(hashes[i, :16].tobytes())        # pap128
            f.write(hashes[i, :64].tobytes())        # pap512

    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 512-bit hash .npy to .bin")
    parser.add_argument("input", type=Path, help="input .npy file (N x 64 uint8)")
    parser.add_argument("output", type=Path, help="output .bin file")
    args = parser.parse_args()
    convert(args.input, args.output)
