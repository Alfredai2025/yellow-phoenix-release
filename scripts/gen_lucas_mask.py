# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Generate Fibonacci-lattice mask for 512-bit -> 256-bit skeleton.
Uses golden ratio phi for low-discrepancy sampling.
Writes:
  data/lucas_mask_256.bin  (512 bytes: 256 uint16 LE indices)
  data/lucas_mask_256.json
"""
import math, json, struct
from pathlib import Path

phi = (1 + math.sqrt(5)) / 2
n_bits, keep = 512, 256

seen = set()
mask = []
k = 0
while len(mask) < keep:
    idx = int((k * n_bits) / phi) % n_bits
    if idx not in seen:
        mask.append(idx)
        seen.add(idx)
    k += 1

Path("data").mkdir(exist_ok=True)

with open('data/lucas_mask_256.bin', 'wb') as f:
    for idx in mask:
        f.write(struct.pack('<H', idx))

with open('data/lucas_mask_256.json', 'w') as f:
    json.dump({'indices': mask, 'phi': phi, 'bits': n_bits, 'keep': keep}, f)

print(f'Mask generated: {len(mask)} unique indices')
print(f'First 10: {mask[:10]}')
gaps = [mask[i+1] - mask[i] for i in range(len(mask)-1)]
print(f'Min gap: {min(gaps)}, Max gap: {max(gaps)}')
print('Phase 1B-A complete.')
