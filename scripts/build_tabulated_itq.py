# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Build tabulated ITQ: precompute centered+projected token prototypes.

Reads:
  • data/itq_model_512.npz  (mean, proj/W/R)
  • data/token_embeddings_30k_384.npy  (V x 384)

Writes:
  • data/itq_token_table_30k_512_f32.bin  (V x 512 float32)
  • data/itq_token_table_meta.npz

The table stores P_t = (E_t - mean) @ W for every token. At query time,
summing P_t for the tokens in a text and taking sign gives the same 512-bit
hash as the standard MiniLM+ITQ pipeline.
"""
import numpy as np
from pathlib import Path

ITQ_PATH = Path("itq_model_512.npz")
E_PATH = Path("data/token_embeddings_30k_384.npy")
OUT_BIN = Path("data/itq_token_table_30k_512_f32.bin")
OUT_META = Path("data/itq_token_table_meta.npz")

if not ITQ_PATH.exists():
    raise FileNotFoundError(f"ITQ model not found: {ITQ_PATH}")
if not E_PATH.exists():
    raise FileNotFoundError(f"Token embeddings not found: {E_PATH}")

print("Loading ITQ model ...")
itq = np.load(ITQ_PATH)
mean = itq["mean"].astype(np.float32) if "mean" in itq else itq["m"].astype(np.float32)
input_dim = mean.shape[0]

# Match yp_engine's selection logic: first key whose first dim == input_dim.
W = None
for key in ("proj", "W", "R"):
    if key in itq and itq[key].shape[0] == input_dim:
        W = itq[key].astype(np.float32)
        print(f"Using projection '{key}' shape {W.shape}")
        break
if W is None:
    raise ValueError("No projection matrix with input_dim rows found in ITQ model")

print(f"Mean shape: {mean.shape}")

print("Loading token embeddings ...")
E = np.load(E_PATH).astype(np.float32)
print(f"Token embeddings: {E.shape}")

# Precompute prototypes: P = (E - mean) @ W
print("Computing token prototypes P = (E - mean) @ W ...")
P = (E - mean) @ W
print(f"P shape: {P.shape}, range=[{P.min():.4f}, {P.max():.4f}]")

# Save flat binary (float32, native endian)
P.astype(np.float32).tofile(OUT_BIN)
print(f"Saved: {OUT_BIN} ({OUT_BIN.stat().st_size / 1e6:.1f} MB)")

np.savez(OUT_META,
         vocab_size=P.shape[0],
         dim=P.shape[1],
         dtype="float32",
         bytes_per_row=P.shape[1] * 4,
         projection_key=key)
print(f"Saved: {OUT_META}")
print("Phase 1A complete.")
