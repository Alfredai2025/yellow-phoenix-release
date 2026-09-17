# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""
Build tabulated prototype table for static MiniLM-input ITQ.

T[token_id] = (E[token_id] - mean_doc) @ W
where E is the MiniLM word embedding matrix.
At query time: sum T[token_ids] and threshold at 0.

Outputs:
  data/itq_token_table_minilm_static_512_f32.bin
  data/itq_token_table_minilm_static_meta.npz
"""
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel

MINILM_PATH = "models/all-MiniLM-L6-v2"
ITQ_MODEL = "data/itq_model_minilm_static_512.npz"
OUT_TABLE = Path("data/itq_token_table_minilm_static_512_f32.bin")
OUT_META = Path("data/itq_token_table_minilm_static_meta.npz")


def main():
    print("Loading MiniLM word embeddings ...")
    minilm = AutoModel.from_pretrained(MINILM_PATH, local_files_only=True)
    E = minilm.embeddings.word_embeddings.weight.detach().cpu().numpy()
    print(f"E shape: {E.shape}")

    print("Loading ITQ model ...")
    itq = np.load(ITQ_MODEL)
    mean = itq["mean"].astype(np.float32)
    W = itq["W"].astype(np.float32)
    dim = int(itq["dim"])
    n_bits = int(itq["n_bits"])

    print(f"Computing T = (E - mean) @ W ...")
    T = (E - mean) @ W  # (vocab_size, 512)
    T = T.astype(np.float32)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    T.tofile(OUT_TABLE)

    np.savez(
        OUT_META,
        vocab_size=E.shape[0],
        dim=n_bits,
        embedding_dim=dim,
        dtype="float32",
        bytes_per_row=n_bits * 4,
        model="minilm_input_static",
    )
    print(f"Saved: {OUT_TABLE} ({OUT_TABLE.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved: {OUT_META}")


if __name__ == "__main__":
    main()
