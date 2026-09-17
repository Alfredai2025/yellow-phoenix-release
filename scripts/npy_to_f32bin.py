import numpy as np
import sys
from pathlib import Path

npy_path = sys.argv[1] if len(sys.argv) > 1 else "data/paper_embeddings_100k.npy"
out_path = Path(npy_path).with_suffix('.f32bin')

emb = np.load(npy_path)
if emb.dtype != np.float32:
    emb = emb.astype(np.float32)
n, dim = emb.shape

with open(out_path, 'wb') as f:
    f.write(np.array([n, dim], dtype=np.int64).tobytes())
    f.write(emb.tobytes())

print(f"[+] {npy_path} -> {out_path}  ({n} vectors, dim={dim})")
