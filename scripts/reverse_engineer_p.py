
import os, sys
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer

print("[+] Loading MiniLM and ITQ...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
itq = np.load('data/itq_model_512.npz')
print("ITQ keys:", list(itq.keys()))
for k in itq.keys():
    print(f"  {k}: shape {itq[k].shape}")

mean = itq['mean']
W = itq.get('W')
R = itq.get('R')
proj = itq.get('proj')
n_bits = int(itq.get('n_bits', 512))

# Determine the projection matrix M: 384 -> 512
# M should be (384, 512) such that ITQ_prethreshold = (emb - mean) @ M
if proj is not None and proj.shape == (384, 512):
    M = proj
    print(f"[+] Using 'proj' as M: {M.shape}")
elif W is not None and R is not None and W.shape[0] == 384 and W.shape[1] == 512 and R.shape == (512, 512):
    M = W @ R
    print(f"[+] Using W @ R as M: {M.shape}")
elif W is not None and W.shape == (384, 512):
    M = W
    print(f"[+] Using 'W' as M: {M.shape}")
else:
    print("[-] Could not determine M. Available shapes:")
    for k in itq.keys():
        print(f"    {k}: {itq[k].shape}")
    raise ValueError("Cannot construct (384, 512) projection matrix")

# Load unique tokens
print("[+] Sampling tokens...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cur = conn.cursor()
cur.execute("SELECT title, abstract FROM papers LIMIT 50000")
texts = [f"{t or ''} {a or ''}".strip() for t, a in cur.fetchall()]
conn.close()

tokens = set()
for text in texts:
    tokens.update(text.lower().split())
tokens = list(tokens)
print(f"[+] {len(tokens)} unique tokens")

# Embed tokens
print("[+] Embedding tokens (~2 min)...")
embs = model.encode(tokens, convert_to_numpy=True, show_progress_bar=True)

# Compute per-token ITQ contributions: Q[t][i] = emb(t) @ M[:,i]
print("[+] Computing Q matrix...")
Q = embs @ M  # (N_tokens, 512)

# BipolarEncoder hash
def hash_token(seed, token):
    h = seed
    for b in token.encode():
        h = (h * 0x9e3779b97f4a7c15 + b) & 0xFFFFFFFFFFFFFFFF
    return h

def splitmix64(state):
    state = (state + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return int(z ^ (z >> 31))

# Compute per-bucket averages
for chunk in range(4):
    bit_start = chunk * 128
    bucket_sums = np.zeros((128, 128), dtype=np.float64)
    bucket_counts = np.zeros((128, 128), dtype=np.int64)
    
    for t_idx, token in enumerate(tokens):
        h = hash_token(42, token)
        h2 = splitmix64(h)
        row = int(h2 % 128)
        for bit in range(128):
            bucket_sums[row][bit] += Q[t_idx][bit_start + bit]
            bucket_counts[row][bit] += 1
    
    P = bucket_sums / np.maximum(bucket_counts, 1)
    P.astype(np.float32).tofile(f'data/bipolar_proj_{chunk}.f32')
    
    # Bias: negative mean contribution, scaled by avg doc length (~15 tokens)
    # This centers the threshold so random docs are near zero
    avg_doc_len = np.mean([len(t.split()) for t in texts])
    bit_end = bit_start + 128
    bias = -(mean @ M[:, bit_start:bit_end]) * (avg_doc_len / 2.0)
    bias.astype(np.float32).tofile(f'data/bipolar_bias_{chunk}.f32')
    
    print(f"[+] Chunk {chunk}: P shape {P.shape}, bias shape {bias.shape}")

print("[+] Done.")
