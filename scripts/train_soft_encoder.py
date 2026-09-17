import os, numpy as np, sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans

print("[+] Loading MiniLM and ITQ...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
itq = np.load('data/itq_model_512.npz')
mean = itq['mean']
M = itq['proj']

# Load tokens
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cur = conn.cursor()
cur.execute("SELECT title, abstract FROM papers LIMIT 50000")
texts = [f"{t or ''} {a or ''}".strip() for t, a in cur.fetchall()]
conn.close()

tokens = list(set(t.lower() for text in texts for t in text.split()))
print(f"[+] {len(tokens)} tokens")

# Embed
embs = model.encode(tokens, convert_to_numpy=True, show_progress_bar=True)
embs_centered = embs - mean

# Train 32 codebooks
N_SUBCHUNKS = 32
DIM = 16
K = 16  # clusters per sub-chunk

for sub in range(N_SUBCHUNKS):
    bit_start = sub * DIM
    projected = embs_centered @ M[:, bit_start:bit_start + DIM]
    
    print(f"[+] Sub-chunk {sub}: k-means {K} clusters in {DIM}D...")
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=1000, random_state=42, n_init=3)
    kmeans.fit(projected)
    
    # Save centers: K rows x DIM cols
    kmeans.cluster_centers_.astype(np.float32).tofile(f'data/soft_centers_{sub}.f32')
    
    # Save zero bias
    np.zeros(DIM, dtype=np.float32).tofile(f'data/soft_bias_{sub}.f32')

print("[+] Done.")
