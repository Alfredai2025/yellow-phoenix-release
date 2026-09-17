# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Benchmark MiniLM embedding generation."""
import time
import sys

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing sentence-transformers...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
    from sentence_transformers import SentenceTransformer

QUERIES = [
    "neural networks for natural language processing",
    "quantum computing algorithms",
    "climate change mitigation strategies",
    "protein folding prediction methods",
    "distributed systems consensus protocols",
] * 20

BATCH_SIZES = [1, 4, 8, 16, 32, 64]

print("=" * 60)
print("YELLOW PHOENIX -- MiniLM Embedding Generation")
print("=" * 60)

print("\nLoading model: all-MiniLM-L6-v2...")
t0 = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded in {:.1f}s".format(time.time() - t0))

for batch_size in BATCH_SIZES:
    _ = model.encode(QUERIES[:batch_size], convert_to_numpy=True, show_progress_bar=False)

    n_batches = len(QUERIES) // batch_size
    times = []
    for i in range(n_batches):
        batch = QUERIES[i*batch_size:(i+1)*batch_size]
        t0 = time.perf_counter()
        _ = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times)/len(times)
    qps = batch_size / avg_time
    print("Batch {:>2} | Time: {:>6.1f} ms | QPS: {:>7.0f}".format(
        batch_size, avg_time * 1000, qps))

print("\n" + "=" * 60)
