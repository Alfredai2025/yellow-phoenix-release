#!/usr/bin/env python3
"""Minimal repro: load droplet then init YPEngine, printing every step unbuffered."""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import yp_engine

print("[test] loading droplet embeddings...", flush=True)
t0 = time.time()
droplet_emb = np.load("data/droplet_embeddings_aligned.npy")
print(f"[test] droplet loaded {droplet_emb.shape} in {time.time()-t0:.2f}s", flush=True)

print("[test] Initialising YPEngine(verbose=True)...", flush=True)
t0 = time.time()
engine = yp_engine.YPEngine(verbose=True)
print(f"[test] YPEngine init done in {time.time()-t0:.2f}s", flush=True)
print(f"[test] engine papers={len(engine.pids)} sources={len(engine.sources)}", flush=True)
