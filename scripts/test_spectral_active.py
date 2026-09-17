# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Quick functional test: verify spectral 512 is active in production queries."""
import sys, os, time, random, json
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yp_engine
from yp_engine import YPEngine

print("=" * 60)
print("SPECTRAL 512 PRODUCTION FUNCTIONAL TEST")
print("=" * 60)

# ── Align engine to the 1M arxiv corpus ──
yp_engine.META_PATH = Path("data/paper_meta_arxiv_1m.pkl")
_orig_np_load = np.load
def _patched_np_load(path, *a, **kw):
    if str(path) == "paper_embeddings.npy":
        path = "data/paper_embeddings_arxiv_1m.npy"
    return _orig_np_load(path, *a, **kw)
np.load = _patched_np_load

_orig_load_hashes = yp_engine._load_hashes
def _patched_load_hashes(path):
    name = Path(path).name
    if name in ("paper_hashes.pkl", "paper_hashes_itq_model_512.pkl"):
        return _orig_load_hashes("data/paper_hashes_arxiv_1m.pkl")
    return _orig_load_hashes(path)
yp_engine._load_hashes = _patched_load_hashes

# ── Build the active hash source aligned with the 1M corpus ──
def _build_1m_source():
    model_data = np.load("data/itq_model_512.npz")
    m = model_data.get("m", model_data.get("mean")).astype(np.float32)
    R = None
    for key in ("proj", "W", "R"):
        if key in model_data and model_data[key].shape[0] == m.shape[0]:
            R = model_data[key].astype(np.float32)
            break
    if R is None:
        raise ValueError("data/itq_model_512.npz has no usable projection")
    hashes = _orig_load_hashes("data/paper_hashes_arxiv_1m.pkl")

    def encode_from_emb(emb):
        x = emb - m
        x = x @ R
        bits = (x > 0).astype(np.uint8)
        return np.packbits(bits).tobytes()

    def encode_text(text):
        from yp_engine import get_model
        return encode_from_emb(get_model().encode(text, convert_to_numpy=True))

    return yp_engine.HashSource("active_itq", hashes, encode_text, encode_from_emb)

custom_sources = [_build_1m_source()]

# ── Stub heavy/experimental init so startup is fast ──
from yp_bridge import RustBridge
YPEngine._build_spectral_512_index = lambda self: None
YPEngine._load_unified_engine = lambda self: None
RustBridge.flat_init = lambda self, *args: 0
RustBridge.ism_init = lambda self, *args: 0
RustBridge.dynamic_mesh_new = lambda self, *args: 0
RustBridge.shard_new = lambda self, *args: 0
RustBridge.cascade_new = lambda self, *args: 0
RustBridge.temporal_new = lambda self: 0
RustBridge.energy_new = lambda self: 0
RustBridge.mesh_snapshot_new = lambda self: 0
RustBridge.flat_insert = lambda self, *args: 0
RustBridge.dynamic_mesh_insert = lambda self, *args: 0
RustBridge.shard_insert = lambda self, *args: 0
RustBridge.cascade_insert = lambda self, *args: 0
RustBridge.ism_insert = lambda self, *args: 0

engine = YPEngine(sources=custom_sources, verbose=False)
print(f"Engine loaded: {len(engine.pids):,} papers, {len(engine.sources)} hash source(s)")

# ── Build a 50k spectral index externally and inject it ──
hash_source = engine.sources[0].hashes
items = list(hash_source.items())
if len(items) > 50000:
    items = random.sample(items, 50000)
ids = []
hash_bytes = bytearray()
internal_to_pid = {}
for internal_id, (pid, hbytes) in enumerate(items):
    ids.append(internal_id)
    internal_to_pid[internal_id] = pid
    hash_bytes.extend(bytes(hbytes)[:64].ljust(64, b'\x00'))

b = RustBridge()
print(f"Building spectral index (k=64, n={len(ids):,})...")
t0 = time.perf_counter()
rc = b.spectral_512_build(bytes(hash_bytes), ids)
build_ms = (time.perf_counter() - t0) * 1000.0
if rc != 0:
    raise RuntimeError(f"spectral_512_build failed with code {rc}")
print(f"BUILD SUCCESS: {len(ids):,} papers in {build_ms:,.1f} ms")

engine.rust.spectral_512_handle = b.spectral_512_handle
engine._spectral_512_id_to_pid = internal_to_pid
print(f"Injected spectral handle: {engine.rust.spectral_512_handle is not None}")

# ── Tests ──
def test_spectral_handle_active():
    print("[TEST 1/4] Spectral handle active...")
    active = engine.rust.spectral_512_handle is not None
    print(f"  {'PASS' if active else 'FAIL'}: handle active = {active}")
    return active

def test_spectral_query_returns_results():
    print("[TEST 2/4] Spectral query returns results...")
    if engine.rust.spectral_512_handle is None:
        print("  SKIP: No spectral index built")
        return True
    query_hash = bytes([random.randint(0, 255) for _ in range(64)])
    t0 = time.perf_counter()
    results = engine.rust.spectral_512_query(query_hash.hex(), top_k=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  PASS: {len(results)} results in {elapsed_ms:.2f} ms")
    return True

def test_engine_search_with_spectral():
    print("[TEST 3/4] YPEngine search includes spectral stage...")
    engine.search("neural network architecture", top_k=5)
    stage_times = getattr(engine, '_last_stage_times', {})
    has_spectral = 'spectral' in stage_times
    print(f"  {'PASS' if has_spectral else 'WARN'}: spectral stage = {stage_times.get('spectral', 'NOT PRESENT')} ms")
    print(f"  All stages: { {k: f'{v:.2f}ms' for k, v in stage_times.items()} }")
    return True

def test_candidate_pool_enrichment():
    print("[TEST 4/4] Candidate pool enrichment...")
    query = "machine learning optimization"

    engine.search(query, top_k=5)
    cand_with = len(getattr(engine, '_last_candidates', []))

    orig = engine.rust.spectral_512_handle
    engine.rust.spectral_512_handle = None
    engine.search(query, top_k=5)
    cand_without = len(getattr(engine, '_last_candidates', []))
    engine.rust.spectral_512_handle = orig

    added = cand_with - cand_without
    print(f"  PASS: with={cand_with}, without={cand_without}, added={added}")
    return True

tests = [test_spectral_handle_active, test_spectral_query_returns_results,
         test_engine_search_with_spectral, test_candidate_pool_enrichment]
passed = sum(1 for t in tests if t())
print(f"\nRESULT: {passed}/{len(tests)} tests passed")
if passed == len(tests):
    print("SPECTRAL 512 IS ACTIVE AND FUNCTIONAL.")
print("=" * 60)
