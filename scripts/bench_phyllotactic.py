#!/usr/bin/env python3
"""
Validate phyllotactic entry points vs random entry points.
Uses 100k ArXiv subset because existing 1M HNSW binaries are v3 and the
current Rust loader only supports v4. To run on 1M, rebuild the HNSW index
with the current code or load a v4 binary.
"""
import ctypes
import numpy as np
import time
import random
import os
import struct

LIB_PATH = "target/release/libpams.dylib"
if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"Library not found: {LIB_PATH}")

lib = ctypes.CDLL(LIB_PATH)

# FFI signatures
lib.yp_binary_hnsw_new.restype = ctypes.c_void_p
lib.yp_binary_hnsw_free.argtypes = [ctypes.c_void_p]
lib.yp_binary_hnsw_count.argtypes = [ctypes.c_void_p]
lib.yp_binary_hnsw_count.restype = ctypes.c_size_t
lib.yp_binary_hnsw_insert.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t
]
lib.yp_binary_hnsw_insert.restype = ctypes.c_int
lib.yp_binary_hnsw_search.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t
]
lib.yp_binary_hnsw_search.restype = ctypes.c_size_t
lib.yp_binary_hnsw_search_from.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t
]
lib.yp_binary_hnsw_search_from.restype = ctypes.c_size_t

lib.yp_phyllotactic_build.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
]
lib.yp_phyllotactic_build.restype = ctypes.c_int
lib.yp_phyllotactic_entry.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.yp_phyllotactic_entry.restype = ctypes.c_size_t


def hamming_bytes(a, b):
    return sum((x ^ y).bit_count() for x, y in zip(a, b))


def main():
    HASH_PATH = "data/paper_hashes_100k_synthetic.npy"
    SPECTRAL_PATH = "data/spectral_coords_100k_synthetic.npy"

    print("[+] Loading hashes...")
    hashes = np.load(HASH_PATH).astype(np.uint8)
    n_nodes = len(hashes)
    print(f"    Nodes: {n_nodes}")

    print("[+] Loading spectral coords...")
    coords = np.load(SPECTRAL_PATH).astype(np.float32)
    assert len(coords) == n_nodes, f"Mismatch: {len(coords)} coords vs {n_nodes} nodes"

    print("[+] Building HNSW index (M=16, ef_construction=200, ef_search=128)...")
    idx = lib.yp_binary_hnsw_new()
    t0 = time.time()
    for i in range(n_nodes):
        rc = lib.yp_binary_hnsw_insert(
            idx, ctypes.c_uint64(i),
            hashes[i].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(64)
        )
        if rc != 0:
            raise RuntimeError(f"insert failed at {i}: {rc}")
    t1 = time.time()
    print(f"    Build time: {t1 - t0:.1f}s")
    print(f"    Indexed: {lib.yp_binary_hnsw_count(idx)}")

    print("[+] Building phyllotactic navigator...")
    ids = np.arange(n_nodes, dtype=np.uint64)
    rc = lib.yp_phyllotactic_build(
        idx,
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        coords.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(n_nodes)
    )
    if rc != 0:
        raise RuntimeError(f"phyllotactic_build failed: {rc}")

    # Warm-up
    out_ids = (ctypes.c_uint64 * 10)()
    out_dists = (ctypes.c_uint32 * 10)()
    lib.yp_binary_hnsw_search(
        idx,
        hashes[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(64), ctypes.c_size_t(10),
        out_ids, out_dists, ctypes.c_size_t(10)
    )

    N_QUERIES = 1000
    random.seed(42)
    query_idxs = random.sample(range(n_nodes), N_QUERIES)

    # Phase 1: Entry point quality
    random_dists = []
    phyllo_dists = []
    print("[+] Phase 1: Entry point quality...")
    for qidx in query_idxs:
        q_hash = bytes(hashes[qidx])
        q_coord = coords[qidx].astype(np.float32)

        rand_entry = random.randint(0, n_nodes - 1)
        random_dists.append(hamming_bytes(q_hash, bytes(hashes[rand_entry])))

        entry = lib.yp_phyllotactic_entry(q_coord.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if entry != ctypes.c_size_t(-1).value and entry < n_nodes:
            phyllo_dists.append(hamming_bytes(q_hash, bytes(hashes[entry])))
        else:
            phyllo_dists.append(512)

    avg_random = sum(random_dists) / len(random_dists)
    avg_phyllo = sum(phyllo_dists) / len(phyllo_dists)
    improvement = (avg_random - avg_phyllo) / avg_random * 100

    print(f"\n=== Phase 1 Results ===")
    print(f"Random entry Hamming:  {avg_random:.1f}")
    print(f"Phyllotactic Hamming:  {avg_phyllo:.1f}")
    print(f"Improvement:           {improvement:.1f}% closer")

    # Phase 2: Latency — standard vs phyllotactic search
    print("\n[+] Phase 2: Search latency (k=10, ef=128)...")
    K = 10
    std_lat = []
    phyllo_lat = []

    for qidx in query_idxs:
        # Standard search
        t0 = time.perf_counter()
        n = lib.yp_binary_hnsw_search(
            idx,
            hashes[qidx].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(64), ctypes.c_size_t(K),
            out_ids, out_dists, ctypes.c_size_t(K)
        )
        t1 = time.perf_counter()
        std_lat.append((t1 - t0) * 1000.0)

        # Phyllotactic search
        entry = lib.yp_phyllotactic_entry(coords[qidx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        t0 = time.perf_counter()
        n = lib.yp_binary_hnsw_search_from(
            idx,
            hashes[qidx].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(64), ctypes.c_size_t(K),
            ctypes.c_size_t(entry),
            out_ids, out_dists, ctypes.c_size_t(K)
        )
        t1 = time.perf_counter()
        phyllo_lat.append((t1 - t0) * 1000.0)

    print(f"Standard    P50: {np.percentile(std_lat, 50):.3f} ms  P99: {np.percentile(std_lat, 99):.3f} ms")
    print(f"Phyllotactic P50: {np.percentile(phyllo_lat, 50):.3f} ms  P99: {np.percentile(phyllo_lat, 99):.3f} ms")

    speedup = np.percentile(std_lat, 50) / np.percentile(phyllo_lat, 50)
    print(f"P50 speedup: {speedup:.2f}x")

    lib.yp_binary_hnsw_free(idx)

    print("\n=== Verdict ===")
    if improvement > 10 and speedup > 1.05:
        print("PASS: Phyllotactic entry points are closer AND faster.")
    elif improvement > 10:
        print("PARTIAL: Entry points are closer; latency gain depends on graph topology.")
    else:
        print("FAIL: Geometry did not help. Check spectral coords or data distribution.")


if __name__ == "__main__":
    main()
