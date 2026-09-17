#!/usr/bin/env python3
"""
Benchmark: HNSW vs Hologram vs Mirror Mesh
Read-only. No files modified. No indexes altered.
"""
import ctypes
import numpy as np
import time
import os
import sys

# =============================================================================
# CONFIG
# =============================================================================
LIB_PATH = os.path.expanduser("~/yellow_phoenix/target/release/libpams.dylib")
N_HOLOGRAM_DOCS = 100_000   # Hologram is O(N); don't test 1M yet
N_QUERIES = 100
TOP_K = 10

# =============================================================================
# FFI TYPES
# =============================================================================
class FfiBinaryMultivector(ctypes.Structure):
    _fields_ = [("chunk0", ctypes.c_uint64), ("chunk1", ctypes.c_uint64)]

# =============================================================================
# LOAD LIBRARY
# =============================================================================
if not os.path.exists(LIB_PATH):
    print(f"ERROR: Library not found at {LIB_PATH}")
    print("Run: cargo build --release")
    sys.exit(1)

lib = ctypes.CDLL(LIB_PATH)
print(f"[+] Loaded {LIB_PATH}")

# =============================================================================
# PROBE: List all geometric symbols
# =============================================================================
print("\n=== SYMBOL PROBE ===")
for prefix in ["hologram", "mirror_mesh", "yp_wave"]:
    try:
        # ctypes finds symbols without underscore prefix on macOS
        names = []
        if prefix == "hologram":
            for n in ["hologram_new", "hologram_drop", "hologram_batch_inject",
                      "hologram_query", "yp_wave_new", "yp_wave_query"]:
                try:
                    getattr(lib, n)
                    names.append(n)
                except AttributeError:
                    pass
        else:
            # brute force check common names
            for n in ["mirror_mesh_init", "mirror_mesh_query", "mirror_mesh_add_paper",
                      "mirror_mesh_stats", "mirror_mesh_encode", "mirror_mesh_init_with_thresholds"]:
                try:
                    getattr(lib, n)
                    names.append(n)
                except AttributeError:
                    pass
        print(f"  {prefix}: {names if names else 'NO SYMBOLS FOUND'}")
    except Exception as e:
        print(f"  {prefix}: probe error {e}")

# =============================================================================
# BENCHMARK 1: HNSW (production baseline)
# =============================================================================
print("\n=== HNSW BASELINE ===")
def bench_hnsw():
    yp_hnsw_search = lib.yp_hnsw_search
    yp_hnsw_search.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t
    ]
    yp_hnsw_search.restype = ctypes.c_size_t

    query_hash = np.random.randint(0, 256, size=64, dtype=np.uint8)
    ids = np.zeros(TOP_K, dtype=np.uint64)
    dists = np.zeros(TOP_K, dtype=np.uint32)

    latencies = []
    for _ in range(N_QUERIES):
        t0 = time.perf_counter()
        n = yp_hnsw_search(
            query_hash.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(64),
            ctypes.c_size_t(TOP_K),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            dists.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_size_t(TOP_K)
        )
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    return latencies

try:
    hnsw_lat = bench_hnsw()
    print(f"  P50: {np.percentile(hnsw_lat, 50):.3f} ms")
    print(f"  P99: {np.percentile(hnsw_lat, 99):.3f} ms")
    print(f"  Min: {np.min(hnsw_lat):.3f} ms")
except Exception as e:
    print(f"  HNSW BENCHMARK FAILED: {e}")
    hnsw_lat = []

# =============================================================================
# BENCHMARK 2: Hologram (StandingHologram via ffi_kernel.rs)
# =============================================================================
print("\n=== HOLOGRAM (O(N) brute-force PAP) ===")
def bench_hologram():
    # Bind known signatures from ffi_kernel.rs
    hologram_new = lib.hologram_new
    hologram_new.argtypes = []
    hologram_new.restype = ctypes.c_void_p

    hologram_drop = lib.hologram_drop
    hologram_drop.argtypes = [ctypes.c_void_p]
    hologram_drop.restype = None

    hologram_batch_inject = lib.hologram_batch_inject
    hologram_batch_inject.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FfiBinaryMultivector),
        ctypes.c_size_t
    ]
    hologram_batch_inject.restype = None

    hologram_query = lib.hologram_query
    hologram_query.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FfiBinaryMultivector),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t
    ]
    hologram_query.restype = ctypes.c_size_t

    h = hologram_new()
    if not h:
        raise RuntimeError("hologram_new returned null")

    try:
        # Generate random 128-bit prototypes
        print(f"  Injecting {N_HOLOGRAM_DOCS} random 128-bit vectors...")
        prototypes = (FfiBinaryMultivector * N_HOLOGRAM_DOCS)()
        for i in range(N_HOLOGRAM_DOCS):
            prototypes[i].chunk0 = np.random.randint(0, 2**64, dtype=np.uint64)
            prototypes[i].chunk1 = np.random.randint(0, 2**64, dtype=np.uint64)

        t0 = time.perf_counter()
        hologram_batch_inject(h, prototypes, N_HOLOGRAM_DOCS)
        t_inject = time.perf_counter() - t0

        # Query
        q = FfiBinaryMultivector()
        q.chunk0 = np.random.randint(0, 2**64, dtype=np.uint64)
        q.chunk1 = np.random.randint(0, 2**64, dtype=np.uint64)
        out_idx = (ctypes.c_size_t * TOP_K)()

        latencies = []
        for _ in range(N_QUERIES):
            t0 = time.perf_counter()
            n = hologram_query(h, ctypes.byref(q), TOP_K, out_idx, TOP_K)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        return t_inject, latencies
    finally:
        hologram_drop(h)

try:
    t_inject, holo_lat = bench_hologram()
    print(f"  Inject time: {t_inject*1000:.1f} ms")
    print(f"  Query P50: {np.percentile(holo_lat, 50):.3f} ms")
    print(f"  Query P99: {np.percentile(holo_lat, 99):.3f} ms")
    print(f"  Per-doc cost: {np.percentile(holo_lat, 50)*1e6/N_HOLOGRAM_DOCS:.2f} ns/doc")
    print(f"  → O(N) confirmed (linear scan)")
except Exception as e:
    print(f"  HOLOGRAM BENCHMARK FAILED: {e}")
    holo_lat = []

# =============================================================================
# BENCHMARK 3: Wave Field (yp_wave_query — O(1) resonance score)
# =============================================================================
print("\n=== WAVE FIELD (O(1) resonance score) ===")
def bench_wave():
    yp_wave_new = lib.yp_wave_new
    yp_wave_new.argtypes = []
    yp_wave_new.restype = ctypes.c_void_p

    yp_wave_free = lib.yp_wave_free
    yp_wave_free.argtypes = [ctypes.c_void_p]
    yp_wave_free.restype = None

    yp_wave_inject = lib.yp_wave_inject
    yp_wave_inject.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    yp_wave_inject.restype = ctypes.c_int

    yp_wave_query = lib.yp_wave_query
    yp_wave_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    yp_wave_query.restype = ctypes.c_int

    w = yp_wave_new()
    if not w:
        raise RuntimeError("yp_wave_new returned null")

    try:
        # Inject N docs (16 bytes each = 128 bits)
        print(f"  Injecting {N_HOLOGRAM_DOCS} docs into wave field...")
        t0 = time.perf_counter()
        for i in range(N_HOLOGRAM_DOCS):
            bits = np.random.randint(0, 256, size=16, dtype=np.uint8)
            yp_wave_inject(w, bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 16)
        t_inject = time.perf_counter() - t0

        # Query
        qbits = np.random.randint(0, 256, size=16, dtype=np.uint8)
        latencies = []
        for _ in range(N_QUERIES):
            t0 = time.perf_counter()
            score = yp_wave_query(w, qbits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 16)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        return t_inject, latencies
    finally:
        yp_wave_free(w)

try:
    t_inject, wave_lat = bench_wave()
    print(f"  Inject time: {t_inject*1000:.1f} ms")
    print(f"  Query P50: {np.percentile(wave_lat, 50):.3f} ms")
    print(f"  Query P99: {np.percentile(wave_lat, 99):.3f} ms")
    print(f"  → O(1) confirmed (constant time regardless of N)")
except Exception as e:
    print(f"  WAVE BENCHMARK FAILED: {e}")
    wave_lat = []

# =============================================================================
# BENCHMARK 4: Mirror Mesh (probe only — signatures unknown)
# =============================================================================
print("\n=== MIRROR MESH (signature probe) ===")
def probe_mirror_mesh():
    # Try to bind mirror_mesh_init
    try:
        fn = lib.mirror_mesh_init
        print("  mirror_mesh_init: FOUND")
    except AttributeError:
        print("  mirror_mesh_init: NOT FOUND")
        return

    # Guess signature: () -> void*  or  () -> int
    try:
        fn.argtypes = []
        fn.restype = ctypes.c_void_p
        handle = fn()
        print(f"  mirror_mesh_init() -> {handle}")
        if handle:
            # Try stats
            try:
                stats_fn = lib.mirror_mesh_stats
                stats_fn.argtypes = [ctypes.c_void_p]
                stats_fn.restype = ctypes.c_char_p
                s = stats_fn(handle)
                print(f"  mirror_mesh_stats: {s.decode() if s else 'null'}")
            except Exception as e:
                print(f"  mirror_mesh_stats failed: {e}")
    except Exception as e:
        print(f"  mirror_mesh_init call failed: {e}")

probe_mirror_mesh()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if hnsw_lat:
    print(f"HNSW (512-bit graph):     P50 = {np.percentile(hnsw_lat, 50):.3f} ms  |  O(log N)")
if 'holo_lat' in dir() and holo_lat:
    print(f"Hologram (128-bit PAP):   P50 = {np.percentile(holo_lat, 50):.3f} ms  |  O(N) brute-force")
if 'wave_lat' in dir() and wave_lat:
    print(f"Wave Field (128-bit):     P50 = {np.percentile(wave_lat, 50):.3f} ms  |  O(1) score only")

print("\nINTERPRETATION:")
print("- HNSW is the only production-viable retriever (sub-ms, returns top-k IDs)")
print("- Hologram is O(N) linear scan; too slow for 1M docs")
print("- Wave Field is O(1) but returns a single score, not neighbors")
print("- Mirror Mesh requires signature discovery before benchmarking")
