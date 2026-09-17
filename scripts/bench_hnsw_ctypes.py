import ctypes, numpy as np, time, os

lib = ctypes.CDLL(os.path.expanduser("~/yellow_phoenix/target/release/libpams.dylib"))
N = 100000
K = 10
Q = 100

print("Loaded library")

# HNSW baseline
hnsw = lib.yp_hnsw_search
hnsw.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t,
                 ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.c_size_t]
hnsw.restype = ctypes.c_size_t

qhash = np.random.randint(0, 256, 64, dtype=np.uint8)
ids = np.zeros(K, dtype=np.uint64)
dists = np.zeros(K, dtype=np.uint32)
hnsw_lat = []
for _ in range(Q):
    t0 = time.perf_counter()
    hnsw(qhash.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 64, K,
         ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
         dists.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), K)
    hnsw_lat.append((time.perf_counter() - t0) * 1000)
print(f"HNSW P50: {np.percentile(hnsw_lat, 50):.3f} ms")

# Wave Field
w = lib.yp_wave_new()
wi = lib.yp_wave_inject
wi.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
wq = lib.yp_wave_query
wq.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

for i in range(N):
    b = np.random.randint(0, 256, 16, dtype=np.uint8)
    wi(w, b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 16)

qbits = np.random.randint(0, 256, 16, dtype=np.uint8)
wave_lat = []
for _ in range(Q):
    t0 = time.perf_counter()
    sc = wq(w, qbits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 16)
    wave_lat.append((time.perf_counter() - t0) * 1000)
lib.yp_wave_free(w)
print(f"Wave P50: {np.percentile(wave_lat, 50):.3f} ms (score={sc})")

# Hologram
class M(ctypes.Structure):
    _fields_ = [("c0", ctypes.c_uint64), ("c1", ctypes.c_uint64)]

h = lib.hologram_new()
p = (M * N)()
for i in range(N):
    p[i].c0 = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
    p[i].c1 = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)

t0 = time.perf_counter()
lib.hologram_batch_inject(h, p, N)
t_inj = (time.perf_counter() - t0) * 1000

q = M()
q.c0 = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
q.c1 = np.random.randint(0, 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
out = (ctypes.c_size_t * K)()

holo_lat = []
for _ in range(Q):
    t0 = time.perf_counter()
    lib.hologram_query(h, ctypes.byref(q), K, out, K)
    holo_lat.append((time.perf_counter() - t0) * 1000)
lib.hologram_drop(h)
print(f"Hologram inject {N}: {t_inj:.1f} ms | Query P50: {np.percentile(holo_lat, 50):.3f} ms | Per-doc: {np.percentile(holo_lat, 50) * 1e6 / N:.1f} ns")

# Mirror Mesh probe
print("Mirror Mesh symbols:", end=" ")
for name in ["mirror_mesh_init", "mirror_mesh_query", "mirror_mesh_add_paper", "mirror_mesh_stats"]:
    try:
        getattr(lib, name)
        print(name, end=" ")
    except:
        pass
print()

print("\n=== VERDICT ===")
print(f"HNSW:     {np.percentile(hnsw_lat, 50):.3f} ms | O(log N) | Production retriever")
print(f"Wave:     {np.percentile(wave_lat, 50):.3f} ms | O(1)     | Single score only")
print(f"Hologram: {np.percentile(holo_lat, 50):.3f} ms | O(N)     | Brute-force PAP scan")
