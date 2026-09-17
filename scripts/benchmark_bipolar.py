import os, sys, sqlite3, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import RustBridge, BinaryHNSW
import ctypes, hashlib

N_INDEX = 10000
N_TEST = 1000
DB_PATH = os.path.abspath("data/phoenix_arxiv_1m.db")

def id_to_u64(pid):
    """Hash any string ID to a u64 for HNSW compatibility."""
    return int(hashlib.sha256(pid.encode()).hexdigest()[:16], 16)

def load_papers():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, abstract FROM papers ORDER BY RANDOM() LIMIT ?", (N_INDEX + N_TEST,))
    rows = cur.fetchall()
    conn.close()
    return rows[:N_INDEX], rows[N_INDEX:]

def build_bipolar_index(index_papers, bridge):
    lib = bridge.lib
    lib.yp_bipolar_load.argtypes = [ctypes.c_char_p]
    lib.yp_bipolar_load.restype = ctypes.c_int
    lib.yp_bipolar_encode.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.yp_bipolar_encode.restype = ctypes.c_int
    
    data_dir = os.path.abspath("data")
    rc = lib.yp_bipolar_load(data_dir.encode())
    if rc != 0:
        raise RuntimeError(f"yp_bipolar_load failed: {rc}")
    
    hnsw = BinaryHNSW(bridge)
    ids = []
    hashes = []
    for p in index_papers:
        text = f"{p[1] or ''} {p[2] or ''}".strip()
        if not text:
            continue
        out = np.zeros(64, dtype=np.uint8)
        rc = lib.yp_bipolar_encode(text.encode(), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 64)
        if rc != 0:
            continue
        ids.append(id_to_u64(p[0]))
        hashes.append(out.tobytes())
    
    if ids:
        ids_arr = np.array(ids, dtype=np.uint64)
        hashes_arr = np.frombuffer(b"".join(hashes), dtype=np.uint8)
        hnsw.insert_batch(ids_arr, hashes_arr)
    return hnsw, ids

def main():
    index_papers, test_papers = load_papers()
    print(f"[+] Index: {len(index_papers)}  Test: {len(test_papers)}")
    
    bridge = RustBridge()
    hnsw, index_ids = build_bipolar_index(index_papers, bridge)
    print(f"[+] HNSW has {len(index_ids)} nodes")
    
    correct = 0
    total = 0
    latencies = []
    
    for q in test_papers:
        text = f"{q[1] or ''} {q[2] or ''}".strip()
        if not text:
            continue
        
        t0 = time.perf_counter()
        out = np.zeros(64, dtype=np.uint8)
        rc = bridge.lib.yp_bipolar_encode(text.encode(), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 64)
        if rc != 0:
            continue
        results = hnsw.search(out.tobytes(), k=1)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
        if not results:
            continue
        pred_id = results[0][0]
        expected_id = id_to_u64(q[0])
        if pred_id == expected_id:
            correct += 1
        total += 1
    
    r1 = correct / total if total else 0.0
    print(f"\nR@1: {r1*100:.2f}%  ({correct}/{total})")
    print(f"P50: {np.median(latencies):.2f} ms")
    print(f"Mean: {np.mean(latencies):.2f} ms")

if __name__ == "__main__":
    main()
