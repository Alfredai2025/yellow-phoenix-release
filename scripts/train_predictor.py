#!/usr/bin/env python3
"""
Train the Trinity predictor on real paper data.
For each paper: compute actual bucket -> record query -> teach predictor.
"""
import sys, os, sqlite3, ctypes, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import RustBridge, YPEngine

print("=" * 65)
print("PREDICTOR TRAINING")
print("=" * 65)

# ── Load engine (mesh must be loaded for bucket_hash FFI) ──
print("\n[1/3] Loading engine...")
engine = YPEngine()
print(f"      Papers: {len(engine.cache)}")

# ── Init Trinity ──
print("\n[2/3] Initializing Trinity...")
bridge = engine.rust if hasattr(engine, 'rust') else RustBridge()
lib = bridge.lib

if hasattr(lib, 'yp_trinity_init'):
    lib.yp_trinity_init()
    print("      Trinity initialized")

# Set up FFI signatures
lib.yp_compute_bucket_128.argtypes = [ctypes.c_char_p]
lib.yp_compute_bucket_128.restype = ctypes.c_uint32

lib.yp_trinity_record_query.argtypes = [ctypes.c_char_p]
lib.yp_trinity_record_query.restype = ctypes.c_uint64

lib.yp_trinity_predict_bucket.argtypes = [ctypes.c_uint64]
lib.yp_trinity_predict_bucket.restype = ctypes.c_int

lib.yp_trinity_record_result.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
lib.yp_trinity_record_result.restype = ctypes.c_int

# ── Load papers from DB ──
print("\n[3/3] Training on papers...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cursor = conn.cursor()
cursor.execute("SELECT id, title, yp_hash512 FROM papers WHERE yp_hash512 IS NOT NULL LIMIT 12000")
rows = cursor.fetchall()
conn.close()

print(f"      Training samples: {len(rows)}")

correct_before = 0
trained = 0

for i, (pid, title, h) in enumerate(rows):
    try:
        h_bytes = bytes.fromhex(h)
    except ValueError:
        continue

    # Compute actual bucket from PAP hash (via real Rust bucket_hash)
    pap_128 = h_bytes[:16]
    buf = ctypes.create_string_buffer(pap_128)
    actual_bucket = lib.yp_compute_bucket_128(buf)

    # Get canonical query_id from title
    title_bytes = (title or "unknown").encode('utf-8', errors='ignore')
    title_buf = ctypes.create_string_buffer(title_bytes)
    query_id = lib.yp_trinity_record_query(title_buf)

    # Check what predictor currently says
    predicted = lib.yp_trinity_predict_bucket(query_id)
    if predicted == actual_bucket:
        correct_before += 1

    # TEACH: record actual result
    lib.yp_trinity_record_result(query_id, predicted, actual_bucket)
    trained += 1

    if (i + 1) % 1000 == 0:
        acc = correct_before / (i + 1) * 100
        print(f"      {i+1}/{len(rows)} | pre-train accuracy: {acc:.2f}%")

print(f"\n  Training complete: {trained} samples")
print(f"  Pre-training accuracy (stub): {correct_before/len(rows)*100:.2f}%")
print(f"  Predictor now memorized {trained} query_id -> bucket mappings")
