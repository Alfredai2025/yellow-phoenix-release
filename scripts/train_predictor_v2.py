#!/usr/bin/env python3
"""
Train Trinity predictor on real paper data.
For each paper: compute actual bucket -> record query -> teach predictor.
"""
import sys, os, sqlite3, ctypes, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

print("=" * 65)
print("PREDICTOR TRAINING v2")
print("=" * 65)

# ── Load engine ──
print("\n[1/4] Loading engine...")
engine = YPEngine()
print(f"      Papers: {len(engine.cache)}")

# ── Init Trinity ──
print("\n[2/4] Initializing Trinity...")
lib = engine.rust.lib
if hasattr(lib, 'yp_trinity_init'):
    lib.yp_trinity_init()
    print("      Trinity initialized")

# Set up FFI
lib.yp_compute_bucket_128.argtypes = [ctypes.c_char_p]
lib.yp_compute_bucket_128.restype = ctypes.c_uint32

lib.yp_trinity_record_query.argtypes = [ctypes.c_char_p]
lib.yp_trinity_record_query.restype = ctypes.c_uint64

lib.yp_trinity_predict_bucket.argtypes = [ctypes.c_uint64]
lib.yp_trinity_predict_bucket.restype = ctypes.c_int

lib.yp_trinity_record_result.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
lib.yp_trinity_record_result.restype = ctypes.c_int

# ── Load papers ──
print("\n[3/4] Loading papers from DB...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cursor = conn.cursor()
cursor.execute("SELECT id, title, yp_hash512 FROM papers WHERE yp_hash512 IS NOT NULL LIMIT 12000")
rows = cursor.fetchall()
conn.close()
print(f"      Samples: {len(rows)}")

# ── Train ──
print("\n[4/4] Training...")
correct_before = 0
trained = 0

for i, (pid, title, h) in enumerate(rows):
    try:
        h_bytes = bytes.fromhex(h)
    except ValueError:
        continue

    # Compute actual bucket
    buf = ctypes.create_string_buffer(h_bytes[:16])
    actual_bucket = lib.yp_compute_bucket_128(buf)

    # Get query_id from title
    title_buf = ctypes.create_string_buffer((title or "unknown").encode('utf-8', errors='ignore'))
    query_id = lib.yp_trinity_record_query(title_buf)

    # Check pre-training prediction
    predicted = lib.yp_trinity_predict_bucket(query_id)
    if predicted == actual_bucket:
        correct_before += 1

    # TEACH: record actual result
    lib.yp_trinity_record_result(query_id, predicted, actual_bucket)
    trained += 1

    if (i + 1) % 1000 == 0:
        acc = correct_before / (i + 1) * 100
        print(f"      {i+1}/{len(rows)} | pre-train accuracy: {acc:.2f}%")

# ── Verify ──
print("\n" + "=" * 65)
print("VERIFICATION")
print("=" * 65)

# Spot-check 10 random papers
import random
random.seed(42)
sample = random.sample(rows, min(10, len(rows)))

correct_after = 0
for pid, title, h in sample:
    try:
        h_bytes = bytes.fromhex(h)
    except:
        continue
    buf = ctypes.create_string_buffer(h_bytes[:16])
    actual = lib.yp_compute_bucket_128(buf)
    title_buf = ctypes.create_string_buffer((title or "").encode('utf-8', errors='ignore'))
    qid = lib.yp_trinity_record_query(title_buf)
    predicted = lib.yp_trinity_predict_bucket(qid)
    match = "✅" if predicted == actual else "❌"
    print(f"  {match} '{title[:50]}...' | actual={actual} | predicted={predicted}")
    if predicted == actual:
        correct_after += 1

print(f"\n  Spot-check accuracy: {correct_after}/{len(sample)} ({correct_after/len(sample)*100:.0f}%)")
print(f"  Total trained: {trained}")
print(f"  Pre-training accuracy (stub): {correct_before/len(rows)*100:.2f}%")

# Save stats
import json
out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'trained_samples': trained,
    'pre_train_accuracy_pct': round(correct_before / len(rows) * 100, 2),
    'spot_check_accuracy_pct': round(correct_after / len(sample) * 100, 2),
}
os.makedirs('logs', exist_ok=True)
with open('logs/predictor_training.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: logs/predictor_training.json")
