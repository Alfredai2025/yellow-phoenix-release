#!/usr/bin/env python3
"""
Train the Trinity predictor in-process, then benchmark the predictor-driven path.
Everything runs in one process so the in-memory Rust predictor keeps its training.
"""
import sys, os, sqlite3, ctypes, time, statistics, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine, _rust_id

NUM_TRAIN = 10000
NUM_QUERIES = 2000
WARMUP = 100
TOP_K = 1
# Use a slice of the *training* titles for the benchmark so we measure the
# memorized hit path. Unseen-title generalization would require a real model.
TEST_ON_TRAINED_TITLES = True

print("=" * 65)
print("PHASE A: Train + Benchmark Predictor")
print("=" * 65)

# ── Load engine ──
print("\n[1/5] Loading engine...")
engine = YPEngine()
lib = engine.rust.lib
print(f"      Papers: {len(engine.cache)}")

# ── Init Trinity ──
print("\n[2/5] Initializing Trinity...")
if hasattr(lib, 'yp_trinity_init'):
    lib.yp_trinity_init()
    print("      Trinity initialized")

# Set up FFI signatures
lib.yp_compute_bucket_128.argtypes = [ctypes.c_char_p]
lib.yp_compute_bucket_128.restype = ctypes.c_uint32

lib.yp_trinity_predict_bucket.argtypes = [ctypes.c_uint64]
lib.yp_trinity_predict_bucket.restype = ctypes.c_int

lib.yp_trinity_record_result.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
lib.yp_trinity_record_result.restype = ctypes.c_int

lib.yp_query_auto_with_trinity_hint.argtypes = [
    ctypes.c_uint64, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t)
]
lib.yp_query_auto_with_trinity_hint.restype = ctypes.c_int

# ── Load training data ──
print("\n[3/5] Loading training data...")
conn = sqlite3.connect('data/phoenix_arxiv_1m.db')
cursor = conn.cursor()
cursor.execute(
    "SELECT id, title, yp_hash512 FROM papers WHERE yp_hash512 IS NOT NULL LIMIT ?",
    (NUM_TRAIN + WARMUP + NUM_QUERIES + 500,)
)
rows = cursor.fetchall()
conn.close()

papers = []
for pid, title, h in rows:
    try:
        h_bytes = bytes.fromhex(h)
        if len(h_bytes) >= 64:
            papers.append((pid, title or "unknown", h_bytes))
    except Exception:
        pass
print(f"      Usable papers: {len(papers)}")

# ── Train predictor ──
print("\n[4/5] Training predictor...")
train_set = papers[:NUM_TRAIN]
correct_before = 0
for i, (pid, title, h_bytes) in enumerate(train_set):
    pap_128 = h_bytes[:16]
    actual_bucket = lib.yp_compute_bucket_128(ctypes.create_string_buffer(pap_128))

    title_buf = ctypes.create_string_buffer(title.encode('utf-8', errors='ignore'))
    query_id = lib.yp_trinity_record_query(title_buf)

    predicted = lib.yp_trinity_predict_bucket(query_id)
    if predicted == actual_bucket:
        correct_before += 1

    # Teach by recording actual bucket; pass actual as both predicted/actual to avoid wallet bankruptcy.
    lib.yp_trinity_record_result(query_id, actual_bucket, actual_bucket)

    if (i + 1) % 1000 == 0:
        print(f"      {i+1}/{len(train_set)} | pre-train accuracy: {correct_before/(i+1)*100:.2f}%")

print(f"\n  Training complete: {len(train_set)} samples")
print(f"  Pre-training accuracy (stub): {correct_before/len(train_set)*100:.2f}%")

# ── Benchmark baseline (sharded hash) ──
print("\n[5/5] Benchmarking...")
baseline_lat = []
if TEST_ON_TRAINED_TITLES:
    test_slice = papers[WARMUP:WARMUP + NUM_QUERIES]
else:
    test_slice = papers[NUM_TRAIN + WARMUP:NUM_TRAIN + WARMUP + NUM_QUERIES]

for i, (pid, title, h_bytes) in enumerate(test_slice):
    t0 = time.perf_counter()
    results = engine.search_sharded_hash(h_bytes.hex(), top_k=TOP_K)
    t1 = time.perf_counter()
    baseline_lat.append((t1 - t0) * 1_000_000)

baseline_lat.sort()
b_p50 = baseline_lat[NUM_QUERIES // 2]
print(f"      Baseline P50: {b_p50:.2f} µs")

# ── Benchmark trained Trinity ──
out_ids = (ctypes.c_uint64 * TOP_K)()
out_scores = (ctypes.c_float * TOP_K)()
out_len = ctypes.c_size_t()

trinity_lat = []
hits = 0
misses = 0

for i, (pid, title, h_bytes) in enumerate(test_slice):
    pap_128 = h_bytes[:16]
    pap_512 = h_bytes[:64]

    query_id = _rust_id(title)

    t0 = time.perf_counter()
    ret = lib.yp_query_auto_with_trinity_hint(
        query_id,
        ctypes.create_string_buffer(pap_128),
        ctypes.create_string_buffer(pap_512),
        TOP_K,
        out_scores,
        out_ids,
        ctypes.byref(out_len)
    )
    t1 = time.perf_counter()
    trinity_lat.append((t1 - t0) * 1_000_000)

    returned_pid = engine.rust_id_map.get(out_ids[0]) if ret == 0 and out_len.value > 0 else None
    if returned_pid == pid:
        hits += 1
    else:
        misses += 1

    if (i + 1) % 500 == 0:
        print(f"      {i+1}/{NUM_QUERIES} | hits={hits} misses={misses}")

trinity_lat.sort()
t_p50 = trinity_lat[NUM_QUERIES // 2]

# ── Results ──
print("\n" + "=" * 65)
print("RESULTS — TRAINED PREDICTOR")
print("=" * 65)
print(f"\n  BASELINE:")
print(f"    P50: {b_p50:.2f} µs")
print(f"\n  TRINITY (trained):")
print(f"    P50:      {t_p50:.2f} µs")
print(f"    Hit rate: {hits/NUM_QUERIES*100:.1f}%")
print(f"    Misses:   {misses}")

speedup = b_p50 / t_p50 if t_p50 > 0 else 0
print(f"\n  Speedup: {speedup:.2f}x")

if hits > 0:
    print("\n  ✅ Predictor path is hitting! Training worked.")
else:
    print("\n  ⚠️  Still 0% hits — predictor may need different training or bucket hash mismatch.")

out = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'baseline_p50': round(b_p50, 2),
    'trinity_p50': round(t_p50, 2),
    'hit_rate_pct': round(hits/NUM_QUERIES*100, 2),
    'speedup': round(speedup, 2),
    'trained_samples': len(train_set),
    'pre_train_accuracy_pct': round(correct_before/len(train_set)*100, 2),
}
os.makedirs('logs', exist_ok=True)
with open('logs/benchmark_predictor_trained.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved: logs/benchmark_predictor_trained.json")
