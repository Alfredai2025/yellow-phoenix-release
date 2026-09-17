import os, sys, time, traceback
import numpy as np
import logging
logging.getLogger("yp_bridge").setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yp_bridge import RustBridge

print("="*70)
print("BATCH 1: TOP 10 EXPERIMENTAL MODULES")
print("="*70)

def test_module(name, test_fn):
    print(f"\nTESTING: {name}")
    t0 = time.time()
    try:
        result = test_fn()
        elapsed = time.time() - t0
        print(f"  STATUS: PASS ({elapsed:.3f}s)")
        print(f"  RESULT: {result}")
        return {"status": "PASS", "time": elapsed, "result": str(result)}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  STATUS: FAIL ({elapsed:.3f}s)")
        print(f"  ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"status": "FAIL", "time": elapsed, "error": f"{type(e).__name__}: {e}"}

results = {}

# 1. TENSOR_SPECTRAL
def test_tensor_spectral():
    bridge = RustBridge()
    emb = np.random.randn(100, 384).astype(np.float32)
    emb_bytes = emb.tobytes()
    rc = bridge.tensor_spectral_build(emb_bytes, 100, 384)
    q = emb[0].tobytes()
    result = bridge.tensor_spectral_query(q)
    return f"build_rc={rc}, query_type={type(result)}"

results["tensor_spectral"] = test_module("tensor_spectral", test_tensor_spectral)

# 2. CRYSTAL_MESH_384
def test_crystal_mesh_384():
    bridge = RustBridge()
    handle = bridge.crystal_create(1024, 384)
    emb = np.random.randn(10, 384).astype(np.float32)
    for i in range(10):
        bridge.crystal_insert(handle, i, emb[i].tobytes())
    result = bridge.crystal_query_crystal(handle, emb[0].tobytes(), 5)
    bridge.crystal_destroy(handle)
    return f"handle={handle}, query_result={result}"

results["crystal_mesh_384"] = test_module("crystal_mesh_384", test_crystal_mesh_384)

# 3. SPECTRAL_COORDS
def test_spectral_coords():
    bridge = RustBridge()
    emb = np.random.randn(100, 384).astype(np.float32)
    emb_bytes = emb.tobytes()
    rc = bridge.spectral_coords_build(emb_bytes, 100, 384)
    q = emb[0].tobytes()
    result = bridge.spectral_coords_query(rc, q, 384, 5)
    bridge.spectral_coords_drop(rc)
    return f"build_handle={rc}, query_type={type(result)}"

results["spectral_coords"] = test_module("spectral_coords", test_spectral_coords)

# 4. SPECTRAL_STAGE
def test_spectral_stage():
    bridge = RustBridge()
    emb = np.random.randn(100, 384).astype(np.float32)
    emb_bytes = emb.tobytes()
    rc = bridge.spectral_stage_build(emb_bytes, 100, 384)
    q = emb[0].tobytes()
    result = bridge.spectral_stage_query(rc, q, 384, 5)
    bridge.spectral_stage_drop(rc)
    return f"build_handle={rc}, query_type={type(result)}"

results["spectral_stage"] = test_module("spectral_stage", test_spectral_stage)

# 5. LEARNED_ROUTER
def test_learned_router():
    bridge = RustBridge()
    handle = bridge.learned_router_new(10)
    for i in range(10):
        emb = np.random.randn(384).astype(np.float32)
        bridge.learned_router_insert(handle, i, emb.tobytes())
    q = np.random.randn(384).astype(np.float32)
    result = bridge.learned_router_query(handle, q.tobytes(), 5)
    bridge.learned_router_drop(handle)
    return f"handle={handle}, query_result={result}"

results["learned_router"] = test_module("learned_router", test_learned_router)

# 6. SELF_TUNER
def test_self_tuner():
    bridge = RustBridge()
    handle = bridge.self_tuner_new(5)
    bridge.self_tuner_submit_observation(handle, 0.5, 1.0)
    result = bridge.self_tuner_propose(handle)
    bridge.self_tuner_drop(handle)
    return f"handle={handle}, proposal={result}"

results["self_tuner"] = test_module("self_tuner", test_self_tuner)

# 7. DRIFT_DETECTOR
def test_drift_detector():
    bridge = RustBridge()
    handle = bridge.drift_detector_new(100)
    for i in range(10):
        emb = np.random.randn(384).astype(np.float32)
        bridge.drift_detector_record(handle, emb.tobytes())
    result = bridge.drift_detector_status(handle)
    bridge.drift_detector_drop(handle)
    return f"handle={handle}, status={result}"

results["drift_detector"] = test_module("drift_detector", test_drift_detector)

# 8. RESULT_CACHE
def test_result_cache():
    bridge = RustBridge()
    handle = bridge.result_cache_new(1000)
    bridge.result_cache_insert(handle, b"test_key", b"test_value")
    result = bridge.result_cache_get(handle, b"test_key")
    bridge.result_cache_drop(handle)
    return f"handle={handle}, cached_value={result}"

results["result_cache"] = test_module("result_cache", test_result_cache)

# 9. ENGINE_FEEDER
def test_engine_feeder():
    bridge = RustBridge()
    handle = bridge.engine_feeder_new(10)
    emb = np.random.randn(384).astype(np.float32)
    rc = bridge.engine_feeder_submit(handle, emb.tobytes(), 42)
    bridge.engine_feeder_drop(handle)
    return f"handle={handle}, submit_rc={rc}"

results["engine_feeder"] = test_module("engine_feeder", test_engine_feeder)

# 10. QUERY
def test_query_module():
    bridge = RustBridge()
    emb = np.random.randn(100, 384).astype(np.float32)
    emb_bytes = emb.tobytes()
    rc = bridge.query_build(emb_bytes, 100, 384)
    q = emb[0].tobytes()
    result = bridge.query_search(rc, q, 384, 5)
    bridge.query_drop(rc)
    return f"build_handle={rc}, query_type={type(result)}"

results["query"] = test_module("query", test_query_module)

print("\n" + "="*70)
print("BATCH 1 SUMMARY")
print("="*70)
for name, data in results.items():
    icon = "PASS" if data["status"] == "PASS" else "FAIL"
    print(f"{icon:4s} {name:25s} {data['time']:.3f}s")

pass_count = sum(1 for d in results.values() if d["status"] == "PASS")
fail_count = len(results) - pass_count
print(f"\nPASS: {pass_count}/10  FAIL: {fail_count}/10")

with open("logs/batch1_experimental_test.json", "w") as f:
    import json
    json.dump(results, f, indent=2)
print("Saved to logs/batch1_experimental_test.json")
