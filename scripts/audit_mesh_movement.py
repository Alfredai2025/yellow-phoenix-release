#!/usr/bin/env python3
"""
Audit: Check if mesh edges match query co-occurrence.
If edges are static (built at load time), they won't match live usage patterns.
"""
import sqlite3
import json
import os
from collections import defaultdict, Counter

DB_PATH = "data/phoenix_arxiv_1m.db"
QUERY_LOG = "logs/query_bucket_log.jsonl"

def _rust_id(paper_id):
    """FNV-1a 64-bit hash — matches Rust hash_id_to_u64 exactly."""
    FNV_BASIS = 0xcbf29ce484222325
    FNV_PRIME = 0x100000001b3
    h = FNV_BASIS
    for b in str(paper_id).encode('utf-8'):
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def audit():
    print("=== MESH MOVEMENT AUDIT ===\n")

    # 1. Check if query log exists
    if not os.path.exists(QUERY_LOG):
        print(f"[AUDIT] No query log at {QUERY_LOG}")
        print("[AUDIT] Run some searches first, then re-run this audit.")
        return

    # 2. Build co-occurrence from query logs
    bucket_hits = Counter()

    with open(QUERY_LOG) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            bucket = entry.get("bucket", 0)
            bucket_hits[bucket] += 1

    print(f"[AUDIT] Query log entries: {sum(bucket_hits.values())}")
    print(f"[AUDIT] Hot buckets (top 5): {bucket_hits.most_common(5)}")

    # 3. Check if mesh snapshot has gravity fields
    mesh_files = [f for f in os.listdir(".") if "mesh" in f and f.endswith(".bin")]
    print(f"[AUDIT] Mesh snapshots found: {mesh_files}")

    # 4. Check autopoiesis log for movement proposals
    auto_log = "logs/autopoiesis_state.json"
    if os.path.exists(auto_log):
        with open(auto_log) as f:
            state = json.load(f)
        proposals = state.get("proposals", [])
        print(f"[AUDIT] Autopoiesis proposals: {len(proposals)}")
        for p in proposals[:3]:
            print(f"  - {p}")
    else:
        print("[AUDIT] No autopoiesis log found")

    # 5. Gravity check: load engine and inspect live gravity_map
    print("\n=== GRAVITY CHECK ===")
    gravity_pids = set()
    total_gravity = 0
    engine = None
    try:
        import sys
        sys.path.insert(0, ".")
        from yp_bridge import YPEngine
        engine = YPEngine()
    except Exception as e:
        print(f"[AUDIT] Could not load engine for gravity check: {e}")

    if engine and sum(bucket_hits.values()) >= 10:
        # Replay a sample of unique queries and check gravity of returned papers
        unique_queries = list({entry.get("query", "") for entry in
                               (json.loads(l) for l in open(QUERY_LOG) if l.strip())
                               if entry.get("query", "")})
        sample_queries = unique_queries[:20]
        print(f"[AUDIT] Replaying {len(sample_queries)} unique queries to sample gravity...")

        fn = engine.rust.available.get("yp_mesh_get_gravity")
        if fn:
            for q in sample_queries:
                try:
                    results = engine.search(q, top_k=5)
                    if not results:
                        continue
                    for r in results:
                        if isinstance(r, tuple) and len(r) == 2:
                            score, payload = r
                            if isinstance(payload, tuple) and len(payload) == 2:
                                pid, title = payload
                                g = fn(_rust_id(pid))
                                if g > 0:
                                    gravity_pids.add(pid)
                                    total_gravity += g
                except Exception:
                    pass
            print(f"[AUDIT] Papers with gravity > 0: {len(gravity_pids)}")
            print(f"[AUDIT] Total gravity accumulated: {total_gravity}")

            # Trigger one autopoiesis re-bucketing cycle and report moves
            try:
                from yp_bridge import _run_rebucket_cycle
                moved = _run_rebucket_cycle(engine)
                dynamic_count = engine.rust.available.get("yp_mesh_dynamic_bucket_count", lambda: 0)()
                print(f"[AUDIT] Papers moved this cycle: {moved}")
                print(f"[AUDIT] Papers in dynamic buckets: {dynamic_count}")
            except Exception as e:
                print(f"[AUDIT] Re-bucketing check failed: {e}")
        else:
            print("[AUDIT] yp_mesh_get_gravity not available")

    # 6. VERDICT
    print("\n=== VERDICT ===")
    if sum(bucket_hits.values()) < 100:
        print("  NOT ENOUGH DATA: Run at least 100 searches first.")
    elif len(gravity_pids) > 0:
        print(f"  DYNAMIC: Mesh gravity detected on {len(gravity_pids)} papers.")
        print(f"  Papers are accumulating query_count and co-occurrence edges.")
        print(f"  LIVING MESH: Autopoiesis has moved papers between buckets.")
    elif not os.path.exists(auto_log):
        print("  STATIC: Mesh has no autopoiesis feedback loop.")
        print("  Papers are frozen in their original buckets.")
    else:
        print("  Check proposals above. If none mention 'move' or 'edge', mesh is static.")

    print("\n=== NEXT: Run searches, then Phase 2 to build gravity ===")

if __name__ == "__main__":
    audit()
