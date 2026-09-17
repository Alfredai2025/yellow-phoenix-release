#!/usr/bin/env python3
"""
Diversity test: 5,000 searches across ALL topics in the corpus.
Uses real paper titles/abstracts to generate queries, not just AI.
"""
import sys
import time
import random
import sqlite3
sys.path.insert(0, ".")
from yp_bridge import YPEngine

DB_PATH = "data/phoenix_arxiv_1m.db"
N_SEARCHES = 5000

def load_diverse_queries(db_path, n=200):
    """Extract topic keywords from all domains in the corpus."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Sample papers from different domains by looking at title keywords
    cur.execute("""
        SELECT title FROM papers 
        WHERE title IS NOT NULL 
        AND length(title) > 20
        ORDER BY RANDOM()
        LIMIT 500
    """)
    titles = [row[0] for row in cur.fetchall()]
    conn.close()
    
    # Extract 2-4 word phrases from titles
    queries = []
    for t in titles:
        words = t.lower().split()
        if len(words) >= 3:
            # Take 2-4 consecutive words as a query
            start = random.randint(0, max(0, len(words) - 4))
            end = min(start + random.randint(2, 4), len(words))
            q = " ".join(words[start:end])
            queries.append(q)
    
    return queries[:n]

def main():
    print("=" * 60)
    print("DIVERSITY TEST: 5K searches across ALL topics")
    print("=" * 60)
    
    engine = YPEngine()
    
    # Load diverse queries
    queries = load_diverse_queries(DB_PATH)
    print(f"\nLoaded {len(queries)} diverse query phrases from corpus")
    print(f"Sample: {', '.join(queries[:5])}")
    
    # BEFORE state
    before = engine.rust.lib.yp_mesh_dynamic_bucket_count()
    print(f"\nDynamic buckets BEFORE: {before}")
    
    # Run 5K searches
    print(f"\nRunning {N_SEARCHES} searches...")
    t0 = time.time()
    
    for i in range(N_SEARCHES):
        q = random.choice(queries)
        try:
            engine.search(q, top_k=5)
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N_SEARCHES} done ({time.time()-t0:.1f}s)")
    
    elapsed = time.time() - t0
    print(f"\nSearches complete in {elapsed:.1f}s ({elapsed/N_SEARCHES*1000:.2f} ms avg)")
    
    # AFTER searches (before re-bucketing)
    mid = engine.rust.lib.yp_mesh_dynamic_bucket_count()
    print(f"Dynamic buckets after searches: {mid}")
    
    # Trigger re-bucketing
    print("\nTriggering re-bucketing...")
    engine._run_rebucket_cycle()
    
    # FINAL
    after = engine.rust.lib.yp_mesh_dynamic_bucket_count()
    moved = after - before
    print(f"\n{'='*60}")
    print(f"  Dynamic buckets BEFORE: {before}")
    print(f"  Dynamic buckets AFTER:  {after}")
    print(f"  Papers moved: {moved}")
    print(f"  Percentage of 12K corpus: {moved/12702*100:.2f}%")
    
    # Adaptive threshold: median of expected movement based on search volume
    expected_min = max(5, N_SEARCHES // 500)  # at least 1 move per 500 searches
    ai_baseline = 17  # previous AI-only run

    if moved > 50:
        print("\n  ✅ STRONG MESH: Diverse queries moved many papers.")
    elif moved >= expected_min:
        print(f"\n  ✅ GROWING MESH: {moved} papers moved (expected min: {expected_min}, AI baseline: {ai_baseline})")
    elif moved > ai_baseline:
        print(f"\n  ✅ IMPROVING: {moved} > AI-only baseline ({ai_baseline}), but below expected {expected_min}")
    else:
        print(f"\n  ⚠️  LOW: {moved} moved. Expected {expected_min}, AI baseline {ai_baseline}. Check gravity logging.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
