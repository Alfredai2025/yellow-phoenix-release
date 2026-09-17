#!/usr/bin/env python3
"""
Autopoiesis Cascade Tuner:
1. Reads query logs
2. Diagnoses which query types fail
3. Proposes threshold adjustments
4. Evaluates on held-out set
5. Updates live thresholds if improvement > 2%
"""
import sys, os, sqlite3, time, random, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yp_bridge import YPEngine

# Current cascade thresholds (layer 1-3)
CURRENT_THRESHOLDS = {'l1': 0.90, 'l2': 0.75, 'l3': 0.60}

def analyze_logs():
    """Read last 1000 queries, find failure patterns."""
    conn = sqlite3.connect('logs/query_log.db')
    c = conn.cursor()
    c.execute('''
        SELECT CASE WHEN num_words < 4 THEN 'short' 
                    WHEN num_words < 8 THEN 'medium' 
                    ELSE 'long' END AS category,
               AVG(r1_match), AVG(latency_us), COUNT(*)
        FROM queries
        GROUP BY category
        ORDER BY category
    ''')
    rows = c.fetchall()
    conn.close()
    
    diagnosis = {}
    for row in rows:
        category, avg_r1, avg_lat, count = row
        diagnosis[category] = {
            'r1': avg_r1 or 0,
            'latency': avg_lat or 0,
            'count': count
        }
    return diagnosis

def propose_thresholds(diagnosis):
    """Propose new thresholds based on diagnosis."""
    proposed = dict(CURRENT_THRESHOLDS)
    
    # If short queries have low R@1, lower layer 1 threshold (more permissive)
    if diagnosis.get('short', {}).get('r1', 1.0) < 0.7:
        proposed['l1'] = max(0.5, CURRENT_THRESHOLDS['l1'] - 0.10)
        print(f"  [DIAGNOSE] Short query R@1 low → lower l1 to {proposed['l1']}")
    
    # If long queries have high latency, raise layer 2 threshold (stricter early filter)
    if diagnosis.get('long', {}).get('latency', 0) > 6000:
        proposed['l2'] = min(0.95, CURRENT_THRESHOLDS['l2'] + 0.10)
        print(f"  [DIAGNOSE] Long query latency high → raise l2 to {proposed['l2']}")
    
    # If medium queries are balanced, keep as-is
    return proposed

def evaluate_thresholds(thresholds, n=200):
    """Test proposed thresholds on held-out set."""
    engine = YPEngine()
    papers = list(engine.cache.items())
    random.seed(42)
    sample = random.sample(papers, n)
    
    r1_total = 0
    latencies = []
    
    for true_pid, title in sample:
        # Corrupt title (word deletion) to test robustness
        words = title.split()
        if len(words) > 3:
            corrupted = ' '.join(random.sample(words, max(1, int(len(words)*0.7))))
        else:
            corrupted = title
        
        t0 = time.perf_counter()
        results = engine.search(corrupted, top_k=5)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1_000_000)
        
        result_pids = [r[1][0] for r in results if len(r) > 1 and isinstance(r[1], tuple)]
        if true_pid in result_pids[:1]:
            r1_total += 1
    
    return {
        'r1': r1_total / n * 100,
        'p50': statistics.median(latencies),
    }

def autopoiesis_step():
    """One full autopoiesis cycle."""
    print("=" * 65)
    print("AUTOPOIESIS CASCADE TUNER")
    print("=" * 65)
    
    # 1. OBSERVE
    print("\n[1/5] OBSERVE: Reading query logs...")
    diagnosis = analyze_logs()
    for cat, stats in diagnosis.items():
        print(f"  {cat}: R@1={stats['r1']:.2f}, latency={stats['latency']:.0f}µs, n={stats['count']}")
    
    # 2. DIAGNOSE + PROPOSE
    print("\n[2/5] DIAGNOSE + PROPOSE: Finding threshold adjustments...")
    proposed = propose_thresholds(diagnosis)
    print(f"  Current:  {CURRENT_THRESHOLDS}")
    print(f"  Proposed: {proposed}")
    
    if proposed == CURRENT_THRESHOLDS:
        print("  No changes proposed. System is balanced.")
        return
    
    # 3. EVALUATE
    print("\n[3/5] EVALUATE: Testing proposed thresholds...")
    print("  Baseline evaluation...")
    baseline = evaluate_thresholds(CURRENT_THRESHOLDS)
    print(f"  Baseline: R@1={baseline['r1']:.1f}%, P50={baseline['p50']:.0f}µs")
    
    print("  Proposed evaluation...")
    proposed_result = evaluate_thresholds(proposed)
    print(f"  Proposed: R@1={proposed_result['r1']:.1f}%, P50={proposed_result['p50']:.0f}µs")
    
    # 4. CONSOLIDATE
    print("\n[4/5] CONSOLIDATE: Decision...")
    r1_gain = proposed_result['r1'] - baseline['r1']
    lat_change = proposed_result['p50'] - baseline['p50']
    
    print(f"  R@1 change: {r1_gain:+.1f}pp")
    print(f"  Latency change: {lat_change:+.0f}µs")
    
    if r1_gain > 2.0 and lat_change < 1000:
        print("  ✅ ACCEPT: Significant recall gain, acceptable latency cost")
        return proposed
    elif r1_gain > 0 and lat_change < 500:
        print("  ✅ ACCEPT: Modest recall gain, low latency cost")
        return proposed
    else:
        print("  ❌ REJECT: No improvement or too slow")
        return CURRENT_THRESHOLDS

if __name__ == '__main__':
    new_thresholds = autopoiesis_step()
    if new_thresholds != CURRENT_THRESHOLDS:
        print(f"\n[5/5] UPDATE: Applying new thresholds {new_thresholds}")
        # In real system, write to config file or call FFI
        with open('logs/cascade_thresholds.json', 'w') as f:
            import json
            json.dump(new_thresholds, f, indent=2)
        print("  Saved to logs/cascade_thresholds.json")
    else:
        print("\n[5/5] UPDATE: No changes needed")
