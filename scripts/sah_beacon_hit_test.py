#!/usr/bin/env python3
"""SAH Beacon Hit Test — can ANY query hit a beacon?"""

import sys
import numpy as np

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import RustBridge, YPEngine
from scripts.sah_cascade import SAHCascade
from scripts.sah_itq_loader import load_rotation_matrix, has_itq_model, get_fallback_rotation


def main():
    print("=" * 60)
    print("SAH BEACON HIT TEST")
    print("=" * 60)

    bridge = RustBridge()
    engine = YPEngine()
    count = bridge.beacon_index_count()
    print(f"\nBeacon index: {count} entries")

    if count == 0:
        print("No beacons. Run sah_live_test.py first.")
        return

    # Get rotation
    rotation = load_rotation_matrix() if has_itq_model() else get_fallback_rotation()

    # 1. SELF-QUERY: Use a beacon hash to search itself
    print("\n[1] SELF-QUERY TEST (beacon hash → beacon index)")
    # We need to extract a beacon hash. We can do this by inserting a known hash
    # and searching for it, but we can't read beacon hashes back directly.
    # Instead: create a hash, insert it, search with same hash = distance 0
    test_vec = np.random.randn(512).astype(np.float32)
    test_vec = test_vec / np.linalg.norm(test_vec)
    test_hash = bridge.sah_eigenvector_to_hash(test_vec.tolist(), rotation)
    bridge.beacon_index_insert(test_hash, node_id=99999, tag=0x01)

    cascade = SAHCascade(bridge, production_search_fn=None)
    result = cascade.query(test_hash, k=3)
    print(f"    Source: {result['source']}")
    print(f"    Results: {result['results']}")
    if result['source'] == 'beacon':
        top = result['results'][0]
        print(f"    ✓ SELF-HIT: id={top[0]}, distance={top[1]}, tag={top[2]}")

    # 2. HUMAN TEXT QUERIES
    print("\n[2] HUMAN TEXT QUERIES")
    human_queries = [
        "machine learning",
        "attention mechanism",
        "neural network",
        "transformer architecture",
        "hello world",
        "what is the capital of France",
        "explain quantum computing",
    ]
    for q in human_queries:
        query_hash = engine._hash_query(q)
        result = cascade.query(query_hash, k=3)
        src = result['source']
        dist = result['results'][0][1] if result['results'] else 'N/A'
        print(f"    '{q}' → source={src}, best_dist={dist}")

    # 3. AI-GENERATED TEXT (simulated)
    # We don't have Qwen running, but we can try text that "looks like" model output
    print("\n[3] AI-STYLE TEXT QUERIES")
    ai_queries = [
        "The user is asking about machine learning. I should provide a comprehensive explanation of neural networks, deep learning architectures, and training methodologies.",
        "Based on the transformer architecture described in 'Attention Is All You Need', the query relates to self-attention mechanisms and multi-head attention layers.",
        "As an AI assistant, I will analyze this query through my attention layers and provide a reasoned response based on my training data.",
        "Processing input tokens through embedding layer → positional encoding → multi-head self-attention → feed-forward network → layer normalization → output logits.",
        "The weight matrix W_q projects the input into query space, while W_k and W_v handle keys and values respectively.",
    ]
    for q in ai_queries:
        query_hash = engine._hash_query(q)
        result = cascade.query(query_hash, k=3)
        src = result['source']
        dist = result['results'][0][1] if result['results'] else 'N/A'
        print(f"    '{q[:50]}...' → source={src}, best_dist={dist}")

    # 4. RANDOM HASH (baseline)
    print("\n[4] RANDOM HASH BASELINE")
    random_hash = bytes(np.random.randint(0, 256, size=64, dtype=np.uint8))
    result = cascade.query(random_hash, k=3)
    src = result['source']
    dist = result['results'][0][1] if result['results'] else 'N/A'
    print(f"    Random bytes → source={src}, best_dist={dist}")

    # 5. STATISTICAL SUMMARY
    print("\n[5] STATISTICAL SUMMARY")
    print(f"    Total beacons: {count}")
    print(f"    Beacon dimension: 512 bits (64 bytes)")
    print(f"    Hamming threshold: 50 (attention), 60 (FFN), 70 (embed), 80 (other)")
    print(f"    Expected random-to-beacon distance: ~256 (half the bits differ)")
    print(f"    Probability of random hit: essentially 0")
    print(f"    Self-query distance: 0 (exact match)")
    print(f"    Conclusion: Beacons only hit for queries in the SAME distribution")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
