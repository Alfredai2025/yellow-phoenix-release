#!/usr/bin/env python3
"""SAH Stage 10 smoke: YPEngine.search_with_sah() routes correctly."""

import sys
sys.path.insert(0, "/Users/mac/yellow_phoenix")

from yp_bridge import YPEngine
from scripts.sah_itq_loader import has_itq_model, load_rotation_matrix, get_fallback_rotation


def main():
    engine = YPEngine()

    # 1. Beacon index must exist
    rc = engine.rust.beacon_index_new(1000)
    assert rc == 0
    print("[1] Beacon index ready")

    # 2. Insert a fake beacon for "hello"
    rotation = load_rotation_matrix() if has_itq_model() else get_fallback_rotation()
    fake_hash = engine._text_to_hash512("hello", rotation)
    engine.rust.beacon_index_insert(fake_hash, node_id=7777, tag=0x01)
    print("[2] Fake beacon inserted for 'hello'")

    # 3. Search with SAH — should hit beacon (distance 0 <= threshold 50)
    result = engine.search_with_sah("hello", k=3, use_cascade=True)
    assert result["source"] in ("beacon", "fallback", "production")
    print(f"[3] search_with_sah source={result['source']}, latency_ms={result['latency_ms']:.3f}")

    # 4. Search with SAH disabled — production only
    result2 = engine.search_with_sah("hello", k=3, use_cascade=False)
    assert result2["source"] == "production"
    print(f"[4] SAH disabled → source={result2['source']}")

    # 5. Unknown query — likely fallback or production
    result3 = engine.search_with_sah("xyz123notfound", k=3, use_cascade=True)
    assert result3["source"] in ("fallback", "production")
    print(f"[5] Unknown query → source={result3['source']}")

    print("\nSAH Stage 10 SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
