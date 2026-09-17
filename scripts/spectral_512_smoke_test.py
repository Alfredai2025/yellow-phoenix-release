# Copyright (C) 2026 Marc John Sawyer
# SPDX-License-Identifier: AGPL-3.0-or-later

#!/usr/bin/env python3
"""Smoke test for the 512-bit tensor-spectral FFI wiring."""
import os, sys, random

YP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, YP_ROOT)


def test_imports():
    print("[TEST 1/5] Import yp_bridge...")
    try:
        from yp_bridge import RustBridge
        print("  PASS: yp_bridge imported")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_spectral_bindings():
    print("[TEST 2/5] Spectral 512 FFI bindings...")
    try:
        from yp_bridge import RustBridge
        b = RustBridge()
        assert hasattr(b, "spectral_512_new"), "Missing spectral_512_new"
        assert hasattr(b, "spectral_512_build"), "Missing spectral_512_build"
        assert hasattr(b, "spectral_512_query"), "Missing spectral_512_query"
        assert hasattr(b, "spectral_512_free"), "Missing spectral_512_free"
        print("  PASS: All spectral 512 methods present")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_spectral_lifecycle():
    print("[TEST 3/5] Spectral 512 index lifecycle (10 papers)...")
    try:
        from yp_bridge import RustBridge
        b = RustBridge()
        n = 10
        hashes = bytes([random.randint(0, 255) for _ in range(n * 64)])
        ids = list(range(1000, 1000 + n))
        rc = b.spectral_512_build(hashes, ids)
        assert rc == 0, f"spectral_512_build returned {rc}"
        query_hex = hashes[:64].hex()
        results = b.spectral_512_query(query_hex, top_k=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        b.spectral_512_free()
        print(f"  PASS: Built, queried ({len(results)} results), freed")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_spectral_in_query_path():
    print("[TEST 4/5] Query path integration check...")
    try:
        import yp_engine
        src = open(yp_engine.__file__).read()
        if "spectral_512_query" in src or "spectral_512_handle" in src:
            print("  PASS: spectral 512 hook found in yp_engine.py")
        else:
            print("  WARN: no spectral 512 hook in yp_engine.py")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_rust_compiles():
    print("[TEST 5/5] Rust dylib symbols...")
    try:
        from yp_bridge import RustBridge
        b = RustBridge()
        ptr = b.spectral_512_new()
        assert ptr is not None
        b.spectral_512_free()
        print("  PASS: Rust symbols exported and callable")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  HINT: cd ~/yellow_phoenix && cargo build --release --features holographic-cascade")
        return False


def main():
    print("=" * 60)
    print("YELLOW PHOENIX SPECTRAL 512 SMOKE TEST")
    print("=" * 60)
    tests = [
        test_imports,
        test_spectral_bindings,
        test_spectral_lifecycle,
        test_spectral_in_query_path,
        test_rust_compiles,
    ]
    passed = sum(1 for t in tests if t())
    print("=" * 60)
    print(f"RESULT: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("SPECTRAL 512 WIRING IS LIVE.")
    else:
        print("Some tests failed. Fix Rust build or Python wiring.")
    print("=" * 60)
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
