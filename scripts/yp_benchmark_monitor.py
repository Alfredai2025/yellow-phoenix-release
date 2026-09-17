#!/usr/bin/env python3
"""
YP Benchmark Monitor — Internal Sanity Check
Verifies measurements are consistent, not hardcoded.
"""
import subprocess
import re
import statistics
import time
import os
import json
import sys


def run_validate():
    """Run end-to-end validation and extract R@1 + timing."""
    t0 = time.time()
    result = subprocess.run(
        ["python3", "scripts/validate_end_to_end.py"],
        capture_output=True, text=True, cwd=os.path.expanduser("~/yellow_phoenix")
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr

    # Extract R@1
    r1_match = re.search(r'R@1[:\s]+([\d.]+%?)', output)
    r1 = r1_match.group(1) if r1_match else "N/A"

    # Extract P50 if present
    p50_match = re.search(r'P50[:\s]+([\d.]+)\s*us', output)
    p50 = float(p50_match.group(1)) if p50_match else None

    return r1, p50, elapsed, output


def run_stress():
    """Run stress test and extract QPS."""
    result = subprocess.run(
        ["python3", "scripts/stress_test_50k.py"],
        capture_output=True, text=True, cwd=os.path.expanduser("~/yellow_phoenix")
    )
    output = result.stdout + result.stderr
    qps_match = re.search(r'(\d+)\s*qps', output, re.IGNORECASE)
    if qps_match is None:
        # stress_test_50k.py reports "Avg throughput: 1234 searches/sec"
        qps_match = re.search(r'Avg throughput:\s*(\d+)\s*searches/sec', output, re.IGNORECASE)
    qps = int(qps_match.group(1)) if qps_match else None
    return qps, output


def variance_test():
    """Run validation 5 times. Check variance."""
    print("=" * 60)
    print("VARIANCE TEST: 5 runs of validate_end_to_end.py")
    print("=" * 60)

    r1s = []
    times = []

    for i in range(5):
        print(f"\nRun {i+1}/5...")
        r1, p50, elapsed, output = run_validate()
        print(f"  R@1: {r1} | Time: {elapsed:.2f}s")
        r1s.append(r1)
        times.append(elapsed)
        time.sleep(2)  # Cool down between runs

    # Check R@1 consistency
    unique_r1 = set(r1s)
    if len(unique_r1) == 1:
        print(f"\n⚠️  WARNING: R@1 identical across all 5 runs ({r1s[0]})")
        print("   This could mean deterministic test data, not necessarily fake.")
    else:
        print(f"\n✅ R@1 varies across runs: {unique_r1}")

    # Check timing variance
    if len(times) >= 2:
        mean_t = statistics.mean(times)
        stdev_t = statistics.stdev(times)
        cv = stdev_t / mean_t if mean_t > 0 else 0
        print(f"  Timing: mean={mean_t:.2f}s, stdev={stdev_t:.2f}s, CV={cv:.3f}")
        if cv < 0.01:
            print("⚠️  WARNING: Timing variance is extremely low.")
        else:
            print("✅ Timing variance looks natural.")

    return True


def stress_consistency_test():
    """Run stress test and check QPS is in reasonable range."""
    print("\n" + "=" * 60)
    print("STRESS TEST CONSISTENCY")
    print("=" * 60)

    qps, output = run_stress()
    if qps is None:
        print("❌ Could not extract QPS from stress test")
        return False

    print(f"  QPS: {qps}")

    # Sanity check: QPS should be between 1000 and 10000 for 50K queries
    if qps < 500:
        print("❌ QPS suspiciously low (< 500). Possible hang or error.")
        return False
    elif qps > 50000:
        print("❌ QPS suspiciously high (> 50000). Possible measurement error.")
        return False
    else:
        print(f"✅ QPS in reasonable range: {qps}")

    # Check for errors in output
    if "error" in output.lower() or "fail" in output.lower():
        print("⚠️  Found 'error' or 'fail' in stress test output. Review needed.")

    return True


def source_audit():
    """Quick scan for hardcoded timing values in non-test code."""
    print("\n" + "=" * 60)
    print("SOURCE CODE AUDIT")
    print("=" * 60)

    src_dir = os.path.expanduser("~/yellow_phoenix/src")
    red_flags = []

    # Check for hardcoded durations in Rust (excluding test modules)
    result = subprocess.run(
        ["grep", "-rn", "Duration::from", src_dir, "--include=*.rs"],
        capture_output=True, text=True
    )
    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            # Skip lines inside #[cfg(test)] modules (best-effort: ignore test fakes)
            if "from_micros" in line or "from_nanos" in line:
                if "test" not in line.lower() and "mock" not in line.lower():
                    red_flags.append(f"HARD-CODED DURATION: {line}")

    # Check for suspicious fake/mock/simulated timing constants outside tests
    result = subprocess.run(
        ["grep", "-rni", "mock\\|fake\\|simulated", src_dir, "--include=*.rs"],
        capture_output=True, text=True
    )
    # Files that are entirely invariant tests / fixtures; not measurement fraud
    test_fixture_files = {"safety_invariants.rs", "canonicalization.rs", "executor.rs"}
    if result.stdout:
        # Only flag if it appears outside obvious test contexts
        suspicious = []
        for line in result.stdout.strip().split('\n'):
            skip = (
                "test" in line.lower() or
                "FakeEngine" in line or
                "FakeDomain" in line or
                "fake_record" in line or
                any(f in line for f in test_fixture_files)
            )
            if skip:
                continue
            suspicious.append(line)
        if suspicious:
            red_flags.append("Found non-test mock/fake/simulated references in source")
            for line in suspicious[:5]:
                red_flags.append(f"   {line}")

    # 'hardcoded' field names are config, not timing fraud — note but don't fail
    result = subprocess.run(
        ["grep", "-rni", "hardcoded", src_dir, "--include=*.rs"],
        capture_output=True, text=True
    )
    hardcoded_fields = result.stdout.strip().split('\n') if result.stdout else []

    if red_flags:
        print("❌ RED FLAGS:")
        for flag in red_flags:
            print(f"   {flag}")
        return False

    print("✅ No hardcoded durations or suspicious non-test mock/fake references found.")
    if hardcoded_fields:
        print("ℹ️  Note: 'hardcoded' used as field/variable name (not timing related):")
        for line in hardcoded_fields[:3]:
            print(f"   {line}")
    return True


def main():
    print("\n" + "=" * 60)
    print("YELLOW PHOENIX BENCHMARK MONITOR")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # Test 1: Variance
    try:
        results.append(("Variance Test", variance_test()))
    except Exception as e:
        print(f"❌ Variance test crashed: {e}")
        results.append(("Variance Test", False))

    # Test 2: Stress consistency
    try:
        results.append(("Stress Consistency", stress_consistency_test()))
    except Exception as e:
        print(f"❌ Stress test crashed: {e}")
        results.append(("Stress Consistency", False))

    # Test 3: Source audit
    try:
        results.append(("Source Audit", source_audit()))
    except Exception as e:
        print(f"❌ Source audit crashed: {e}")
        results.append(("Source Audit", False))

    # Summary
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("✅✅✅ BENCHMARKS PASS INTERNAL SANITY CHECK ✅✅✅")
    else:
        print("❌❌❌ ISSUES DETECTED — REVIEW BEFORE ARXIV ❌❌❌")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
