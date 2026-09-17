#!/usr/bin/env python3
"""Standalone timer sanity check using a pure Python prime sieve."""
import time
import math
import platform


def main():
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("Running prime sieve from 2 to 100,000...")

    start = time.perf_counter()
    count = 0
    for i in range(2, 100_000):
        is_prime = True
        limit = int(math.sqrt(i)) + 1
        for j in range(2, limit):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    elapsed = time.perf_counter() - start

    print(f"Primes found: {count}")
    print(f"Elapsed time: {elapsed:.3f}s")
    print(f"Events/sec (primes/second): {count / elapsed:,.0f}")

    if elapsed < 0.5:
        print("WARNING: completed suspiciously fast; timer may be unreliable.")
    elif 5_000 <= (count / elapsed) <= 15_000:
        print("Timer looks normal for an Apple Silicon Mac.")
    else:
        print("Timer result is outside the expected 5,000-15,000 range.")


if __name__ == "__main__":
    main()
