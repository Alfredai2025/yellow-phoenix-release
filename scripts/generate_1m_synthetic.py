#!/usr/bin/env python3
"""Generate 1M synthetic papers with Zipf topic clustering.

Format matches validate_m1_input.bin:
    u32 count
    for each paper:
        u64 id
        [u8; 16] pap_128
        [u8; 64] pap_512

80% of papers belong to one of the top-1000 topics (Zipf distributed);
20% are random long-tail topics.  Papers in the same topic share hash
prefixes, which produces realistic coarse-bucket collisions for M3.5.
"""

import hashlib
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "1m_papers.bin")
COUNT = 1_000_000
TOPIC_COUNT = 1000
ZIPF_S = 1.5
LONG_TAIL_START = TOPIC_COUNT
LONG_TAIL_END = TOPIC_COUNT * 100


def _zipf_topic(s: float, n: int) -> int:
    """Sample a topic index 0..n-1 from a Zipf distribution."""
    denom = sum(1.0 / (i ** s) for i in range(1, n + 1))
    u = random.random()
    cum = 0.0
    for i in range(1, n + 1):
        cum += 1.0 / (i ** s) / denom
        if u <= cum:
            return i - 1
    return n - 1


def _topic_prefix(topic: int) -> bytes:
    """Deterministic 2-byte prefix shared by every paper in a topic."""
    h = hashlib.sha256(f"topic:{topic}".encode()).digest()
    return h[:2]


def _bytes_from_seed(seed: int, length: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(length))


def main():
    random.seed(42)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    # Cap how many records share the same topic prefix.  This keeps coarse
    # buckets bounded so build_edges() stays fast at 1M scale, while still
    # producing enough same-bucket queries to exercise batch fusion.
    PREFIX_CAP = 500
    topic_counts: dict[int, int] = {}

    with open(OUTPUT, "wb") as f:
        f.write(struct.pack("<I", COUNT))
        for i in range(COUNT):
            # 80% clustered in popular topics, 20% long-tail.
            if random.random() < 0.8:
                topic = _zipf_topic(ZIPF_S, TOPIC_COUNT)
            else:
                topic = random.randint(LONG_TAIL_START, LONG_TAIL_END)

            # Share a topic prefix to create coarse-bucket collisions, but keep
            # the rest of the hash unique per record so exact-match R@1 stays
            # meaningful.  Overflow records get a random prefix so no bucket
            # grows unbounded.
            count = topic_counts.get(topic, 0)
            if count < PREFIX_CAP:
                topic_counts[topic] = count + 1
                prefix = _topic_prefix(topic)
            else:
                prefix = _bytes_from_seed(i, 2)
            tail_128 = _bytes_from_seed(i, 14)
            tail_512 = _bytes_from_seed(i + 1_000_000_000, 62)
            pap_128 = prefix + tail_128
            pap_512 = prefix + tail_512

            f.write(struct.pack("<Q", i))
            f.write(pap_128)
            f.write(pap_512)

            if (i + 1) % 100_000 == 0:
                print(f"  generated {i + 1:,} / {COUNT:,}")

    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"Wrote {OUTPUT}: {COUNT:,} records ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
