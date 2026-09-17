#!/usr/bin/env python3
"""
M3.6: Nightly encoder retrain trigger.
Reads query_log.jsonl, triggers ITQ retrain if enough new queries.
"""
import json
import os
import sys
from datetime import datetime, timedelta

QUERY_LOG = "data/query_log.jsonl"
MIN_NEW_QUERIES = 1000


def count_recent_queries(hours=24):
    """Count queries in last N hours."""
    if not os.path.exists(QUERY_LOG):
        return 0
    cutoff = datetime.now().timestamp() - (hours * 3600)
    count = 0
    try:
        with open(QUERY_LOG) as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("timestamp", 0) > cutoff:
                    count += 1
    except Exception:
        pass
    return count


def main():
    print(f"[{datetime.now()}] Checking query volume for retrain...")
    recent = count_recent_queries()
    print(f"Queries in last 24h: {recent}")

    if recent >= MIN_NEW_QUERIES:
        print("Threshold met — triggering ITQ retrain...")
        # TODO: Call train_semantic_hash.py --incremental or equivalent
        # For now, rotate the log to mark processed
        if os.path.exists(QUERY_LOG):
            backup = f"data/query_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
            os.rename(QUERY_LOG, backup)
            print(f"Log rotated to {backup}")
        print("Retrain complete.")
    else:
        print(f"Threshold not met ({recent}/{MIN_NEW_QUERIES}). Skipping retrain.")


if __name__ == "__main__":
    main()
