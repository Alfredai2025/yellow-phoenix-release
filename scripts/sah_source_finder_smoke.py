#!/usr/bin/env python3
"""SAH source finder smoke: region detection + search + optional download."""

import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, "/Users/mac/yellow_phoenix")

from scripts.sah_source_finder import SAHSourceFinder


def main():
    tmpdir = tempfile.mkdtemp()
    SAHSourceFinder.CACHE_DIR = Path(tmpdir)

    finder = SAHSourceFinder()
    assert finder.region in ("global", "china_mainland", "unknown")
    print(f"[1] Region detected: {finder.region}")

    # Search capability (no crash, may return empty offline)
    results = finder.search_models("qwen", limit=3)
    print(f"[2] Search returned {len(results)} models")

    # Try downloading a tiny model; tolerate offline
    tiny_model = "qwen/Qwen2.5-0.5B-Instruct"
    try:
        path = finder.download_model(tiny_model)
        shards = finder.find_local_shards(path)
        print(f"[3] Downloaded {len(shards)} shards from {tiny_model}")
    except Exception as e:
        print(f"[3] Download skipped/offline: {e}")

    # Cache skip test: calling again should report cached if shards exist
    try:
        path2 = finder.download_model(tiny_model)
        cached_shards = finder.find_local_shards(path2)
        print(f"[4] Cache re-check: {len(cached_shards)} shards")
    except Exception as e:
        print(f"[4] Cache re-check skipped: {e}")

    shutil.rmtree(tmpdir)
    print("\nSAH SOURCE FINDER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
