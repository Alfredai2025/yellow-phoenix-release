# YP-Bench physical watch — 3.9M SEMANTIC SCAN (Tier C) — VERIFIED
Date: 2026-10-02 18:58 · Device: Apple Watch Series 10 (Watch7,8), watchOS 11.6.2
Corpus: yp-corpus-20261001-3e9a2336, first 3,900,000 docs · flat_39m_new512.bin (280.8MB, YISM)
Hash: index_10m_hashes.bin ITQ-512 (2026-10-01 rebuild) · payload_sem_39m.json (695 semantic queries)
Method: Tier C brute-force Hamming scan, mmap from flash (no graph, RAM-light)
RESULT: R@1/R@10 (top-10 containment) 86.5% · p50 34792µs (~8.9ns/record)
Memory: clean exit, 305MB restored. App v1016 (CURRENT_PROJECT_VERSION bump).
BUG FOUND (root cause of 5 failed attempts): flat file cut with head -c kept the
ORIGINAL header count (5,660,333) while holding 3.9M records -> yp_ism_load rejected
it (no row, "graph unavailable"). Delivery was innocent. Fixed header to 3,900,000.
Lessons: (1) bench only shows the Flat row when yp_ism_load rc==0 — absence of row
means LOAD FAILURE, not file absence; (2) verify header counts after head -c cuts;
(3) devicectl copy-to needs ABSOLUTE destination paths; (4) watch GUI Run needs
device registered in the dev account; (5) devicectl installs are ephemeral (iPhone
re-pushes) — bench immediately.
Full wrist results tonight: 1M graph fast 69.1% @1.47ms (ef400); 3.9M scan 86.5% @35ms.
Screenshots IMG_9754-9757.
