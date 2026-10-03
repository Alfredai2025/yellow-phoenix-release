# YP-Bench physical watch — FIRST SEMANTIC ON-WRIST RESULT
Date: 2026-10-02 17:22 · Device: Apple Watch Series 10 (Watch7,8), watchOS 11.6.2
Corpus: yp-corpus-20261001-3e9a2336 (first 1,000,000 docs)
Hash: index_10m_hashes.bin ITQ-512 (2026-10-01 rebuild) · graph_1m_new512_m8_v5 (m8/efC200, 282MB)
Payload: payload_sem_1m.json — 175 semantic queries (E-space gt, identity-schema {id,hash})
Install: devicectl (ephemeral) · Memory gate: OK, need ~270MB, avail 305MB · load 0.08s
Ladder (exclude-self): ef50 R@10 66.9% p50 353µs · ef100 66.9% p50 480µs ·
ef200 68.0% p50 840µs · ef400 69.1% p50 1474µs
Exit: clean, graph freed, 304MB avail.
NOTES: bench R@1/R@10 counters are rank-blind containment (identical values = artifact);
meaningful metric = top-10 containment. Mac-side brute-force ceiling on same corpus: R@3 96.9%.
Watch gap = m=8 slim graph + small ef budget (storage/battery constraints). QUANTIFIED, not assumed.
Screenshots: IMG_9734-9738 in this directory.
