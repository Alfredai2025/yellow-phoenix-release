# YP-Bench LINUX leg — TABLE-1 identical 1M — VERIFIED (fourth platform)
Date: 2026-10-02 ~23:56 UTC+8 · Host: DigitalOcean ubuntu-s-2vcpu-2gb (sgp1), x86_64, 2 vCPU / 2GB RAM
Corpus/graph: graph_1m_new512_m8_v5.bin (282,299,760 B, md5 c69e83e6fc26ca32eeb29d080facf407 —
verified identical on Mac and droplet). Queries: payload_sem_1m.json → queries_1m.bin
(175 queries, targets = E-space gt, all inside 1M prefix).
Engine: droplet ~/yp_linux_test, src/binary_hnsw.rs + ffi_binary_hnsw.rs byte-IDENTICAL
(diff-verified) to the /tmp/yp_build snapshot (f9968fe3) that built the graph.
Binary: examples/bench_identical_1m (Rust API direct; calls same load/set_ef/search path
the FFI shim wraps). YP_GENESIS_PHRASE=YP_DEV_BUILD_2026_08_19.

RESULT (warm p50; ef50 row includes one-time mmap page-in of 282MB):
  ef=50  R@10 66.9%  p50 1255us   <- page-in included; steady-state ~466-583us
  ef=100 R@10 66.9%  p50 466us
  ef=200 R@10 68.0%  p50 583us
  ef=400 R@10 69.1%  p50 1098us

RECALL BIT-IDENTICAL to Watch S10 / iPhone 16 / iPhone 17 Pro Max (all 66.9/66.9/68.0/69.1).
Claim upgraded: "one engine, four hosts, two architectures (ARM + x86), two OS families
(watchOS/iOS + Linux)". Linux latency is informational only (2-vCPU shared VM vs A18/M4 —
not a claim row).
