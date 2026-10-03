// SPDX-License-Identifier: AGPL-3.0-or-later
// bench_identical_1m.rs — TABLE-1 identical-N bench for the Linux leg.
// Same graph file, same 175 queries, same protocol as BenchController.swift (.t1):
// for each ef in {50,100,200,400}, count payload-target (id) containment in the
// graph's top-10, report R@10 and p50. Goal: bit-identical recall to
// iPhone 16 / iPhone 17 Pro Max / Watch Series 10 (66.9/66.9/68.0/69.1%).
// Calls the Rust API directly (the crate builds cdylib/staticlib, so the
// extern "C" shim doesn't link from examples — the FFI layer itself was
// externally reviewed and is a thin passthrough over these same functions).

use pams::binary_hnsw::BinaryHNSW;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bench_identical_1m <graph.bin> <queries.bin>");
        std::process::exit(2);
    }
    let hnsw = BinaryHNSW::load(&args[1]).expect("graph load failed");
    println!("[LINUX-BENCH] graph loaded: {} nodes (expect 1000000)", hnsw.len());

    let raw = std::fs::read(&args[2]).expect("queries file");
    const REC: usize = 8 + 8 + 64;
    assert!(raw.len() % REC == 0, "bad queries file size");
    let nq = raw.len() / REC;
    let mut ids = Vec::with_capacity(nq);
    let mut ids0 = Vec::with_capacity(nq);
    let mut hashes: Vec<[u8; 64]> = Vec::with_capacity(nq);
    for i in 0..nq {
        let off = i * REC;
        ids.push(u64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
        ids0.push(u64::from_le_bytes(raw[off + 8..off + 16].try_into().unwrap()));
        hashes.push(raw[off + 16..off + REC].try_into().unwrap());
    }
    println!("[LINUX-BENCH] {} queries", nq);

    for &ef in &[50usize, 100, 200, 400] {
        hnsw.set_ef_search(ef);
        // warm-up pass (first-touch), same spirit as the device bench
        for i in 0..nq.min(20) {
            let _ = hnsw.search(&hashes[i], 10);
        }
        let (mut hits, mut hits0) = (0usize, 0usize);
        let mut lats: Vec<f64> = Vec::with_capacity(nq);
        for i in 0..nq {
            let t0 = Instant::now();
            let results = hnsw.search(&hashes[i], 10);
            lats.push(t0.elapsed().as_micros() as f64);
            let hit = results.iter().take(10).any(|(_, ni, _)| {
                hnsw.node(*ni).map(|n| n.id) == Some(ids[i])
            });
            let hit0 = results.iter().take(10).any(|(_, ni, _)| {
                hnsw.node(*ni).map(|n| n.id) == Some(ids0[i])
            });
            if hit { hits += 1; }
            if hit0 { hits0 += 1; }
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = if nq % 2 == 1 {
            lats[nq / 2]
        } else {
            (lats[nq / 2 - 1] + lats[nq / 2]) / 2.0
        };
        println!("[LINUX-BENCH] HNSW ef={} member {:.1}% gt0 {:.1}% p50 {:.0}us",
                 ef, 100.0 * hits as f64 / nq as f64, 100.0 * hits0 as f64 / nq as f64, p50);
    }
    println!("[LINUX-BENCH] DONE");
}
