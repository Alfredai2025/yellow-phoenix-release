// SPDX-License-Identifier: AGPL-3.0-or-later
// bench_flat_identical.rs — identical-N FLAT SCAN bench for the Linux leg.
// Same flat file, same 695 queries, same protocol as BenchController.swift
// Tier C: payload-target (id) containment in the exhaustive top-10.
// Goal: match Watch S10 / iPhone (86.5% containment, R@10).

use pams::ism_flat::IsmFlatIndex;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: bench_flat_identical <flat.bin> <queries.bin>");
        std::process::exit(2);
    }
    let idx = IsmFlatIndex::load(&args[1]).expect("flat load failed");
    println!("[LINUX-FLAT] loaded: {} records (expect 3900000)", idx.count());

    let raw = std::fs::read(&args[2]).expect("queries file");
    const REC: usize = 8 + 64;
    assert!(raw.len() % REC == 0);
    let nq = raw.len() / REC;
    let mut ids = Vec::with_capacity(nq);
    let mut hashes: Vec<[u8; 64]> = Vec::with_capacity(nq);
    for i in 0..nq {
        let off = i * REC;
        ids.push(u64::from_le_bytes(raw[off..off + 8].try_into().unwrap()));
        hashes.push(raw[off + 8..off + REC].try_into().unwrap());
    }
    println!("[LINUX-FLAT] {} queries", nq);

    // warm-up (page-in)
    for i in 0..nq.min(10) {
        let _ = idx.search(&hashes[i], 10);
    }
    let mut hits = 0usize;
    let mut lats: Vec<f64> = Vec::with_capacity(nq);
    for i in 0..nq {
        let t0 = Instant::now();
        let results = idx.search(&hashes[i], 10);
        lats.push(t0.elapsed().as_micros() as f64);
        if results.iter().take(10).any(|(id, _)| *id == ids[i]) {
            hits += 1;
        }
    }
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = if nq % 2 == 1 { lats[nq / 2] } else { (lats[nq / 2 - 1] + lats[nq / 2]) / 2.0 };
    println!("[LINUX-FLAT] FLAT R@10 containment {:.1}% p50 {:.0}us",
             100.0 * hits as f64 / nq as f64, p50);
    println!("[LINUX-FLAT] DONE");
}
