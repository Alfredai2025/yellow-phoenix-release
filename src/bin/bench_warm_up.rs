// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Measure cold-start warm-up: cold first query vs post-warm-up query.
//! Usage: cargo run --release --bin bench_warm_up -- <v5.bin>
use pams::binary_hnsw::BinaryHNSW;
use std::env;
use std::time::Instant;

fn main() {
    let path = env::args().nth(1).expect("usage: bench_warm_up <v5.bin>");

    let t = Instant::now();
    let mut idx = BinaryHNSW::load(&path).expect("load failed");
    println!("load:            {:?}", t.elapsed());
    println!("nodes:           {}", idx.len());

    let probe = idx.node(0).unwrap().hash;

    // Cold: drop all clean mmap pages, then time queries.
    idx.advise_dontneed();
    let t = Instant::now();
    let _ = idx.search(&probe, 10);
    println!("cold first query:{:?}", t.elapsed());

    idx.advise_dontneed();
    let t = Instant::now();
    for i in (0..idx.len()).step_by((idx.len() / 10).max(1)) {
        let h = idx.node(i as u32).unwrap().hash;
        let _ = idx.search(&h, 10);
    }
    let cold10 = t.elapsed();
    println!("10 cold queries: {:?}", cold10);

    // Warm-up from a cold state.
    idx.advise_dontneed();
    let t = Instant::now();
    idx.warm_up(50, 400);
    let warm_t = t.elapsed();
    println!("warm_up(50,400): {:?}", warm_t);

    let t = Instant::now();
    let _ = idx.search(&probe, 10);
    println!("warm first query:{:?}", t.elapsed());

    let t = Instant::now();
    for i in (0..idx.len()).step_by((idx.len() / 10).max(1)) {
        let h = idx.node(i as u32).unwrap().hash;
        let _ = idx.search(&h, 10);
    }
    let warm10 = t.elapsed();
    println!("10 warm queries: {:?}", warm10);
}
