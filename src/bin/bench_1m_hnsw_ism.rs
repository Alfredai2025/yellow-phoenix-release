// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Benchmark 1M multi-edge HNSW vs ISM flat exact search.
//! Usage: cargo run --release --bin bench_1m_hnsw_ism -- <hnsw.bin> <ism.yism.bin>

use pams::binary_hnsw::{BinaryHNSW, HASH512_BITS};
use pams::ism_flat::IsmFlatIndex;
use rand::Rng;
use std::env;
use std::time::Instant;

fn perturb_hash(hash: &[u8; 64], bits: usize, rng: &mut rand::rngs::ThreadRng) -> [u8; 64] {
    let mut out = *hash;
    for _ in 0..bits {
        let bit = rng.random_range(0..HASH512_BITS as usize);
        out[bit / 8] ^= 1 << (bit % 8);
    }
    out
}

fn percentile(sorted_us: &[f64], p: f64) -> f64 {
    let idx = ((sorted_us.len() - 1) as f64 * p) as usize;
    sorted_us[idx]
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <hnsw.bin> <ism.yism.bin>", args[0]);
        std::process::exit(1);
    }
    let hnsw_path = &args[1];
    let ism_path = &args[2];

    eprintln!("Loading HNSW from {}...", hnsw_path);
    let hnsw = BinaryHNSW::load(hnsw_path).expect("load HNSW");
    eprintln!("HNSW nodes: {}", hnsw.len());

    eprintln!("Loading ISM from {}...", ism_path);
    let ism = IsmFlatIndex::load(ism_path).expect("load ISM");
    eprintln!("ISM records: {}", ism.count());

    let n_queries = 100;
    let top_k = 10;
    let mut rng = rand::rng();

    // Build queries by perturbing random existing hashes.
    let mut queries: Vec<[u8; 64]> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let idx = rng.random_range(0..hnsw.len());
        let node = hnsw.node(idx as u32).expect("valid node");
        queries.push(perturb_hash(&node.hash, 50, &mut rng));
    }

    // HNSW benchmark
    let mut hnsw_times = Vec::with_capacity(n_queries);
    let mut hnsw_r1_hits = 0usize;
    let mut hnsw_r5_hits = 0usize;
    let mut hnsw_r10_hits = 0usize;

    // Warmup
    for q in queries.iter().take(5) {
        let _ = hnsw.search(q, top_k);
    }

    for q in &queries {
        let t0 = Instant::now();
        let hnsw_res = hnsw.search(q, top_k);
        let elapsed = t0.elapsed().as_secs_f64() * 1e6;
        hnsw_times.push(elapsed);

        let exact_res = ism.search(q, top_k);
        let gt_ids: Vec<u64> = exact_res.iter().map(|(id, _)| *id).collect();
        let hnsw_ids: Vec<u64> = hnsw_res.iter().map(|(_, idx, _)| hnsw.node(*idx).map(|n| n.id).unwrap_or(0)).collect();

        if let Some(gt) = gt_ids.first() {
            if hnsw_ids.iter().take(1).any(|id| id == gt) { hnsw_r1_hits += 1; }
            if hnsw_ids.iter().take(5).any(|id| id == gt) { hnsw_r5_hits += 1; }
            if hnsw_ids.iter().take(10).any(|id| id == gt) { hnsw_r10_hits += 1; }
        }
    }

    hnsw_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hnsw_p50 = percentile(&hnsw_times, 0.50);
    let hnsw_p95 = percentile(&hnsw_times, 0.95);
    let hnsw_avg = hnsw_times.iter().sum::<f64>() / hnsw_times.len() as f64;

    // ISM flat benchmark (same queries, exact)
    let mut ism_times = Vec::with_capacity(n_queries);
    for q in &queries {
        let t0 = Instant::now();
        let _ = ism.search(q, top_k);
        let elapsed = t0.elapsed().as_secs_f64() * 1e6;
        ism_times.push(elapsed);
    }

    ism_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ism_p50 = percentile(&ism_times, 0.50);
    let ism_p95 = percentile(&ism_times, 0.95);
    let ism_avg = ism_times.iter().sum::<f64>() / ism_times.len() as f64;

    println!("\n=== 1M multi-edge HNSW vs ISM flat exact ({} queries, top-{}) ===", n_queries, top_k);
    println!("HNSW P50: {:.3} µs | P95: {:.3} µs | avg: {:.3} µs", hnsw_p50, hnsw_p95, hnsw_avg);
    println!("ISM  P50: {:.3} µs | P95: {:.3} µs | avg: {:.3} µs", ism_p50, ism_p95, ism_avg);
    println!("Speedup (ISM P50 / HNSW P50): {:.1}×", ism_p50 / hnsw_p50);
    println!("Recall@1: {:.2}%  Recall@5: {:.2}%  Recall@10: {:.2}%",
        100.0 * hnsw_r1_hits as f64 / n_queries as f64,
        100.0 * hnsw_r5_hits as f64 / n_queries as f64,
        100.0 * hnsw_r10_hits as f64 / n_queries as f64);
}
