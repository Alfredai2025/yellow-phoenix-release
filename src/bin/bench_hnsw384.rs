// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use pams::binary_hnsw_384::{BinaryHNSW384, PAP_384_BYTES};
use rand::Rng;
use std::time::Instant;

fn random_pap(rng: &mut impl Rng) -> [u8; PAP_384_BYTES] {
    let mut p = [0u8; PAP_384_BYTES];
    rng.fill_bytes(&mut p);
    p
}

fn main() {
    let scales = [100_000, 500_000, 1_000_000];
    let mut rng = rand::thread_rng();

    println!("=== Rust Native HNSW384 Benchmark ===");
    println!("Params: M=16, ef_construction=200, ef_search=50");
    println!("Distance: 6×u64 XOR+POPCNT Hamming (384-bit)");
    println!("");

    for &n in &scales {
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            items.push((i as u64, random_pap(&mut rng)));
        }

        // Build
        let mut hnsw = BinaryHNSW384::with_params(16, 200, 50);
        let t0 = Instant::now();
        hnsw.insert_batch(&items);
        let build = t0.elapsed();

        let ins_per_sec = n as f64 / build.as_secs_f64();
        let ms_per_doc = build.as_secs_f64() * 1000.0 / n as f64;

        println!("Scale: {:>10} docs", n);
        println!("  Build:      {:?} ({:.0} docs/s, {:.3} ms/doc)", build, ins_per_sec, ms_per_doc);

        // Query latency
        let mut latencies = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let q = random_pap(&mut rng);
            let t0 = Instant::now();
            let _ = hnsw.search(&q, 10);
            latencies.push(t0.elapsed().as_secs_f64());
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = latencies[latencies.len() / 2] * 1e6;
        let p95 = latencies[(latencies.len() as f64 * 0.95) as usize] * 1e6;
        let qps = 1000.0 / latencies.iter().sum::<f64>();

        println!("  Query P50:  {:.0} µs", p50);
        println!("  Query P95:  {:.0} µs", p95);
        println!("  QPS:        {:.0}", qps);
        println!("  Nodes:      {}", hnsw.len());
        println!("  Max level:  {}", hnsw.max_level);
        println!("");
    }
}
