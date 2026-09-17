// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use pams::binary_hnsw_384::BinaryHNSW384;
use rand::Rng;
use std::time::Instant;

fn random_pap_384(rng: &mut impl Rng) -> [u8; 48] {
    let mut p = [0u8; 48];
    rng.fill(&mut p[..]);
    p
}

fn main() {
    let scales = [100_000, 500_000, 1_000_000];
    let mut rng = rand::rng();

    println!("=== BinaryHNSW384 Batch Insert Benchmark ===\n");

    for &n in &scales {
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            items.push((i as u64, random_pap_384(&mut rng)));
        }

        // Phase 3 sweet spot: M=16, efConstruction=200, efSearch=50
        let mut hnsw = BinaryHNSW384::with_params(16, 200, 50);

        let t0 = Instant::now();
        hnsw.insert_batch(&items);
        let elapsed = t0.elapsed();

        let ins_per_sec = n as f64 / elapsed.as_secs_f64();
        let ms_per_doc = elapsed.as_secs_f64() * 1000.0 / n as f64;

        println!("Scale: {:>10} docs", n);
        println!("  Time:       {:?}", elapsed);
        println!("  Throughput: {:>10.0} docs/s", ins_per_sec);
        println!("  Per doc:    {:>10.3} ms", ms_per_doc);
        println!("  Nodes:      {}", hnsw.len());
        println!();

        // Quick smoke test: query 100 random docs
        let t0 = Instant::now();
        for _ in 0..100 {
            let q = random_pap_384(&mut rng);
            let _ = hnsw.search(&q, 10);
        }
        let q_time = t0.elapsed();
        println!("  100 queries (k=10): {:?}  ({:.0} qps)", q_time, 100.0 / q_time.as_secs_f64());
        println!("---");
    }
}
