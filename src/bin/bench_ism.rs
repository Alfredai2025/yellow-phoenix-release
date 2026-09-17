// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::time::Instant;
use pams::intelligent_shard_manager::IntelligentShardManager;

fn main() {
    let scales = [1_000_000usize, 10_000_000, 200_000_000];
    
    for &n in &scales {
        println!("\n=== {}M papers ===", n / 1_000_000);
        
        // Generate in Rust — no Python, no FFI copy
        let mut papers = Vec::with_capacity(n);
        let t0 = Instant::now();
        for i in 0..n {
            papers.push((i as u64, [0u8; 32]));
        }
        let gen_time = t0.elapsed();
        println!("  Generate: {:.3}s", gen_time.as_secs_f64());
        
        // Build ISM
        let mut mgr = IntelligentShardManager::new();
        let t1 = Instant::now();
        let rc = mgr.build_parallel(papers);
        let build_time = t1.elapsed();
        println!("  Build:    {:.3}s | rc={:?}", build_time.as_secs_f64(), rc);
        
        // Query benchmark
        let queries = 100_000usize;
        let t2 = Instant::now();
        for q in 0..queries {
            let _ = mgr.query_parallel(q as u64 % n as u64);
        }
        let query_time = t2.elapsed();
        let avg_us = query_time.as_secs_f64() / queries as f64 * 1e6;
        println!("  Query:    {} calls | avg={:.2}µs | total={:.3}s", queries, avg_us, query_time.as_secs_f64());
        
        // Status
        println!("  Status:   {}", mgr.status_json());
    }
}
