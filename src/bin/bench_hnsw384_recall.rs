// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use pams::binary_hnsw_384::{BinaryHNSW384, Hash384, PAP_384_BYTES};
use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::time::Instant;

fn read_hash_bin(path: &str) -> IoResult<Vec<(u64, Hash384)>> {
    let mut f = File::open(path)?;
    let mut count_buf = [0u8; 4];
    f.read_exact(&mut count_buf)?;
    let count = u32::from_le_bytes(count_buf) as usize;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        let mut id_buf = [0u8; 8];
        f.read_exact(&mut id_buf)?;
        let id = u64::from_le_bytes(id_buf);
        let mut pap = [0u8; PAP_384_BYTES];
        f.read_exact(&mut pap)?;
        items.push((id, pap));
    }
    Ok(items)
}

fn main() {
    println!("=== Rust HNSW384 Real-Hash Recall Benchmark ===");

    let db = read_hash_bin("data/paper_hashes_100k_384.bin").expect("DB hash file");
    let queries = read_hash_bin("data/query_hashes_1k_384.bin").expect("Query hash file");

    let n_db = db.len();
    let n_q = queries.len();
    println!("Loaded: {} DB hashes, {} queries", n_db, n_q);

    // Build HNSW
    let mut hnsw = BinaryHNSW384::with_params(16, 200, 50);
    let t0 = Instant::now();
    for &(id, pap) in &db {
        hnsw.insert(id, pap);
    }
    let build = t0.elapsed();
    println!("Build: {:?} ({:.0} docs/s)", build, n_db as f64 / build.as_secs_f64());

    // Query top-1 Hamming
    let mut correct = 0;
    let mut latencies = Vec::with_capacity(n_q);
    for &(gt_id, pap) in &queries {
        let t0 = Instant::now();
        let res = hnsw.search(&pap, 1);
        let lat = t0.elapsed().as_secs_f64();
        latencies.push(lat);

        if !res.is_empty() {
            let found_id = hnsw.node(res[0].1).unwrap().id;
            if found_id == gt_id {
                correct += 1;
            }
        }
    }

    let r1 = correct as f64 / n_q as f64;
    let p50 = latencies[n_q / 2] * 1e6;
    let p95 = latencies[(n_q * 95) / 100] * 1e6;

    println!("\n=== Top-1 Hamming Results ===");
    println!("R@1:  {:.4}", r1);
    println!("P50:  {:.0} us", p50);
    println!("P95:  {:.0} us", p95);
    println!("QPS:  {:.0}", n_q as f64 / latencies.iter().sum::<f64>());

    // Query top-50 Hamming -> measure recall@50
    let mut correct50 = 0;
    for &(gt_id, pap) in &queries {
        let res = hnsw.search(&pap, 50);
        if res.iter().any(|(_, idx)| hnsw.node(*idx).unwrap().id == gt_id) {
            correct50 += 1;
        }
    }
    let r50 = correct50 as f64 / n_q as f64;
    println!("R@50: {:.4}", r50);

    // Save
    let out = format!(
        "{{\"scale\":\"100K\",\"r1_top1_hamming\":{:.4},\"r50_hamming\":{:.4},\"p50_us\":{:.0},\"p95_us\":{:.0},\"build_s\":{:.1},\"M\":16,\"ef_construction\":200,\"ef_search\":50}}",
        r1, r50, p50, p95, build.as_secs_f64()
    );
    std::fs::write("logs/benchmark_hnsw384_recall.json", out).unwrap();
    println!("[+] Saved to logs/benchmark_hnsw384_recall.json");
}
