// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! benchmark_sharded.rs — benchmark the memory-optimized sharded mesh.
//!
//! Loads the 1M synthetic dataset and either:
//!   * (default) builds an OptimizedShard directly, or
//!   * (--compare) builds a CollaborativeEngine, measures non-sharded queries,
//!     then enables sharding and measures the sharded path.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pams::collaborative_engine::{CollaborativeEngine, QueryContext};
use pams::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};
use pams::sharded_mesh::OptimizedShard;
use pams::spectral_coords::SpectralCoords;

const QUERIES: usize = 10_000;
const HOT_RATIO: f64 = 0.10;

fn load_records(path: &Path) -> Vec<(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])> {
    let mut file = File::open(path).expect("open input");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read input");

    let mut cursor = 0usize;
    let count = u32::from_le_bytes([buf[cursor], buf[cursor + 1], buf[cursor + 2], buf[cursor + 3]]) as usize;
    cursor += 4;

    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let id = u64::from_le_bytes([
            buf[cursor], buf[cursor + 1], buf[cursor + 2], buf[cursor + 3],
            buf[cursor + 4], buf[cursor + 5], buf[cursor + 6], buf[cursor + 7],
        ]);
        cursor += 8;

        let mut pap_128 = [0u8; PAP_128_BYTES];
        pap_128.copy_from_slice(&buf[cursor..cursor + PAP_128_BYTES]);
        cursor += PAP_128_BYTES;

        let mut pap_512 = [0u8; PAP_512_BYTES];
        pap_512.copy_from_slice(&buf[cursor..cursor + PAP_512_BYTES]);
        cursor += PAP_512_BYTES;

        records.push((id, pap_128, pap_512));
    }
    records
}

#[cfg(target_os = "macos")]
fn peak_rss_kb() -> i64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
        usage.ru_maxrss / 1024
    } else {
        -1
    }
}

#[cfg(target_os = "linux")]
fn peak_rss_kb() -> i64 {
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage>() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
        usage.ru_maxrss
    } else {
        -1
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn peak_rss_kb() -> i64 {
    -1
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run_standalone(records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])]) {
    let cold_path = PathBuf::from(".tmp/sharded_cold.bin");
    std::fs::create_dir_all(".tmp").unwrap();
    let _ = std::fs::remove_file(&cold_path);

    let build_t0 = Instant::now();
    let shard = OptimizedShard::build(
        records.iter().map(|(id, _, pap_512)| (*id, *pap_512, SpectralCoords::from_hash(pap_512))),
        HOT_RATIO,
        &cold_path,
    )
    .expect("build optimized shard");
    let build_s = build_t0.elapsed().as_secs_f64();

    let paper_size = pams::sharded_mesh::CompressedPaper::COMPRESSED_SIZE;
    let mem_bytes = shard.memory_bytes();

    println!(
        "Built shard: hot {} / cold {} records in {:.2}s",
        shard.hot_len(),
        shard.cold_len(),
        build_s
    );

    let fault_t0 = Instant::now();
    let touched = shard.pre_fault_cold_pages();
    println!(
        "Pre-faulted {} cold pages in {:.2}ms",
        touched,
        fault_t0.elapsed().as_secs_f64() * 1000.0
    );

    let mut latencies = Vec::with_capacity(QUERIES);
    let mut correct = 0usize;
    let start = Instant::now();
    for i in 0..QUERIES {
        let (expected_id, _, pap_512) = &records[i % records.len()];
        let t0 = Instant::now();
        let result = shard.query(pap_512);
        latencies.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        if let Some((id, _)) = result {
            if id == *expected_id {
                correct += 1;
            }
        }
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let peak_rss = peak_rss_kb();
    let correct_ratio = correct as f64 / QUERIES as f64;

    println!("\n=== Sharded Mesh Benchmark ===");
    println!("Records loaded: {}", records.len());
    println!("Hot ratio: {}%", HOT_RATIO * 100.0);
    println!("Queries: {}", QUERIES);
    println!("R@1: {:.4}", correct_ratio);
    println!("P50 latency: {:.3} µs", percentile(&latencies, 0.50));
    println!("P99 latency: {:.3} µs", percentile(&latencies, 0.99));
    println!("Mean latency: {:.3} µs", latencies.iter().sum::<f64>() / latencies.len() as f64);
    println!("QPS: {:.0}", QUERIES as f64 / elapsed_s);
    println!("Peak RSS: {} KB ({:.2} GB)", peak_rss, peak_rss as f64 / (1024.0 * 1024.0));
    println!("Estimated hot RAM: {} KB ({:.2} GB)", mem_bytes / 1024, mem_bytes as f64 / (1024.0 * 1024.0 * 1024.0));

    let report = serde_json::json!({
        "records": records.len(),
        "hot_ratio": HOT_RATIO,
        "queries": QUERIES,
        "r1": correct_ratio,
        "p50_us": percentile(&latencies, 0.50),
        "p99_us": percentile(&latencies, 0.99),
        "mean_us": latencies.iter().sum::<f64>() / latencies.len() as f64,
        "qps": QUERIES as f64 / elapsed_s,
        "peak_rss_kb": peak_rss,
        "hot_ram_kb": mem_bytes / 1024,
        "compressed_per_paper_bytes": paper_size,
    });
    println!("\n{}", serde_json::to_string_pretty(&report).unwrap());
}

fn ctx_for(record: &(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])) -> QueryContext {
    QueryContext {
        pap_128: record.1,
        pap_512: record.2,
        features: [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        expected_id: Some(record.0),
    }
}

fn measure_engine<F>(name: &str, engine: &mut CollaborativeEngine, records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])], query_fn: F)
where
    F: Fn(&mut CollaborativeEngine, &QueryContext) -> pams::collaborative_engine::ConsensusResult,
{
    let mut latencies = Vec::with_capacity(QUERIES);
    let mut correct = 0usize;
    let start = Instant::now();
    for i in 0..QUERIES {
        let expected_id = records[i % records.len()].0;
        let ctx = ctx_for(&records[i % records.len()]);
        let t0 = Instant::now();
        let result = query_fn(engine, &ctx);
        latencies.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        if result.top1.map(|(id, _)| id == expected_id).unwrap_or(false) {
            correct += 1;
        }
    }
    let elapsed_s = start.elapsed().as_secs_f64();
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let peak_rss = peak_rss_kb();

    println!("\n=== {} ===", name);
    println!("R@1: {:.4}", correct as f64 / QUERIES as f64);
    println!("P50 latency: {:.3} µs", percentile(&latencies, 0.50));
    println!("P99 latency: {:.3} µs", percentile(&latencies, 0.99));
    println!("Mean latency: {:.3} µs", latencies.iter().sum::<f64>() / latencies.len() as f64);
    println!("QPS: {:.0}", QUERIES as f64 / elapsed_s);
    println!("Peak RSS: {} KB ({:.2} GB)", peak_rss, peak_rss as f64 / (1024.0 * 1024.0));
}

fn run_compare(records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])]) {
    let mut mesh = HybridMesh::new(records.len().next_power_of_two(), records.len().next_power_of_two());
    let build_t0 = Instant::now();
    for (id, pap_128, pap_512) in records {
        mesh.insert_dual(*id, pap_128, pap_512);
    }
    mesh.build_edges(8);
    println!("Built HybridMesh in {:.2}s", build_t0.elapsed().as_secs_f64());

    let mut engine = CollaborativeEngine::new(mesh);
    measure_engine("CollaborativeEngine (non-sharded)", &mut engine, records, |e, ctx| e.query_lazy(ctx));

    let cold_path = PathBuf::from(".tmp/sharded_cold_engine.bin");
    std::fs::create_dir_all(".tmp").unwrap();
    let _ = std::fs::remove_file(&cold_path);

    let shard_t0 = Instant::now();
    engine.enable_sharding(&cold_path, HOT_RATIO, true).expect("enable sharding");
    engine.set_sharding_threshold(1); // force sharded path for this benchmark
    println!("Enabled sharding in {:.2}s", shard_t0.elapsed().as_secs_f64());

    measure_engine("CollaborativeEngine (sharded)", &mut engine, records, |e, ctx| e.query(ctx));

    if let Some(ref shard) = engine.sharded_mesh {
        println!("\nSharded hot records: {}, cold records: {}", shard.hot_len(), shard.cold_len());
    }
}

fn main() {
    let input = PathBuf::from("data/1m_papers.bin");
    if !input.exists() {
        eprintln!("Input not found: {}", input.display());
        eprintln!("Generate it with: python3 scripts/generate_1m_synthetic.py");
        std::process::exit(1);
    }

    let records = load_records(&input);
    println!("Loaded {} records from {}", records.len(), input.display());

    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--compare" {
        run_compare(&records);
    } else {
        run_standalone(&records);
    }
}
