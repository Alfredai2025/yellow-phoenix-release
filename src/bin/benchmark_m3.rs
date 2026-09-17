// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! benchmark_m3.rs — M3.1 speed validation harness.
//!
//! Compares the M1-style full hash path against the M3 fast bucket path
//! on the same 15K-record input used by validate_m1.rs.

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::{Path, PathBuf};
use std::time::Instant;

use pams::collaborative_engine::{CollaborativeEngine, QueryContext};
use pams::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};

const QUERIES: usize = 10_000;

fn load_records(path: &Path) -> IoResult<Vec<(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let mut cursor = 0usize;
    let count = u32::from_le_bytes([buf[cursor], buf[cursor + 1], buf[cursor + 2], buf[cursor + 3]]) as usize;
    cursor += 4;

    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let mut id_bytes = [0u8; 8];
        id_bytes.copy_from_slice(&buf[cursor..cursor + 8]);
        let id = u64::from_le_bytes(id_bytes);
        cursor += 8;

        let mut pap_128 = [0u8; PAP_128_BYTES];
        pap_128.copy_from_slice(&buf[cursor..cursor + PAP_128_BYTES]);
        cursor += PAP_128_BYTES;

        let mut pap_512 = [0u8; PAP_512_BYTES];
        pap_512.copy_from_slice(&buf[cursor..cursor + PAP_512_BYTES]);
        cursor += PAP_512_BYTES;

        records.push((id, pap_128, pap_512));
    }

    Ok(records)
}

fn build_engine(records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])]) -> CollaborativeEngine {
    let mut mesh = HybridMesh::new(records.len().next_power_of_two(), records.len().next_power_of_two());
    for (id, pap_128, pap_512) in records {
        mesh.insert_dual(*id, pap_128, pap_512);
    }
    mesh.coarse.build_edges(8);
    mesh.fine.build_edges(8);
    CollaborativeEngine::new(mesh)
}

fn ctx_from_record(record: &(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])) -> QueryContext {
    QueryContext {
        pap_128: record.1,
        pap_512: record.2,
        features: features_from_pap(&record.1),
        expected_id: Some(record.0),
    }
}

fn features_from_pap(pap_128: &[u8; PAP_128_BYTES]) -> [f32; 8] {
    let mut f = [0.0f32; 8];
    for i in 0..8 {
        f[i] = (pap_128[i * 2] as f32) / 255.0;
    }
    f
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Peak resident set size in KiB.  macOS reports bytes, Linux reports KiB.
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

fn benchmark(
    engine: &mut CollaborativeEngine,
    records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])],
    lazy: bool,
) -> serde_json::Value {
    let mut latencies = Vec::with_capacity(QUERIES);
    let mut correct = 0usize;
    let start = Instant::now();

    for i in 0..QUERIES {
        let ctx = ctx_from_record(&records[i % records.len()]);
        let t0 = Instant::now();
        let result = if lazy {
            engine.query_lazy(&ctx)
        } else {
            engine.query(&ctx)
        };
        // Report fractional microseconds so sub-microsecond lazy paths don't show as 0.00.
        latencies.push(t0.elapsed().as_nanos() as f64 / 1000.0);

        if let Some((top_id, _)) = result.top1 {
            if top_id == records[i % records.len()].0 {
                correct += 1;
            }
        }
    }

    let elapsed_s = start.elapsed().as_secs_f64();
    latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    serde_json::json!({
        "queries": QUERIES,
        "correct": correct,
        "r1": correct as f64 / QUERIES as f64,
        "p50_us": percentile(&latencies, 0.50),
        "p99_us": percentile(&latencies, 0.99),
        "mean_us": latencies.iter().sum::<f64>() / latencies.len() as f64,
        "qps": QUERIES as f64 / elapsed_s,
    })
}

/// M3.5: multi-threaded benchmark that uses the batch-fusion queue.
///
/// Queries are submitted in asynchronous bursts so the 1ms worker window can
/// actually accumulate multiple same-bucket lookups and fuse them into a single
/// mesh call.
fn benchmark_batch(
    engine: &CollaborativeEngine,
    records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])],
) -> serde_json::Value {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let queries_per_thread = QUERIES / n_threads;
    const BURST: usize = 64;

    let start = Instant::now();
    let (latencies, correct) = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for t in 0..n_threads {
            handles.push(s.spawn(move || {
                let mut local_latencies = Vec::with_capacity(queries_per_thread);
                let mut local_correct = 0usize;
                for chunk_start in (0..queries_per_thread).step_by(BURST) {
                    let chunk_end = (chunk_start + BURST).min(queries_per_thread);
                    let chunk_len = chunk_end - chunk_start;

                    let mut ctxs = Vec::with_capacity(chunk_len);
                    let mut expected = Vec::with_capacity(chunk_len);
                    for i in chunk_start..chunk_end {
                        let idx = t * queries_per_thread + i;
                        ctxs.push(ctx_from_record(&records[idx % records.len()]));
                        expected.push(records[idx % records.len()].0);
                    }

                    let t0 = Instant::now();
                    let results = engine.query_batch_many(&ctxs);
                    let chunk_us = t0.elapsed().as_nanos() as f64 / 1000.0;
                    let per_query_us = chunk_us / chunk_len as f64;

                    for (result, exp) in results.iter().zip(expected.iter()) {
                        local_latencies.push(per_query_us);
                        if let Some((top_id, _)) = result.top1 {
                            if top_id == *exp {
                                local_correct += 1;
                            }
                        }
                    }
                }
                (local_latencies, local_correct)
            }));
        }

        let mut all_latencies = Vec::with_capacity(QUERIES);
        let mut total_correct = 0usize;
        for h in handles {
            let (lat, corr) = h.join().unwrap();
            all_latencies.extend(lat);
            total_correct += corr;
        }
        all_latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        (all_latencies, total_correct)
    });

    let elapsed_s = start.elapsed().as_secs_f64();

    serde_json::json!({
        "queries": QUERIES,
        "correct": correct,
        "r1": correct as f64 / QUERIES as f64,
        "p50_us": percentile(&latencies, 0.50),
        "p99_us": percentile(&latencies, 0.99),
        "mean_us": latencies.iter().sum::<f64>() / latencies.len() as f64,
        "qps": QUERIES as f64 / elapsed_s,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut batch_mode = false;
    let mut scale = None;
    let mut input_arg = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--batch" => batch_mode = true,
            "--scale" => {
                i += 1;
                scale = args.get(i).cloned();
            }
            a => input_arg = Some(a.to_string()),
        }
        i += 1;
    }

    let input_path: PathBuf = if let Some(path) = input_arg {
        PathBuf::from(path)
    } else {
        match scale.as_deref() {
            Some("1m") => PathBuf::from("data/1m_papers.bin"),
            Some("100k") => PathBuf::from("data/100k_papers.bin"),
            _ => PathBuf::from("data/validate_m1_input.bin"),
        }
    };

    if !input_path.exists() {
        eprintln!("Input file not found: {}", input_path.display());
        eprintln!("Generate it with: python3 scripts/generate_1m_synthetic.py");
        std::process::exit(1);
    }

    let records = load_records(&input_path).expect("load input records");
    println!("Loaded {} records from {}", records.len(), input_path.display());

    // M1-style baseline: low confidence, full feature chain (query).
    let build_t0 = Instant::now();
    let mut baseline = build_engine(&records);
    let build_elapsed = build_t0.elapsed().as_secs_f64();
    baseline.router().force_baseline();
    let base_report = benchmark(&mut baseline, &records, false);
    println!("Baseline: P50={:.2}us, QPS={:.0}", base_report["p50_us"].as_f64().unwrap(), base_report["qps"].as_f64().unwrap());

    // Drop the baseline mesh before building the M3 mesh so peak RSS reflects
    // a single engine instance.
    drop(baseline);

    // M3 fast path: high confidence triggers bucket-only lookup via query_lazy.
    let mut m3 = build_engine(&records);
    m3.router().force_fast_path();

    let (m3_report, fusion_rate) = if batch_mode {
        m3.enable_batch_fusion(1);
        let report = benchmark_batch(&m3, &records);
        let rate = m3.batch_fusion_rate();
        println!("M3 batch: P50={:.2}us, QPS={:.0}, fusion_rate={:.2}", report["p50_us"].as_f64().unwrap(), report["qps"].as_f64().unwrap(), rate);
        (report, Some(rate))
    } else {
        let report = benchmark(&mut m3, &records, true);
        println!("M3:      P50={:.2}us, QPS={:.0}", report["p50_us"].as_f64().unwrap(), report["qps"].as_f64().unwrap());
        (report, None)
    };

    // Cap displayed speedup to avoid Infinity/null in JSON when M3 P50 rounds to 0.
    let base_p50 = base_report["p50_us"].as_f64().unwrap();
    let m3_p50 = m3_report["p50_us"].as_f64().unwrap();
    let speedup = if m3_p50 < 0.001 {
        1000.0
    } else {
        (base_p50 / m3_p50).min(1000.0)
    };
    let peak_rss = peak_rss_kb();
    let mut report = serde_json::json!({
        "input": input_path.display().to_string(),
        "records_loaded": records.len(),
        "queries": QUERIES,
        "build_time_s": build_elapsed,
        "peak_rss_kb": peak_rss,
        "baseline": base_report,
        "m3": m3_report,
        "speedup_p50": speedup,
        "batch_mode": batch_mode,
    });
    if let Some(rate) = fusion_rate {
        report["fusion_rate"] = serde_json::json!(rate);
    }

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
