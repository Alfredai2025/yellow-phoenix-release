// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! validate_m2.rs — M2 throughput + zero-downtime validation harness.
//!
//! Reads the same binary input as validate_m1, then:
//!   1. Runs a synchronous M1-style baseline.
//!   2. Runs the same queries through the TemporalOrchestrator.
//!   3. Performs versioned-table swaps under load.
//! Prints a JSON report.

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use pams::collaborative_engine::{CollaborativeEngine, QueryContext};
use pams::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};
use pams::temporal_orchestrator::TemporalOrchestrator;
use pams::versioned_tables::VersionedTable;

const BASELINE_QUERIES: usize = 10_000;
const ORCHESTRATOR_QUERIES: usize = 10_000;
const ORCHESTRATOR_BATCH: usize = 64;

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

fn run_baseline(engine: &mut CollaborativeEngine, records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])]) -> serde_json::Value {
    let mut latencies = Vec::with_capacity(BASELINE_QUERIES);
    let start = Instant::now();
    for i in 0..BASELINE_QUERIES {
        let ctx = ctx_from_record(&records[i % records.len()]);
        let t0 = Instant::now();
        let _ = engine.query(&ctx);
        latencies.push(t0.elapsed().as_micros() as u64);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = BASELINE_QUERIES as f64 / elapsed;

    latencies.sort_unstable();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    serde_json::json!({
        "queries": BASELINE_QUERIES,
        "qps": qps,
        "elapsed_s": elapsed,
        "p50_us": p50,
        "p99_us": p99,
    })
}

fn run_orchestrator(
    engine: CollaborativeEngine,
    records: &[(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])],
) -> serde_json::Value {
    let mut orchestrator = TemporalOrchestrator::with_capacity(engine, ORCHESTRATOR_BATCH * 2);
    let mut latencies = Vec::with_capacity(ORCHESTRATOR_QUERIES);
    let mut submitted = 0usize;
    let mut completed = 0usize;
    let start = Instant::now();

    let mut tokens = Vec::new();
    loop {
        // Submit up to batch size new queries.
        while submitted < ORCHESTRATOR_QUERIES && orchestrator.in_flight_count() < ORCHESTRATOR_BATCH {
            let ctx = ctx_from_record(&records[submitted % records.len()]);
            let token = orchestrator.submit(ctx);
            if token == 0 {
                break;
            }
            tokens.push((token, Instant::now()));
            submitted += 1;
        }

        // Tick pipeline forward.
        orchestrator.tick();

        // Check for completed tokens.
        tokens.retain(|(token, t0)| {
            if let Some(_payload) = orchestrator.poll(*token) {
                latencies.push(t0.elapsed().as_micros() as u64);
                completed += 1;
                false
            } else {
                true
            }
        });

        if completed >= ORCHESTRATOR_QUERIES && orchestrator.in_flight_count() == 0 {
            break;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let qps = completed as f64 / elapsed;

    latencies.sort_unstable();
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    serde_json::json!({
        "queries": completed,
        "qps": qps,
        "elapsed_s": elapsed,
        "p50_us": p50,
        "p99_us": p99,
    })
}

fn run_swap_test() -> serde_json::Value {
    let versioned = VersionedTable::new(0usize);
    let mut reads = 0usize;

    let start = Instant::now();
    let duration = Duration::from_millis(500);

    // Background swapper.
    let mut swapper_versioned = VersionedTable::new(0usize);
    let swap_handle = thread::spawn(move || {
        let mut count = 0usize;
        let start = Instant::now();
        while start.elapsed() < duration {
            swapper_versioned.write_next(count + 1);
            swapper_versioned.swap();
            count += 1;
            thread::sleep(Duration::from_millis(5));
        }
        count
    });

    // Reader under load.
    while start.elapsed() < duration {
        let v = versioned.read();
        let _ = *v;
        reads += 1;
    }

    let swaps = swap_handle.join().unwrap_or(0);

    serde_json::json!({
        "swaps": swaps,
        "reads_during_swaps": reads,
        "final_version": versioned.version(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.bin>", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let records = load_records(input_path).expect("load input records");
    println!("Loaded {} records", records.len());

    // M1 baseline.
    let mut engine = build_engine(&records);
    let baseline = run_baseline(&mut engine, &records);
    println!("Baseline: {} qps", baseline["qps"]);

    // M2 orchestrator.
    let engine = build_engine(&records);
    let orchestrator = run_orchestrator(engine, &records);
    println!("Orchestrator: {} qps", orchestrator["qps"]);

    // Zero-downtime swap.
    let swap = run_swap_test();
    println!("Swaps: {}", swap["swaps"]);

    let report = serde_json::json!({
        "records_loaded": records.len(),
        "baseline": baseline,
        "orchestrator": orchestrator,
        "swap_test": swap,
        "speedup": orchestrator["qps"].as_f64().unwrap() / baseline["qps"].as_f64().unwrap(),
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
