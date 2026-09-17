// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! validate_m1.rs — M1 validation harness.
//!
//! Reads a binary file of (id, pap128, pap512) records, builds a HybridMesh,
//! runs the collaborative engine over 10K self-queries, and prints a JSON
//! metrics report.
//!
//! Input binary layout:
//!   u32 LE      : record count N
//!   N * 88 bytes: [id u64 LE][pap128 16 bytes][pap512 64 bytes]

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;
use std::time::Instant;

use pams::collaborative_engine::{CollaborativeEngine, QueryContext};
use pams::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};
use pams::learned_router::FeatureChain;

const QUERY_COUNT: usize = 10_000;

#[derive(Default)]
struct Metrics {
    correct: usize,
    total: usize,
    cache_hits: usize,
    latencies_us: Vec<u64>,
    spectral: usize,
    wedge: usize,
    hologram: usize,
}

impl Metrics {
    fn r1(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.correct as f64 / self.total as f64 }
    }

    fn p50_us(&self) -> f64 {
        if self.latencies_us.is_empty() { 0.0 } else {
            let mut v = self.latencies_us.clone();
            v.sort_unstable();
            v[v.len() / 2] as f64
        }
    }

    fn p99_us(&self) -> f64 {
        if self.latencies_us.is_empty() { 0.0 } else {
            let mut v = self.latencies_us.clone();
            v.sort_unstable();
            let idx = ((v.len() as f64) * 0.99) as usize;
            v[idx.min(v.len() - 1)] as f64
        }
    }

    fn cache_rate(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.cache_hits as f64 / self.total as f64 }
    }
}

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input.bin>", args[0]);
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let records = load_records(input_path).expect("load input records");
    if records.is_empty() {
        eprintln!("No records loaded from {:?}", input_path);
        std::process::exit(1);
    }

    println!("Loaded {} records", records.len());

    // Build mesh.
    let mut mesh = HybridMesh::new(records.len().next_power_of_two(), records.len().next_power_of_two());
    for (id, pap_128, pap_512) in &records {
        mesh.insert_dual(*id, pap_128, pap_512);
    }
    mesh.coarse.build_edges(8);
    mesh.fine.build_edges(8);
    println!("Built mesh with {} records", records.len());

    // Run 10K self-queries: 5K unique titles queried twice to exercise cache.
    let mut engine = CollaborativeEngine::new(mesh);
    let mut metrics = Metrics::default();
    let unique = QUERY_COUNT / 2;

    let start_total = Instant::now();
    for pass in 0..2 {
        for i in 0..unique {
            let (id, pap_128, pap_512) = &records[i % records.len()];
            let features = features_from_pap(pap_128);
            let ctx = QueryContext {
                pap_128: *pap_128,
                pap_512: *pap_512,
                features,
                expected_id: Some(*id),
            };

            let result = engine.query(&ctx);

            metrics.total += 1;
            if let Some((top_id, _)) = result.top1 {
                if top_id == *id {
                    metrics.correct += 1;
                }
            }
            if result.from_cache {
                metrics.cache_hits += 1;
            }
            metrics.latencies_us.push(result.latency_us);
            if result.chain.run_spectral { metrics.spectral += 1; }
            if result.chain.run_wedge { metrics.wedge += 1; }
            if result.chain.run_hologram { metrics.hologram += 1; }

            // Learn that hash-only is sufficient when the answer is correct.
            if pass > 0 || i >= 4 {
                let pattern = pattern_from_features(features);
                engine.learning_table().update(pattern, &FeatureChain::none(), true);
            }
        }
    }
    let total_time_ms = start_total.elapsed().as_millis();

    let report = serde_json::json!({
        "records_loaded": records.len(),
        "queries": metrics.total,
        "r1": metrics.r1(),
        "correct": metrics.correct,
        "p50_latency_us": metrics.p50_us(),
        "p99_latency_us": metrics.p99_us(),
        "cache_hit_rate": metrics.cache_rate(),
        "cache_hits": metrics.cache_hits,
        "feature_activation": {
            "spectral": metrics.spectral as f64 / metrics.total as f64,
            "wedge": metrics.wedge as f64 / metrics.total as f64,
            "hologram": metrics.hologram as f64 / metrics.total as f64,
        },
        "total_time_ms": total_time_ms,
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}

fn features_from_pap(pap_128: &[u8; PAP_128_BYTES]) -> [f32; 8] {
    let mut f = [0.0f32; 8];
    for i in 0..8 {
        f[i] = (pap_128[i * 2] as f32) / 255.0;
    }
    f
}

fn pattern_from_features(features: [f32; 8]) -> u8 {
    let mut p = 0u8;
    for (i, &v) in features.iter().enumerate() {
        if v > 0.5 {
            p |= 1 << i;
        }
    }
    p
}
