// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! functor_audit.rs — Run YP Phase B functor measurements on a real mesh.
//!
//! Input binary layout (same as validate_m1):
//!   u32 LE      : record count N
//!   N * 88 bytes: [id u64 LE][pap128 16 bytes][pap512 64 bytes]
//!
//! Usage:
//!   cargo run --release --bin functor_audit -- data/validate_m1_input.bin
//!   cargo run --release --bin functor_audit -- data/1m_papers.bin --limit 100000 --f1-pairs 5000 --output logs/functor_audit_100k.json
//!
//! Flags:
//!   --limit N          only load the first N records
//!   --f1-pairs N       pairs for F1 (default 1000)
//!   --f2-queries N     queries for F2 (default 100)
//!   --f3-queries N     queries for F3 (default 100)
//!   --f4-docs N        test ids for F4 (default 100)
//!   --output PATH      write JSON report to file (default: stdout)

use pams::binary_hnsw::BinaryHNSW;
use pams::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};
use pams::math::functor_bounds::{
    check_f3_order_preservation, compute_f4_delta, measure_f1_lipschitz, measure_f2_bound,
};
use pams::math::tensor_spectral_512::TensorSpectralIndex512;
use std::env;
use std::fs::File;
use std::io::{Read, Result as IoResult, Write};
use std::path::Path;
use std::time::Instant;

const RECORD_BYTES: usize = 8 + PAP_128_BYTES + PAP_512_BYTES;

fn load_records(
    path: &Path,
    limit: Option<usize>,
) -> IoResult<Vec<(u64, [u8; PAP_128_BYTES], [u8; PAP_512_BYTES])>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    if buf.len() < 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too short",
        ));
    }
    let file_count = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let count = limit.map(|l| l.min(file_count)).unwrap_or(file_count);
    let expected = 4 + count * RECORD_BYTES;
    if buf.len() < expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "expected at least {} bytes for {} records, got {}",
                expected,
                count,
                buf.len()
            ),
        ));
    }

    let mut records = Vec::with_capacity(count);
    let mut cursor = 4usize;
    for _ in 0..count {
        let id = u64::from_le_bytes([
            buf[cursor],
            buf[cursor + 1],
            buf[cursor + 2],
            buf[cursor + 3],
            buf[cursor + 4],
            buf[cursor + 5],
            buf[cursor + 6],
            buf[cursor + 7],
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

    Ok(records)
}

#[derive(Debug, Default)]
struct Args {
    input: String,
    limit: Option<usize>,
    f1_pairs: usize,
    f2_queries: usize,
    f3_queries: usize,
    f4_docs: usize,
    output: Option<String>,
}

fn parse_args() -> Args {
    let raw: Vec<String> = env::args().skip(1).collect();
    let mut args = Args {
        f1_pairs: 1000,
        f2_queries: 100,
        f3_queries: 100,
        f4_docs: 100,
        ..Default::default()
    };

    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--limit" => {
                i += 1;
                if i < raw.len() {
                    args.limit = raw[i].parse().ok();
                }
            }
            "--f1-pairs" => {
                i += 1;
                if i < raw.len() {
                    args.f1_pairs = raw[i].parse().unwrap_or(args.f1_pairs);
                }
            }
            "--f2-queries" => {
                i += 1;
                if i < raw.len() {
                    args.f2_queries = raw[i].parse().unwrap_or(args.f2_queries);
                }
            }
            "--f3-queries" => {
                i += 1;
                if i < raw.len() {
                    args.f3_queries = raw[i].parse().unwrap_or(args.f3_queries);
                }
            }
            "--f4-docs" => {
                i += 1;
                if i < raw.len() {
                    args.f4_docs = raw[i].parse().unwrap_or(args.f4_docs);
                }
            }
            "--output" => {
                i += 1;
                if i < raw.len() {
                    args.output = Some(raw[i].clone());
                }
            }
            other => {
                if !other.starts_with('-') && args.input.is_empty() {
                    args.input = other.to_string();
                }
            }
        }
        i += 1;
    }

    args
}

fn main() {
    let args = parse_args();
    if args.input.is_empty() {
        eprintln!(
            "Usage: functor_audit <input.bin> [options]\n\
             Options:\n\
             --limit N          only load first N records\n\
             --f1-pairs N\n\
             --f2-queries N\n\
             --f3-queries N\n\
             --f4-docs N\n\
             --output PATH"
        );
        std::process::exit(1);
    }

    let input_path = Path::new(&args.input);
    let records = load_records(input_path, args.limit).expect("load input records");
    if records.is_empty() {
        eprintln!("No records loaded from {:?}", input_path);
        std::process::exit(1);
    }
    println!(
        "Loaded {} records from {:?}",
        records.len(),
        input_path
    );

    // Build HybridMesh.
    let t0 = Instant::now();
    let mut mesh = HybridMesh::new(
        records.len().next_power_of_two(),
        records.len().next_power_of_two(),
    );
    for (id, pap_128, pap_512) in &records {
        mesh.insert_dual(*id, pap_128, pap_512);
    }
    mesh.coarse.build_edges(8);
    mesh.fine.build_edges(8);
    println!("Built HybridMesh in {:?}", t0.elapsed());

    // Build BinaryHNSW.
    let t0 = Instant::now();
    let mut hnsw = BinaryHNSW::new();
    for (id, _, pap_512) in &records {
        hnsw.insert(*id, *pap_512);
    }
    println!(
        "Built BinaryHNSW ({} nodes) in {:?}",
        hnsw.len(),
        t0.elapsed()
    );

    // Build 512-bit spectral index.
    let t0 = Instant::now();
    let spectral_k = env::var("SPECTRAL_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64usize);
    let mut pap_512_bytes: Vec<u8> = Vec::with_capacity(records.len() * PAP_512_BYTES);
    for (_, _, pap_512) in &records {
        pap_512_bytes.extend_from_slice(pap_512);
    }
    let spectral = TensorSpectralIndex512::build(&pap_512_bytes, spectral_k);
    println!(
        "Built TensorSpectralIndex512 (k={}) in {:?}",
        spectral_k,
        t0.elapsed()
    );

    let cap = records.len();
    let f1_pairs = args.f1_pairs.min(cap);
    let f2_queries = args.f2_queries.min(cap);
    let f3_queries = args.f3_queries.min(cap);
    let f4_ids = args.f4_docs.min(cap);

    let mut report = serde_json::json!({
        "records": records.len(),
        "input": input_path.to_string_lossy().to_string(),
        "limit": args.limit,
    });

    // F1: Hash -> Spectral.
    let t0 = Instant::now();
    match measure_f1_lipschitz(&mesh, &spectral, f1_pairs) {
        Ok((k1, r2)) => {
            report["F1"] = serde_json::json!({
                "k1": k1,
                "r_squared": r2,
                "sample_pairs": f1_pairs,
                "time_ms": t0.elapsed().as_millis(),
                "status": "ok",
            });
            println!("F1 k1={:.6} R^2={:.4} ({:?})", k1, r2, t0.elapsed());
        }
        Err(e) => {
            report["F1"] = serde_json::json!({"status": "err", "error": e});
            eprintln!("F1 failed: {}", e);
        }
    }

    // F2: Spectral -> HNSW.
    let t0 = Instant::now();
    match measure_f2_bound(&spectral, &hnsw, &mesh, f2_queries) {
        Ok((mean, p95)) => {
            report["F2"] = serde_json::json!({
                "mean_rank": mean,
                "p95_rank": p95,
                "queries": f2_queries,
                "time_ms": t0.elapsed().as_millis(),
                "status": "ok",
            });
            println!(
                "F2 mean_rank={:.2} p95_rank={:.2} ({:?})",
                mean,
                p95,
                t0.elapsed()
            );
        }
        Err(e) => {
            report["F2"] = serde_json::json!({"status": "err", "error": e});
            eprintln!("F2 failed: {}", e);
        }
    }

    // F3: HNSW -> Geometric.
    let t0 = Instant::now();
    match check_f3_order_preservation(&hnsw, &mesh, f3_queries) {
        Ok(tau) => {
            report["F3"] = serde_json::json!({
                "kendall_tau": tau,
                "queries": f3_queries,
                "time_ms": t0.elapsed().as_millis(),
                "status": "ok",
            });
            println!("F3 tau={:.4} ({:?})", tau, t0.elapsed());
        }
        Err(e) => {
            report["F3"] = serde_json::json!({"status": "err", "error": e});
            eprintln!("F3 failed: {}", e);
        }
    }

    // F4: End-to-end delta.
    let test_ids: Vec<u64> = records.iter().map(|(id, _, _)| *id).take(f4_ids).collect();
    let t0 = Instant::now();
    match compute_f4_delta(&mesh, &hnsw, &test_ids) {
        Ok(delta) => {
            report["F4"] = serde_json::json!({
                "delta": delta,
                "test_ids": f4_ids,
                "time_ms": t0.elapsed().as_millis(),
                "status": "ok",
            });
            println!("F4 delta={:.4} ({:?})", delta, t0.elapsed());
        }
        Err(e) => {
            report["F4"] = serde_json::json!({"status": "err", "error": e});
            eprintln!("F4 failed: {}", e);
        }
    }

    let json = serde_json::to_string_pretty(&report).unwrap();
    if let Some(out_path) = args.output {
        let path = Path::new(&out_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut file = File::create(path).expect("create output file");
        file.write_all(json.as_bytes()).expect("write output");
        println!("\n[+] Report written to {}", out_path);
    } else {
        println!("\n{}", json);
    }
}
