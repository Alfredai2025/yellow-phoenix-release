// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Build a 106K BinaryHNSW index from the same source as the ISM flat file.
//!
//! Reads `yp_edge_106k.bin` (YISM format) and writes
//! `binary_hnsw_106k.bin` in BinaryHNSW format.
//!
//! Also prints a quick diagnostic comparing the first few hashes against
//! an existing HNSW index (e.g. the 1M arxiv index) to confirm mismatch.
//!
//! Usage:
//!     cd ~/yellow_phoenix
//!     cargo run --release --bin shootout_build_hnsw_106k -- \
//!         ~/yellow_phoenix_mobile/YPPhone/Resources/yp_edge_106k.bin \
//!         ~/yellow_phoenix_mobile/YPPhone/Resources/binary_hnsw_106k.bin \
//!         [~/yellow_phoenix_mobile/YPPhone/Resources/binary_hnsw_arxiv1m_m16.bin]

use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES, hamming_distance};

const YISM_MAGIC: &[u8] = b"YISM";
const YISM_VERSION: u8 = 1;
const YISM_HEADER_LEN: usize = 14; // magic(4) + version(1) + count(8) + hash_len(1)

fn load_yism(path: &str) -> Vec<(u64, [u8; HASH512_BYTES])> {
    let mut f = File::open(path).expect("open YISM file");
    let mut header = [0u8; YISM_HEADER_LEN];
    f.read_exact(&mut header).expect("read header");
    assert_eq!(&header[0..4], YISM_MAGIC, "bad magic");
    assert_eq!(header[4], YISM_VERSION, "bad version");

    let count = u64::from_le_bytes(header[5..13].try_into().unwrap()) as usize;
    let hash_len = header[13] as usize;
    assert_eq!(hash_len, HASH512_BYTES, "unsupported hash length");

    let record_size = 8 + HASH512_BYTES;
    let expected_size = YISM_HEADER_LEN + count * record_size;
    let actual_size = std::fs::metadata(path).map(|m| m.len() as usize).unwrap_or(0);
    assert_eq!(expected_size, actual_size, "file size mismatch");

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let mut record = [0u8; 8 + HASH512_BYTES];
        f.read_exact(&mut record).expect("read record");
        let id = u64::from_le_bytes(record[0..8].try_into().unwrap());
        let mut hash = [0u8; HASH512_BYTES];
        hash.copy_from_slice(&record[8..]);
        out.push((id, hash));
        if (i + 1) % 10000 == 0 {
            eprintln!("  loaded {} / {}", i + 1, count);
        }
    }
    out
}

fn diagnose(existing_hnsw_path: &str, records: &[(u64, [u8; HASH512_BYTES])]) {
    eprintln!("\n=== Diagnostic against {} ===", existing_hnsw_path);
    let hnsw = BinaryHNSW::load(existing_hnsw_path).expect("load existing HNSW");
    eprintln!("Existing HNSW has {} nodes", hnsw.len());

    let mut self_found = 0usize;
    let mut better_in_hnsw = 0usize;
    let samples = records.len().min(20);

    for i in 0..samples {
        let (id, hash) = &records[i];
        let exact_dist_in_flat = records
            .iter()
            .map(|(_, h)| hamming_distance(hash, h))
            .min()
            .unwrap_or(u32::MAX);

        let hnsw_results = hnsw.search(hash, 1);
        let (hnsw_dist, _hnsw_idx, _tag) = hnsw_results.first().copied().unwrap_or((u32::MAX, 0, 0));

        if hnsw_dist == 0 {
            self_found += 1;
        }
        if hnsw_dist < exact_dist_in_flat {
            better_in_hnsw += 1;
        }

        eprintln!(
            "  query #{:4} (id={}): flat exact min dist = {:3}, hnsw dist = {:3} {}",
            i,
            id,
            exact_dist_in_flat,
            hnsw_dist,
            if hnsw_dist == 0 { "[SELF FOUND]" } else { "" }
        );
    }

    eprintln!("\nSelf-found (dist 0) in existing HNSW: {}/{}", self_found, samples);
    if self_found == 0 {
        eprintln!("CONFIRMED: existing HNSW does NOT contain these exact hashes.");
    }
    if better_in_hnsw > 0 {
        eprintln!("HNSW found closer points in {}/{} queries (different dataset).", better_in_hnsw, samples);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <yp_edge_106k.bin> <binary_hnsw_106k_out.bin> [existing_hnsw_for_diag.bin]",
            args[0]
        );
        std::process::exit(1);
    }

    let yism_path = &args[1];
    let out_path = &args[2];

    eprintln!("Loading YISM records from {}...", yism_path);
    let t0 = Instant::now();
    let records = load_yism(yism_path);
    eprintln!("Loaded {} records in {:.2}s", records.len(), t0.elapsed().as_secs_f64());

    if args.len() >= 4 {
        diagnose(&args[3], &records);
    }

    eprintln!("\nBuilding BinaryHNSW (m=32, ef_construction=200)...");
    let mut hnsw = BinaryHNSW::with_params(32, 200, 128);
    let t1 = Instant::now();
    for (i, (id, hash)) in records.iter().enumerate() {
        hnsw.insert(*id, *hash);
        if (i + 1) % 10000 == 0 {
            eprintln!("  inserted {} / {}", i + 1, records.len());
        }
    }
    eprintln!("Built graph in {:.2}s", t1.elapsed().as_secs_f64());

    eprintln!("Saving to {}...", out_path);
    let t2 = Instant::now();
    hnsw.save(out_path).expect("save HNSW");
    let size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved {} ({:.1} MB) in {:.2}s",
        out_path,
        size as f64 / (1024.0 * 1024.0),
        t2.elapsed().as_secs_f64()
    );
}
