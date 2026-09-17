// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Build a BinaryHNSW index from a synthetic ISM file.
//!
//! Reads the ISM format produced by generate_all_synthetic.py:
//!   count(u64 LE) + count * 64-byte hashes + count * u64 ids
//!
//! Writes binary_hnsw.bin in the project's native BinaryHNSW format.
//!
//! Usage:
//!     cargo run --release --bin build_hnsw_from_ism -- \
//!         ~/yellow_phoenix/data/synth_10m.ism \
//!         ~/yellow_phoenix/data/synth_10m_hnsw.bin

use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn load_ism(path: &str) -> Vec<(u64, [u8; HASH512_BYTES])> {
    let mut f = File::open(path).expect("open ISM file");

    let mut count_bytes = [0u8; 8];
    f.read_exact(&mut count_bytes).expect("read count");
    let count = u64::from_le_bytes(count_bytes) as usize;

    let mut records = Vec::with_capacity(count);
    let mut hash_buf = [0u8; HASH512_BYTES];
    let mut id_buf = [0u8; 8];

    for i in 0..count {
        f.read_exact(&mut hash_buf).expect("read hash");
        records.push((0u64, hash_buf));
        if (i + 1) % 100_000 == 0 {
            eprintln!("  loaded {} / {} hashes", i + 1, count);
        }
    }

    for i in 0..count {
        f.read_exact(&mut id_buf).expect("read id");
        records[i].0 = u64::from_le_bytes(id_buf);
        if (i + 1) % 100_000 == 0 {
            eprintln!("  loaded {} / {} ids", i + 1, count);
        }
    }

    records
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: {} <input.ism> <output_hnsw.bin>",
            args[0]
        );
        std::process::exit(1);
    }

    let ism_path = &args[1];
    let out_path = &args[2];

    eprintln!("Loading ISM from {}...", ism_path);
    let t0 = Instant::now();
    let records = load_ism(ism_path);
    eprintln!(
        "Loaded {} records in {:.2}s",
        records.len(),
        t0.elapsed().as_secs_f64()
    );

    // Allow tuning via env vars so large-scale builds can finish in reasonable time.
    let m = env::var("HNSW_M").ok().and_then(|s| s.parse().ok()).unwrap_or(16usize);
    let ef_construction = env::var("HNSW_EF_CONSTRUCTION").ok().and_then(|s| s.parse().ok()).unwrap_or(200usize);
    let ef_search = env::var("HNSW_EF_SEARCH").ok().and_then(|s| s.parse().ok()).unwrap_or(128usize);
    eprintln!(
        "\nBuilding BinaryHNSW (m={}, ef_construction={}, ef_search={})...",
        m, ef_construction, ef_search
    );
    let mut hnsw = BinaryHNSW::with_params(m, ef_construction, ef_search);
    let t1 = Instant::now();
    for (i, (id, hash)) in records.iter().enumerate() {
        hnsw.insert(*id, *hash);
        if (i + 1) % 100_000 == 0 {
            eprintln!("  inserted {} / {}", i + 1, records.len());
        }
    }
    eprintln!("Built graph in {:.2}s", t1.elapsed().as_secs_f64());

    eprintln!("Saving to {}...", out_path);
    let t2 = Instant::now();
    // v5 (mmap-backed): multi-GB graphs must not be fully resident in RAM
    // on 8GB devices — v4 BufReader load jetsams the app (observed 2026-09-10).
    hnsw.save_v5(out_path).expect("save HNSW v5");
    let size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved {} ({:.2} GB) in {:.2}s",
        out_path,
        size as f64 / (1024.0 * 1024.0 * 1024.0),
        t2.elapsed().as_secs_f64()
    );
}
