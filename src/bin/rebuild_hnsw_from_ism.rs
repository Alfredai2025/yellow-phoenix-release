// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Rebuild HNSW from an ISM flat index, preserving ID alignment.
//!
//! Usage:
//!   cargo run --bin rebuild_hnsw_from_ism --release -- \
//!     data/synth_10m.ism data/synth_10m_hnsw_m8.bin 8 200 128
//!
//! Arguments:
//!   1. input .ism path
//!   2. output .bin path
//!   3. M (default 16)
//!   4. ef_construction (default 200)
//!   5. ef_search (default 128)

use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <input.ism> <output.bin> [M] [ef_construction] [ef_search]",
            args[0]
        );
        std::process::exit(1);
    }

    let ism_path = &args[1];
    let out_path = &args[2];
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let ef_construction: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let ef_search: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(128);

    eprintln!("Reading ISM: {}", ism_path);
    let t0 = Instant::now();

    // ISM format: [u64 count LE][count * 64-byte hashes][count * u64 ids]
    let mut file = File::open(ism_path).expect("open ISM");
    let mut count_buf = [0u8; 8];
    file.read_exact(&mut count_buf).expect("read count");
    let count = u64::from_le_bytes(count_buf) as usize;
    eprintln!("  {} entries", count);

    let mut hashes = vec![0u8; count * HASH512_BYTES];
    file.read_exact(&mut hashes).expect("read hashes");

    let mut id_buf = vec![0u8; count * 8];
    file.read_exact(&mut id_buf).expect("read IDs");

    eprintln!(
        "  parsed in {:?}\nBuilding HNSW (M={}, ef_construction={}, ef_search={})...",
        t0.elapsed(),
        m,
        ef_construction,
        ef_search
    );

    let mut hnsw = BinaryHNSW::with_params(m, ef_construction, ef_search);
    let t1 = Instant::now();

    for i in 0..count {
        let hash: [u8; HASH512_BYTES] = hashes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES]
            .try_into()
            .expect("hash slice");
        let id = u64::from_le_bytes(id_buf[i * 8..(i + 1) * 8].try_into().expect("id slice"));
        hnsw.insert(id, hash);

        if (i + 1) % 100_000 == 0 || i + 1 == count {
            eprint!(
                "\r  inserted {}/{}  ({:.1}%)",
                i + 1,
                count,
                (i + 1) as f64 / count as f64 * 100.0
            );
        }
    }
    eprintln!("\nBuilt in {:?}", t1.elapsed());

    hnsw.save(out_path).expect("save HNSW");
    eprintln!(
        "Saved to: {}  ({:.1} MB)",
        out_path,
        std::fs::metadata(out_path)
            .map(|m| m.len() as f64 / 1_048_576.0)
            .unwrap_or(0.0)
    );
}
