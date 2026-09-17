// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Verify that HNSW IDs and hashes align with the ISM source.

use std::env;
use std::fs::File;
use std::io::Read;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES, hamming_distance};

struct FlatIndex {
    ids: Vec<u64>,
    hashes: Vec<u8>,
}

impl FlatIndex {
    fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut count_buf = [0u8; 8];
        file.read_exact(&mut count_buf)?;
        let count = u64::from_le_bytes(count_buf) as usize;

        let mut hashes = vec![0u8; count * HASH512_BYTES];
        file.read_exact(&mut hashes)?;

        let mut ids = vec![0u64; count];
        let mut id_buf = vec![0u8; count * 8];
        file.read_exact(&mut id_buf)?;
        for i in 0..count {
            ids[i] = u64::from_le_bytes(id_buf[i * 8..i * 8 + 8].try_into().unwrap());
        }
        Ok(Self { ids, hashes })
    }

    fn count(&self) -> usize {
        self.ids.len()
    }
    fn hash_at(&self, i: usize) -> &[u8] {
        &self.hashes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES]
    }
    fn id_at(&self, i: usize) -> u64 {
        self.ids[i]
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.ism> <input.bin>", args[0]);
        std::process::exit(1);
    }

    let flat = FlatIndex::load(&args[1]).expect("load ISM");
    let hnsw = BinaryHNSW::load(&args[2]).expect("load HNSW");

    assert_eq!(flat.count(), hnsw.len(), "COUNT MISMATCH");

    let check_n = flat.count().min(10_000);
    let mut id_mismatch = 0;
    let mut hash_mismatch = 0;
    let mut recall_hits = 0;

    for i in 0..check_n {
        let hash: [u8; HASH512_BYTES] = flat.hash_at(i).try_into().unwrap();
        let id = flat.id_at(i);

        if let Some(hnsw_hash) = hnsw.hash_by_id(id) {
            if hamming_distance(&hash, hnsw_hash) != 0 {
                hash_mismatch += 1;
            }
        } else {
            id_mismatch += 1;
        }

        let res = hnsw.search(&hash, 1);
        if let Some((_, idx, _)) = res.first() {
            let found_id = hnsw.node(*idx).map(|n| n.id).unwrap_or(0);
            if found_id == id {
                recall_hits += 1;
            }
        }
    }

    println!("Checked {} / {} nodes", check_n, flat.count());
    println!("  ID not found in HNSW:      {} (should be 0)", id_mismatch);
    println!(
        "  Hash mismatch for same ID: {} (should be 0)",
        hash_mismatch
    );
    println!(
        "  Self-query recall R@1:     {:.2}% (should be ~99%+)",
        recall_hits as f64 / check_n as f64 * 100.0
    );

    if id_mismatch > 0 || hash_mismatch > 0 || recall_hits < check_n * 95 / 100 {
        println!("\nFAIL — alignment is broken. Rebuild the HNSW from the ISM.");
        std::process::exit(1);
    } else {
        println!("\nPASS — IDs and hashes are aligned.");
    }
}
