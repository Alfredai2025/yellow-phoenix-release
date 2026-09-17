// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Extract hashes and IDs from a BinaryHNSW file and write a flat .ism file.
//! Usage: cargo run --release --bin hnsw_extract_ism -- <input.bin> <output.ism>

use std::env;
use pams::binary_hnsw::BinaryHNSW;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <binary_hnsw.bin> <output.ism>", args[0]);
        std::process::exit(1);
    }
    let hnsw_path = &args[1];
    let out_path = &args[2];

    eprintln!("Loading HNSW from {}...", hnsw_path);
    let hnsw = BinaryHNSW::load(hnsw_path).expect("load HNSW");
    let count = hnsw.len();
    eprintln!("Loaded {} nodes", count);

    let mut ids = Vec::with_capacity(count);
    let mut hashes: Vec<u8> = Vec::with_capacity(count * 64);
    for i in 0..count {
        let node = hnsw.node(i as u32).expect("valid node");
        ids.push(node.id);
        hashes.extend_from_slice(&node.hash);
    }

    let mut out = std::fs::File::create(out_path).expect("create output");
    use std::io::Write;
    out.write_all(&(count as u64).to_le_bytes()).unwrap();
    out.write_all(&hashes).unwrap();
    for id in ids {
        out.write_all(&id.to_le_bytes()).unwrap();
    }
    let size = std::fs::metadata(out_path).unwrap().len();
    eprintln!("Wrote {} ({} MB)", out_path, size as f64 / (1024.0 * 1024.0));
}
