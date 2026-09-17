// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use pams::binary_hnsw::BinaryHNSW;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input.bin> <output.bin>", args[0]);
        std::process::exit(1);
    }
    let hnsw = BinaryHNSW::load(&args[1]).expect("load HNSW");
    eprintln!("Loaded {} nodes", hnsw.len());
    hnsw.save(&args[2]).expect("save stripped");
    eprintln!("Saved stripped graph");
}
