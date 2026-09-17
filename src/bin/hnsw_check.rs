// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::env;
use pams::binary_hnsw::BinaryHNSW;

fn main() {
    let path = env::args().nth(1).expect("usage: hnsw_check <hnsw.bin>");
    println!("Loading {}...", path);
    match BinaryHNSW::load(&path) {
        Ok(h) => {
            println!("Loaded {} nodes, memory {} bytes", h.len(), h.memory_bytes());
            let q = [0u8; 64];
            let start = std::time::Instant::now();
            let res = h.search(&q, 10);
            println!("Search took {:?}, got {} results", start.elapsed(), res.len());
        }
        Err(e) => {
            eprintln!("Load failed: {}", e);
            std::process::exit(1);
        }
    }
}
