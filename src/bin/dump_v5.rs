// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Print Rust-loaded v5 node0 hash vs file bytes.

use std::env;
use pams::binary_hnsw::BinaryHNSW;

fn main() {
    let args: Vec<String> = env::args().collect();
    let v5 = BinaryHNSW::load(&args[1]).unwrap();
    let n0 = v5.node(0).unwrap();
    println!("rust v5 node0 hash[:16]: {:02x?}", &n0.hash[..16]);
    // raw file bytes
    let bytes = std::fs::read(&args[1]).unwrap();
    println!("file v5 node0 hash[:16]: {:02x?}", &bytes[56 + 8..56 + 24]);
    let n1 = v5.node(1).unwrap();
    println!("rust v5 node1 hash[:16]: {:02x?}", &n1.hash[..16]);
    println!("file v5 node1 hash[:16]: {:02x?}", &bytes[56 + 112 + 8..56 + 112 + 24]);
}
