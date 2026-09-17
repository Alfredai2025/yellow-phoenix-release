// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Verify AbstractStore against papers_5m.db (spot ids).
//! Usage: cargo run --release --bin test_abstracts_decode -- <store.bin> [id...]
use pams::abstract_store::AbstractStore;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let store = AbstractStore::open(&args[1]).expect("open store");
    println!("count: {}", store.len());
    let ids: Vec<u64> = if args.len() > 2 {
        args[2..].iter().map(|s| s.parse().unwrap()).collect()
    } else {
        vec![0, 1265414, 3561125, 4495304, 2_875_034, 5_107_507, 42]
    };
    for id in ids {
        match store.decode(id) {
            Some(t) => println!("id={} len={} | {}", id, t.len(), t.chars().take(100).collect::<String>().replace('\n', " ")),
            None => println!("id={} MISSING", id),
        }
    }
}
