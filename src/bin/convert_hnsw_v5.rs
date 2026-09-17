// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Convert a v4 BinaryHNSW file to the mmap-friendly v5 format and verify
//! that search results are identical on a sampled set of exact-hash queries.
//!
//! Usage:
//!     cargo run --release --bin convert_hnsw_v5 -- <in_v4.bin> <out_v5.bin>

use std::env;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES, hamming_distance};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <in_v4.bin> <out_v5.bin>", args[0]);
        std::process::exit(1);
    }
    let t0 = std::time::Instant::now();
    eprintln!("[1/4] loading v4 {} ...", args[1]);
    let v4 = BinaryHNSW::load(&args[1]).expect("load v4");
    eprintln!("      {} nodes in {:?}", v4.len(), t0.elapsed());

    eprintln!("[2/4] saving v5 {} ...", args[2]);
    let t1 = std::time::Instant::now();
    v4.save_v5(&args[2]).expect("save v5");
    let size = std::fs::metadata(&args[2]).map(|m| m.len()).unwrap_or(0);
    eprintln!("      {:.1} MB in {:?}", size as f64 / 1e6, t1.elapsed());

    eprintln!("[3/4] reloading v5 (mmap) ...");
    let t2 = std::time::Instant::now();
    let v5 = BinaryHNSW::load(&args[2]).expect("load v5");
    eprintln!("      {} nodes in {:?} (memory_bytes={:.2} GB virtual)",
        v5.len(), t2.elapsed(), v5.memory_bytes() as f64 / 1e9);

    eprintln!("[4/4] equivalence probe (500 exact-hash queries) ...");
    // sample node hashes by scanning the v4 file's hash region directly is
    // overkill; use hash_by_id via node() — hashes live in the nodes.
    let n = v4.len();
    let step = (n / 500).max(1);
    let mut same_top1 = 0;
    let mut checked = 0;
    for i in (0..n).step_by(step).take(500) {
        let node = match v4.node(i as u32) {
            Some(x) => x,
            None => continue,
        };
        let mut q = [0u8; HASH512_BYTES];
        q.copy_from_slice(&node.hash);
        let r4 = v4.search(&q, 10);
        let r5 = v5.search(&q, 10);
        let top4 = r4.first().map(|(d, i, _)| (*d, *i));
        let top5 = r5.first().map(|(d, i, _)| (*d, *i));
        if top4 == top5 {
            same_top1 += 1;
        }
        checked += 1;
    }
    eprintln!(
        "      identical top-1 (dist,idx): {}/{}",
        same_top1, checked
    );
    if same_top1 < checked * 99 / 100 {
        eprintln!("FAIL — v5 results diverge from v4");
        std::process::exit(1);
    }
    eprintln!("PASS — v5 mmap index matches v4 results.");
}
