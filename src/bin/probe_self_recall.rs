// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Probe exact-hash self-recall of a BinaryHNSW index at a given ef_search.
//! Usage: probe_self_recall <in.ism> <hnsw.bin> <ef_search> [n_samples]

use std::env;
use std::fs::File;
use std::io::Read;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <in.ism> <hnsw.bin> <ef_search> [n]", args[0]);
        std::process::exit(1);
    }
    let ef: usize = args[3].parse().unwrap();
    let n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let mut r = File::open(&args[1]).expect("open ism");
    let count = read_u64(&mut r).expect("count") as usize;
    let mut hashes = vec![0u8; count * HASH512_BYTES];
    r.read_exact(&mut hashes).expect("hashes");
    let mut idb = vec![0u8; count * 8];
    r.read_exact(&mut idb).expect("ids");

    let mut hnsw = BinaryHNSW::load(&args[2]).expect("load hnsw");
    hnsw.set_ef_search(ef);

    // deterministic stride sample across the whole index
    let step = (count / n).max(1);
    let mut hits = 0;
    let mut checked = 0;
    let t = std::time::Instant::now();
    let mut i = 0;
    while i < count && checked < n {
        let mut h = [0u8; HASH512_BYTES];
        h.copy_from_slice(&hashes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES]);
        let id = u64::from_le_bytes(idb[i * 8..i * 8 + 8].try_into().unwrap());
        let res = hnsw.search(&h, 1);
        if let Some((d, _, _)) = res.first() {
            if *d == 0 && hnsw.node(res[0].1).map(|x| x.id) == Some(id) {
                hits += 1;
            }
        }
        checked += 1;
        i += step;
    }
    println!(
        "ef={} self-recall R@1: {:.2}% ({}/{}) in {:?} ({} queries, avg {:?})",
        ef,
        hits as f64 / checked as f64 * 100.0,
        hits,
        checked,
        t.elapsed(),
        checked,
        t.elapsed() / checked as u32
    );
}
