// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Tiny repro: build 20K-node HNSW, save v4, convert to v5, compare searches.
//! Usage: repro_v5 <ism> [n]

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
    let ism = &args[1];
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    let mut r = File::open(ism).unwrap();
    let count = read_u64(&mut r).unwrap() as usize;
    let mut hashes = vec![0u8; count * HASH512_BYTES];
    r.read_exact(&mut hashes).unwrap();

    let mut h = BinaryHNSW::with_params(32, 200, 128);
    for i in 0..n {
        let mut hh = [0u8; HASH512_BYTES];
        hh.copy_from_slice(&hashes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES]);
        h.insert(i as u64, hh);
    }

    h.save("/tmp/repro_v4.bin").unwrap();
    h.save_v5("/tmp/repro_v5.bin").unwrap();

    let v4 = BinaryHNSW::load("/tmp/repro_v4.bin").unwrap();
    let v5 = BinaryHNSW::load("/tmp/repro_v5.bin").unwrap();
    eprintln!("v4 nodes={}  v5 nodes={} (max_layers v4={} v5={})",
        v4.len(), v5.len(), v4.top_layer(), v5.top_layer());

    let mut same = 0;
    let mut self4 = 0;
    let mut self5 = 0;
    for i in (0..n).step_by(n / 100) {
        let mut q = [0u8; HASH512_BYTES];
        q.copy_from_slice(&hashes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES]);
        let r4 = v4.search(&q, 5);
        let r5 = v5.search(&q, 5);
        let t4 = r4.first().copied();
        let t5 = r5.first().copied();
        if t4 == t5 { same += 1; }
        if t4.map(|(d, _, _)| d) == Some(0) { self4 += 1; }
        if t5.map(|(d, _, _)| d) == Some(0) { self5 += 1; }
        if i == 0 {
            eprintln!("query0 v4 top: {:?}  v5 top: {:?}", t4, t5);
        }
    }
    eprintln!("identical top1: {}/100 | v4 self-hit: {} | v5 self-hit: {}", same, self4, self5);
}
