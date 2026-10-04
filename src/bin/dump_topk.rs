// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dump top-K candidate ids per query for two-stage re-rank measurement.
//! Usage: dump_topk x <graph.bin> <queries.ism> <out.bin> [K] [ef]
//!   (arg1 "x" is a placeholder kept for CLI symmetry with eval_graph)
//! queries.ism: BLOCK layout — [u64 count][count x 64B hashes][count x u64 ids]
//!   (ids are NOT read per query; hashes are contiguous, eval_graph-style)
//! out.bin: [u64 n_queries][u64 K][n_queries * K * u64 ids]
//! Hard assert: search must return >= K candidates (catches k>ef misuse).
use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let g = BinaryHNSW::load(&a[2]).expect("load graph");
    let k: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);
    let ef: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(&a[3]).expect("open queries"));
    let n = read_u64(&mut r).expect("count") as usize;
    let mut out = File::create(&a[4]).expect("create out");
    out.write_all(&(n as u64).to_le_bytes()).unwrap();
    out.write_all(&(k as u64).to_le_bytes()).unwrap();
    let mut hb = [0u8; HASH512_BYTES];
    for i in 0..n {
        // hashes are a contiguous block; do NOT skip per-row ids
        r.read_exact(&mut hb).expect("hash");
        let top = g.search_with_ef(&hb, k, ef);
        let mut ids: Vec<u64> = top.iter().map(|(_, idx, _)| g.node_label(*idx)).collect();
        ids.resize(k, u64::MAX);
        assert!(!ids.contains(&u64::MAX), "query {}: search returned fewer than k candidates (k>ef?)", i);
        for id in ids.iter().take(k) {
            out.write_all(&id.to_le_bytes()).unwrap();
        }
        if (i + 1) % 1000 == 0 { eprintln!("  {} / {}", i + 1, n); }
    }
    eprintln!("done");
}
