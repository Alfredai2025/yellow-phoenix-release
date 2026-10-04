// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Persistent query server for ann-benchmarks integration.
//! Holds in RAM: 512-bit BinaryHNSW graph + float32 payload + fused ITQ matrix.
//! Protocol (stdin/stdout, binary, little-endian):
//!   request:  [128 x f32]  (raw query vector)
//!   response: [10 x u64]   (top-10 ids by two-stage: graph propose K -> exact L2 verify)
//! ef/K fixed at server start (one server process per operating point).
//!
//! Usage: yp_ann_server <graph.bin> <payload.f32> <W.f32> <mu.f32> [ef] [K]
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn load_f32(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let graph = BinaryHNSW::load(&a[1]).expect("load graph");
    let payload = load_f32(&a[2]);
    let w = load_f32(&a[3]);           // 128 x 512 fused (mu already centered? NO - mu subtracted here)
    let mu = load_f32(&a[4]);          // 128
    let ef: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(200);
    assert!(w.len() == 128 * 512 && mu.len() == 128, "W/mu shape");
    eprintln!("yp_ann_server ready: ef={ef} K={k}, payload {} vecs", payload.len()/128);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut inp = stdin.lock();
    let mut out = stdout.lock();
    let mut q = [0f32; 128];
    let mut qb = [0u8; 512];
    loop {
        if inp.read_exact(&mut qb).is_err() { break; } // EOF
        for i in 0..128 { q[i] = f32::from_le_bytes(qb[i*4..i*4+4].try_into().unwrap()); }
        // encode: (q - mu) @ W  ->  512 bits
        let mut code = [0u8; HASH512_BYTES];
        for b in 0..512 {
            let mut s = 0f32;
            for d in 0..128 {
                s += (q[d] - mu[d]) * w[d * 512 + b];
            }
            if s >= 0.0 { code[b / 8] |= 1 << (7 - (b % 8)); }
        }
        let top = graph.search_with_ef(&code, k, ef);
        let cand: Vec<u64> = top.iter().map(|(_, idx, _)| graph.node_label(*idx)).collect();
        let mut best: Vec<(f32, u64)> = cand.iter()
            .map(|&id| (l2(&payload[id as usize * 128..id as usize * 128 + 128], &q), id))
            .collect();
        best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        best.truncate(10);
        let mut ob = [0u8; 80];
        for (i, (_, id)) in best.iter().enumerate() {
            ob[i * 8..(i + 1) * 8].copy_from_slice(&id.to_le_bytes());
        }
        out.write_all(&ob).unwrap();
        out.flush().unwrap();
    }
    eprintln!("yp_ann_server: client disconnected");
}
