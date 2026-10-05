// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Persistent query server for the Int8Hnsw engine (ann-benchmarks protocol).
//! Reads 128 f32 (512B) per query on stdin, writes 10 x u64 (80B) top-10 ids.
//! Usage: int8_server <graph.bin> <payload.f32> [ef] [K] [prune_alpha] [--ccep] [--gateway]

use std::io::{Read, Write};
use pams::int8_hnsw::I8Hnsw;

fn load_f32(path: &str) -> Vec<f32> {
    let mut f = std::fs::File::open(path).expect("open f32");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut g = I8Hnsw::load(&a[1]).expect("load graph");
    let payload = load_f32(&a[2]);
    let n_vecs = payload.len() / 128;
    let ef: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let alpha: f32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let ccep = a.iter().any(|s| s == "--ccep");
    let no_gateway = a.iter().any(|s| s == "--no-gateway");
    if alpha > 0.0 {
        let removed = if ccep { g.prune_diverse_ccep(alpha) } else { g.prune_diverse(alpha) };
        eprintln!("int8_server: pruned {} edges (alpha={}{})", removed, alpha, if ccep { ", ccep" } else { "" });
    }
    g.use_gateway = !no_gateway;
    eprintln!("int8_server ready: ef={ef} K={k} gateway={} vecs={}", g.use_gateway, n_vecs);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut inp = stdin.lock();
    let mut out = stdout.lock();
    let mut qbuf = [0u8; 512];
    loop {
        match inp.read_exact(&mut qbuf) {
            Ok(()) => {}
            Err(_) => break, // EOF: done
        }
        let mut q = [0f32; 128];
        for i in 0..128 {
            q[i] = f32::from_le_bytes([qbuf[i * 4], qbuf[i * 4 + 1], qbuf[i * 4 + 2], qbuf[i * 4 + 3]]);
        }
        let mut ctx = g.encode_query(&q);
        let top = g.search_with_ef(&mut ctx, k, ef);
        // exact float L2 verify, bounded top-10 insertion
        let mut best: Vec<(f32, u64)> = Vec::with_capacity(10);
        for &(_, idx) in &top {
            let id = g.node_label(idx);
            let mut d = 0f32;
            let base = id as usize * 128;
            for i in 0..128 { let diff = payload[base + i] - q[i]; d += diff * diff; }
            let pos = best.partition_point(|&(bd, _)| bd < d);
            if pos < 10 { best.insert(pos.min(best.len()), (d, id)); best.truncate(10); }
        }
        let mut ob = [0u8; 80];
        for i in 0..10 {
            let id = best.get(i).map(|&(_, id)| id).unwrap_or(u64::MAX);
            ob[i * 8..i * 8 + 8].copy_from_slice(&id.to_le_bytes());
        }
        out.write_all(&ob).expect("write");
        out.flush().expect("flush");
    }
}
