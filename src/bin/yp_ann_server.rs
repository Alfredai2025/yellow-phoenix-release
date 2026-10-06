// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Persistent query server for ann-benchmarks integration.
//! Holds in RAM: 512-bit BinaryHNSW graph + float32 payload + fused ITQ matrix.
//! Protocol (stdin/stdout, binary, little-endian):
//!   request:  [128 x f32]  (raw query vector)
//!   response: [10 x u64]   (top-10 ids by two-stage: graph propose K -> exact L2 verify)
//! ef/K fixed at server start (one server process per operating point).
//!
//! Usage: yp_ann_server <graph.bin> <payload.f32> <W.f32> <mu.f32> [ef] [K] [--mode raw]
//!   default mode: 512B f32 query in -> ITQ encode -> graph -> float verify -> 10 ids
//!   --mode raw:   64B code in -> graph search only -> 10 ids (hamming datasets;
//!                 caller's codes must be padded to 64B; ordering preserved)
//!   --funnel <books> <codes> <V> <mu> <narrow>: after the graph proposes K
//!                 candidates, rotate the query, build OPQ-ADC tables, narrow
//!                 to <narrow>, then exact-verify only those (crown funnel).
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
    let w = load_f32(&a[3]);           // 128 x 512 fused, d-major as stored
    let mu = load_f32(&a[4]);          // d floats (input dimension)
    let d = mu.len();
    let ef: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(200);
    let raw_mode = a.iter().any(|s| s == "--mode" ) && a.iter().any(|s| s == "raw");
    // funnel: --funnel <books.bin> <codes.bin> <V.f32> <mu.f32> <narrow_to>
    let funnel = a.iter().position(|s| s == "--funnel").map(|p| {
        assert!(d == 128, "funnel path is 128-d only");
        let books = load_pq_books(&a[p + 1]);
        let (n_codes, nb, codes) = load_pq_codes(&a[p + 2]);
        let vf = load_f32_exact(&a[p + 3], 128 * 128);
        let mf = load_f32_exact(&a[p + 4], 128);
        let narrow: usize = a.get(p + 5).and_then(|s| s.parse().ok()).unwrap_or(100);
        (books, n_codes, nb, codes, vf, mf, narrow)
    });
    if let Some((b, nc, nb, _, _, _, nw)) = &funnel {
        assert!(*nc * 128 == payload.len() as u64 && b.0 == *nb, "funnel artifact mismatch");
        eprintln!("funnel: {} blocks, narrow to {}", b.0, nw);
    }
    assert!(w.len() == d * 512, "W/mu shape");
    // transpose W to bit-major rows for the 128-wide SIMD kernel (kept for the
    // original 128-d path); generic d uses a plain per-bit dot below.
    let wt128: Option<Box<[[f32; 128]; 512]>> = if d == 128 {
        let mut wt: Box<[[f32; 128]; 512]> = Box::new([[0f32; 128]; 512]);
        for dd in 0..128 {
            for b in 0..512 {
                wt[b][dd] = w[dd * 512 + b];
            }
        }
        Some(wt)
    } else { None };
    // generic bit-major W for any d
    let wtg: Vec<Vec<f32>> = (0..512).map(|b| (0..d).map(|dd| w[dd * 512 + b]).collect()).collect();
    eprintln!("yp_ann_server ready: ef={ef} K={k} raw={raw_mode} d={d}, payload {} vecs", payload.len()/d);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut inp = stdin.lock();
    let mut out = stdout.lock();
    let mut q = vec![0f32; d];
    let mut qb = vec![0u8; d * 4];
    let mut code_in = [0u8; HASH512_BYTES];
    loop {
        let code: [u8; HASH512_BYTES] = if raw_mode {
            if inp.read_exact(&mut code_in).is_err() { break; }
            code_in
        } else {
            if inp.read_exact(&mut qb).is_err() { break; }
            for i in 0..d { q[i] = f32::from_le_bytes(qb[i*4..i*4+4].try_into().unwrap()); }
            // encode: (q - mu) @ W
            let mut qm = vec![0f32; d];
            for dd in 0..d { qm[dd] = q[dd] - mu[dd]; }
            if let Some(wt) = &wt128 {
                let qm_arr: [f32; 128] = qm.try_into().unwrap();
                pams::simd_kernels::encode512(&qm_arr, wt)
            } else {
                let mut bits = [0u8; HASH512_BYTES];
                for (b, row) in wtg.iter().enumerate() {
                    let mut s = 0f32;
                    for dd in 0..d { s += qm[dd] * row[dd]; }
                    if s >= 0.0 { bits[b / 8] |= 1 << (7 - (b % 8)); }  // MSB-first: matches numpy.packbits
                }
                bits
            }
        };
        let top = graph.search_with_ef(&code, k, ef);
        let mut cand: Vec<u64> = top.iter().map(|(_, idx, _)| graph.node_label(*idx)).collect();
        if let Some((books, _nc, nb, codes, vf, mf, narrow)) = &funnel {
            // rotate query into OPQ space: qrot = (q - mf) @ V
            let mut qr = [0f32; 128];
            for d in 0..128 { qr[d] = q[d] - mf[d]; }
            let mut qrot = [0f32; 128];
            for b in 0..128 {
                let mut s = 0f32;
                for d in 0..128 { s += qr[d] * vf[d * 128 + b]; }
                qrot[b] = s;
            }
            let (nblk, kbook, blen, ref bookf) = *books;
            let kk = kbook as usize;
            let mut tabs = vec![0f32; nblk as usize * kk];
            for b in 0..nblk as usize {
                let qoff = b * blen as usize;
                for c in 0..kk {
                    let mut s = 0f32;
                    for dd in 0..blen as usize {
                        let diff = bookf[(b * kk + c) * blen as usize + dd] - qrot[qoff + dd];
                        s += diff * diff;
                    }
                    tabs[b * kk + c] = s;
                }
            }
            let nb = *nb as usize;
            let mut scored: Vec<(f32, u64)> = cand.iter().map(|&id| {
                let off = id as usize * nb;
                let mut s = 0f32;
                for b in 0..nb { s += tabs[b * kk + codes[off + b] as usize]; }
                (s, id)
            }).collect();
            scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            scored.truncate((*narrow).min(scored.len()));
            cand = scored.into_iter().map(|(_, id)| id).collect();
        }
        let best_ids: Vec<u64> = if raw_mode || cand.len() <= 10 {
            cand.into_iter().take(10).collect()
        } else {
            let mut best: Vec<(f32, u64)> = cand.iter()
                .map(|&id| (pams::simd_kernels::l2_sq_f32(&payload[id as usize * d..id as usize * d + d], &q), id))
                .collect();
            best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            best.truncate(10);
            best.into_iter().map(|(_, id)| id).collect()
        };
        let mut ob = [0u8; 80];
        for i in 0..10 {
            ob[i * 8..(i + 1) * 8].copy_from_slice(&best_ids.get(i).copied().unwrap_or(u64::MAX).to_le_bytes());
        }
        out.write_all(&ob).unwrap();
        out.flush().unwrap();
    }
    eprintln!("yp_ann_server: client disconnected");
}

// ---- PQ funnel artifact loaders (kept with the binary for simplicity) ----
fn load_f32_exact(path: &str, expect: usize) -> Vec<f32> {
    let v = load_f32_any(path);
    assert!(v.len() == expect, "f32 artifact size mismatch");
    v
}
fn load_f32_any(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32 artifact");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32 artifact");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn load_pq_books(path: &str) -> (u32, u32, u32, Vec<f32>) {
    let mut f = File::open(path).expect("open books");
    let mut h = [0u8; 12];
    f.read_exact(&mut h).unwrap();
    let nb = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
    let k = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
    let bl = u32::from_le_bytes([h[8], h[9], h[10], h[11]]);
    let mut v = Vec::new();
    f.read_to_end(&mut v).unwrap();
    let data: Vec<f32> = v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert!(data.len() == nb as usize * k as usize * bl as usize, "books size");
    (nb, k, bl, data)
}
fn load_pq_codes(path: &str) -> (u64, u32, Vec<u8>) {
    let mut f = File::open(path).expect("open codes");
    let mut h = [0u8; 16];
    f.read_exact(&mut h).unwrap();
    let n = u64::from_le_bytes(h[0..8].try_into().unwrap());
    let nb = u32::from_le_bytes(h[8..12].try_into().unwrap());
    let cb = u32::from_le_bytes(h[12..16].try_into().unwrap());
    assert!(cb == 1, "v1 codes loader supports 1-byte codes");
    let mut v = vec![0u8; n as usize * nb as usize];
    f.read_exact(&mut v).unwrap();
    (n, nb, v)
}
