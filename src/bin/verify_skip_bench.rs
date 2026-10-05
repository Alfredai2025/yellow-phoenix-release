// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Verify-skip test: ITQ hamming->L2 lower-bound certificate.
//! Calibrates P(L2 | hamming) on random + neighbor node pairs, then at query
//! time skips float re-verification of candidates whose calibrated LB
//! (minus k*sigma) exceeds the current 10th-best verified L2.
//! Usage: verify_skip_bench <graph.bin> <payload.f32> <qcodes.ism> <gt.bin> <queries.f32> <out.json> [margin_sigma]

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};
use pams::binary_hnsw::hamming_distance;

fn load_f32(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
/// hamming on the first 16 bytes (top-128 PCA-ordered bits = highest-variance ITQ dims)
fn h128(a: &[u8], b: &[u8]) -> u32 {
    let mut d = 0u32;
    for c in 0..2 {
        let x = u64::from_le_bytes(a[c * 8..c * 8 + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[c * 8..c * 8 + 8].try_into().unwrap());
        d += x.count_ones();
    }
    d
}

const NBINS: usize = 513;

fn main() {
    let a: Vec<String> = env::args().collect();
    let g = BinaryHNSW::load(&a[1]).expect("load graph");
    let payload = load_f32(&a[2]);
    let n_vecs = payload.len() / 128;
    let margin: f32 = if a.len() > 7 { a[7].parse().expect("margin") } else { 3.0 };
    let w128 = a.len() > 8 && a[8] == "w128";
    if w128 { eprintln!("mode: weighted prefix-128 hamming LUT"); }
    let mut r = BufReader::with_capacity(8 << 20, File::open(&a[3]).expect("open qcodes"));
    let nq = read_u64(&mut r).expect("n") as usize;
    let mut qcodes = vec![0u8; nq * HASH512_BYTES];
    r.read_exact(&mut qcodes).expect("qcodes");
    let mut gtf = File::open(&a[4]).expect("gt");
    let mut gt = vec![[0u64; 10]; nq];
    for row in gt.iter_mut() {
        for id in row.iter_mut() { *id = read_u64(&mut gtf).expect("gt"); }
    }
    let qraw = load_f32(&a[5]);

    // ---------- calibration: P(L2 | dH) on node pairs ----------
    // xorshift for reproducibility
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rng = move || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    let mut sum = vec![0f64; NBINS];
    let mut sq = vec![0f64; NBINS];
    let mut cnt = vec![0u64; NBINS];
    let n_pairs = 400_000usize;
    for t in 0..n_pairs {
        let i = (rng() % n_vecs as u64) as usize;
        let near = t % 2 == 0;
        let j = if near {
            // neighbor regime: random neighbor of i from layer 0
            let nb = g.neighbors(i as u32, 0);
            if nb.is_empty() { (rng() % n_vecs as u64) as usize } else { nb[(rng() as usize) % nb.len()] as usize }
        } else {
            (rng() % n_vecs as u64) as usize
        };
        if i == j { continue; }
        let dh = if w128 {
            h128(&g.node_hash(i as u32)[..16], &g.node_hash(j as u32)[..16]) as usize
        } else {
            hamming_distance(g.node_hash(i as u32), g.node_hash(j as u32)) as usize
        };
        let d = l2(&payload[i * 128..i * 128 + 128], &payload[j * 128..j * 128 + 128]) as f64;
        sum[dh] += d; sq[dh] += d * d; cnt[dh] += 1;
    }
    // LB estimate per bin: mean - margin*sigma (sigma from sample std)
    let mut lb = vec![f32::MAX; NBINS];
    for b in 0..NBINS {
        if cnt[b] >= 8 {
            let m = sum[b] / cnt[b] as f64;
            let var = (sq[b] / cnt[b] as f64 - m * m).max(0.0);
            lb[b] = (m - (margin as f64) * var.sqrt()) as f32;
        } else if b > 0 {
            lb[b] = lb[b - 1]; // fill sparse bins conservatively from below
        }
    }
    let mut calibrated_max = 0usize;
    for b in 0..NBINS { if cnt[b] >= 8 { calibrated_max = b; } }
    eprintln!("calibration: {} pairs, bins>=8 up to dH={}, margin={}", n_pairs, calibrated_max, margin);

    // ---------- query eval ----------
    let cells = [(256usize, 500usize), (1000, 1000)];
    let mut lines = Vec::new();
    for &(ef, k) in &cells {
        for i in 0..20.min(nq) {
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            let _ = g.search_with_ef(q, k, ef);
        }
        let mut f_rec = 0f64; let mut s_rec = 0f64; let mut skips = 0u64; let mut total = 0u64;
        let mut f_vt = Vec::new(); let mut s_vt = Vec::new();
        for i in 0..nq {
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            let qq = &qraw[i * 128..i * 128 + 128];
            let top = g.search_with_ef(q, k, ef);
            // full verify
            let t0 = std::time::Instant::now();
            let mut scored: Vec<(f32, u64)> = top.iter()
                .map(|&(d, idx, _)| (l2(&payload[g.node_label(idx) as usize * 128..g.node_label(idx) as usize * 128 + 128], qq), g.node_label(idx)))
                .collect();
            scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            scored.truncate(10);
            f_vt.push(t0.elapsed().as_micros());
            f_rec += scored.iter().filter(|(_, id)| gt[i].contains(id)).count() as f64 / 10.0;
            // skip verify: iterate in hamming order (top is sorted by d), maintain 10th-best
            let t1 = std::time::Instant::now();
            let mut best: Vec<(f32, u64)> = Vec::with_capacity(10);
            for &(dh, idx, _) in &top {
                let bound = if w128 {
                    let d128 = h128(&qcodes[i * HASH512_BYTES..i * HASH512_BYTES + 16], &g.node_hash(idx)[..16]) as usize;
                    let bb = d128.min(NBINS - 1);
                    if bb <= calibrated_max { lb[bb] } else { lb[calibrated_max] }
                } else {
                    let bb = (dh as usize).min(NBINS - 1);
                    if bb <= calibrated_max { lb[bb] } else { lb[calibrated_max] }
                };
                let worst = if best.len() < 10 { f32::MAX } else { best.last().unwrap().0 };
                total += 1;
                if bound > worst { skips += 1; continue; } // certificate: skip float verify
                let id = g.node_label(idx);
                let d = l2(&payload[id as usize * 128..id as usize * 128 + 128], qq);
                let pos = best.partition_point(|&(bd, _)| bd < d);
                if pos < 10 { best.insert(pos.min(best.len()), (d, id)); best.truncate(10); }
            }
            s_vt.push(t1.elapsed().as_micros());
            s_rec += best.iter().filter(|(_, id)| gt[i].contains(id)).count() as f64 / 10.0;
        }
        let med = |v: &mut Vec<u128>| { v.sort_unstable(); v[v.len() / 2] };
        let (mut fv, mut sv) = (f_vt.clone(), s_vt.clone());
        let line = format!(
            "\"ef{ef}_K{k}\":{{\"full_rec\":{:.4},\"skip_rec\":{:.4},\"skip_rate\":{:.3},\"full_vmed_us\":{},\"skip_vmed_us\":{}}}",
            f_rec / nq as f64, s_rec / nq as f64,
            if total > 0 { skips as f64 / total as f64 } else { 0.0 },
            med(&mut fv), med(&mut sv));
        eprintln!("{line}");
        lines.push(line);
    }
    let out = format!("{{{}}}", lines.join(","));
    let mut fo = File::create(&a[6]).unwrap();
    fo.write_all(out.as_bytes()).unwrap();
    println!("{out}");
}
