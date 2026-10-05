// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Bench Int8HNSW: raw query -> honest int8-L2 search -> top-K -> exact float verify -> recall@10.
//! Usage: int8_bench <graph.bin> <payload.f32> <queries.f32> <gt.bin> <out.json> [ef] [K] [alpha] [savepath]

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use pams::int8_hnsw::I8Hnsw;

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

fn main() {
    let a: Vec<String> = env::args().collect();
    let mut g = I8Hnsw::load(&a[1]).expect("load graph");
    let payload = load_f32(&a[2]);
    let n_vecs = payload.len() / 128;
    let qf = load_f32(&a[3]);
    let nq = qf.len() / 128;
    let mut gtf = File::open(&a[4]).expect("open gt");
    let mut gt = vec![[0u64; 10]; nq];
    for row in gt.iter_mut() {
        for id in row.iter_mut() { *id = read_u64(&mut gtf).expect("gt"); }
    }
    let ef: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(128);
    let k: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(200);
    let alpha: f32 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    if std::env::var("I8_PRE").is_ok() { g.pre_centered = true; eprintln!("PRE-CENTERED distance mode"); }
    if std::env::var("I8_GATEWAY").is_ok() { g.use_gateway = true; eprintln!("T7 cell-gateway entry mode"); }
    let mut p2: Option<[f32; 256]> = None;
    if let Ok(path) = std::env::var("I8_P2") {
        let raw = load_f32(&path);
        assert!(raw.len() >= 256, "P2 file");
        let mut arr = [0f32; 256];
        arr.copy_from_slice(&raw[..256]);
        g.attach_sketch(&arr);
        p2 = Some(arr);
        eprintln!("T6 sketch fusion attached");
    }
    let sk_shift: u32 = std::env::var("I6_SHIFT").ok().and_then(|v| v.parse().ok()).unwrap_or(18);
    if alpha > 0.0 {
        let removed = if std::env::var("I8_CCEP").is_ok() { g.prune_diverse_ccep(alpha) } else { g.prune_diverse(alpha) };
        eprintln!("int8 prune alpha={}: removed {} edges", alpha, removed);
        if let Some(p) = a.get(9) { if !p.is_empty() { g.save(p).expect("save"); eprintln!("saved {}", p); } }
    }
    let presence_mode = std::env::var("I8_PRESENCE").is_ok();
    let mut lat: Vec<u128> = Vec::with_capacity(nq);
    let mut recall = 0f64;
    let mut presence = 0f64;
    for i in 0..50.min(nq) {
        let q = &qf[i * 128..(i + 1) * 128];
        let mut ctx = g.encode_query(q);
        let _ = g.search_with_ef(&mut ctx, k, ef);
    }
    for i in 0..nq {
        let q = &qf[i * 128..(i + 1) * 128];
        let t0 = std::time::Instant::now();
        let mut ctx = g.encode_query(q);
        let top = g.search_with_ef_fused(&mut ctx, k, ef, &p2, sk_shift);
        let cand: Vec<u64> = top.iter().map(|&(_, idx)| g.node_label(idx)).collect();
        if presence_mode {
            presence += cand.iter().filter(|id| gt[i].contains(id)).count() as f64 / 10.0;
        }
        let mut best: Vec<(f32, u64)> = cand.iter()
            .map(|&id| (l2(&payload[id as usize * 128..id as usize * 128 + 128], q), id))
            .collect();
        best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        best.truncate(10);
        lat.push(t0.elapsed().as_micros());
        recall += best.iter().filter(|(_, id)| gt[i].contains(id)).count() as f64 / 10.0;
    }
    lat.sort_unstable();
    let p50 = lat[nq / 2] as f64;
    let p95 = lat[nq * 95 / 100] as f64;
    let p99 = lat[(nq as f64 * 0.99) as usize] as f64;
    let qps = 1e6 / (lat.iter().map(|&x| x as f64).sum::<f64>() / nq as f64);
    if presence_mode { eprintln!("PRESENCE@{} in raw beam = {:.4}", k, presence / nq as f64); }
    let arm = if alpha > 0.0 { format!("pruned_a{}", alpha) } else { "base".into() };
    let out = format!(
        "{{\"kind\":\"int8hnsw\",\"arm\":\"{arm}\",\"ef\":{ef},\"K\":{k},\"recall_at_10\":{:.6},\"p50_us\":{:.2},\"p95_us\":{:.2},\"p99_us\":{:.2},\"qps\":{:.1},\"n_queries\":{nq},\"n_vecs\":{n_vecs}}}",
        recall / nq as f64, p50, p95, p99, qps);
    let mut fo = File::create(&a[5]).expect("create out");
    fo.write_all(out.as_bytes()).unwrap();
    println!("{}", out);
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    pams::simd_kernels::l2_sq_f32(a, b)
}
