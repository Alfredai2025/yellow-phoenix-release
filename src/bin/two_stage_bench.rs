// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Two-stage / three-stage ANN bench.
//! Stage 1: 512-bit binary HNSW proposes top-K candidates.
//! Stage 2 (optional, --qc128 <file>): narrow to top-NARROW by 1024-bit hamming.
//! Stage 3: exact float32 L2 verify, top-10 reported.
//! Measures recall@10 vs GT + p50/p95/QPS per (ef, K) point.
//!
//! Usage:
//!   two_stage_bench <graph.bin> <payload.f32> <queries.ism> <queries.f32> <gt.bin> <out.json> [--qc128 <q128file>] [--narrow 50]
//! payload.f32: n*128 f32 LE, row index == graph node label
//! queries.ism: [u64 n][n*64B 512-bit codes] (ids block ignored, block layout)
//! queries.f32: n*128 f32 LE, same row order
//! q128file:    [u64 n][n*128B 1024-bit query codes], same row order
//! gt.bin:      n*10*u64 LE
use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn load_f32_matrix(path: &str, dim: usize) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32 matrix");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32 matrix");
    assert!(v.len() % (dim * 4) == 0, "f32 matrix size mismatch");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn load_codes(path: &str, rowbytes: usize) -> Vec<u8> {
    let mut r = BufReader::new(File::open(path).expect("open codes"));
    let n = read_u64(&mut r).expect("count") as usize;
    let mut v = vec![0u8; n * rowbytes];
    r.read_exact(&mut v).expect("codes");
    v
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    pams::simd_kernels::l2_sq_f32(a, b)
}

#[inline]
fn hamming128(a: &[u8], b: &[u8]) -> u32 {
    let mut s = 0u32;
    for i in 0..128 {
        s += (a[i] ^ b[i]).count_ones();
    }
    s
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let graph = BinaryHNSW::load(&a[1]).expect("load graph");
    let payload = load_f32_matrix(&a[2], 128);
    let n_vecs = payload.len() / 128;

    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(&a[3]).expect("open queries.ism"));
    let nq = read_u64(&mut r).expect("count") as usize;
    let mut qcodes = vec![0u8; nq * HASH512_BYTES];
    r.read_exact(&mut qcodes).expect("query codes");

    let qf = load_f32_matrix(&a[4], 128);
    assert!(qf.len() == nq * 128, "queries.f32 row mismatch");

    let mut gtf = File::open(&a[5]).expect("open gt");
    let mut gt = vec![[0u64; 10]; nq];
    for row in gt.iter_mut() {
        for id in row.iter_mut() {
            *id = read_u64(&mut gtf).expect("gt id");
        }
    }

    // optional 1024-bit codes: train sidecar + query codes + narrow width
    let side_train: Option<Vec<u8>> = a.iter().position(|s| s == "--sidecar128")
        .map(|p| load_codes(&a[p + 1], 128));
    let qc128: Option<Vec<u8>> = a.iter().position(|s| s == "--qc128")
        .map(|p| load_codes(&a[p + 1], 128));
    let narrow_to: usize = a.iter().position(|s| s == "--narrow")
        .and_then(|p| a.get(p + 1)).and_then(|s| s.parse().ok()).unwrap_or(50);
    if qc128.is_some() {
        assert!(side_train.is_some(), "--qc128 requires --sidecar128");
        assert!(qc128.as_ref().unwrap().len() == nq * 128, "qc128 row mismatch");
        assert!(side_train.as_ref().unwrap().len() == n_vecs * 128, "sidecar128 row mismatch");
        eprintln!("3-stage mode: narrow to top-{} by 1024-bit hamming", narrow_to);
    }

    let points: Vec<(usize, usize)> = vec![
        (64, 50), (64, 100),
        (128, 100), (128, 200),
        (256, 200), (256, 500),
        (500, 500), (1000, 1000),
    ];

    let mut results = Vec::new();
    for &(ef, k) in &points {
        let mut lat: Vec<u128> = Vec::with_capacity(nq);
        let mut recall = 0f64;
        for i in 0..50.min(nq) {
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            let _ = graph.search_with_ef(q, k, ef);
        }
        for i in 0..nq {
            let t0 = std::time::Instant::now();
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            let top = graph.search_with_ef(q, k, ef);
            let mut cand: Vec<u64> = top.iter().map(|(_, idx, _)| graph.node_label(*idx)).collect();
            if let (Some(q128), Some(strain)) = (&qc128, &side_train) {
                let qq = &q128[i * 128..(i + 1) * 128];
                cand.sort_by_key(|&id| hamming128(&strain[id as usize * 128..id as usize * 128 + 128], qq));
                cand.truncate(narrow_to.min(cand.len()));
            }
            let qqf = &qf[i * 128..(i + 1) * 128];
            let mut best: Vec<(f32, u64)> = cand.iter()
                .map(|&id| (l2(&payload[id as usize * 128..id as usize * 128 + 128], qqf), id))
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
        eprintln!("ef={ef} K={k}: recall={:.4} p50={:.1}us QPS={:.0}", recall / nq as f64, p50, qps);
        results.push(format!(
            "{{\"ef\":{ef},\"K\":{k},\"recall_at_10\":{:.6},\"p50_us\":{:.2},\"p95_us\":{:.2},\"p99_us\":{:.2},\"qps\":{:.1}}}",
            recall / nq as f64, p50, p95, p99, qps));
    }

    let out = format!("{{\"n_queries\":{nq},\"n_vecs\":{n_vecs},\"three_stage\":{},\"points\":[{}]}}",
        qc128.is_some(), results.join(","));
    let mut fo = File::create(&a[6]).expect("create out");
    fo.write_all(out.as_bytes()).unwrap();
    println!("{}", out);
}
