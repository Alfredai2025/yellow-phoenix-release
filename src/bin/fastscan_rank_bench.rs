// M2 end-to-end: graph proposes -> PQ4-32B ADC ranks candidates -> float verify.
// Compares against the 1024-bit narrow stage: ADC ranking is ~exact for the
// codec's approximation, so candidate pools can grow (500-2000) at near-zero
// rank cost (scalar LUT today; block fast-scan only pays on full-pool scans).
//
//   fastscan_rank_bench <graph.bin> <payload.f32> <queries.ism> <queries.f32> <gt.bin> <out.json> \
//       --pq4 <codes.bin> --books <books.f32> [--points ef:K,ef:K,...]
#![allow(clippy::missing_safety_doc)]

use std::fs::File;
use std::io::{BufReader, Read, Write};

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn load_f32_matrix(path: &str, dim: usize) -> Vec<f32> {
    let mut buf = Vec::new();
    BufReader::with_capacity(8 * 1024 * 1024, File::open(path).expect("open f32"))
        .read_to_end(&mut buf)
        .unwrap();
    assert!(buf.len() % (dim * 4) == 0);
    buf.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn argpos(a: &[String], key: &str) -> Option<usize> {
    a.iter().position(|s| s == key)
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[inline]
fn top_iter(g: &BinaryHNSW, q: &[u8; HASH512_BYTES], k: usize, ef: usize) -> Vec<u32> {
    g.search_with_ef(q, k, ef).into_iter().map(|(_, idx, _)| idx).collect()
}

#[inline]
fn top_iter_from(g: &BinaryHNSW, q: &[u8; HASH512_BYTES], k: usize, ef: usize, ep: u32) -> Vec<u32> {
    g.search_with_ef_from(q, k, ef, ep).into_iter().map(|(_, idx, _)| idx).collect()
}

const D: usize = 128;

/// Parse --mdims "2:32,1:64" -> per-subq dims list (default: all 2-dim, M subq).
fn parse_mdims(spec: Option<&String>, m: usize) -> Vec<usize> {
    match spec {
        Some(s) => {
            let mut v = Vec::new();
            for part in s.split(',') {
                let mut it = part.split(':');
                let dim: usize = it.next().unwrap().parse().unwrap();
                let cnt: usize = it.next().unwrap().parse().unwrap();
                v.extend(std::iter::repeat(dim).take(cnt));
            }
            v
        }
        None => vec![2; m],
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() >= 7, "usage: fastscan_rank_bench <graph> <payload> <queries.ism> <queries.f32> <gt.bin> <out.json> --pq4 codes.bin --books books.f32 [--points ef:K,...]");
    let graph = BinaryHNSW::load(&a[1]).expect("load graph");
    let payload = load_f32_matrix(&a[2], D);
    let n_vecs = payload.len() / D;

    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(&a[3]).expect("open queries.ism"));
    let mut hdr = [0u8; 8];
    r.read_exact(&mut hdr).unwrap();
    let nq = u64::from_le_bytes(hdr) as usize;
    let mut qcodes = vec![0u8; nq * HASH512_BYTES];
    r.read_exact(&mut qcodes).unwrap();
    let qf = load_f32_matrix(&a[4], D);
    let mut gtbuf = Vec::new();
    BufReader::new(File::open(&a[5]).expect("open gt")).read_to_end(&mut gtbuf).unwrap();
    let gt: Vec<std::collections::HashSet<u64>> = gtbuf
        .chunks_exact(80)
        .map(|c| c.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect())
        .collect();

    let pq4 = std::fs::read(&a[argpos(&a, "--pq4").unwrap() + 1]).expect("read pq4 codes");
    let m_subq: usize = argpos(&a, "--subq").and_then(|p| a.get(p + 1)).and_then(|s| s.parse().ok()).unwrap_or(64);
    assert_eq!(pq4.len(), n_vecs * m_subq, "pq4 row mismatch");
    let mdims = parse_mdims(argpos(&a, "--mdims").and_then(|p| a.get(p + 1)), m_subq);
    assert_eq!(mdims.len(), m_subq);
    assert_eq!(mdims.iter().sum::<usize>(), D, "mdims must sum to 128");
    let moffs: Vec<usize> = {
        let mut v = Vec::with_capacity(m_subq + 1);
        let mut acc = 0;
        for &dm in &mdims { v.push(acc); acc += dm; }
        v.push(acc);
        v
    };
    let books = load_f32_matrix(&a[argpos(&a, "--books").unwrap() + 1], 2); // (M,16,2 max)
    assert_eq!(books.len(), m_subq * 16 * 2);
    // centroids are mean-centered: queries MUST be centered with the same mu.
    // --rot R.f32: optional OPQ rotation (128x128 row-major); query centered
    // then rotated, LUT built in rotated space against rotated books.
    let mu_path = argpos(&a, "--mu").map(|p| a[p + 1].clone());
    let mu: Vec<f32> = match &mu_path {
        Some(p) => load_f32_matrix(p, D),
        None => vec![0.0; D],
    };
    let rot: Vec<f32> = argpos(&a, "--rot")
        .map(|p| load_f32_matrix(&a[p + 1], D))
        .unwrap_or_else(|| {
            let mut id = vec![0f32; D * D];
            for d in 0..D { id[d * D + d] = 1.0; }
            id
        });

    let points: Vec<(usize, usize)> = argpos(&a, "--points")
        .and_then(|p| a.get(p + 1))
        .map(|s| {
            s.split(',')
                .map(|pair| {
                    let mut it = pair.split(':');
                    (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(64, 500), (128, 500), (128, 1000), (256, 1000), (256, 2000), (500, 2000)]);
    // council discrimination tests: --no-adc (verify raw graph candidates),
    // --verify N (float-verify top-N instead of top-50)
    let no_adc = argpos(&a, "--no-adc").is_some();
    let verify_n: usize = argpos(&a, "--verify")
        .and_then(|p| a.get(p + 1)).and_then(|s| s.parse().ok()).unwrap_or(50);
    // multi-probe: P independent walks from deterministic-diverse entry points
    let probes: usize = argpos(&a, "--probes").and_then(|p| a.get(p + 1)).and_then(|s| s.parse().ok()).unwrap_or(1);
    let pef: usize = argpos(&a, "--pef").and_then(|p| a.get(p + 1)).and_then(|s| s.parse().ok()).unwrap_or(200);
    let nnodes = graph.len() as u64;

    let mut results = Vec::new();
    for &(ef, k) in &points {
        // warm-up
        for i in 0..50.min(nq) {
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            let _ = graph.search_with_ef(q, k, ef);
        }
        let mut lat: Vec<u128> = Vec::with_capacity(nq);
        let mut recall = 0f64;
        for i in 0..nq {
            let t0 = std::time::Instant::now();
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            // candidate pool: single walk, or P probes merged by union on node idx
            let mut cand_idx: Vec<u32> = Vec::new();
            if probes <= 1 {
                cand_idx.extend(top_iter(&graph, q, k, ef));
            } else {
                for p in 0..probes as u64 {
                    // deterministic per-(query,probe) diverse entry: xorshift of i and p
                    let mut h: u64 = (i as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ p.wrapping_mul(0xC2B2AE3D27D4EB4F) ^ 0x165667B19E3779F9;
                    h ^= h >> 33; h = h.wrapping_mul(0xFF51AFD7ED558CCD); h ^= h >> 33;
                    let ep = (h % nnodes) as u32;
                    cand_idx.extend(top_iter_from(&graph, q, pef, pef, ep));
                }
                cand_idx.sort_unstable();
                cand_idx.dedup();
            }
            let cand: Vec<u64> = cand_idx.iter().map(|&ix| graph.node_label(ix)).collect();

            // per-query ADC LUT: lut[m][c] = ||q_sub - cent_m[c]||^2
            let qqf = &qf[i * D..(i + 1) * D];
            let mut qc = [0f32; D];
            for d in 0..D {
                let mut s = 0f32;
                for e in 0..D {
                    s += rot[d * D + e] * (qqf[e] - mu[e]);
                }
                qc[d] = s;
            }
            let mut lut = vec![[0f32; 16]; m_subq];
            for m in 0..m_subq {
                let off = moffs[m];
                for c in 0..16 {
                    let mut s = 0f32;
                    for dd in 0..mdims[m] {
                        let dv = qc[off + dd] - books[(m * 16 + c) * 2 + dd];
                        s += dv * dv;
                    }
                    lut[m][c] = s;
                }
            }
            // ADC-rank candidates (unless council test --no-adc), keep top-verify_n
            let mut scored: Vec<(f32, u64)> = if no_adc {
                cand.iter().map(|&id| (0f32, id)).collect()
            } else {
                cand.iter()
                    .map(|&id| {
                        let row = &pq4[id as usize * m_subq..id as usize * m_subq + m_subq];
                        let mut s = 0f32;
                        for m in 0..m_subq {
                            s += lut[m][row[m] as usize];
                        }
                        (s, id)
                    })
                    .collect()
            };
            scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            scored.truncate(verify_n.min(scored.len()));

            let mut best: Vec<(f32, u64)> = scored
                .iter()
                .map(|&(_, id)| (l2(&payload[id as usize * D..id as usize * D + D], qqf), id))
                .collect();
            best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            best.truncate(10);
            lat.push(t0.elapsed().as_micros());
            recall += best.iter().filter(|(_, id)| gt[i].contains(id)).count() as f64 / 10.0;
        }
        lat.sort_unstable();
        let p50 = lat[nq / 2] as f64;
        let p95 = lat[nq * 95 / 100] as f64;
        let qps = 1e6 / (lat.iter().map(|&x| x as f64).sum::<f64>() / nq as f64);
        eprintln!("ef={ef} K={k}: recall={:.4} p50={:.1}us QPS={:.0}", recall / nq as f64, p50, qps);
        results.push(format!(
            "{{\"ef\":{ef},\"K\":{k},\"recall_at_10\":{:.6},\"p50_us\":{:.2},\"p95_us\":{:.2},\"qps\":{:.1}}}",
            recall / nq as f64, p50, p95, qps));
    }

    let out = format!("{{\"n_queries\":{nq},\"n_vecs\":{n_vecs},\"stage2\":\"pq4_adc\",\"points\":[{}]}}", results.join(","));
    let mut fo = File::create(&a[6]).expect("create out");
    fo.write_all(out.as_bytes()).unwrap();
    println!("{}", out);
}
