// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Graph evaluator for the kill-ceiling experiment (Experiment B).
//!
//! Usage:
//!   eval_graph <hnsw|vamana> <graph_path> <queries.ism> <gt.bin> <out.json>
//!
//! queries.ism: count u64 + n*64 hash bytes + n u64 ids (ids unused for recall,
//!              rows pair 1:1 with gt rows)
//! gt.bin:      n * 10 * u64 little-endian (top-10 GT ids restricted to the subset)
//!
//! For ef in {16,64,256}: k=10 search, strict set recall@10 vs GT,
//! median/p95 latency, getrusage(RUSAGE_SELF) minor-fault deltas, ru_maxrss.
//! Warm-up: first 50 queries discarded per ef level.

use std::collections::{BinaryHeap, HashSet};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

type Hash512 = [u8; HASH512_BYTES];

fn load_queries(path: &str) -> Vec<Hash512> {
    let mut f = File::open(path).expect("open queries");
    let mut cb = [0u8; 8];
    f.read_exact(&mut cb).expect("count");
    let n = u64::from_le_bytes(cb) as usize;
    let mut out = Vec::with_capacity(n);
    let mut hb = [0u8; HASH512_BYTES];
    for _ in 0..n {
        f.read_exact(&mut hb).expect("hash");
        out.push(hb);
    }
    out
}

fn load_gt(path: &str, n: usize) -> Vec<[u64; 10]> {
    let mut f = File::open(path).expect("open gt");
    let mut out = Vec::with_capacity(n);
    let mut b = [0u8; 8];
    for _ in 0..n {
        let mut row = [0u64; 10];
        for j in 0..10 {
            f.read_exact(&mut b).expect("gt id");
            row[j] = u64::from_le_bytes(b);
        }
        out.push(row);
    }
    out
}

struct VamanaG {
    n: usize,
    r: usize,
    medoid: u32,
    hashes: Vec<Hash512>,
    out: Vec<Vec<u32>>,
}

fn load_vamana(path: &str) -> VamanaG {
    let mut f = File::open(path).expect("open vamana");
    let mut magic = [0u8; 5];
    f.read_exact(&mut magic).expect("magic");
    assert_eq!(&magic, b"YPVA1", "bad magic");
    let mut b1 = [0u8; 1];
    f.read_exact(&mut b1).unwrap(); // version
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8).unwrap();
    let n = u64::from_le_bytes(b8) as usize;
    let mut b4 = [0u8; 4];
    f.read_exact(&mut b4).unwrap();
    let r = u32::from_le_bytes(b4) as usize;
    f.read_exact(&mut b4).unwrap();
    let medoid = u32::from_le_bytes(b4);
    let rec_len = 64 + 2 + 2 + 4 * r;
    let mut hashes = vec![[0u8; HASH512_BYTES]; n];
    let mut out = vec![Vec::new(); n];
    let mut rec = vec![0u8; rec_len];
    for i in 0..n {
        f.read_exact(&mut rec).expect("record");
        hashes[i].copy_from_slice(&rec[0..64]);
        let deg = u16::from_le_bytes([rec[64], rec[65]]) as usize;
        for j in 0..deg {
            let off = 68 + 4 * j;
            out[i].push(u32::from_le_bytes([rec[off], rec[off + 1], rec[off + 2], rec[off + 3]]));
        }
    }
    VamanaG { n, r, medoid, hashes, out }
}

fn getrusage() -> (libc::c_long, libc::c_long) {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        (ru.ru_minflt, ru.ru_maxrss)
    }
}

fn hdist(a: &Hash512, b: &Hash512) -> u32 {
    pams::binary_hnsw::hamming_distance(a, b)
}

fn vamana_search(g: &VamanaG, q: &Hash512, k: usize, ef: usize) -> Vec<u64> {
    use std::cmp::Reverse;
    let l = ef.max(k);
    let mut visited: HashSet<u32> = HashSet::new();
    let mut cand: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let mut top: BinaryHeap<(u32, u32)> = BinaryHeap::new();
    let d0 = hdist(q, &g.hashes[g.medoid as usize]);
    cand.push(Reverse((d0, g.medoid)));
    visited.insert(g.medoid);
    while let Some(Reverse((dc, c))) = cand.pop() {
        if top.len() >= l && dc > top.peek().map(|t| t.0).unwrap_or(u32::MAX) {
            break;
        }
        top.push((dc, c));
        if top.len() > l {
            top.pop();
        }
        for &j in &g.out[c as usize] {
            if visited.insert(j) {
                let dj = hdist(q, &g.hashes[j as usize]);
                if top.len() < l || dj < top.peek().map(|t| t.0).unwrap_or(u32::MAX) {
                    cand.push(Reverse((dj, j)));
                }
            }
        }
    }
    let mut vis: Vec<(u32, u32)> = top.into_iter().collect();
    vis.sort();
    vis.truncate(k);
    vis.iter().map(|&(_, v)| v as u64).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        eprintln!("Usage: {} <hnsw|vamana> <graph> <queries.ism> <gt.bin> <out.json>", args[0]);
        std::process::exit(1);
    }
    let kind = args[1].clone();
    let queries = load_queries(&args[3]);
    let gt = load_gt(&args[4], queries.len());
    let n = queries.len();

    // search closure returning top-10 ids
    let mut results = serde_json::json!({});
    let mut maxrss_kb: libc::c_long = 0;

    let mut run_level = |
        name: &str,
        search: &mut dyn FnMut(&Hash512, usize, usize) -> Vec<u64>,
        results: &mut serde_json::Value,
    | {
        for &ef in &[16usize, 64, 256] {
            let key = format!("{}_ef{}", name, ef);
            // warm-up 50
            for i in 0..n.min(50) {
                search(&queries[i], 10, ef);
            }
            let (flt0, _) = getrusage();
            let t0 = Instant::now();
            let mut lats = Vec::with_capacity(n);
            let mut recall = 0.0f64;
            for i in 0..n {
                let tq = Instant::now();
                let top = search(&queries[i], 10, ef);
                lats.push(tq.elapsed().as_nanos() as f64 / 1e3);
                let gset: HashSet<u64> = gt[i].iter().cloned().collect();
                let hit = top.iter().filter(|x| gset.contains(x)).count();
                recall += hit as f64 / 10.0;
            }
            let (flt1, rss) = getrusage();
            maxrss_kb = maxrss_kb.max(rss);
            lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = lats[n / 2];
            let p95 = lats[(n as f64 * 0.95) as usize];
            results[&key] = serde_json::json!({
                "ef": ef,
                "recall_at_10": recall / n as f64,
                "p50_us": p50,
                "p95_us": p95,
                "total_wall_s": t0.elapsed().as_secs_f64(),
                "minor_faults_total": flt1 - flt0,
                "minor_faults_per_query": (flt1 - flt0) as f64 / n as f64,
                "ru_maxrss_kb": rss,
            });
        }
    };

    match kind.as_str() {
        "hnsw" => {
            let g = BinaryHNSW::load(&args[2]).expect("load hnsw");
            let mut f = |q: &Hash512, k: usize, ef: usize| -> Vec<u64> {
                g.search_with_ef(q, k, ef).iter().map(|(_, idx, _)| g.node_label(*idx)).collect()
            };
            run_level("hnsw", &mut f, &mut results);
        }
        "vamana" => {
            let g = load_vamana(&args[2]);
            let mut f = |q: &Hash512, k: usize, ef: usize| -> Vec<u64> {
                // beam ef: emulate with candidate queue seeded ef-wide from medoid greedy
                vamana_search(&g, q, k, ef)
            };
            run_level("vamana", &mut f, &mut results);
        }
        _ => panic!("unknown kind"),
    }

    let file_bytes = std::fs::metadata(&args[2]).map(|m| m.len()).unwrap_or(0);
    let out = serde_json::json!({
        "kind": kind,
        "graph": args[2],
        "n_queries": n,
        "file_bytes": file_bytes,
        "ru_maxrss_bytes_max": maxrss_kb,
        "ru_maxrss_units": "bytes (macOS rusage convention)",
        "levels": results,
    });
    let mut fo = File::create(&args[5]).expect("create out");
    fo.write_all(serde_json::to_string_pretty(&out).unwrap().as_bytes()).unwrap();
    println!("{}", serde_json::to_string(&out).unwrap());
}
