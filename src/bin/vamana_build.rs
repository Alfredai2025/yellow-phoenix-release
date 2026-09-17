// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Vamana-style graph construction over 512-bit hashes (single layer).
//!
//! Two variants (one build per invocation):
//!   plain  : RobustPrune with alpha-RNG pruning in Hamming space
//!   paged  : same, but neighbor SELECTION is scored by
//!            (hamming - LAMBDA * page_dist) so that edges prefer nodes in the
//!            same 4KB page under insertion-order layout; LAMBDA is auto-tuned
//!            (x2 steps) until >=60% of edges land in-page or LAMBDA cap hit.
//!
//! Usage:
//!   vamana_build <input.ism> <out.bin> <R> <L> <alpha_milli> <lambda_x10> [paged]
//!
//! Output format (all little-endian):
//!   magic "YPVA1" (5B) + version u8, u64 count, u32 R, u32 medoid,
//!   then count records of: hash[64] + deg u16 + pad u16 + neighbors u32[R]
//! Page geometry derives from the record length: nodes_per_page = 4096 / rec_len.
//! Prints a one-line JSON stats object at the end.

use std::collections::{HashSet, BinaryHeap};
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;

use pams::binary_hnsw::{hamming_distance, HASH512_BYTES};
use rand::seq::SliceRandom;
use rand::SeedableRng;

type Hash512 = [u8; HASH512_BYTES];

fn load_ism(path: &str) -> Vec<(u64, Hash512)> {
    let mut f = File::open(path).expect("open ISM");
    let mut cb = [0u8; 8];
    f.read_exact(&mut cb).expect("count");
    let count = u64::from_le_bytes(cb) as usize;
    let mut recs = Vec::with_capacity(count);
    let mut hb = [0u8; HASH512_BYTES];
    let mut ib = [0u8; 8];
    for _ in 0..count {
        f.read_exact(&mut hb).expect("hash");
        recs.push((0u64, hb));
    }
    for i in 0..count {
        f.read_exact(&mut ib).expect("id");
        recs[i].0 = u64::from_le_bytes(ib);
    }
    recs
}

struct Vamana {
    hashes: Vec<Hash512>,
    ids: Vec<u64>,
    out: Vec<Vec<u32>>,
    r: usize,
    alpha: f64,
    lambda: f64,        // hamming units per page step (0 = plain)
    nodes_per_page: usize,
    medoid: u32,
}

impl Vamana {
    /// Selection score for pruning: LOW is good. Equivalent to the spec's
    /// (hamming_quality - lambda*page_distance) with quality = -distance:
    ///   score = d + lambda * page_dist   (same-page neighbors favored)
    fn score(&self, p: usize, c: usize) -> f64 {
        let d = hamming_distance(&self.hashes[p], &self.hashes[c]) as f64;
        if self.lambda > 0.0 {
            let pd = (p as i64 - c as i64).abs() as usize / self.nodes_per_page.max(1);
            d + self.lambda * pd as f64
        } else {
            d
        }
    }
    fn raw_dist(&self, a: usize, b: usize) -> u32 {
        hamming_distance(&self.hashes[a], &self.hashes[b])
    }

    /// Beam search (Vamana GreedySearch): expand while the closest candidate
    /// beats the worst of the running best-L set. Returns best-L visited.
    fn greedy(&self, q: usize, l: usize) -> Vec<u32> {
        use std::cmp::Reverse;
        let mut visited: HashSet<u32> = HashSet::new();
        let mut cand: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new(); // min-heap
        let mut top: BinaryHeap<(u32, u32)> = BinaryHeap::new();           // max-heap, best-l
        let ep = self.medoid as usize;
        let d0 = self.raw_dist(q, ep);
        cand.push(Reverse((d0, ep as u32)));
        visited.insert(ep as u32);
        while let Some(Reverse((dc, c))) = cand.pop() {
            if top.len() >= l && dc > top.peek().map(|t| t.0).unwrap_or(u32::MAX) {
                break;
            }
            top.push((dc, c));
            if top.len() > l {
                top.pop();
            }
            for &j in &self.out[c as usize] {
                if visited.insert(j) {
                    let dj = self.raw_dist(q, j as usize);
                    if top.len() < l || dj < top.peek().map(|t| t.0).unwrap_or(u32::MAX) {
                        cand.push(Reverse((dj, j)));
                    }
                }
            }
        }
        let mut vis: Vec<(u32, u32)> = top.into_iter().collect();
        vis.sort();
        vis.into_iter().take(l.max(1)).map(|(_, v)| v).collect()
    }

    /// RobustPrune(p, cand, alpha, R). Selection order uses score(); the
    /// alpha-RNG elimination uses raw Hamming distances (as per Vamana).
    fn prune(&self, p: usize, cand: Vec<u32>, stats: &mut PruneStats) -> Vec<u32> {
        let mut cand: Vec<u32> = cand;
        cand.sort_by(|&a, &b| self.score(p, a as usize).partial_cmp(&self.score(p, b as usize)).unwrap());
        cand.dedup();
        cand.retain(|&c| c as usize != p);
        let mut keep: Vec<u32> = Vec::with_capacity(self.r);
        let mut removed = vec![false; cand.len()];
        for (i, &c) in cand.iter().enumerate() {
            if removed[i] {
                continue;
            }
            if keep.len() >= self.r {
                break;
            }
            keep.push(c);
            let dc = self.raw_dist(p, c as usize);
            for (j, &c2) in cand.iter().enumerate().skip(i + 1) {
                if removed[j] {
                    continue;
                }
                let d2 = self.raw_dist(p, c2 as usize);
                // RNG: c2 is covered by c iff d(c,c2) <= alpha * d(p,c2)... standard
                // formulation: remove c2 if d(c, c2) < alpha * d(p, c2) with alpha >= 1
                // (here: keep c2 only if it is NOT dominated).
                if (self.raw_dist(c as usize, c2 as usize) as f64) < self.alpha * d2 as f64 {
                    removed[j] = true;
                }
            }
        }
        let _ = stats;
        keep
    }
}

struct PruneStats;

fn same_page_frac(g: &Vamana) -> f64 {
    let mut same = 0u64;
    let mut tot = 0u64;
    for (i, nbrs) in g.out.iter().enumerate() {
        for &j in nbrs {
            tot += 1;
            if (i as i64 - j as i64).abs() < g.nodes_per_page.max(1) as i64 {
                same += 1;
            }
        }
    }
    if tot == 0 {
        0.0
    } else {
        same as f64 / tot as f64
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 7 {
        eprintln!("Usage: {} <in.ism> <out.bin> <R> <L> <alpha_milli> <lambda_x10> [paged]", args[0]);
        std::process::exit(1);
    }
    let recs = load_ism(&args[1]);
    let out_path = args[2].clone();
    let r: usize = args[3].parse().unwrap();
    let l: usize = args[4].parse().unwrap();
    let alpha = args[5].parse::<u64>().unwrap() as f64 / 1000.0;
    let paged = args.get(7).map(|s| s == "paged").unwrap_or(false);

    let n = recs.len();
    let rec_len = 64 + 2 + 2 + 4 * r;
    let nodes_per_page = (4096 / rec_len).max(1);
    eprintln!("vamana: n={} R={} L={} alpha={} paged={} nodes_per_page={}",
              n, r, l, alpha, paged, nodes_per_page);

    let mut lambda0 = args[6].parse::<u64>().unwrap() as f64 / 10.0;

    let t0 = Instant::now();
    // approximate medoid over two fixed samples
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5DEECE);
    use rand::seq::IndexedRandom;
    let all: Vec<u32> = (0..n as u32).collect();
    let samp_a: Vec<u32> = all.choose_multiple(&mut rng, 4096.min(n)).cloned().collect();
    let samp_b: Vec<u32> = all.choose_multiple(&mut rng, 2048.min(n)).cloned().collect();
    let mut medoid = 0u32;
    let mut best_avg = f64::MAX;
    for &a in &samp_a {
        let da: u64 = samp_b.iter().map(|&b| hamming_distance(&recs[a as usize].1, &recs[b as usize].1) as u64).sum();
        let avg = da as f64 / samp_b.len() as f64;
        if avg < best_avg {
            best_avg = avg;
            medoid = a;
        }
    }
    eprintln!("medoid={} avg_hamming={:.1}", medoid, best_avg);

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(&mut rng);

    // auto-tune lambda for the paged variant
    let mut lambda = lambda0;
    let mut frac = 0.0;
    for attempt in 0..4 {
        let mut g = Vamana {
            hashes: recs.iter().map(|r| r.1).collect(),
            ids: recs.iter().map(|r| r.0).collect(),
            out: vec![Vec::new(); n],
            r, alpha,
            lambda: if paged { lambda } else { 0.0 },
            nodes_per_page, medoid,
        };
        let tp = Instant::now();
        for pass in 0..2 {
            let ord: Vec<u32> = if pass == 0 { order.clone() } else { order.iter().rev().cloned().collect() };
            for (k, &p) in ord.iter().enumerate() {
                let mut cand = g.greedy(p as usize, l);
                cand.extend_from_slice(&g.out[p as usize]);
                let mut st = PruneStats;
                let keep = g.prune(p as usize, cand, &mut st);
                g.out[p as usize] = keep.clone();
                for &j in &keep {
                    g.out[j as usize].push(p);
                    if g.out[j as usize].len() > r {
                        let c2: Vec<u32> = g.out[j as usize].clone();
                        let pruned = g.prune(j as usize, c2, &mut st);
                        g.out[j as usize] = pruned;
                    }
                }
                if (k + 1) % 100000 == 0 {
                    eprintln!("  pass{} {}/{} ({:?})", pass, k + 1, n, tp.elapsed());
                }
            }
        }
        let build_s = t0.elapsed().as_secs_f64();
        frac = same_page_frac(&g);
        let avg_deg = g.out.iter().map(|v| v.len()).sum::<usize>() as f64 / n as f64;
        eprintln!("attempt {}: lambda={} same_page_frac={:.3} avg_deg={:.2} build_s={:.1}",
                  attempt, lambda, frac, avg_deg, build_s);

        if !paged || frac >= 0.60 || attempt == 3 {
            // serialize
            let mut f = File::create(&out_path).expect("create out");
            f.write_all(b"YPVA1").unwrap();
            f.write_all(&[1u8]).unwrap();
            f.write_all(&(n as u64).to_le_bytes()).unwrap();
            f.write_all(&(r as u32).to_le_bytes()).unwrap();
            f.write_all(&medoid.to_le_bytes()).unwrap();
            for i in 0..n {
                f.write_all(&g.hashes[i]).unwrap();
                f.write_all(&(g.out[i].len() as u16).to_le_bytes()).unwrap();
                f.write_all(&[0u8; 2]).unwrap();
                for &j in &g.out[i] {
                    f.write_all(&j.to_le_bytes()).unwrap();
                }
                for _ in g.out[i].len()..r {
                    f.write_all(&u32::MAX.to_le_bytes()).unwrap();
                }
            }
            let size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
            println!("{{\"build_s\":{:.1},\"n\":{},\"R\":{},\"L\":{},\"alpha\":{},\"lambda\":{},\"paged\":{},\"same_page_frac\":{:.4},\"avg_deg\":{:.3},\"file_bytes\":{},\"nodes_per_page\":{}}}",
                     build_s, n, r, l, alpha, lambda, paged, frac, avg_deg, size, nodes_per_page);
            return;
        }
        lambda *= 2.0;
        lambda0 = lambda;
    }
}
