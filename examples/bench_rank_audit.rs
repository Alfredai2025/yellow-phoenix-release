// SPDX-License-Identifier: AGPL-3.0-or-later
// bench_rank_audit.rs — rank-position audit for paper-3 fact-check (2026-10-03).
// Copies bench_rerank_sweep.rs (the certified Mac harness) and additionally
// records the RANK of the payload target (member) and the yardstick reference
// rank-1 (gt[0]) so S@1..S@10 curves and gt[0] containment can be verified
// against the paper. Results are NOT self-filtered, exactly like the certified
// harness; self-occupancy is reported separately.
//
// Caller supplies (matching queries.bin order):
//   /tmp/qpos_list.bin    u64 LE per query (payload qpos)
//   /tmp/own_rowids.bin   u64 LE per query (canonical_ids[qpos])

use pams::binary_hnsw::BinaryHNSW;
use memmap2::Mmap;
use std::fs::File;
use std::time::Instant;

const REC: usize = 8 + 64 + 128;

struct Sidecar1k {
    base: *const u8,
    #[allow(dead_code)]
    hold: Mmap,
}
unsafe impl Send for Sidecar1k {}
unsafe impl Sync for Sidecar1k {}
impl Sidecar1k {
    fn load(path: &str) -> Self {
        let f = File::open(path).expect("sidecar open");
        let m = unsafe { Mmap::map(&f).expect("sidecar mmap") };
        assert!(&m[0..4] == b"YCSC", "bad sidecar magic");
        let base = m.as_ptr();
        Sidecar1k { base, hold: m }
    }
    #[inline(always)]
    unsafe fn hash_for_rowid(&self, rowid: u64) -> Option<*const u64> {
        let n = u64::from_le_bytes(std::slice::from_raw_parts(self.base.add(5), 8).try_into().unwrap()) as usize;
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            let off = 16 + mid * 136;
            let stored = u64::from_le_bytes(std::slice::from_raw_parts(self.base.add(off), 8).try_into().unwrap());
            if stored == rowid {
                return Some(self.base.add(off + 8) as *const u64);
            } else if stored < rowid {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        None
    }
}

#[inline(always)]
fn ham1024(a: *const u64, b: &[u64; 16]) -> u32 {
    let mut d = 0u32;
    for k in 0..16 {
        d += unsafe { (*a.add(k)) ^ b[k] }.count_ones();
    }
    d
}

fn read_u64_le(path: &str) -> Vec<u64> {
    let b = std::fs::read(path).expect(path);
    assert!(b.len() % 8 == 0);
    b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn rank_of(list: &[u64], target: u64) -> Option<usize> {
    list.iter().position(|&x| x == target).map(|p| p + 1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: bench_rank_audit <graph.bin> <queries.bin> <sidecar.bin> <gt0_rowids.bin>");
        std::process::exit(2);
    }
    let hnsw = BinaryHNSW::load(&args[1]).expect("graph load");
    println!("[AUDIT] graph: {} nodes", hnsw.len());
    let sc = Sidecar1k::load(&args[3]);
    println!("[AUDIT] sidecar loaded");

    let raw = std::fs::read(&args[2]).expect("queries");
    assert!(raw.len() % REC == 0);
    let nq = raw.len() / REC;
    let mut ids = Vec::with_capacity(nq);
    let mut hashes: Vec<[u8; 64]> = Vec::with_capacity(nq);
    let mut h1ks = Vec::with_capacity(nq);
    for i in 0..nq {
        let o = i * REC;
        ids.push(u64::from_le_bytes(raw[o..o + 8].try_into().unwrap()));
        hashes.push(raw[o + 8..o + 72].try_into().unwrap());
        let mut h = [0u64; 16];
        for k in 0..16 {
            h[k] = u64::from_le_bytes(raw[o + 72 + k * 8..o + 80 + k * 8].try_into().unwrap());
        }
        h1ks.push(h);
    }
    println!("[AUDIT] {} queries", nq);

    let qpos_list = read_u64_le("/tmp/qpos_list.bin");
    let own_list = read_u64_le("/tmp/own_rowids.bin");
    let gt0_list = read_u64_le(&args[4]);
    assert!(qpos_list.len() == nq && own_list.len() == nq && gt0_list.len() == nq,
        "side lists must match nq");
    let gt0s: Vec<Option<u64>> = gt0_list.iter().map(|&g| if g == 0 { None } else { Some(g) }).collect();
    let gt0_n = gt0s.iter().filter(|g| g.is_some()).count();
    println!("[AUDIT] gt[0] resolved for {}/{} queries", gt0_n, nq);

    // RAW ef50 with rank histograms
    hnsw.set_ef_search(50);
    for i in 0..nq.min(20) { let _ = hnsw.search(&hashes[i], 10); }
    let mut mem_hist = [0usize; 11];
    let mut gt0_hist = [0usize; 11];
    let mut self_hits = 0usize;
    let mut lat: Vec<f64> = Vec::with_capacity(nq);
    for i in 0..nq {
        let t0 = Instant::now();
        let r = hnsw.search(&hashes[i], 10);
        lat.push(t0.elapsed().as_micros() as f64);
        let idlist: Vec<u64> = r.iter().map(|(_, ni, _)| hnsw.node(*ni).map(|n| n.id).unwrap_or(0)).collect();
        if idlist.iter().any(|&x| x == own_list[i]) { self_hits += 1; }
        match rank_of(&idlist, ids[i]) {
            Some(rk) if rk <= 10 => mem_hist[rk - 1] += 1,
            _ => mem_hist[10] += 1,
        }
        if let Some(g0) = gt0s[i] {
            match rank_of(&idlist, g0) {
                Some(rk) if rk <= 10 => gt0_hist[rk - 1] += 1,
                _ => gt0_hist[10] += 1,
            }
        }
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cum = |h: &[usize; 11], k: usize| -> f64 {
        100.0 * h[..k.min(10)].iter().sum::<usize>() as f64 / nq as f64
    };
    println!("[AUDIT] RAW ef50 member curve: S@1 {:.1} S@2 {:.1} S@3 {:.1} S@5 {:.1} S@10 {:.1} (absent {}) p50 {:.0}us",
        cum(&mem_hist, 1), cum(&mem_hist, 2), cum(&mem_hist, 3), cum(&mem_hist, 5), cum(&mem_hist, 10),
        mem_hist[10], lat[nq / 2]);
    println!("[AUDIT] self-occupancy: own record in raw top-10 for {}/{} queries", self_hits, nq);
    if gt0_n > 0 {
        let c = |k: usize| 100.0 * gt0_hist[..k.min(10)].iter().sum::<usize>() as f64 / gt0_n as f64;
        println!("[AUDIT] RAW ef50 gt0 curve (n={}): S@1 {:.1} S@3 {:.1} S@10 {:.1}",
            gt0_n, c(1), c(3), c(10));
    }

    // RAW k=2000 shortlist-membership curve (for the plateau figure: member rank in top-2000)
    hnsw.set_ef_search(50);
    for i in 0..nq.min(10) { let _ = hnsw.search(&hashes[i], 2000); }
    let mut k_hist = [0usize; 2001];
    for i in 0..nq {
        let r = hnsw.search(&hashes[i], 2000);
        let idlist: Vec<u64> = r.iter().map(|(_, ni, _)| hnsw.node(*ni).map(|n| n.id).unwrap_or(0)).collect();
        match rank_of(&idlist, ids[i]) {
            Some(rk) if rk <= 2000 => k_hist[rk] += 1,
            _ => {}
        }
    }
    for &k in &[200usize, 500, 1000, 2000] {
        let c: usize = k_hist[..=k].iter().sum();
        println!("[AUDIT] member in top-{} at ef50: {:.1}%", k, 100.0 * c as f64 / nq as f64);
    }

    // RERANK: member + gt0 S@10
    for &(ef, k) in &[(50usize, 200usize), (50, 500), (50, 1000), (400, 200)] {
        hnsw.set_ef_search(ef);
        for i in 0..nq.min(10) { let _ = hnsw.search(&hashes[i], k); }
        let (mut mem_hits, mut gt0_hits, mut lat) = (0usize, 0usize, Vec::with_capacity(nq));
        for i in 0..nq {
            let t0 = Instant::now();
            let r = hnsw.search(&hashes[i], k);
            let mut scored: Vec<(u32, u64)> = r.iter().map(|(_, ni, _)| {
                let id = hnsw.node(*ni).map(|n| n.id).unwrap_or(0);
                let d = match unsafe { sc.hash_for_rowid(id) } {
                    Some(p) => ham1024(p, &h1ks[i]),
                    None => u32::MAX,
                };
                (d, id)
            }).collect();
            scored.sort_by(|a, b| a.0.cmp(&b.0));
            lat.push(t0.elapsed().as_micros() as f64);
            let idlist: Vec<u64> = scored.iter().map(|&(_, id)| id).collect();
            if idlist.iter().take(10).any(|&x| x == ids[i]) { mem_hits += 1; }
            if let Some(g0) = gt0s[i] {
                if idlist.iter().take(10).any(|&x| x == g0) { gt0_hits += 1; }
            }
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("[AUDIT] ef={} K={} +1024 member S@10 {:.1}% gt0 S@10 {:.1}% (gt0 n={}) p50 {:.0}us",
            ef, k, 100.0 * mem_hits as f64 / nq as f64,
            if gt0_n > 0 { 100.0 * gt0_hits as f64 / gt0_n as f64 } else { 0.0 },
            gt0_n, lat[nq / 2]);
    }
    println!("[AUDIT] DONE");
}
