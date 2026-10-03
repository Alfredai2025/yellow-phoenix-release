// SPDX-License-Identifier: AGPL-3.0-or-later
// bench_rerank_sweep.rs — device-equivalent +1024 re-rank harness (Mac leg).
// Reproduces the Swift bench's +1024 protocol exactly: 512-bit graph proposes
// top-K candidates; 1024-bit hamming (from sidecar_1k_49m.bin) re-orders them;
// S@10 = payload-target containment in the re-ranked top-10.
// Sweeps ef x K to measure the re-rank ceiling vs shortlist size.
// Query records: id(8B LE) + hash(64B) + h1k(128B) = 200B (queries_49m.bin).
// Sidecar: "YCSC" ver u64LE-count, records at 16+136*i = rowid LE (8B) + hash (128B),
// rowids 1..4.9M sorted contiguous (hexdump-verified).

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
        // sidecar rowids are INDEX rowids (ascending, with quarantine gaps) —
        // binary search like the Swift loader; direct rowid-1 indexing is WRONG
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: bench_rerank_sweep <graph.bin> <queries.bin> <sidecar.bin>");
        std::process::exit(2);
    }
    let hnsw = BinaryHNSW::load(&args[1]).expect("graph load");
    println!("[RERANK] graph: {} nodes", hnsw.len());
    let sc = Sidecar1k::load(&args[3]);
    println!("[RERANK] sidecar loaded");

    let raw = std::fs::read(&args[2]).expect("queries");
    assert!(raw.len() % REC == 0);
    let nq = raw.len() / REC;
    let mut ids = Vec::with_capacity(nq);
    let mut hashes: Vec<[u8; 64]> = Vec::with_capacity(nq);
    let mut h1ks: Vec<[u64; 16]> = Vec::with_capacity(nq);
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
    println!("[RERANK] {} queries", nq);

    // raw baseline (k=10) at each ef
    for &ef in &[50usize, 100, 200, 400] {
        hnsw.set_ef_search(ef);
        for i in 0..nq.min(20) { let _ = hnsw.search(&hashes[i], 10); }
        let (mut hits, mut lat) = (0usize, Vec::with_capacity(nq));
        for i in 0..nq {
            let t0 = Instant::now();
            let r = hnsw.search(&hashes[i], 10);
            lat.push(t0.elapsed().as_micros() as f64);
            if r.iter().take(10).any(|(_, ni, _)| hnsw.node(*ni).map(|n| n.id) == Some(ids[i])) { hits += 1; }
        }
        lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("[RERANK] RAW  ef={:3} S@10 {:5.1}% p50 {:5.0}us", ef, 100.0 * hits as f64 / nq as f64, lat[nq / 2]);
    }

    // rerank sweep
    for &ef in &[50usize, 100, 200, 400] {
        hnsw.set_ef_search(ef);
        for &k in &[200usize, 500, 1000] {
            for i in 0..nq.min(10) { let _ = hnsw.search(&hashes[i], k); }
            let (mut hits, mut lat) = (0usize, Vec::with_capacity(nq));
            for i in 0..nq {
                let t0 = Instant::now();
                let r = hnsw.search(&hashes[i], k);
                // rerank by 1024-bit hamming
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
                if scored.iter().take(10).any(|(_, id)| *id == ids[i]) { hits += 1; }
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("[RERANK] ef={:3} K={:4} +1024 S@10 {:5.1}% p50 {:5.0}us", ef, k, 100.0 * hits as f64 / nq as f64, lat[nq / 2]);
        }
    }
    println!("[RERANK] DONE");
}
