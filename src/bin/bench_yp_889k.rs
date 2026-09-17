// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! YP BinaryHNSW (ITQ-512) bench at 889K — controlled twin of
//! bench_mac_baselines_889k.py (same rows, same 256 held-out queries).
//!
//! Usage:
//!   cargo run --release --bin bench_yp_889k -- \
//!       <base.ism> <queries.ism> <out.json> [m] [ef_construction]
//!
//! ISM layout: [u64 count LE][count x 64B hashes][count x 8B u64 ids]

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_ism(path: &str) -> (Vec<[u8; HASH512_BYTES]>, Vec<u64>) {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path).expect("open"));
    let count = read_u64(&mut r).expect("count") as usize;
    let mut hashes = vec![[0u8; HASH512_BYTES]; count];
    for h in hashes.iter_mut() {
        r.read_exact(h).expect("hash");
    }
    let mut ids = Vec::with_capacity(count);
    for _ in 0..count {
        ids.push(read_u64(&mut r).expect("id"));
    }
    (hashes, ids)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <base.ism> <queries.ism> <out.json> [m] [ef_construction]", args[0]);
        std::process::exit(1);
    }
    let m: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let efc: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);

    let (hashes, ids) = read_ism(&args[1]);
    let (qhashes, qids) = read_ism(&args[2]);
    let n = hashes.len();
    let nq = qhashes.len();
    eprintln!("base {} queries {} | M={} ef_construction={}", n, nq, m, efc);

    let mut hnsw = BinaryHNSW::with_params(m, efc, 100);
    let t0 = Instant::now();
    for (i, h) in hashes.iter().enumerate() {
        hnsw.insert(ids[i], *h);
    }
    let build_s = t0.elapsed().as_secs_f64();
    let mem = hnsw.memory_bytes();
    eprintln!("build {:.1}s | {:.0} docs/s | {} bytes ({:.0} B/doc)",
              build_s, n as f64 / build_s, mem, mem as f64 / n as f64);

    let mut out = String::new();
    out.push_str(&format!(
        "{{\n  \"n_vectors\": {},\n  \"n_queries\": {},\n  \"m\": {},\n  \"ef_construction\": {},\n  \"build_s\": {:.1},\n  \"memory_bytes\": {},\n  \"bytes_per_doc\": {:.1},\n  \"ef_sweep\": [\n",
        n, nq, m, efc, build_s, mem, mem as f64 / n as f64
    ));

    let efs = [16u32, 32, 64, 128, 256, 512];
    for (ei, &ef) in efs.iter().enumerate() {
        hnsw.set_ef_search(ef as usize);
        // one throwaway query to warm the new ef path
        let _ = hnsw.search(&qhashes[0], 10);
        let mut lats = Vec::with_capacity(nq);
        let mut all_ids: Vec<Vec<u64>> = Vec::with_capacity(nq);
        for (qi, qh) in qhashes.iter().enumerate() {
            let t = Instant::now();
            let res = hnsw.search(qh, 10);
            lats.push(t.elapsed().as_secs_f64());
            let ids10: Vec<u64> = res
                .iter()
                .map(|&(_score, idx, _tag)| hnsw.node(idx).map(|nd| nd.id).unwrap_or(u64::MAX))
                .collect();
            let _ = qi;
            all_ids.push(ids10);
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = lats[nq / 2] * 1e6;
        let p95 = lats[(nq as f64 * 0.95) as usize] * 1e6;
        eprintln!("ef={:>3}  p50 {:>7.0} us  p95 {:>9.0} us", ef, p50, p95);
        out.push_str(&format!(
            "    {{\"ef\": {}, \"p50_us\": {:.1}, \"p95_us\": {:.1}, \"top10_ids\": [{}]}}{}\n",
            ef, p50, p95,
            all_ids.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>().join(","),
            if ei + 1 == efs.len() { "" } else { "," }
        ));
    }
    out.push_str("  ]\n}\n");
    File::create(&args[3]).expect("create out").write_all(out.as_bytes()).expect("write out");
    eprintln!("wrote {}", args[3]);
}
