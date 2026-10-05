// SPDX-License-Identifier: AGPL-3.0-or-later
//! Build a PQHNSW graph from raw vectors + PQ codec (books/V/mu).
//! Usage: build_pq_graph <payload.f32> <books.bin> <V.f32> <mu.f32> <itqW.f32> <itqmu.f32> <out.bin> [m] [efC] [fuse_w] [sketch_w] [seed]
//! Insertion order: shuffled (HNSW paper recommendation for graph quality).
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use pams::pq_hnsw::{PqCodec, PqHnsw};

fn load_f32(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let payload = load_f32(&a[1]);
    let d = 128;
    let n = payload.len() / d;
    // codec files
    let mut r = BufReader::new(File::open(&a[2]).expect("open books"));
    let mut h = [0u8; 12];
    r.read_exact(&mut h).unwrap();
    let nb = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize;
    let k = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize;
    let bl = u32::from_le_bytes([h[8], h[9], h[10], h[11]]) as usize;
    let mut bv = Vec::new();
    r.read_to_end(&mut bv).unwrap();
    let books: Vec<f32> = bv.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let v = load_f32(&a[3]);
    let mu = load_f32(&a[4]);
    assert!(v.len() == d * d && mu.len() == d);
    let itq_w = load_f32(&a[5]);
    let itq_mu = load_f32(&a[6]);
    let m: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(16);
    let efc: usize = a.get(9).and_then(|s| s.parse().ok()).unwrap_or(200);
    let fuse_w: f32 = a.get(10).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let sketch_w: f32 = a.get(11).and_then(|s| s.parse().ok()).unwrap_or(100.0);
    // transpose ITQ W (128x512 d-major) to 512x128 bit-major
    let mut itq_wt = vec![0f32; 512 * 128];
    for dd in 0..128 { for b in 0..512 { itq_wt[b * 128 + dd] = itq_w[dd * 512 + b]; } }
    let codec = PqCodec { nb, k, bl, books, v, mu, d };
    eprintln!("building fusion+sketch PQHNSW: n={n} nb={nb} m={m} efC={efc} fuse_w={fuse_w} sketch_w={sketch_w}");

    let mut g = PqHnsw::new(m, efc, codec, itq_wt, itq_mu, fuse_w, sketch_w);
    let mut order: Vec<u32> = (0..n as u32).collect();
    let seed: u64 = a.get(12).and_then(|s| s.parse().ok()).unwrap_or(0xDEADBEEF);
    let mut rng = seed;
    for i in (1..order.len()).rev() {  // Fisher-Yates
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let j = (rng % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let t0 = std::time::Instant::now();
    for (cnt, &i) in order.iter().enumerate() {
        g.insert(i as u64, &payload[i as usize * d..i as usize * d + d], &payload);
        if (cnt + 1) % 100_000 == 0 {
            eprintln!("  {} / {} ({:.0}/s)", cnt + 1, n,
                (cnt + 1) as f64 / t0.elapsed().as_secs_f64());
        }
    }
    g.save(&a[7]).expect("save");
    eprintln!("saved {} in {:.1}s", a[7], t0.elapsed().as_secs_f64());
}
