// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Build a BinaryHNSW (YPH5v4) index from a FlatIndex-format ISM file.
//!
//! ISM layout: [u64 count LE][count x 64B hashes][count x 8B u64 ids]
//! (this is exactly what `yp_shootout_load_ism` / FlatIndex::load reads).
//!
//! Usage:
//!     cargo run --release --bin build_hnsw_from_flat_ism -- \
//!         <in.ism> <out.bin> [m] [ef_construction]

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES};

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <in.ism> <out.bin> [m] [ef_construction]", args[0]);
        std::process::exit(1);
    }
    let in_path = &args[1];
    let out_path = &args[2];
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32);
    let ef: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);

    eprintln!("Loading {} ...", in_path);
    let t0 = Instant::now();
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, File::open(in_path).expect("open ism"));
    let count = read_u64(&mut reader).expect("count") as usize;
    eprintln!("  {} records", count);

    // Read hashes then ids (same layout FlatIndex::load uses).
    let mut hashes = vec![0u8; count * HASH512_BYTES];
    reader.read_exact(&mut hashes).expect("hashes");
    let mut id_bytes = vec![0u8; count * 8];
    reader.read_exact(&mut id_bytes).expect("ids");
    let ids: Vec<u64> = id_bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    eprintln!("  loaded in {:.2}s", t0.elapsed().as_secs_f64());
    drop(id_bytes);

    eprintln!("Building BinaryHNSW (m={}, ef_construction={})...", m, ef);
    let mut hnsw = BinaryHNSW::with_params(m, ef, 128);
    let t1 = Instant::now();
    for i in 0..count {
        let mut h = [0u8; HASH512_BYTES];
        h.copy_from_slice(&hashes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES]);
        hnsw.insert(ids[i], h);
        if (i + 1) % 100_000 == 0 {
            eprintln!("  inserted {} / {} ({:.0}/s)", i + 1, count,
                      (i + 1) as f64 / t1.elapsed().as_secs_f64().max(1e-9));
        }
    }
    eprintln!("  built in {:.2}s", t1.elapsed().as_secs_f64());
    drop(hashes);

    hnsw.save(out_path).expect("save");
    let size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("Saved {} ({:.1} MB)", out_path, size as f64 / (1024.0 * 1024.0));
}
