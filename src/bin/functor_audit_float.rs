// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use pams::binary_hnsw::BinaryHNSW;
use pams::hybrid_mesh::HybridMesh;
use pams::math::functor_bounds_float::{measure_f1_lipschitz_float, measure_f2_bound_float};
use pams::math::tensor_spectral_float::TensorSpectralIndexFloat;
use std::env;
use std::fs::File;
use std::io::{Read, Result as IoResult, Write};
use std::time::Instant;

fn main() -> IoResult<()> {
    let args: Vec<String> = env::args().collect();
    let bin_path = args.get(1).map(|s| s.as_str()).unwrap_or("data/paper_hashes_100k.bin");
    let emb_path = args.get(2).map(|s| s.as_str()).unwrap_or("data/paper_embeddings_100k.f32bin");

    let f1_pairs: usize = env::var("F1_PAIRS").ok().and_then(|s| s.parse().ok()).unwrap_or(10000);
    let f2_queries: usize = env::var("F2_QUERIES").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let spectral_k: usize = env::var("SPECTRAL_K").ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    // Load hashes + build mesh + HNSW
    let t0 = Instant::now();
    let records = read_bin_records(bin_path)?;
    let mut mesh = HybridMesh::new(1024, 65536);
    let mut hnsw = BinaryHNSW::new();
    for (id, pap_128, pap_512) in &records {
        mesh.insert_dual(*id, pap_128, pap_512);
        hnsw.insert(*id, *pap_512);
    }
    println!("[+] Loaded {} records, mesh+HNSW built in {:?}", records.len(), t0.elapsed());

    // Load float embeddings
    let (embeddings, dim) = read_f32bin(emb_path)?;
    println!("[+] Embeddings: {} vectors, dim={}", embeddings.len() / dim, dim);

    // Build spectral index on float embeddings
    let t0 = Instant::now();
    let spectral = TensorSpectralIndexFloat::build(&embeddings, dim, spectral_k);
    println!("[+] Spectral index (k={}) built in {:?}", spectral_k, t0.elapsed());

    // F1: Hamming vs Embedding Cosine
    let t0 = Instant::now();
    let (k1, r2) = measure_f1_lipschitz_float(&mesh, &embeddings, dim, f1_pairs)
        .unwrap_or((0.0, 0.0));
    println!("\nF1  Hamming-vs-Embedding  k1={:.6}  R\u{b2}={:.6}  ({:?})", k1, r2, t0.elapsed());

    // F2: Spectral-NN rank in HNSW
    let t0 = Instant::now();
    let (mean_rank, p95_rank) = measure_f2_bound_float(&spectral, &hnsw, &mesh, &embeddings, dim, f2_queries)
        .unwrap_or((50.0, 50.0));
    println!("F2  Spectral-NN-in-HNSW   mean={:.2}  p95={:.2}  ({:?})", mean_rank, p95_rank, t0.elapsed());

    // Summary JSON
    let out = format!(
        r#"{{"F1":{{"k1":{:.6},"r_squared":{:.6}}},"F2":{{"mean_rank":{:.2},"p95_rank":{:.2}}}}}"#,
        k1, r2, mean_rank, p95_rank
    );
    println!("\nJSON: {}", out);

    if let Ok(mut f) = File::create("logs/functor_audit_float.json") {
        let _ = f.write_all(out.as_bytes());
        println!("[+] Saved logs/functor_audit_float.json");
    }

    Ok(())
}

fn read_bin_records(path: &str) -> IoResult<Vec<(u64, [u8; 16], [u8; 64])>> {
    let mut f = File::open(path)?;
    let mut count_buf = [0u8; 4];
    f.read_exact(&mut count_buf)?;
    let count = u32::from_le_bytes(count_buf) as usize;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let mut id_buf = [0u8; 8];
        f.read_exact(&mut id_buf)?;
        let id = u64::from_le_bytes(id_buf);
        let mut pap128 = [0u8; 16];
        f.read_exact(&mut pap128)?;
        let mut pap512 = [0u8; 64];
        f.read_exact(&mut pap512)?;
        records.push((id, pap128, pap512));
    }
    Ok(records)
}

fn read_f32bin(path: &str) -> IoResult<(Vec<f32>, usize)> {
    let mut f = File::open(path)?;
    let mut header = [0u8; 16];
    f.read_exact(&mut header)?;
    let n = i64::from_le_bytes([header[0], header[1], header[2], header[3],
                                header[4], header[5], header[6], header[7]]) as usize;
    let dim = i64::from_le_bytes([header[8], header[9], header[10], header[11],
                                  header[12], header[13], header[14], header[15]]) as usize;
    let mut bytes = vec![0u8; n * dim * 4];
    f.read_exact(&mut bytes)?;
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((floats, dim))
}
