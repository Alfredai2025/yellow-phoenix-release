// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Float-embedding functor measurements for YP Phase B.
//! F1: Hamming(512-bit hash) vs Cosine(float embedding)
//! F2: Spectral NN(embedding space) rank in HNSW(hash)

use crate::binary_hnsw::BinaryHNSW;
use crate::hybrid_mesh::{HybridMesh, Slot512};
use crate::math::tensor_spectral_float::TensorSpectralIndexFloat;
use rand::Rng;
use rand::thread_rng;
use std::collections::HashMap;

fn hamming_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn spectral_angle(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let cos = (dot / (na * nb)).clamp(-1.0, 1.0);
    cos.acos() as f64
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 { return (1.0, 0.0, 0.0); }
    let sx = x.iter().sum::<f64>();
    let sy = y.iter().sum::<f64>();
    let sxy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
    let sx2 = x.iter().map(|a| a * a).sum::<f64>();
    let d = n * sx2 - sx * sx;
    if d.abs() < 1e-12 { return (1.0, 0.0, 0.0); }
    let slope = (n * sxy - sx * sy) / d;
    let intercept = (sy - slope * sx) / n;
    let my = sy / n;
    let sst: f64 = y.iter().map(|yi| (yi - my).powi(2)).sum();
    let ssr: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2)).sum();
    let r2 = if sst < 1e-12 { 1.0 } else { 1.0 - (ssr / sst) };
    (slope, intercept, r2)
}

/// F1: Hamming distance on 512-bit hashes vs spectral angle on float embeddings.
pub fn measure_f1_lipschitz_float(
    mesh: &HybridMesh,
    embeddings: &[f32],
    dim: usize,
    sample_pairs: usize,
) -> Result<(f64, f64), String> {
    let fine_by_id: HashMap<u64, &Slot512> = mesh.fine.slots.iter().map(|s| (s.id, s)).collect();
    let all_ids: Vec<u64> = fine_by_id.keys().copied().collect();
    if all_ids.len() < 2 {
        return Err("F1: need >=2 docs".to_string());
    }
    let mut rng = thread_rng();
    let mut d_h = Vec::with_capacity(sample_pairs);
    let mut d_theta = Vec::with_capacity(sample_pairs);
    for _ in 0..sample_pairs {
        let id1 = all_ids[rng.random_range(0..all_ids.len())];
        let id2 = all_ids[rng.random_range(0..all_ids.len())];
        if id1 == id2 { continue; }
        let h = hamming_bytes(&fine_by_id[&id1].pap, &fine_by_id[&id2].pap) as f64;
        let emb1 = &embeddings[id1 as usize * dim..(id1 as usize + 1) * dim];
        let emb2 = &embeddings[id2 as usize * dim..(id2 as usize + 1) * dim];
        d_h.push(h);
        d_theta.push(spectral_angle(emb1, emb2));
    }
    if d_h.len() < 10 {
        return Err("F1: insufficient samples".to_string());
    }
    let (k1, _, r2) = linear_regression(&d_h, &d_theta);
    Ok((k1, r2))
}

/// F2: Spectral nearest-neighbor (embedding space) rank in HNSW (hash space).
/// The query document itself is excluded from the spectral NN.
pub fn measure_f2_bound_float(
    spectral: &TensorSpectralIndexFloat,
    hnsw_graph: &BinaryHNSW,
    mesh: &HybridMesh,
    embeddings: &[f32],
    dim: usize,
    queries: usize,
) -> Result<(f64, f64), String> {
    let fine_by_id: HashMap<u64, &Slot512> = mesh.fine.slots.iter().map(|s| (s.id, s)).collect();
    let all_ids: Vec<u64> = fine_by_id.keys().copied().collect();
    if all_ids.is_empty() {
        return Err("F2: mesh empty".to_string());
    }
    let mut rng = thread_rng();
    let mut ranks = Vec::with_capacity(queries);
    for _ in 0..queries {
        let qid = all_ids[rng.random_range(0..all_ids.len())];
        let qemb = &embeddings[qid as usize * dim..(qid as usize + 1) * dim];
        let spec_results = spectral.query(qemb, 2);
        let spec_nn = spec_results
            .iter()
            .find(|(_, id)| *id != qid)
            .map(|(_, id)| *id)
            .unwrap_or(qid);
        let pap = &fine_by_id[&qid].pap;
        let hnsw_res = hnsw_graph.search(pap, 500);
        let rank = hnsw_res.iter().position(|(_, idx, _)| {
            hnsw_graph.node(*idx).map(|n| n.id == spec_nn).unwrap_or(false)
        }).map(|p| p as f64).unwrap_or(50.0);
        ranks.push(rank);
    }
    let mean = ranks.iter().sum::<f64>() / ranks.len() as f64;
    ranks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95 = ranks[(((ranks.len() as f64) * 0.95) as usize).min(ranks.len() - 1)];
    Ok((mean, p95))
}
