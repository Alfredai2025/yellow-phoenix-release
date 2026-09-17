// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Functor bounds measurement for YP Phase B
//! Computes empirical Lipschitz constants and order-preservation checks
//! using the real Rust engine APIs.

use crate::binary_hnsw::BinaryHNSW;
use crate::hybrid_mesh::{HybridMesh, Slot128, Slot512, PAP_128_BYTES, PAP_512_BYTES};
use crate::math::tensor_spectral_512::TensorSpectralIndex512;
use rand::Rng;
use rand::thread_rng;
use std::collections::HashMap;

// ==================== MATH HELPERS ====================

/// Hamming distance on byte slices (generic)
fn hamming_distance_bytes(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Spectral angle between two f32 vectors (arccos of normalized dot product)
fn spectral_angle(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    let cos = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    cos.acos() as f64
}

/// Linear regression: returns (slope, intercept, r_squared)
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = x.len() as f64;
    if n < 2.0 {
        return (1.0, 0.0, 0.0);
    }
    let sum_x = x.iter().sum::<f64>();
    let sum_y = y.iter().sum::<f64>();
    let sum_xy = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();
    let sum_x2 = x.iter().map(|a| a * a).sum::<f64>();
    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return (1.0, 0.0, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    let mean_y = sum_y / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - (slope * xi + intercept)).powi(2))
        .sum();
    let r2 = if ss_tot < 1e-12 {
        1.0
    } else {
        1.0 - (ss_res / ss_tot)
    };
    (slope, intercept, r2)
}

/// Kendall tau correlation between two rankings
fn kendall_tau(rank_a: &[usize], rank_b: &[usize]) -> f64 {
    let n = rank_a.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0usize;
    let mut discordant = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let a_order = rank_a[i].cmp(&rank_a[j]);
            let b_order = rank_b[i].cmp(&rank_b[j]);
            match (a_order, b_order) {
                (std::cmp::Ordering::Equal, _) | (_, std::cmp::Ordering::Equal) => {}
                (oa, ob) if oa == ob => concordant += 1,
                _ => discordant += 1,
            }
        }
    }
    let total = concordant + discordant;
    if total == 0 {
        0.0
    } else {
        (concordant as f64 - discordant as f64) / total as f64
    }
}

/// Convert a 512-bit PAP hash into a 512-dimensional 0/1 f32 vector.
fn hash512_to_f32(hash: &[u8; PAP_512_BYTES]) -> Vec<f32> {
    let mut out = Vec::with_capacity(PAP_512_BYTES * 8);
    for byte in hash.iter() {
        for bit in 0..8 {
            out.push(((byte >> bit) & 1) as f32);
        }
    }
    out
}

/// Geometric structure score: scalar part + 0.1 * bivector norm.
/// This mirrors `yp_geometric_score` in `ffi_unified.rs`.
fn geometric_score(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na2 = 0.0f32;
    let mut nb2 = 0.0f32;
    for i in 0..a.len() {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        na2 += ai * ai;
        nb2 += bi * bi;
    }
    let bivector_norm = (na2 * nb2 - dot * dot).max(0.0).sqrt();
    dot + 0.1 * bivector_norm
}

// ==================== F1: HASH -> SPECTRAL ====================

pub fn measure_f1_lipschitz(
    mesh: &HybridMesh,
    spectral: &TensorSpectralIndex512,
    sample_pairs: usize,
) -> Result<(f64, f64), String> {
    // F1 uses the full 512-bit hash Hamming distance vs the 512-bit spectral angle.
    let fine_by_id: HashMap<u64, &Slot512> =
        mesh.fine.slots.iter().map(|s| (s.id, s)).collect();
    let all_ids: Vec<u64> = fine_by_id.keys().copied().collect();
    if all_ids.len() < 2 {
        return Err("F1: need at least 2 documents in mesh.fine.slots".to_string());
    }

    let mut rng = thread_rng();
    let mut d_h = Vec::with_capacity(sample_pairs);
    let mut d_theta = Vec::with_capacity(sample_pairs);

    for _ in 0..sample_pairs {
        let id1 = all_ids[rng.random_range(0..all_ids.len())];
        let id2 = all_ids[rng.random_range(0..all_ids.len())];
        if id1 == id2 {
            continue;
        }

        let hash1 = &fine_by_id[&id1].pap;
        let hash2 = &fine_by_id[&id2].pap;
        let h = hamming_distance_bytes(hash1, hash2) as f64;

        let spec1 = spectral
            .vector_by_id(id1)
            .ok_or_else(|| format!("F1: missing spectral vector for id {id1}"))?;
        let spec2 = spectral
            .vector_by_id(id2)
            .ok_or_else(|| format!("F1: missing spectral vector for id {id2}"))?;
        let theta = spectral_angle(spec1, spec2);

        d_h.push(h);
        d_theta.push(theta);
    }

    if d_h.len() < 10 {
        return Err("F1: insufficient samples after filtering.".to_string());
    }

    let (k1, _intercept, r2) = linear_regression(&d_h, &d_theta);
    Ok((k1, r2))
}

// ==================== F2: SPECTRAL -> HNSW ====================

pub fn measure_f2_bound(
    spectral: &TensorSpectralIndex512,
    hnsw_graph: &BinaryHNSW,
    mesh: &HybridMesh,
    queries: usize,
) -> Result<(f64, f64), String> {
    let fine_by_id: HashMap<u64, &Slot512> =
        mesh.fine.slots.iter().map(|s| (s.id, s)).collect();
    let all_ids: Vec<u64> = fine_by_id.keys().copied().collect();
    if all_ids.is_empty() {
        return Err("F2: mesh.fine.slots is empty".to_string());
    }

    let mut rng = thread_rng();
    let mut ranks = Vec::with_capacity(queries);

    for _ in 0..queries {
        let query_id = all_ids[rng.random_range(0..all_ids.len())];
        let pap_512 = &fine_by_id[&query_id].pap;

        // Spectral nearest neighbor, excluding the query itself.
        let spec_results = spectral.query(pap_512, 2);
        let spec_nn_id = spec_results
            .iter()
            .find(|(_, id)| *id != query_id)
            .map(|(_, id)| *id)
            .unwrap_or(query_id);

        // HNSW search returns (hamming_distance, internal_node_index), sorted ascending
        let hnsw_results = hnsw_graph.search(pap_512, 50);
        let rank = hnsw_results
            .iter()
            .position(|(_, idx, _)| {
                hnsw_graph
                    .node(*idx)
                    .map(|n| n.id == spec_nn_id)
                    .unwrap_or(false)
            })
            .map(|pos| pos as f64)
            .unwrap_or(50.0);
        ranks.push(rank);
    }

    if ranks.is_empty() {
        return Err("F2: no measurements collected.".to_string());
    }

    let mean = ranks.iter().sum::<f64>() / ranks.len() as f64;
    ranks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p95_idx = ((ranks.len() as f64) * 0.95) as usize;
    let p95 = ranks[p95_idx.min(ranks.len() - 1)];

    Ok((mean, p95))
}

// ==================== F3: HNSW -> GEOMETRIC ====================

pub fn check_f3_order_preservation(
    hnsw_graph: &BinaryHNSW,
    mesh: &HybridMesh,
    test_queries: usize,
) -> Result<f64, String> {
    let fine_by_id: HashMap<u64, &Slot512> =
        mesh.fine.slots.iter().map(|s| (s.id, s)).collect();
    let all_ids: Vec<u64> = fine_by_id.keys().copied().collect();
    if all_ids.is_empty() {
        return Err("F3: mesh.fine.slots is empty".to_string());
    }

    let mut rng = thread_rng();
    let mut taus = Vec::with_capacity(test_queries);

    for _ in 0..test_queries {
        let query_id = all_ids[rng.random_range(0..all_ids.len())];
        let query_hash = &fine_by_id[&query_id].pap;
        let query_vec = hash512_to_f32(query_hash);

        let hnsw_results = hnsw_graph.search(query_hash, 20);
        if hnsw_results.len() < 4 {
            continue;
        }

        // HNSW ranking: 0 = closest (smallest Hamming distance)
        let mut hnsw_order: Vec<usize> = (0..hnsw_results.len()).collect();
        hnsw_order.sort_by(|a, b| hnsw_results[*a].0.cmp(&hnsw_results[*b].0));

        // Geometric scores for the same HNSW candidates
        let mut geo_scores: Vec<f32> = Vec::with_capacity(hnsw_results.len());
        for (_, idx, _) in &hnsw_results {
            let score = hnsw_graph
                .node(*idx)
                .and_then(|n| fine_by_id.get(&n.id))
                .map(|slot| geometric_score(&query_vec, &hash512_to_f32(&slot.pap)))
                .unwrap_or(0.0);
            geo_scores.push(score);
        }

        // Geometric ranking: 0 = highest geometric score
        let mut geo_order: Vec<usize> = (0..geo_scores.len()).collect();
        geo_order.sort_by(|a, b| geo_scores[*b].partial_cmp(&geo_scores[*a]).unwrap());

        taus.push(kendall_tau(&hnsw_order, &geo_order));
    }

    if taus.is_empty() {
        return Err("F3: no valid queries processed.".to_string());
    }

    let avg_tau = taus.iter().sum::<f64>() / taus.len() as f64;
    Ok(avg_tau)
}

// ==================== F4: END-TO-END DELTA ====================

pub fn compute_f4_delta(
    mesh: &HybridMesh,
    hnsw_graph: &BinaryHNSW,
    test_doc_ids: &[u64],
) -> Result<f64, String> {
    if test_doc_ids.is_empty() {
        return Err("F4: no test doc ids provided.".to_string());
    }

    let coarse_by_id: HashMap<u64, &Slot128> =
        mesh.coarse.slots.iter().map(|s| (s.id, s)).collect();
    let fine_by_id: HashMap<u64, &Slot512> =
        mesh.fine.slots.iter().map(|s| (s.id, s)).collect();

    let mut hash_correct = 0usize;
    let mut geo_correct = 0usize;

    for &id in test_doc_ids {
        let pap_128 = coarse_by_id
            .get(&id)
            .map(|s| &s.pap)
            .ok_or_else(|| format!("F4: missing 128-bit PAP for id {id}"))?;
        let pap_512 = fine_by_id
            .get(&id)
            .map(|s| &s.pap)
            .ok_or_else(|| format!("F4: missing 512-bit PAP for id {id}"))?;

        // Ground truth: nearest neighbour by exact 512-bit Hamming (excluding self)
        let mut best_id = id;
        let mut best_dist = u32::MAX;
        for slot in &mesh.fine.slots {
            if slot.id == id {
                continue;
            }
            let d = hamming_distance_bytes(pap_512, &slot.pap);
            if d < best_dist {
                best_dist = d;
                best_id = slot.id;
            }
        }
        let truth_id = best_id;

        // Hash/cascade retrieval via HybridMesh::query_auto (skip self if returned)
        let hash_results = mesh.query_auto(pap_128, pap_512, 10);
        let hash_top1 = hash_results.iter().find(|(_, result_id)| *result_id != id).map(|(_, result_id)| *result_id);
        if hash_top1 == Some(truth_id) {
            hash_correct += 1;
        }

        // Geometric retrieval: HNSW top-50 candidates reranked by geometric score (skip self)
        let hnsw_results = hnsw_graph.search(pap_512, 50);
        let query_vec = hash512_to_f32(pap_512);
        let mut scored: Vec<(f32, u64)> = hnsw_results
            .iter()
            .filter_map(|(_, idx, _)| {
                let node = hnsw_graph.node(*idx)?;
                if node.id == id {
                    return None;
                }
                let slot = fine_by_id.get(&node.id)?;
                let score = geometric_score(&query_vec, &hash512_to_f32(&slot.pap));
                Some((score, node.id))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let geo_top1 = scored.first().map(|(_, id)| *id);
        if geo_top1 == Some(truth_id) {
            geo_correct += 1;
        }
    }

    let n = test_doc_ids.len() as f64;
    let r1_hash = hash_correct as f64 / n;
    let r1_geo = geo_correct as f64 / n;
    let delta = r1_hash - r1_geo;

    Ok(delta)
}

// ==================== TESTS ====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let a = vec![0b11110000u8, 0b00001111];
        let b = vec![0b11110000u8, 0b00001111];
        assert_eq!(hamming_distance_bytes(&a, &b), 0);

        let c = vec![0b11110000u8, 0b00001111];
        let d = vec![0b00001111u8, 0b11110000];
        assert_eq!(hamming_distance_bytes(&c, &d), 16);
    }

    #[test]
    fn test_spectral_angle_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        let angle = spectral_angle(&v, &v);
        assert!(angle < 0.001);
    }

    #[test]
    fn test_spectral_angle_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let angle = spectral_angle(&a, &b);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 0.001);
    }

    #[test]
    fn test_kendall_tau_perfect() {
        let a = vec![0, 1, 2, 3];
        let b = vec![0, 1, 2, 3];
        assert!((kendall_tau(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_kendall_tau_reverse() {
        let a = vec![0, 1, 2, 3];
        let b = vec![3, 2, 1, 0];
        assert!((kendall_tau(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let (slope, _intercept, r2) = linear_regression(&x, &y);
        assert!((slope - 2.0).abs() < 0.001);
        assert!((r2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_linear_regression_noisy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let (slope, _intercept, r2) = linear_regression(&x, &y);
        assert!(slope > 1.8 && slope < 2.2);
        assert!(r2 > 0.95);
    }

    #[test]
    fn test_hash512_to_f32_and_geometric_score() {
        let mut h = [0u8; 64];
        h[0] = 0b00000011;
        let v = hash512_to_f32(&h);
        assert_eq!(v.len(), 512);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
        for x in &v[2..] {
            assert_eq!(*x, 0.0);
        }

        let score_identical = geometric_score(&v, &v);
        assert!(score_identical > 0.0);

        let zeros = vec![0.0f32; 512];
        let score_orthogonal = geometric_score(&v, &zeros);
        assert!(score_orthogonal.abs() < 0.001);
    }
}
