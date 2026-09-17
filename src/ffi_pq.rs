// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge for 16-byte Product Quantization (PQ) Discovery Mode.
//!
//! This is intentionally a lightweight static singleton: one PQ index is loaded
//! globally and queried with `yp_pq_search`.  The API matches the iOS-side
//! expectation (`yp_pq_load` / `yp_pq_encode` / `yp_pq_search`) rather than the
//! older handle-based API.

use std::ffi::{c_char, CStr};
use std::sync::RwLock;

const PQ_BYTES: usize = 16;      // 16-byte PQ code
const PQ_CENTROIDS: usize = 256; // 8 bits per subquantizer
const PQ_DIMS: usize = 8;        // 8 subquantizers
const PQ_SUB_DIM_DEFAULT: usize = 48; // MiniLM-L6 output: 384 dims / 8 subqs

struct PQIndex {
    codes: Vec<[u8; PQ_BYTES]>, // one 16-byte code per vector
    ids: Vec<u64>,
    // Per-subquantizer flat centroid table: 256 centroids × sub_dim floats.
    centroids: Vec<Vec<f32>>,
    sub_dim: usize,
}

impl PQIndex {
    fn asymmetric_distance(&self, query: &[f32], code: &[u8; PQ_BYTES]) -> f32 {
        let mut dist = 0.0f32;
        for (subq, &ci) in code.iter().enumerate() {
            let centroid = &self.centroids[subq][ci as usize * self.sub_dim..(ci as usize + 1) * self.sub_dim];
            let q_sub = &query[subq * self.sub_dim..(subq + 1) * self.sub_dim];
            for (q, c) in q_sub.iter().zip(centroid.iter()) {
                let d = q - c;
                dist += d * d;
            }
        }
        dist
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.codes.is_empty() {
            return Vec::new();
        }
        let k = k.min(self.codes.len());
        let mut scored: Vec<(f32, u64)> = self
            .codes
            .iter()
            .zip(self.ids.iter())
            .map(|(code, &id)| (self.asymmetric_distance(query, code), id))
            .collect();
        scored.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .map(|(dist, id)| (id, 1.0 / (1.0 + dist.sqrt())))
            .collect()
    }
}

static PQ: RwLock<Option<PQIndex>> = RwLock::new(None);

/// Load (or reload) the global PQ index from disk.
///
/// TODO: parse real file format:
///     [n:u64][n×16 bytes codes][n×8 bytes IDs][centroids]
/// For now creates an empty placeholder index so the UI toggle can be wired
/// without blocking the build.
#[no_mangle]
pub extern "C" fn yp_pq_load(path: *const c_char) -> i32 {
    let _path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    let sub_dim = PQ_SUB_DIM_DEFAULT;
    *PQ.write().unwrap() = Some(PQIndex {
        codes: Vec::new(),
        ids: Vec::new(),
        centroids: vec![vec![0.0f32; PQ_CENTROIDS * sub_dim]; PQ_DIMS],
        sub_dim,
    });
    0
}

/// Encode a float embedding into a 16-byte PQ code.
///
/// TODO: real OPQ rotation + quantization against loaded centroids.
/// Stub: down-sample `dim` floats into 16 bytes.
#[no_mangle]
pub extern "C" fn yp_pq_encode(embedding: *const f32, dim: usize, out_code: *mut u8) -> i32 {
    if embedding.is_null() || out_code.is_null() || dim < PQ_BYTES {
        return -1;
    }
    let emb = unsafe { std::slice::from_raw_parts(embedding, dim) };
    let code = unsafe { std::slice::from_raw_parts_mut(out_code, PQ_BYTES) };
    let stride = dim / PQ_BYTES;
    for i in 0..PQ_BYTES {
        code[i] = emb[i * stride] as u8;
    }
    0
}

/// Search the loaded PQ index.
///
/// `query` must be at least `PQ_DIMS * sub_dim` floats (384 for MiniLM-L6).
/// Writes up to `out_cap` (id, score) pairs; score is in (0,1].
#[no_mangle]
pub extern "C" fn yp_pq_search(
    query: *const f32,
    dim: usize,
    k: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_cap: usize,
) -> usize {
    let guard = PQ.read().unwrap();
    let pq = match guard.as_ref() {
        Some(p) => p,
        None => return 0,
    };
    if query.is_null() || out_ids.is_null() || out_scores.is_null() || out_cap == 0 {
        return 0;
    }
    if dim % PQ_DIMS != 0 {
        return 0;
    }
    let q = unsafe { std::slice::from_raw_parts(query, dim) };
    let results = pq.search(q, k.min(out_cap));
    let n = results.len();
    let ids = unsafe { std::slice::from_raw_parts_mut(out_ids, out_cap) };
    let scores = unsafe { std::slice::from_raw_parts_mut(out_scores, out_cap) };
    for (i, (id, score)) in results.iter().enumerate() {
        ids[i] = *id;
        scores[i] = *score;
    }
    n
}
