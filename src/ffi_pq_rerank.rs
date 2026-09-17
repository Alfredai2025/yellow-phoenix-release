// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge for the OPQ-PQ asymmetric-distance re-ranker.
//!
//! Mirrors the global-singleton pattern used by the other iOS FFI modules:
//! one PQ re-rank index is loaded per prefix and then queried with candidate
//! IDs supplied by the binary Hamming HNSW index.

use std::ffi::{c_char, CStr};
use std::sync::RwLock;

use crate::pq_rerank::PqRerankIndex;

static PQ_RERANK: RwLock<Option<PqRerankIndex>> = RwLock::new(None);

/// Load (or reload) the global PQ re-rank index.
///
/// `prefix` is the shared path prefix used by the three index files:
///   `{prefix}_opq.bin`
///   `{prefix}_codebooks.bin`
///   `{prefix}_index.bin`
/// An optional `{prefix}_ids.bin` may be supplied to map record IDs to rows.
///
/// Returns 0 on success, -1 on null/invalid path, -2 on IO error, -3 on format error.
#[no_mangle]
pub extern "C" fn yp_pq_rerank_load(prefix: *const c_char) -> i32 {
    if prefix.is_null() {
        return -1;
    }
    let path = unsafe {
        match CStr::from_ptr(prefix).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    match PqRerankIndex::load(path) {
        Ok(idx) => {
            *PQ_RERANK.write().unwrap() = Some(idx);
            0
        }
        Err(e) => {
            eprintln!("yp_pq_rerank_load failed for {}: {}", path, e);
            if e.contains("failed to read") || e.contains("failed to open") {
                -2
            } else {
                -3
            }
        }
    }
}

/// Unload the global PQ re-rank index, freeing its memory.
#[no_mangle]
pub extern "C" fn yp_pq_rerank_unload() -> i32 {
    *PQ_RERANK.write().unwrap() = None;
    0
}

/// Re-rank a candidate list using the loaded PQ index.
///
/// * `query` — pointer to `dim` floats (must be 384).
/// * `dim` — dimensionality of the query vector.
/// * `candidate_ids` — pointer to `n_candidates` record IDs from the Hamming index.
/// * `n_candidates` — number of candidates.
/// * `k` — number of re-ranked results desired.
/// * `out_ids` / `out_scores` — caller-allocated buffers of length `out_cap`.
/// * `out_cap` — capacity of the output buffers.
///
/// Returns the number of results written (at most `k` and `out_cap`).
/// Scores are cosine approximations in `[0, 1]`.
#[no_mangle]
pub extern "C" fn yp_pq_rerank_search(
    query: *const f32,
    dim: usize,
    candidate_ids: *const u64,
    n_candidates: usize,
    k: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_cap: usize,
) -> usize {
    if query.is_null()
        || candidate_ids.is_null()
        || out_ids.is_null()
        || out_scores.is_null()
        || out_cap == 0
        || n_candidates == 0
    {
        return 0;
    }

    let guard = PQ_RERANK.read().unwrap();
    let idx = match guard.as_ref() {
        Some(i) => i,
        None => return 0,
    };

    let q = unsafe { std::slice::from_raw_parts(query, dim) };
    let candidates = unsafe { std::slice::from_raw_parts(candidate_ids, n_candidates) };
    let results = idx.rerank(q, candidates, k.min(out_cap));

    let ids = unsafe { std::slice::from_raw_parts_mut(out_ids, out_cap) };
    let scores = unsafe { std::slice::from_raw_parts_mut(out_scores, out_cap) };
    for (i, (id, score)) in results.iter().enumerate() {
        ids[i] = *id;
        scores[i] = *score;
    }
    results.len()
}
