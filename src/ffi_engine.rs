// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Unified query engine: single FFI hop for HNSW search + cosine re-rank.
//!
//! This module wraps an existing `BinaryHNSW` handle and `RerankEngine` handle
//! (both owned by Python) in a synchronized context, then exposes one call that
//! does HNSW search, optional cosine re-rank over mmap'd embeddings, and
//! optional fast-mode (HNSW-only) return.

use std::ffi::c_void;
use std::ptr::NonNull;
use std::slice;
use std::sync::Mutex;

use hashbrown::HashMap;

use crate::binary_hnsw::{BinaryHNSW, HASH512_BITS, HASH512_BYTES};
use crate::rerank_engine::RerankEngine;

/// Synchronized context that borrows (does not own) a `BinaryHNSW` and a
/// `RerankEngine`. Python creates the underlying handles and keeps them alive
/// for the lifetime of the context.
pub struct EngineContext {
    /// Borrowed HNSW handle. Access is serialized by `guard`.
    hnsw: NonNull<BinaryHNSW>,
    /// Borrowed re-rank engine handle. Access is serialized by `guard`.
    rerank: NonNull<RerankEngine>,
    /// Maps HNSW node id (rust_id / user id) to embedding row index.
    node_to_row: HashMap<u64, u64>,
    /// Maps embedding row index back to HNSW node id. Dense vector indexed by
    /// row; length equals the number of indexed papers.
    row_to_node: Vec<u64>,
    /// Serialize all queries through one context. Parallel queries are deferred
    /// to Phase 5 (frozen/read-only HNSW).
    guard: Mutex<()>,
}

// Safety: all mutable access to the borrowed handles is protected by `guard`.
unsafe impl Send for EngineContext {}
unsafe impl Sync for EngineContext {}

impl EngineContext {
    /// Create a context from externally-owned handles.
    ///
    /// # Safety
    /// `hnsw` and `rerank` must remain valid and alive until the context is
    /// freed. They must not be freed while the context is in use.
    pub unsafe fn new(hnsw: *mut BinaryHNSW, rerank: *mut RerankEngine) -> Option<Self> {
        if hnsw.is_null() || rerank.is_null() {
            return None;
        }
        Some(Self {
            hnsw: NonNull::new_unchecked(hnsw),
            rerank: NonNull::new_unchecked(rerank),
            node_to_row: HashMap::new(),
            row_to_node: Vec::new(),
            guard: Mutex::new(()),
        })
    }

    /// Build the node-id → row-index mapping used for re-ranking.
    ///
    /// * `node_ids` — HNSW user ids (e.g. rust_id(pid)).
    /// * `row_indices` — corresponding row in the mmap'd embedding matrix.
    pub fn build_id_map(&mut self, node_ids: &[u64], row_indices: &[u64]) {
        self.node_to_row.clear();
        self.node_to_row.reserve(node_ids.len());
        let mut max_row = 0usize;
        for (&node, &row) in node_ids.iter().zip(row_indices.iter()) {
            self.node_to_row.insert(node, row);
            if (row as usize) > max_row {
                max_row = row as usize;
            }
        }
        self.row_to_node.clear();
        self.row_to_node.resize(max_row + 1, u64::MAX);
        for (&node, &row) in node_ids.iter().zip(row_indices.iter()) {
            self.row_to_node[row as usize] = node;
        }
    }

    /// Single unified query.
    ///
    /// * `query_hash` — 64-byte packed ITQ hash.
    /// * `query_embedding` — L2-normalized query embedding (`D` floats).
    /// * `top_k` — number of final results requested.
    /// * `candidate_pool` — number of HNSW candidates to retrieve before re-rank.
    /// * `bypass_rerank` — if true, return HNSW Hamming results directly.
    /// * `out_ids` / `out_scores` — caller-allocated output buffers.
    ///
    /// Returns the number of results written.
    pub fn query_unified(
        &self,
        query_hash: &[u8; HASH512_BYTES],
        query_embedding: &[f32],
        top_k: usize,
        candidate_pool: usize,
        bypass_rerank: bool,
        out_ids: &mut [u64],
        out_scores: &mut [f32],
    ) -> usize {
        let _lock = self.guard.lock().unwrap();

        // SAFETY: we hold the lock and the handles are guaranteed valid by
        // the context's safety contract.
        let hnsw = unsafe { self.hnsw.as_ref() };

        let want = candidate_pool.max(top_k);
        let hnsw_results = hnsw.search(query_hash, want);

        if bypass_rerank {
            let cap = out_ids.len().min(hnsw_results.len()).min(top_k);
            for i in 0..cap {
                let (dist, node_idx, _tag) = hnsw_results[i];
                let id = hnsw.node(node_idx).map(|n| n.id).unwrap_or(0);
                out_ids[i] = id;
                out_scores[i] = 1.0 - (dist as f32 / HASH512_BITS as f32);
            }
            return cap;
        }

        let rerank = unsafe { self.rerank.as_ref() };

        // Map HNSW node ids to embedding row indices. Skip ids missing from
        // the map (should not happen for a consistent index).
        let mut candidates: Vec<u64> = Vec::with_capacity(hnsw_results.len());
        for (_dist, node_idx, _tag) in &hnsw_results {
            let id = hnsw.node(*node_idx).map(|n| n.id).unwrap_or(0);
            if id == 0 {
                continue;
            }
            if let Some(&row) = self.node_to_row.get(&id) {
                candidates.push(row);
            }
        }

        if candidates.is_empty() {
            return 0;
        }

        let ranked = rerank.rerank(&candidates, query_embedding, top_k);
        let cap = out_ids.len().min(ranked.len());
        for i in 0..cap {
            // rerank.rerank returns the candidate id we passed in, which is the
            // embedding row index. Map it back to the HNSW node id for Python.
            let row = ranked[i].0 as usize;
            let score = ranked[i].1;
            let node_id = if row < self.row_to_node.len() {
                self.row_to_node[row]
            } else {
                u64::MAX
            };
            out_ids[i] = node_id;
            out_scores[i] = score;
        }
        cap
    }
}

/// Create a unified engine context from existing HNSW and re-rank handles.
///
/// Returns an opaque pointer on success, null on failure. The returned context
/// does **not** take ownership of `hnsw_ptr` or `rerank_ptr`; Python must free
/// those separately after freeing this context.
#[no_mangle]
pub extern "C" fn yp_engine_context_new(
    hnsw_ptr: *mut c_void,
    rerank_ptr: *mut c_void,
) -> *mut c_void {
    let ctx = match unsafe { EngineContext::new(hnsw_ptr as *mut BinaryHNSW, rerank_ptr as *mut RerankEngine) } {
        Some(c) => Box::new(c),
        None => return std::ptr::null_mut(),
    };
    Box::into_raw(ctx) as *mut c_void
}

/// Free a context previously returned by `yp_engine_context_new`.
/// This does **not** free the underlying HNSW or re-rank handles.
#[no_mangle]
pub extern "C" fn yp_engine_context_free(ctx_ptr: *mut c_void) {
    if ctx_ptr.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(ctx_ptr as *mut EngineContext);
    }
}

/// Build the node-id → row-index mapping inside an existing context.
///
/// * `node_ids` — flat array of `n` HNSW user ids.
/// * `row_indices` — flat array of `n` embedding row indices.
/// Returns 0 on success, -1 on null arguments.
#[no_mangle]
pub extern "C" fn yp_engine_context_build_id_map(
    ctx_ptr: *mut c_void,
    node_ids: *const u64,
    row_indices: *const u64,
    n: usize,
) -> i32 {
    if ctx_ptr.is_null() || node_ids.is_null() || row_indices.is_null() {
        return -1;
    }
    let ctx = unsafe { &mut *(ctx_ptr as *mut EngineContext) };
    let ids = unsafe { slice::from_raw_parts(node_ids, n) };
    let rows = unsafe { slice::from_raw_parts(row_indices, n) };
    ctx.build_id_map(ids, rows);
    0
}

/// Unified HNSW + re-rank query.
///
/// Output buffers are caller-allocated. Returns the number of results written
/// (at most `top_k` and `out_cap`).
#[no_mangle]
pub extern "C" fn yp_engine_query_unified(
    ctx_ptr: *mut c_void,
    query_hash: *const u8,
    query_hash_len: usize,
    query_embedding: *const f32,
    embedding_dim: usize,
    top_k: usize,
    candidate_pool: usize,
    bypass_rerank: u32,
    out_ids: *mut u64,
    out_scores: *mut f32,
    out_cap: usize,
) -> usize {
    if ctx_ptr.is_null()
        || query_hash.is_null()
        || query_embedding.is_null()
        || out_ids.is_null()
        || out_scores.is_null()
    {
        return 0;
    }
    if query_hash_len != HASH512_BYTES || embedding_dim == 0 || out_cap == 0 {
        return 0;
    }

    let ctx = unsafe { &*(ctx_ptr as *mut EngineContext) };

    let qhash = unsafe { slice::from_raw_parts(query_hash, query_hash_len) };
    let mut qhash_arr = [0u8; HASH512_BYTES];
    qhash_arr.copy_from_slice(qhash);

    let qemb = unsafe { slice::from_raw_parts(query_embedding, embedding_dim) };
    let out_ids = unsafe { slice::from_raw_parts_mut(out_ids, out_cap) };
    let out_scores = unsafe { slice::from_raw_parts_mut(out_scores, out_cap) };

    ctx.query_unified(
        &qhash_arr,
        qemb,
        top_k,
        candidate_pool,
        bypass_rerank != 0,
        out_ids,
        out_scores,
    )
}
