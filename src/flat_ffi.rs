// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge for Flat Array v0.4.

use crate::binary_hnsw::{BinaryHNSW, HASH512_BYTES};
use crate::flat_array::{FlatArray, FlatError};
use crate::flat_query_engine::FlatQueryEngine;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::{LazyLock, Mutex, RwLock};

static FLAT: LazyLock<RwLock<Option<FlatQueryEngine>>> =
    LazyLock::new(|| RwLock::new(None));
static HNSW_CHUNK: LazyLock<RwLock<Option<BinaryHNSW>>> =
    LazyLock::new(|| RwLock::new(None));
static LAST_ERROR: LazyLock<Mutex<Option<String>>> =
    LazyLock::new(|| Mutex::new(None));

fn flat_err_to_code(e: &FlatError) -> c_int {
    match e {
        FlatError::FileNotFound => 2,
        FlatError::InvalidSize => 2,
        FlatError::OutOfMemory => 3,
        FlatError::ChecksumMismatch => 4,
        FlatError::InvalidVersion => 4,
        FlatError::Io(_) => 2,
    }
}

/// Error slot for the shootout loaders (they historically returned bare -1
/// with no message; the recall runner surfaces this via `yp_last_error`).
pub fn set_shootout_error(msg: String) {
    set_error(msg);
}

fn set_error(msg: String) {
    *LAST_ERROR.lock().unwrap() = Some(msg);
}

/// Return the last loader error (empty string when none), then clear it.
/// The caller owns the returned string — release it with the existing
/// `yp_free_string` FFI.
#[no_mangle]
pub extern "C" fn yp_last_error() -> *mut c_char {
    let msg = LAST_ERROR.lock().unwrap().take().unwrap_or_default();
    CString::new(msg).unwrap_or_default().into_raw()
}

/// Unload the currently-loaded HNSW chunk (releases its mmap and node RAM).
/// Safe to call when nothing is loaded. Call BEFORE loading a different
/// corpus so two multi-GB graphs are never mapped at once — on iOS the
/// second mmap can hit the per-process VA ceiling and fail with ENOMEM,
/// which the loader reports as a generic "corrupt/truncated" error.
#[no_mangle]
pub extern "C" fn yp_hnsw_unload() -> c_int {
    *HNSW_CHUNK.write().unwrap() = None;
    0
}

fn hash_to_hex(hash: &[u8; 32]) -> String {
    hex::encode(hash)
}

/// Initialise (or reset) the global flat array singleton.
#[no_mangle]
pub extern "C" fn yp_flat_init() -> c_int {
    let mut guard = FLAT.write().unwrap();
    *guard = None;
    0
}

/// Build the flat array from a memory-mapped file of fixed-size records.
/// Each record is 8-byte LE id + 32-byte hash.
#[no_mangle]
pub extern "C" fn yp_flat_build_from_file(path: *const c_char, n: c_uint) -> c_int {
    if path.is_null() || n == 0 {
        set_error("invalid arguments".to_string());
        return 2;
    }
    let path = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => {
                set_error("invalid path encoding".to_string());
                return 2;
            }
        }
    };

    match FlatArray::from_mmap(path, n as usize) {
        Ok(array) => match FlatQueryEngine::new(array) {
            Ok(engine) => {
                let mut guard = FLAT.write().unwrap();
                *guard = Some(engine);
                0
            }
            Err(e) => {
                let msg = format!("failed to create query engine: {}", e);
                set_error(msg.clone());
                flat_err_to_code(&e)
            }
        },
        Err(e) => {
            let msg = format!("build failed: {}", e);
            set_error(msg.clone());
            flat_err_to_code(&e)
        }
    }
}

/// Query a single cell id. Returns a hex-encoded hash or null if not found / not initialized.
#[no_mangle]
pub extern "C" fn yp_flat_query(cell_id: u64) -> *mut c_char {
    let guard = FLAT.read().unwrap();
    let json = match guard.as_ref() {
        Some(engine) => match engine.query(cell_id) {
            Some(hash) => hash_to_hex(&hash),
            None => "null".to_string(),
        },
        None => "null".to_string(),
    };
    CString::new(json).unwrap_or_default().into_raw()
}

/// Query a batch of cell ids. Returns a JSON array of hex strings / nulls.
#[no_mangle]
pub extern "C" fn yp_flat_query_batch(cell_ids: *const u64, count: c_uint) -> *mut c_char {
    let guard = FLAT.read().unwrap();
    let json = match guard.as_ref() {
        Some(engine) => {
            let ids = unsafe { std::slice::from_raw_parts(cell_ids, count as usize) };
            let results = engine.query_batch(ids);
            let arr: Vec<serde_json::Value> = results
                .into_iter()
                .map(|r| match r {
                    Some(hash) => serde_json::Value::String(hash_to_hex(&hash)),
                    None => serde_json::Value::Null,
                })
                .collect();
            serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
        }
        None => "[]".to_string(),
    };
    CString::new(json).unwrap_or_default().into_raw()
}

/// Return JSON stats: hits, misses, avg_time_ns, batch_count, total_papers.
#[no_mangle]
pub extern "C" fn yp_flat_stats() -> *mut c_char {
    let guard = FLAT.read().unwrap();
    let json = match guard.as_ref() {
        Some(engine) => {
            let s = engine.stats();
            serde_json::json!({
                "hits": s.hits,
                "misses": s.misses,
                "avg_time_ns": s.avg_time_ns(),
                "total_time_ns": s.total_time_ns,
                "batch_count": s.batch_count,
                "total_papers": engine.len(),
                "memory_bytes": engine.memory_bytes(),
            })
            .to_string()
        }
        None => "{\"error\":\"not initialized\"}".to_string(),
    };
    CString::new(json).unwrap_or_default().into_raw()
}

/// Return JSON health report.
#[no_mangle]
pub extern "C" fn yp_flat_health() -> *mut c_char {
    let guard = FLAT.read().unwrap();
    let json = match guard.as_ref() {
        Some(engine) => {
            let h = engine.health();
            serde_json::json!({
                "status": h.status.to_string(),
                "corruption_detected": h.corruption_detected,
                "memory_pressure": h.memory_pressure,
                "circuit_open": h.circuit_open,
                "avg_query_ns": h.avg_query_ns,
            })
            .to_string()
        }
        None => "{\"error\":\"not initialized\"}".to_string(),
    };
    CString::new(json).unwrap_or_default().into_raw()
}

/// Persist the current flat array to disk atomically.
#[no_mangle]
pub extern "C" fn yp_flat_persist(path: *const c_char) -> c_int {
    if path.is_null() {
        return 2;
    }
    let path = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return 2,
        }
    };
    let guard = FLAT.read().unwrap();
    match guard.as_ref() {
        Some(engine) => match engine.array().persist(path) {
            Ok(()) => 0,
            Err(e) => {
                set_error(format!("persist failed: {}", e));
                flat_err_to_code(&e)
            }
        },
        None => 1,
    }
}

/// Free a string returned by this FFI module.
#[no_mangle]
pub extern "C" fn yp_flat_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Progressive loader: attempt to load up to `max_docs` documents from a binary
/// HNSW index file. The loaded index is stored in a global slot and can be
/// queried with `yp_index_doc_count(NULL)`. Returns the number of documents
/// actually loaded (capped by `max_docs`, 0 on error).
#[no_mangle]
pub extern "C" fn yp_load_index_chunk(path: *const c_char, max_docs: u64) -> u64 {
    if path.is_null() || max_docs == 0 {
        return 0;
    }
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return 0,
        }
    };

    match BinaryHNSW::load(path_str) {
        Ok(loaded) => {
            let count = loaded.len().min(max_docs as usize) as u64;
            *HNSW_CHUNK.write().unwrap() = Some(loaded);
            count
        }
        Err(e) => {
            set_error(format!("yp_load_index_chunk failed for {}: {}", path_str, e));
            0
        }
    }
}

/// Return the document count for an opaque index handle.
/// Pass NULL to query the currently-loaded HNSW chunk (loaded by
/// `yp_load_index_chunk`). Non-null handles are not yet supported.
#[no_mangle]
pub extern "C" fn yp_index_doc_count(handle: *mut c_void) -> u64 {
    if handle.is_null() {
        let guard = HNSW_CHUNK.read().unwrap();
        return guard.as_ref().map(|idx| idx.len() as u64).unwrap_or(0);
    }
    // Opaque handles are not yet implemented.
    0
}

/// Set the query-time `ef_search` parameter on the currently-loaded HNSW chunk.
/// Has no effect if no chunk is loaded. Returns 0 on success, -1 if no chunk.
#[no_mangle]
pub extern "C" fn yp_hnsw_set_ef(ef_search: usize) -> i32 {
    // ef_search is atomic — a read lock suffices. Taking the write lock here
    // blocked the UI thread behind warm-up and froze the app on iOS.
    let guard = HNSW_CHUNK.read().unwrap();
    match guard.as_ref() {
        Some(idx) => {
            idx.set_ef_search(ef_search.max(1));
            0
        }
        None => -1,
    }
}

/// Cold-start warm-up for the currently-loaded HNSW chunk: prefetches the
/// mmap'd neighbor arena and runs `n_queries` sampled self-hash probes at
/// beam width `ef` so the first real query pays no page faults. Run on a
/// background queue right after `yp_load_index_chunk`. Returns 0 on
/// success, -1 if no chunk is loaded.
#[no_mangle]
pub extern "C" fn yp_hnsw_warm_up(n_queries: u64, ef: u64) -> i32 {
    // Warm-up only READS the graph (search_with_ef makes no mutation), so a
    // read lock suffices. Holding the write lock here blocked every real
    // query for the entire warm-up duration and froze the UI on iOS.
    let guard = HNSW_CHUNK.read().unwrap();
    match guard.as_ref() {
        Some(idx) => {
            idx.warm_up(n_queries as usize, ef as usize);
            0
        }
        None => -1,
    }
}

/// Retrieve the stored 512-bit hash for a user-facing record id in the
/// currently-loaded HNSW chunk (for "find similar" self-query).
/// * `out_hash` — caller-allocated 64-byte buffer.
/// Returns 0 on success, -1 if no chunk loaded or id not found.
#[no_mangle]
pub extern "C" fn yp_hnsw_get_hash_current(id: u64, out_hash: *mut u8) -> i32 {
    if out_hash.is_null() {
        return -1;
    }
    let guard = HNSW_CHUNK.read().unwrap();
    match guard.as_ref().and_then(|idx| idx.hash_by_id(id)) {
        Some(h) => {
            unsafe { std::ptr::copy_nonoverlapping(h.as_ptr(), out_hash, HASH512_BYTES) };
            0
        }
        None => -1,
    }
}

/// Search the currently-loaded HNSW chunk.
/// * `query_hash` — pointer to 64 bytes (512-bit query hash).
/// * `hash_len` — length of the query hash (must be 64).
/// * `k` — number of results requested.
/// * `out_ids` — caller-allocated buffer for record IDs (`u64`).
/// * `out_dists` — caller-allocated buffer for Hamming distances (`u32`).
/// * `out_cap` — capacity of both output buffers.
/// Returns the number of results written (at most `k` and `out_cap`).
#[no_mangle]
pub extern "C" fn yp_hnsw_search(
    query_hash: *const u8,
    hash_len: usize,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut u32,
    out_cap: usize,
) -> usize {
    if query_hash.is_null() || out_ids.is_null() || out_dists.is_null() || out_cap == 0 {
        return 0;
    }
    if hash_len != HASH512_BYTES {
        return 0;
    }

    let guard = HNSW_CHUNK.read().unwrap();
    let idx = match guard.as_ref() {
        Some(i) => i,
        None => return 0,
    };

    let qbytes = unsafe { std::slice::from_raw_parts(query_hash, hash_len) };
    let mut qhash = [0u8; HASH512_BYTES];
    qhash.copy_from_slice(qbytes);

    let results = idx.search(&qhash, k.min(out_cap));
    let n = results.len();

    let ids = unsafe { std::slice::from_raw_parts_mut(out_ids, out_cap) };
    let dists = unsafe { std::slice::from_raw_parts_mut(out_dists, out_cap) };

    for (i, (dist, node_idx, _tag)) in results.iter().enumerate() {
        ids[i] = idx.node(*node_idx).map(|n| n.id).unwrap_or(0);
        dists[i] = *dist;
    }

    n
}
