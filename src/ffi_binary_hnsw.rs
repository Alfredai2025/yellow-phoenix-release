// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge for the binary HNSW index.
//!
//! All functions are `extern "C"` and use raw pointers so they can be called
//! from Python via ctypes.  The caller owns the index handle returned by
//! `yp_binary_hnsw_new` and must free it with `yp_binary_hnsw_free`.
//!
//! # Safety
//! All pointer arguments must be valid and properly aligned.  Passing null
//! where a non-null pointer is expected results in an early return / error code.

use std::ffi::c_void;
use std::ptr;
use std::slice;
use std::sync::Mutex;
use std::collections::HashMap;

use crate::binary_hnsw::{BinaryHNSW, HASH512_BITS, HASH512_BYTES, hamming_distance};
use crate::phyllotactic::PhyllotacticNavigator;

/// Create a new binary HNSW index with default parameters.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_new() -> *mut c_void {
    let index = Box::new(BinaryHNSW::new());
    Box::into_raw(index) as *mut c_void
}

/// Free a binary HNSW index previously returned by `yp_binary_hnsw_new`.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_free(index: *mut c_void) {
    if index.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(index as *mut BinaryHNSW);
    }
}

/// Return the number of indexed nodes.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_count(index: *mut c_void) -> usize {
    if index.is_null() {
        return 0;
    }
    let idx = unsafe { &*(index as *mut BinaryHNSW) };
    idx.len()
}

/// Insert a point into the index.
///
/// * `index` — index handle.
/// * `id` — user-facing record identifier.
/// * `hash` — pointer to 64 bytes (512-bit hash).
///
/// Returns 0 on success, -1 on null index/hash, -2 if `hash` is not 64 bytes.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_insert(
    index: *mut c_void,
    id: u64,
    hash: *const u8,
    hash_len: usize,
) -> i32 {
    if index.is_null() {
        return -1;
    }
    if hash.is_null() || hash_len != HASH512_BYTES {
        return -2;
    }

    let idx = unsafe { &mut *(index as *mut BinaryHNSW) };
    let bytes = unsafe { slice::from_raw_parts(hash, hash_len) };
    let mut hash512 = [0u8; HASH512_BYTES];
    hash512.copy_from_slice(bytes);

    idx.insert(id, hash512);
    0
}

/// Search the index for the `k` nearest neighbors of `query`.
///
/// * `index` — index handle.
/// * `query` — pointer to 64 bytes (512-bit query hash).
/// * `k` — number of results requested.
/// * `out_ids` — caller-allocated buffer for record IDs (`u64`).
/// * `out_dists` — caller-allocated buffer for Hamming distances (`u32`).
/// * `out_cap` — capacity of both output buffers.
///
/// Returns the number of results written (at most `k` and `out_cap`).
/// Returns 0 if the index is empty or on error.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_search(
    index: *mut c_void,
    query: *const u8,
    query_len: usize,
    k: usize,
    out_ids: *mut u64,
    out_dists: *mut u32,
    out_cap: usize,
) -> usize {
    if index.is_null() || query.is_null() || out_ids.is_null() || out_dists.is_null() {
        return 0;
    }
    if query_len != HASH512_BYTES {
        return 0;
    }

    let idx = unsafe { &*(index as *mut BinaryHNSW) };
    let qbytes = unsafe { slice::from_raw_parts(query, query_len) };
    let mut qhash = [0u8; HASH512_BYTES];
    qhash.copy_from_slice(qbytes);

    let results = idx.search(&qhash, k.min(out_cap));
    let n = results.len();

    let ids = unsafe { slice::from_raw_parts_mut(out_ids, out_cap) };
    let dists = unsafe { slice::from_raw_parts_mut(out_dists, out_cap) };

    for (i, (dist, node_idx, _tag)) in results.iter().enumerate() {
        ids[i] = idx.node(*node_idx).map(|n| n.id).unwrap_or(0);
        dists[i] = *dist;
    }

    n
}

/// Batch insert into BinaryHNSW.
/// ids: *const u64, length n
/// hashes: *const u8, flat array of n * hash_bytes
/// hash_bytes: must be HASH512_BYTES (64)
/// Returns: n inserted on success, negative error code on failure.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_insert_batch(
    handle: *mut c_void,
    ids: *const u64,
    hashes: *const u8,
    n: usize,
    hash_bytes: usize,
) -> i32 {
    if handle.is_null() || ids.is_null() || hashes.is_null() || n == 0 || hash_bytes != HASH512_BYTES {
        return -1;
    }
    let idx = unsafe { &mut *(handle as *mut BinaryHNSW) };
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, n) };
    let hashes_slice = unsafe { std::slice::from_raw_parts(hashes, n * hash_bytes) };

    for i in 0..n {
        let mut h = [0u8; HASH512_BYTES];
        h.copy_from_slice(&hashes_slice[i * hash_bytes..(i + 1) * hash_bytes]);
        idx.insert(ids_slice[i], h);
    }
    n as i32
}

/// Create a new binary HNSW index with explicit parameters.
/// * `m` — max neighbors per layer (typical: 8, 16, 32, 64)
/// * `ef_construction` — candidate list size during build (typical: 50-400)
/// * `ef_search` — candidate list size during query (typical: 50-400)
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_new_with_params(
    m: usize,
    ef_construction: usize,
    ef_search: usize,
) -> *mut c_void {
    let index = Box::new(BinaryHNSW::with_params(m, ef_construction, ef_search));
    Box::into_raw(index) as *mut c_void
}

/// Set query-time beam width (`efSearch`) without rebuilding the graph.
/// * `index` — index handle.
/// * `ef_search` — new dynamic candidate list size (must be >= 1).
/// Returns 0 on success, -1 on null index.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_set_ef(
    index: *mut c_void,
    ef_search: usize,
) -> i32 {
    if index.is_null() {
        return -1;
    }
    let idx = unsafe { &mut *(index as *mut BinaryHNSW) };
    idx.set_ef_search(ef_search.max(1));
    0
}

/// Retrieve the stored 512-bit hash for a given node_id.
/// * `handle` — index handle.
/// * `node_id` — user-facing record identifier.
/// * `out_hash` — caller-allocated 64-byte buffer.
/// Returns 0 on success, -1 if handle/node not found or out_hash is null.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_get_hash(
    handle: *mut c_void,
    node_id: i64,
    out_hash: *mut u8,
) -> i32 {
    if handle.is_null() || out_hash.is_null() {
        return -1;
    }
    if node_id < 0 {
        return -1;
    }
    let idx = unsafe { &*(handle as *mut BinaryHNSW) };
    let target_id = node_id as u64;

    if let Some(hash) = idx.hash_by_id(target_id) {
        unsafe {
            ptr::copy_nonoverlapping(hash.as_ptr(), out_hash, HASH512_BYTES);
        }
        return 0;
    }
    -1
}

/// Save a binary HNSW index to disk.
/// * `handle` — index handle.
/// * `path` — null-terminated UTF-8 file path.
/// Returns 0 on success, -1 on null handle/path, -2 on IO error.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_save(
    handle: *mut c_void,
    path: *const std::ffi::c_char,
) -> i32 {
    if handle.is_null() || path.is_null() {
        return -1;
    }
    let idx = unsafe { &*(handle as *mut BinaryHNSW) };
    let path = unsafe { std::ffi::CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match idx.save(path) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

/// Load a binary HNSW index from disk, replacing the contents of `handle`.
/// * `handle` — index handle (must have been created with `yp_binary_hnsw_new_with_params`).
/// * `path` — null-terminated UTF-8 file path.
/// Returns 0 on success, -1 on null handle/path, -2 on IO error, -3 on parse error.
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_load(
    handle: *mut c_void,
    path: *const std::ffi::c_char,
) -> i32 {
    if handle.is_null() || path.is_null() {
        return -1;
    }
    let idx = unsafe { &mut *(handle as *mut BinaryHNSW) };
    let path = unsafe { std::ffi::CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match BinaryHNSW::load(path) {
        Ok(loaded) => {
            *idx = loaded;
            0
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => -2,
        Err(_) => -3,
    }
}

/// Compute triple Hamming resonance = geometric product for three 512-bit hashes.
/// * `qhash` — query hash (64 bytes).
/// * `paper_hash` — paper hash (64 bytes).
/// * `mind_hash` — substructure pattern hash (64 bytes).
/// * `out_score` — output geometric resonance score.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn yp_unified_query_blend(
    qhash: *const u8,
    paper_hash: *const u8,
    mind_hash: *const u8,
    out_score: *mut f32,
) -> i32 {
    if qhash.is_null() || paper_hash.is_null() || mind_hash.is_null() || out_score.is_null() {
        return -1;
    }

    let q = unsafe { slice::from_raw_parts(qhash, HASH512_BYTES) };
    let p = unsafe { slice::from_raw_parts(paper_hash, HASH512_BYTES) };
    let m = unsafe { slice::from_raw_parts(mind_hash, HASH512_BYTES) };

    let mut q_arr = [0u8; HASH512_BYTES];
    let mut p_arr = [0u8; HASH512_BYTES];
    let mut m_arr = [0u8; HASH512_BYTES];
    q_arr.copy_from_slice(q);
    p_arr.copy_from_slice(p);
    m_arr.copy_from_slice(m);

    let sim_qp = 1.0 - (hamming_distance(&q_arr, &p_arr) as f32 / HASH512_BITS as f32);
    let sim_qm = 1.0 - (hamming_distance(&q_arr, &m_arr) as f32 / HASH512_BITS as f32);
    let sim_pm = 1.0 - (hamming_distance(&p_arr, &m_arr) as f32 / HASH512_BITS as f32);

    let score = sim_qp * sim_qm * sim_pm;
    unsafe {
        *out_score = score;
    }
    0
}


// ---------------------------------------------------------------------------
// Phyllotactic entry point FFI (2026-08-25)
// ---------------------------------------------------------------------------

static PHYLLO_NAV: Mutex<Option<PhyllotacticNavigator>> = Mutex::new(None);

/// Build phyllotactic navigator from HNSW index + spectral coordinates.
/// `ids`: *const u64, length n
/// `coords`: *const f32, flat array of n * 2 (x,y pairs)
#[no_mangle]
pub extern "C" fn yp_phyllotactic_build(
    hnsw_handle: *mut c_void,
    ids: *const u64,
    coords: *const f32,
    n: usize,
) -> i32 {
    if hnsw_handle.is_null() || ids.is_null() || coords.is_null() || n == 0 {
        return -1;
    }
    let hnsw = unsafe { &*(hnsw_handle as *mut BinaryHNSW) };
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, n) };
    let coords_slice = unsafe { std::slice::from_raw_parts(coords, n * 2) };

    let mut map = HashMap::with_capacity(n);
    for i in 0..n {
        map.insert(ids_slice[i], [coords_slice[i * 2], coords_slice[i * 2 + 1]]);
    }

    let nav = PhyllotacticNavigator::build_from_coords(
        hnsw.len(),
        |idx| hnsw.node(idx as u32).map(|n| n.id),
        &map,
    );

    if let Ok(mut guard) = PHYLLO_NAV.lock() {
        *guard = Some(nav);
    }
    0
}

/// Return the best phyllotactic entry node index for a query spectral coord.
/// `query_coord`: [x, y] as 2 f32s
/// Returns node index, or usize::MAX on error.
#[no_mangle]
pub extern "C" fn yp_phyllotactic_entry(
    query_coord: *const f32,
) -> usize {
    if query_coord.is_null() {
        return usize::MAX;
    }
    let q = unsafe { std::slice::from_raw_parts(query_coord, 2) };
    let nav = match PHYLLO_NAV.lock() {
        Ok(g) => g,
        Err(_) => return usize::MAX,
    };
    nav.as_ref()
        .and_then(|n| n.nearest_entry([q[0], q[1]]))
        .unwrap_or(usize::MAX)
}

/// Search HNSW starting from a phyllotactic entry point.
/// `entry_node_idx` is the internal HNSW node index (from yp_phyllotactic_entry).
/// Returns number of results written (at most k and out_cap).
#[no_mangle]
pub extern "C" fn yp_binary_hnsw_search_from(
    index: *mut c_void,
    query: *const u8,
    query_len: usize,
    k: usize,
    entry_node_idx: usize,
    out_ids: *mut u64,
    out_dists: *mut u32,
    out_cap: usize,
) -> usize {
    if index.is_null() || query.is_null() || out_ids.is_null() || out_dists.is_null() {
        return 0;
    }
    if query_len != HASH512_BYTES {
        return 0;
    }

    let idx = unsafe { &*(index as *mut BinaryHNSW) };
    let qbytes = unsafe { slice::from_raw_parts(query, query_len) };
    let mut qhash = [0u8; HASH512_BYTES];
    qhash.copy_from_slice(qbytes);

    let results = idx.search_from(&qhash, k.min(out_cap), entry_node_idx as u32);
    let n = results.len();

    let ids = unsafe { slice::from_raw_parts_mut(out_ids, out_cap) };
    let dists = unsafe { slice::from_raw_parts_mut(out_dists, out_cap) };

    for (i, (dist, node_idx, _tag)) in results.iter().enumerate() {
        ids[i] = idx.node(*node_idx).map(|n| n.id).unwrap_or(0);
        dists[i] = *dist;
    }

    n
}
