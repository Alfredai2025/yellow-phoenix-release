// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI for 512-bit tensor spectral index.
//!
//! Exports:
//!   yp_tensor_spectral_512_new
//!   yp_tensor_spectral_512_build
//!   yp_tensor_spectral_512_query
//!   yp_tensor_spectral_512_free
//!
//! This interface matches the Python bridge in yp_bridge.py:
//!   build(handle, hashes: *u8, ids: *u64, n: size_t) -> c_int
//!   query(handle, query_hash: *u8, top_k: size_t, out_ids: *u64, out_scores: *f32) -> size_t

use std::ffi::{c_int, c_void};
use std::sync::Mutex;

use libc::size_t;

use crate::math::tensor_spectral_512::TensorSpectralIndex512;

static INDEX_512: Mutex<Option<Box<TensorSpectralIndex512>>> = Mutex::new(None);

#[no_mangle]
pub extern "C" fn yp_tensor_spectral_512_new() -> *mut c_void {
    // The index is global; this just returns a non-null sentinel.
    1usize as *mut c_void
}

#[no_mangle]
pub extern "C" fn yp_tensor_spectral_512_build(
    _handle: *mut c_void,
    hashes: *const u8,
    ids: *const u64,
    n: size_t,
) -> c_int {
    if hashes.is_null() || n == 0 {
        return -1;
    }

    let n = n as usize;
    let total_bytes = match n.checked_mul(64) {
        Some(bytes) => bytes,
        None => return -2,
    };

    let hashes_slice = unsafe { std::slice::from_raw_parts(hashes, total_bytes) };
    let ids_slice = if ids.is_null() {
        &[] as &[u64]
    } else {
        unsafe { std::slice::from_raw_parts(ids, n) }
    };

    let k = 32.min(n);
    let mut index = TensorSpectralIndex512::build(hashes_slice, k);

    // Assign caller-provided ids when available.
    if ids_slice.len() == n {
        index.set_ids(ids_slice);
    }

    if let Ok(mut guard) = INDEX_512.lock() {
        *guard = Some(Box::new(index));
        0
    } else {
        -3
    }
}

#[no_mangle]
pub extern "C" fn yp_tensor_spectral_512_query(
    _handle: *mut c_void,
    query_hash: *const u8,
    top_k: size_t,
    out_ids: *mut u64,
    out_scores: *mut f32,
) -> size_t {
    if query_hash.is_null() || out_ids.is_null() || out_scores.is_null() {
        return 0;
    }

    let top_k = top_k as usize;
    if top_k == 0 {
        return 0;
    }

    let query_bytes = unsafe { std::slice::from_raw_parts(query_hash, 64) };
    let mut pap = [0u8; 64];
    pap.copy_from_slice(query_bytes);

    let guard = INDEX_512.lock().unwrap_or_else(|e| e.into_inner());
    let results = match guard.as_ref() {
        Some(idx) => idx.query(&pap, top_k),
        None => return 0,
    };

    let count = results.len().min(top_k);
    let out_ids_slice = unsafe { std::slice::from_raw_parts_mut(out_ids, top_k) };
    let out_scores_slice = unsafe { std::slice::from_raw_parts_mut(out_scores, top_k) };

    for (i, (score, id)) in results.iter().take(count).enumerate() {
        out_ids_slice[i] = *id;
        out_scores_slice[i] = -score; // query returns negative squared Euclidean distance
    }

    count as size_t
}

#[no_mangle]
pub extern "C" fn yp_tensor_spectral_512_free(_handle: *mut c_void) {
    if let Ok(mut guard) = INDEX_512.lock() {
        *guard = None;
    }
}
