// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_void};
use std::slice;
use crate::float_hnsw::FloatHNSW;

#[no_mangle]
pub extern "C" fn yp_float_hnsw_new(m: usize, ef_construction: usize, ef_search: usize) -> *mut c_void {
    let hnsw = FloatHNSW::new(m, ef_construction, ef_search);
    Box::into_raw(Box::new(hnsw)) as *mut c_void
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_insert(
    handle: *mut c_void,
    id: u64,
    vec: *const f32,
    dim: usize,
) {
    if handle.is_null() || vec.is_null() { return; }
    let hnsw = unsafe { &mut *(handle as *mut FloatHNSW) };
    let slice = unsafe { slice::from_raw_parts(vec, dim) };
    hnsw.insert(id, slice.to_vec());
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_insert_batch(
    handle: *mut c_void,
    ids: *const u64,
    vecs: *const f32,
    n: usize,
    dim: usize,
) {
    if handle.is_null() || ids.is_null() || vecs.is_null() || n == 0 || dim == 0 { return; }
    let hnsw = unsafe { &mut *(handle as *mut FloatHNSW) };
    let ids_slice = unsafe { slice::from_raw_parts(ids, n) };
    let vecs_slice = unsafe { slice::from_raw_parts(vecs, n * dim) };
    hnsw.insert_batch(ids_slice, vecs_slice, dim);
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_search(
    handle: *mut c_void,
    query: *const f32,
    dim: usize,
    k: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
) -> usize {
    if handle.is_null() || query.is_null() || out_ids.is_null() || out_scores.is_null() {
        return 0;
    }
    let hnsw = unsafe { &*(handle as *const FloatHNSW) };
    let q = unsafe { slice::from_raw_parts(query, dim) };
    let results = hnsw.search(q, k);
    
    let out_id_slice = unsafe { slice::from_raw_parts_mut(out_ids, k) };
    let out_score_slice = unsafe { slice::from_raw_parts_mut(out_scores, k) };
    
    for (i, (id, dist)) in results.iter().enumerate() {
        out_id_slice[i] = *id;
        out_score_slice[i] = 1.0 - *dist;
    }
    results.len()
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_set_ef(handle: *mut c_void, ef: usize) {
    if handle.is_null() { return; }
    let hnsw = unsafe { &mut *(handle as *mut FloatHNSW) };
    hnsw.set_ef_search(ef);
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_count(handle: *mut c_void) -> usize {
    if handle.is_null() { return 0; }
    let hnsw = unsafe { &*(handle as *const FloatHNSW) };
    hnsw.len()
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_save(handle: *mut c_void, path: *const c_char) -> i32 {
    if handle.is_null() || path.is_null() { return -1; }
    let hnsw = unsafe { &*(handle as *const FloatHNSW) };
    let path = unsafe { std::ffi::CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match hnsw.save(path) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_load(path: *const c_char) -> *mut c_void {
    if path.is_null() { return std::ptr::null_mut(); }
    let path = unsafe { std::ffi::CStr::from_ptr(path) };
    let path = match path.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match FloatHNSW::load(path) {
        Ok(hnsw) => Box::into_raw(Box::new(hnsw)) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn yp_float_hnsw_free(handle: *mut c_void) {
    if !handle.is_null() {
        let _ = unsafe { Box::from_raw(handle as *mut FloatHNSW) };
    }
}
