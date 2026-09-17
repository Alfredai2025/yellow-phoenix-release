// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge for multi-mirror holographic cascade.
use std::ffi::c_void;
use std::slice;

#[cfg(feature = "holographic-cascade")]
use crate::holographic_cascade::HolographicCascade;

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_new(l1: usize, l2: usize, l3: usize, dim: usize) -> *mut c_void {
    let c = Box::new(HolographicCascade::new(l1 as u8, l2 as u8, l3 as u8, dim));
    Box::into_raw(c) as *mut c_void
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_insert_mirror(ptr: *mut c_void, pid: u64, hash: *const u8, len: usize) -> i32 {
    if ptr.is_null() || hash.is_null() || len != 64 { return -1; }
    let c = unsafe { &mut *(ptr as *mut HolographicCascade) };
    let h = unsafe { slice::from_raw_parts(hash, len) };
    let mut arr = [0u8; 64]; arr.copy_from_slice(h);
    c.insert_mirror(pid, &arr);
    0
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_apply_gravity(ptr: *mut c_void, hashes: *const u8, n: usize) -> i32 {
    if ptr.is_null() || hashes.is_null() { return -1; }
    let c = unsafe { &mut *(ptr as *mut HolographicCascade) };
    let h_slice = unsafe { slice::from_raw_parts(hashes, n * 64) };
    let mut hash_vec: Vec<[u8; 64]> = Vec::with_capacity(n);
    for i in 0..n {
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&h_slice[i*64..(i+1)*64]);
        hash_vec.push(arr);
    }
    c.apply_semantic_gravity(&hash_vec);
    0
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_auto_tune_all(ptr: *mut c_void) -> i32 {
    if ptr.is_null() { return -1; }
    let c = unsafe { &mut *(ptr as *mut HolographicCascade) };
    c.auto_tune_all_phases();
    0
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_query_arrow(
    ptr: *mut c_void, q: *const u8, qlen: usize,
    top_k: usize, out_ids: *mut u64, out_scores: *mut f32, out_cap: usize,
) -> usize {
    if ptr.is_null() || q.is_null() || out_ids.is_null() || out_scores.is_null() { return 0; }
    if qlen != 64 { return 0; }
    let c = unsafe { &*(ptr as *mut HolographicCascade) };
    let qs = unsafe { slice::from_raw_parts(q, qlen) };
    let mut arr = [0u8; 64]; arr.copy_from_slice(qs);
    let results = c.query_arrow(&arr, top_k.min(out_cap));
    let n = results.len();
    unsafe { for (i, (pid, score)) in results.iter().enumerate() { *out_ids.add(i) = *pid; *out_scores.add(i) = *score; } }
    n
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_free(ptr: *mut c_void) {
    if !ptr.is_null() { unsafe { let _ = Box::from_raw(ptr as *mut HolographicCascade); } }
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_len(ptr: *mut c_void) -> usize {
    if ptr.is_null() { return 0; }
    let c = unsafe { &*(ptr as *mut HolographicCascade) };
    c.len()
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_holo_cascade_vault_count(ptr: *mut c_void) -> usize {
    if ptr.is_null() { return 0; }
    let c = unsafe { &*(ptr as *mut HolographicCascade) };
    c.vault_count()
}