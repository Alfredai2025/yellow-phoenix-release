// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge for spectral hologram.
use std::ffi::c_void;
use std::slice;

#[cfg(feature = "holographic-cascade")]
use crate::spectral_hologram::SpectralHologram;

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_spectral_holo_new(
    basis: *const f32, d: usize, k: usize,
    projections: *const f32, n: usize,
) -> *mut c_void {
    if basis.is_null() || projections.is_null() { return std::ptr::null_mut(); }
    let basis_slice = unsafe { slice::from_raw_parts(basis, d * k) };
    let proj_slice = unsafe { slice::from_raw_parts(projections, n * k) };
    let holo = Box::new(SpectralHologram::new(
        basis_slice.to_vec(), d, k,
        proj_slice.to_vec(), n,
    ));
    Box::into_raw(holo) as *mut c_void
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_spectral_holo_query_direct(
    ptr: *mut c_void, query: *const f32, dim: usize,
    top_k: usize, out_ids: *mut u64, out_scores: *mut f32, out_cap: usize,
) -> usize {
    if ptr.is_null() || query.is_null() || out_ids.is_null() || out_scores.is_null() { return 0; }
    let holo = unsafe { &*(ptr as *mut SpectralHologram) };
    let q = unsafe { slice::from_raw_parts(query, dim) };
    let results = holo.query_direct(q, top_k.min(out_cap));
    let n = results.len();
    unsafe { for (i, (pid, score)) in results.iter().enumerate() { *out_ids.add(i) = *pid; *out_scores.add(i) = *score; } }
    n
}

#[cfg(feature = "holographic-diffusion")]
#[no_mangle]
pub extern "C" fn yp_spectral_holo_query_diffusion(
    ptr: *mut c_void, query: *const f32, dim: usize,
    top_k: usize, out_ids: *mut u64, out_scores: *mut f32, out_cap: usize,
) -> usize {
    if ptr.is_null() || query.is_null() || out_ids.is_null() || out_scores.is_null() { return 0; }
    let holo = unsafe { &*(ptr as *mut SpectralHologram) };
    let q = unsafe { slice::from_raw_parts(query, dim) };
    let results = holo.query_diffusion(q, top_k.min(out_cap));
    let n = results.len();
    unsafe { for (i, (pid, score)) in results.iter().enumerate() { *out_ids.add(i) = *pid; *out_scores.add(i) = *score; } }
    n
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_spectral_holo_free(ptr: *mut c_void) {
    if !ptr.is_null() { unsafe { let _ = Box::from_raw(ptr as *mut SpectralHologram); } }
}

#[cfg(feature = "holographic-cascade")]
#[no_mangle]
pub extern "C" fn yp_spectral_holo_len(ptr: *mut c_void) -> usize {
    if ptr.is_null() { return 0; }
    let holo = unsafe { &*(ptr as *mut SpectralHologram) };
    holo.len()
}
