// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! C-compatible FFI for the OPQ+PQ compact semantic index.

use crate::opq_pq_index::OpqPqTier;
use std::os::raw::{c_char, c_float, c_int, c_uint, c_uchar};
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

static TIER1: OnceLock<Mutex<OpqPqTier>> = OnceLock::new();
static TIER2: OnceLock<Mutex<OpqPqTier>> = OnceLock::new();

fn get_tier(tier: c_int) -> Option<&'static Mutex<OpqPqTier>> {
    match tier {
        1 => TIER1.get(),
        2 => TIER2.get(),
        _ => None,
    }
}

/// Initialize an OPQ+PQ tier.
///
/// `rotation` must contain `d_in * d_out` f32 values.
/// `codebooks` must contain `m * (1<<nbits) * (d_out/m)` f32 values.
/// `codes` must contain `n * m` bytes.
#[no_mangle]
pub extern "C" fn yp_opq_pq_init(
    tier: c_int,
    rotation: *const c_float,
    d_in: usize,
    d_out: usize,
    codebooks: *const c_float,
    codes: *const c_uchar,
    n: usize,
    m: usize,
    nbits: c_int,
) -> c_int {
    if rotation.is_null() || codebooks.is_null() || codes.is_null() {
        return -1;
    }
    if d_out == 0 || m == 0 || nbits < 1 || nbits > 16 {
        return -1;
    }
    if d_out % m != 0 {
        return -1;
    }
    let ksub = 1usize << nbits;
    let dsub = d_out / m;

    let rot = unsafe { std::slice::from_raw_parts(rotation, d_in * d_out) }.to_vec();
    let books = unsafe { std::slice::from_raw_parts(codebooks, m * ksub * dsub) }.to_vec();
    let codes_vec = unsafe { std::slice::from_raw_parts(codes, n * m) }.to_vec();

    let tier_obj = OpqPqTier::new(rot, d_in, d_out, books, codes_vec, n, m, ksub);

    let cell = match tier {
        1 => &TIER1,
        2 => &TIER2,
        _ => return -1,
    };
    if cell.set(Mutex::new(tier_obj)).is_err() {
        return -2; // already initialized
    }
    0
}

/// Encode a 384-dim embedding into an M-byte PQ code for the given tier.
/// Returns the number of bytes written to `out_code` (M) or <0 on error.
#[no_mangle]
pub extern "C" fn yp_opq_pq_encode(
    tier: c_int,
    embedding: *const c_float,
    d_in: usize,
    out_code: *mut c_uchar,
) -> c_int {
    if embedding.is_null() || out_code.is_null() {
        return -1;
    }
    let emb = unsafe { std::slice::from_raw_parts(embedding, d_in) };
    let lock = match get_tier(tier) {
        Some(m) => m,
        None => return -1,
    };
    let t = lock.lock().unwrap();
    if d_in != t.d_in {
        return -1;
    }
    let code = t.encode(emb);
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), out_code, code.len());
    }
    code.len() as c_int
}

/// Query a tier and write up to `top_k` (id, distance) pairs into out buffers.
/// Returns the number of results written.
#[no_mangle]
pub extern "C" fn yp_opq_pq_query(
    tier: c_int,
    embedding: *const c_float,
    d_in: usize,
    top_k: usize,
    out_ids: *mut c_uint,
    out_scores: *mut c_float,
) -> c_int {
    if embedding.is_null() || out_ids.is_null() || out_scores.is_null() {
        return -1;
    }
    let emb = unsafe { std::slice::from_raw_parts(embedding, d_in) };
    let lock = match get_tier(tier) {
        Some(m) => m,
        None => return -1,
    };
    let t = lock.lock().unwrap();
    if d_in != t.d_in {
        return -1;
    }
    let results = t.query(emb, top_k);
    for (i, (id, dist)) in results.iter().enumerate() {
        unsafe {
            *out_ids.add(i) = *id;
            *out_scores.add(i) = *dist;
        }
    }
    results.len() as c_int
}

/// Re-rank a candidate list using a tier's ADC distances.
/// Returns the number of results written (<= n_candidates).
#[no_mangle]
pub extern "C" fn yp_opq_pq_rerank(
    tier: c_int,
    embedding: *const c_float,
    d_in: usize,
    candidate_ids: *const c_uint,
    n_candidates: usize,
    out_ids: *mut c_uint,
    out_scores: *mut c_float,
) -> c_int {
    if embedding.is_null() || candidate_ids.is_null() || out_ids.is_null() || out_scores.is_null() {
        return -1;
    }
    let emb = unsafe { std::slice::from_raw_parts(embedding, d_in) };
    let cands = unsafe { std::slice::from_raw_parts(candidate_ids, n_candidates) };
    let lock = match get_tier(tier) {
        Some(m) => m,
        None => return -1,
    };
    let t = lock.lock().unwrap();
    if d_in != t.d_in {
        return -1;
    }
    let results = t.rerank(emb, cands);
    for (i, (id, dist)) in results.iter().enumerate() {
        unsafe {
            *out_ids.add(i) = *id;
            *out_scores.add(i) = *dist;
        }
    }
    results.len() as c_int
}

/// Free a string returned by another FFI function (placeholder for symmetry).
#[no_mangle]
pub extern "C" fn yp_opq_pq_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}
