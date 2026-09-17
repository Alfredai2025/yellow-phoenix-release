// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Minimal FFI wiring for experimental autopoiesis modules.
//! These exports are intentionally small: they let Python call into the modules
//! without crashing, and they return safe placeholders for query operations.

use std::collections::HashMap;
use std::ffi::{c_char, c_double, c_int, c_void, CString};
use std::slice;

// ---------------------------------------------------------------------------
// Internal state containers
// ---------------------------------------------------------------------------

struct RouterState {
    slots: Vec<(u64, Vec<f32>)>,
}

struct TunerState {
    #[allow(dead_code)]
    threshold_us: f64,
    observations: Vec<(f64, f64)>,
}

struct DriftState {
    #[allow(dead_code)]
    window_size: usize,
    window: Vec<Vec<f32>>,
}

struct CacheState {
    map: HashMap<String, Vec<u8>>,
}

struct FeederState {
    #[allow(dead_code)]
    capacity: usize,
    queue: Vec<(Vec<f32>, u64)>,
}

struct SpectralCoordsState {
    embeddings: Vec<Vec<f32>>,
}

struct SpectralStageState {
    embeddings: Vec<Vec<f32>>,
}

struct QueryState {
    embeddings: Vec<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

unsafe fn bytes_to_f32s(ptr: *const c_char, byte_len: usize) -> Vec<f32> {
    if ptr.is_null() || byte_len == 0 {
        return Vec::new();
    }
    let u8_slice = slice::from_raw_parts(ptr as *const u8, byte_len);
    let count = byte_len / 4;
    let f32_slice = slice::from_raw_parts(u8_slice.as_ptr() as *const f32, count);
    f32_slice.to_vec()
}

unsafe fn bytes_to_string(ptr: *const c_char, len: usize) -> String {
    if ptr.is_null() {
        return String::new();
    }
    let slice = slice::from_raw_parts(ptr as *const u8, len);
    String::from_utf8_lossy(slice).into_owned()
}

fn empty_json() -> *mut c_char {
    // Static allocation, no leak, safe to return as mutable pointer for FFI.
    static EMPTY_JSON: &[u8] = b"[]\0";
    EMPTY_JSON.as_ptr() as *mut c_char
}

unsafe fn take_handle<T>(handle: *mut c_void) -> Option<Box<T>> {
    if handle.is_null() {
        return None;
    }
    Some(Box::from_raw(handle as *mut T))
}

// ---------------------------------------------------------------------------
// Learned Router
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn learned_router_new(_capacity: usize) -> *mut c_void {
    let state = Box::new(RouterState { slots: Vec::new() });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn learned_router_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<RouterState>(handle) };
}

#[no_mangle]
pub extern "C" fn learned_router_insert(
    handle: *mut c_void,
    id: u64,
    vec: *const c_char,
    vec_len: usize,
) -> c_int {
    let state = match unsafe { (handle as *mut RouterState).as_mut() } {
        Some(s) => s,
        None => return -1,
    };
    let v = unsafe { bytes_to_f32s(vec, vec_len) };
    state.slots.push((id, v));
    0
}

#[no_mangle]
pub extern "C" fn learned_router_query(
    handle: *mut c_void,
    _vec: *const c_char,
    _vec_len: usize,
    k: usize,
) -> *mut c_char {
    let state = match unsafe { (handle as *mut RouterState).as_mut() } {
        Some(s) => s,
        None => return empty_json(),
    };
    // Return up to k stored ids as a JSON array.
    let ids: Vec<u64> = state
        .slots
        .iter()
        .map(|(id, _)| *id)
        .take(k)
        .collect();
    let json = format!("[{}]", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => empty_json(),
    }
}

// ---------------------------------------------------------------------------
// Self Tuner
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn self_tuner_new(threshold_us: f64) -> *mut c_void {
    let state = Box::new(TunerState {
        threshold_us,
        observations: Vec::new(),
    });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn self_tuner_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<TunerState>(handle) };
}

#[no_mangle]
pub extern "C" fn self_tuner_submit_observation(
    handle: *mut c_void,
    metric: f64,
    cost: f64,
) -> c_int {
    let state = match unsafe { (handle as *mut TunerState).as_mut() } {
        Some(s) => s,
        None => return -1,
    };
    state.observations.push((metric, cost));
    0
}

#[no_mangle]
pub extern "C" fn self_tuner_propose(handle: *mut c_void) -> *mut c_char {
    let state = match unsafe { (handle as *mut TunerState).as_ref() } {
        Some(s) => s,
        None => return empty_json(),
    };
    let n = state.observations.len();
    let json = format!(
        "{{\"phase\":\"propose\",\"observations\":{n},\"action\":\"continue\"}}"
    );
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => empty_json(),
    }
}

// ---------------------------------------------------------------------------
// Drift Detector
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn drift_detector_new(window_size: usize) -> *mut c_void {
    let state = Box::new(DriftState {
        window_size,
        window: Vec::new(),
    });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn drift_detector_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<DriftState>(handle) };
}

#[no_mangle]
pub extern "C" fn drift_detector_record(
    handle: *mut c_void,
    vec: *const c_char,
    vec_len: usize,
) -> c_int {
    let state = match unsafe { (handle as *mut DriftState).as_mut() } {
        Some(s) => s,
        None => return -1,
    };
    let v = unsafe { bytes_to_f32s(vec, vec_len) };
    state.window.push(v);
    0
}

#[no_mangle]
pub extern "C" fn drift_detector_status(handle: *mut c_void) -> *mut c_char {
    let state = match unsafe { (handle as *mut DriftState).as_ref() } {
        Some(s) => s,
        None => return empty_json(),
    };
    let json = format!(
        "{{\"samples\":{},\"drift\":false}}",
        state.window.len()
    );
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => empty_json(),
    }
}

// ---------------------------------------------------------------------------
// Result Cache
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn result_cache_new(_capacity: usize) -> *mut c_void {
    let state = Box::new(CacheState {
        map: HashMap::new(),
    });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn result_cache_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<CacheState>(handle) };
}

#[no_mangle]
pub extern "C" fn result_cache_insert(
    handle: *mut c_void,
    key: *const c_char,
    key_len: usize,
    value: *const c_char,
    value_len: usize,
) -> c_int {
    let state = match unsafe { (handle as *mut CacheState).as_mut() } {
        Some(s) => s,
        None => return -1,
    };
    let k = unsafe { bytes_to_string(key, key_len) };
    let v = unsafe { slice::from_raw_parts(value as *const u8, value_len).to_vec() };
    state.map.insert(k, v);
    0
}

#[no_mangle]
pub extern "C" fn result_cache_get(
    handle: *mut c_void,
    key: *const c_char,
    key_len: usize,
) -> *mut c_char {
    let state = match unsafe { (handle as *mut CacheState).as_ref() } {
        Some(s) => s,
        None => return std::ptr::null_mut(),
    };
    let k = unsafe { bytes_to_string(key, key_len) };
    match state.map.get(&k) {
        Some(v) => match CString::new(v.clone()) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn result_cache_free_value(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = unsafe { CString::from_raw(ptr) };
    }
}

// ---------------------------------------------------------------------------
// Engine Feeder
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn engine_feeder_new(capacity: usize) -> *mut c_void {
    let state = Box::new(FeederState {
        capacity,
        queue: Vec::new(),
    });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn engine_feeder_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<FeederState>(handle) };
}

#[no_mangle]
pub extern "C" fn engine_feeder_submit(
    handle: *mut c_void,
    vec: *const c_char,
    vec_len: usize,
    id: u64,
) -> c_int {
    let state = match unsafe { (handle as *mut FeederState).as_mut() } {
        Some(s) => s,
        None => return -1,
    };
    let v = unsafe { bytes_to_f32s(vec, vec_len) };
    state.queue.push((v, id));
    0
}

// ---------------------------------------------------------------------------
// Spectral Coords / Stage / Query helpers
// ---------------------------------------------------------------------------

unsafe fn build_embeddings(
    data: *const c_char,
    n: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    if data.is_null() || n == 0 || dim == 0 {
        return Vec::new();
    }
    let byte_len = n * dim * 4;
    let flat = bytes_to_f32s(data, byte_len);
    let mut embeddings = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * dim;
        let end = start + dim;
        embeddings.push(flat[start..end].to_vec());
    }
    embeddings
}

unsafe fn query_embeddings(
    state_embeddings: &[Vec<f32>],
    qdata: *const c_char,
    q_len: usize,
    k: usize,
) -> *mut c_char {
    let q = bytes_to_f32s(qdata, q_len);
    let mut scored: Vec<(usize, f32)> = state_embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let dist: f32 = emb
                .iter()
                .zip(q.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            (i, dist)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ids: Vec<usize> = scored.into_iter().map(|(i, _)| i).take(k).collect();
    let json = format!("[{}]", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => empty_json(),
    }
}

// ---------------------------------------------------------------------------
// Spectral Coords
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn spectral_coords_build(
    data: *const c_char,
    n: usize,
    dim: usize,
) -> *mut c_void {
    let embeddings = unsafe { build_embeddings(data, n, dim) };
    let state = Box::new(SpectralCoordsState { embeddings });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn spectral_coords_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<SpectralCoordsState>(handle) };
}

#[no_mangle]
pub extern "C" fn spectral_coords_query(
    handle: *mut c_void,
    qdata: *const c_char,
    q_len: usize,
    k: usize,
) -> *mut c_char {
    let state = match unsafe { (handle as *mut SpectralCoordsState).as_ref() } {
        Some(s) => s,
        None => return empty_json(),
    };
    unsafe { query_embeddings(&state.embeddings, qdata, q_len, k) }
}

// ---------------------------------------------------------------------------
// Spectral Stage
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn spectral_stage_build(
    data: *const c_char,
    n: usize,
    dim: usize,
) -> *mut c_void {
    let embeddings = unsafe { build_embeddings(data, n, dim) };
    let state = Box::new(SpectralStageState { embeddings });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn spectral_stage_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<SpectralStageState>(handle) };
}

#[no_mangle]
pub extern "C" fn spectral_stage_query(
    handle: *mut c_void,
    qdata: *const c_char,
    q_len: usize,
    k: usize,
) -> *mut c_char {
    let state = match unsafe { (handle as *mut SpectralStageState).as_ref() } {
        Some(s) => s,
        None => return empty_json(),
    };
    unsafe { query_embeddings(&state.embeddings, qdata, q_len, k) }
}

// ---------------------------------------------------------------------------
// Query module
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn query_build(
    data: *const c_char,
    n: usize,
    dim: usize,
) -> *mut c_void {
    let embeddings = unsafe { build_embeddings(data, n, dim) };
    let state = Box::new(QueryState { embeddings });
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub extern "C" fn query_drop(handle: *mut c_void) {
    let _ = unsafe { take_handle::<QueryState>(handle) };
}

#[no_mangle]
pub extern "C" fn query_search(
    handle: *mut c_void,
    qdata: *const c_char,
    q_len: usize,
    k: usize,
) -> *mut c_char {
    let state = match unsafe { (handle as *mut QueryState).as_ref() } {
        Some(s) => s,
        None => return empty_json(),
    };
    unsafe { query_embeddings(&state.embeddings, qdata, q_len, k) }
}
