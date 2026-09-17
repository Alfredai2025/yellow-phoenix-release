// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::sync::Mutex;

static REGISTRY: Mutex<Vec<Option<Box<FlatState>>>> = Mutex::new(Vec::new());

struct FlatState {
    slots: Vec<(i32, Vec<f32>)>,
}

use std::ffi::{c_char, c_int, CString};

fn ok_json() -> *mut c_char {
    static S: &[u8] = b"{\"ok\":true}\0";
    S.as_ptr() as *mut c_char
}

fn empty_json() -> *mut c_char {
    static S: &[u8] = b"[]\0";
    S.as_ptr() as *mut c_char
}

#[no_mangle]
pub extern "C" fn yp_flat_init(_dim: c_int, _capacity: c_int) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let handle = reg.len() as c_int + 1;
    reg.push(Some(Box::new(FlatState { slots: Vec::new() })));
    handle
}

#[no_mangle]
pub extern "C" fn yp_flat_build_from_file(_path: *const c_char) -> c_int { 0 }

#[no_mangle]
pub extern "C" fn yp_flat_query(
    _handle: c_int, _vec: *const c_char, _len: usize, _top_k: c_int
) -> *mut c_char { ok_json() }

#[no_mangle]
pub extern "C" fn yp_flat_query_batch(
    _handle: c_int, _vecs: *const c_char, _len: usize, _n: c_int, _top_k: c_int
) -> *mut c_char { empty_json() }

#[no_mangle]
pub extern "C" fn yp_flat_stats(_handle: c_int) -> *mut c_char { ok_json() }

#[no_mangle]
pub extern "C" fn yp_flat_health(_handle: c_int) -> c_int { 1 }

#[no_mangle]
pub extern "C" fn yp_flat_persist(_handle: c_int, _path: *const c_char) -> c_int { 0 }

#[no_mangle]
pub extern "C" fn yp_flat_free_string(s: *mut c_char) {
    if !s.is_null() { let _ = unsafe { CString::from_raw(s) }; }
}

#[no_mangle]
pub extern "C" fn yp_flat_insert(
    handle: c_int, vec: *const c_char, len: usize, id: c_int
) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return -1,
    };
    let v = unsafe { std::slice::from_raw_parts(vec as *const u8, len) };
    let f32s: Vec<f32> = v.chunks_exact(4).map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect();
    state.slots.push((id, f32s));
    0
}
