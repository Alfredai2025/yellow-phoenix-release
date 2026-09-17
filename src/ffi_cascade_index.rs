// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

struct CascadeState { slots: Vec<(u64, Vec<f32>)> }

static REGISTRY: Mutex<Vec<Option<Box<CascadeState>>>> = Mutex::new(Vec::new());

fn empty_json() -> *mut c_char {
    static S: &[u8] = b"[]\0";
    S.as_ptr() as *mut c_char
}

fn alloc(state: CascadeState) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    for (i, slot) in reg.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = Some(Box::new(state));
            return (i as i32) + 1;
        }
    }
    reg.push(Some(Box::new(state)));
    reg.len() as i32
}

#[no_mangle]
pub extern "C" fn yp_cascade_new(_depth: c_int, _width: c_int) -> c_int {
    alloc(CascadeState { slots: Vec::new() })
}

#[no_mangle]
pub extern "C" fn yp_cascade_drop(handle: c_int) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    if idx < reg.len() {
        reg[idx] = None;
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn yp_cascade_insert(
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
    state.slots.push((id as u64, f32s));
    0
}

#[no_mangle]
pub extern "C" fn yp_cascade_query(
    handle: c_int, _vec: *const c_char, _len: usize, k: usize
) -> *mut c_char {
    let reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get(idx).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return empty_json(),
    };
    let ids: Vec<u64> = state.slots.iter().map(|(id, _)| *id).take(k).collect();
    match CString::new(format!("[{}]", ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))) {
        Ok(c) => c.into_raw(),
        Err(_) => empty_json(),
    }
}
