// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

static REGISTRY: Mutex<Vec<Option<Box<TemporalState>>>> = Mutex::new(Vec::new());

struct TemporalState {
    events: Vec<(i32, String, i32)>, // (pid, event_type, timestamp)
}

fn ok_json() -> *mut c_char {
    static S: &[u8] = b"{\"ok\":true}\0";
    S.as_ptr() as *mut c_char
}

fn empty_json() -> *mut c_char {
    static S: &[u8] = b"[]\0";
    S.as_ptr() as *mut c_char
}

#[no_mangle]
pub extern "C" fn yp_temporal_record(
    handle: c_int, pid: c_int, event_type: *const c_char, timestamp: c_int
) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return -1,
    };
    let et = unsafe { std::ffi::CStr::from_ptr(event_type).to_string_lossy().into_owned() };
    state.events.push((pid, et, timestamp));
    0
}

#[no_mangle]
pub extern "C" fn yp_temporal_trending(handle: c_int, _n: c_int) -> *mut c_char {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return empty_json(),
    };
    let mut counts = std::collections::HashMap::new();
    for (pid, _, _) in &state.events {
        *counts.entry(*pid).or_insert(0) += 1;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let ids: Vec<String> = v.into_iter().map(|(pid, _)| pid.to_string()).collect();
    let json = format!("[{}]", ids.join(","));
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn yp_temporal_history(handle: c_int, pid: c_int) -> *mut c_char {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return empty_json(),
    };
    let items: Vec<String> = state.events.iter()
        .filter(|(p, _, _)| *p == pid)
        .map(|(_, et, ts)| format!("{{\"event\":\"{}\",\"ts\":{}}}", et, ts))
        .collect();
    let json = format!("[{}]", items.join(","));
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn yp_temporal_new() -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let handle = reg.len() as c_int + 1;
    reg.push(Some(Box::new(TemporalState { events: Vec::new() })));
    handle
}
