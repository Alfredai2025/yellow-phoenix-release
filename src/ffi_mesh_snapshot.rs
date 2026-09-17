// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

static REGISTRY: Mutex<Vec<Option<Box<MeshSnapshotState>>>> = Mutex::new(Vec::new());

struct MeshSnapshotState {
    data: Vec<(i32, Vec<u8>)>,
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
pub extern "C" fn yp_mesh_snapshot(handle: c_int) -> *mut c_char {
    let reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get(idx).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return empty_json(),
    };
    let items: Vec<String> = state.data.iter()
        .map(|(pid, _)| pid.to_string())
        .collect();
    let json = format!("{{\"count\":{},\"ids\":[{}]}}", items.len(), items.join(","));
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn yp_mesh_restore(
    handle: c_int, snapshot: *const c_char, len: usize
) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return -1,
    };
    let v = unsafe { std::slice::from_raw_parts(snapshot as *const u8, len) };
    state.data.push((handle, v.to_vec()));
    0
}

#[no_mangle]
pub extern "C" fn yp_mesh_delta(handle: c_int, other: c_int) -> *mut c_char {
    let reg = REGISTRY.lock().unwrap();
    let idx1 = (handle - 1).max(0) as usize;
    let idx2 = (other - 1).max(0) as usize;
    let s1 = match reg.get(idx1).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return empty_json(),
    };
    let s2 = match reg.get(idx2).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return empty_json(),
    };
    let ids1: std::collections::HashSet<i32> = s1.data.iter().map(|(pid, _)| *pid).collect();
    let ids2: std::collections::HashSet<i32> = s2.data.iter().map(|(pid, _)| *pid).collect();
    let added: Vec<String> = ids2.difference(&ids1).map(|x| x.to_string()).collect();
    let removed: Vec<String> = ids1.difference(&ids2).map(|x| x.to_string()).collect();
    let json = format!("{{\"added\":[{}],\"removed\":[{}]}}",
        added.join(","), removed.join(","));
    CString::new(json).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn yp_mesh_snapshot_new() -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let handle = reg.len() as c_int + 1;
    reg.push(Some(Box::new(MeshSnapshotState { data: Vec::new() })));
    handle
}
