// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, CString};
use std::sync::Mutex;

static REGISTRY: Mutex<Vec<Option<Box<EnergyState>>>> = Mutex::new(Vec::new());

struct EnergyState {
    energy: std::collections::HashMap<i32, i32>,
}

fn ok_json() -> *mut c_char {
    static S: &[u8] = b"{\"ok\":true}\0";
    S.as_ptr() as *mut c_char
}

#[no_mangle]
pub extern "C" fn yp_energy_energize(handle: c_int, pid: c_int, amount: c_int) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return -1,
    };
    *state.energy.entry(pid).or_insert(0) += amount;
    0
}

#[no_mangle]
pub extern "C" fn yp_energy_decay(handle: c_int, pid: c_int, amount: c_int) -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get_mut(idx).and_then(|s| s.as_mut()) {
        Some(s) => s,
        None => return -1,
    };
    let e = state.energy.entry(pid).or_insert(0);
    *e = (*e - amount).max(0);
    0
}

#[no_mangle]
pub extern "C" fn yp_energy_get(handle: c_int, pid: c_int) -> c_int {
    let reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get(idx).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return -1,
    };
    *state.energy.get(&pid).unwrap_or(&0)
}

#[no_mangle]
pub extern "C" fn yp_energy_total(handle: c_int) -> c_int {
    let reg = REGISTRY.lock().unwrap();
    let idx = (handle - 1).max(0) as usize;
    let state = match reg.get(idx).and_then(|s| s.as_ref()) {
        Some(s) => s,
        None => return -1,
    };
    state.energy.values().sum()
}

#[no_mangle]
pub extern "C" fn yp_energy_new() -> c_int {
    let mut reg = REGISTRY.lock().unwrap();
    let handle = reg.len() as c_int + 1;
    reg.push(Some(Box::new(EnergyState { energy: std::collections::HashMap::new() })));
    handle
}
