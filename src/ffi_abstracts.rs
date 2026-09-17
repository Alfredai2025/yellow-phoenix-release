// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI for the YPA1 abstract store (see abstract_store.rs).
//!
//! Single global store (one abstracts file per app instance):
//!   yp_abstracts_open(path) -> 0 ok / -1 error
//!   yp_abstracts_close()    -> 0
//!   yp_abstract_decode(id, out, cap) -> >=0 text length written (0 = stored
//!        empty abstract), -1 = id not found / no store open / buffer too small
//!   yp_abstract_decode_batch(ids, n, lens, out, cap) -> 0 ok / -1 error
//!        Decodes n abstracts under ONE store lock. lens[i] = decoded byte
//!        length, or -1 on miss. `out` must hold n * cap bytes; abstracts
//!        longer than cap report -1 in lens[i] (same resize contract as the
//!        single decoder). One FFI round-trip per result list instead of n.
//!   yp_abstracts_count()    -> record count per header, 0 if closed

use std::ffi::{c_char, CStr};
use std::sync::{Mutex, OnceLock};

use crate::abstract_store::AbstractStore;

static STORE: OnceLock<Mutex<Option<AbstractStore>>> = OnceLock::new();

fn store() -> &'static Mutex<Option<AbstractStore>> {
    STORE.get_or_init(|| Mutex::new(None))
}

#[no_mangle]
pub extern "C" fn yp_abstracts_open(path: *const c_char) -> i32 {
    if path.is_null() {
        return -1;
    }
    let p = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    match AbstractStore::open(&p) {
        Ok(s) => {
            if let Ok(mut g) = store().lock() {
                *g = Some(s);
                return 0;
            }
            -1
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn yp_abstracts_close() -> i32 {
    if let Ok(mut g) = store().lock() {
        *g = None;
    }
    0
}

#[no_mangle]
pub extern "C" fn yp_abstracts_count() -> u64 {
    if let Ok(g) = store().lock() {
        if let Some(s) = g.as_ref() {
            return s.len();
        }
    }
    0
}

/// Decode abstract for `id` into `out` (up to `cap` bytes, UTF-8, no NUL).
/// Returns length on success (fits exactly or was truncated-check semantics:
/// returns -1 if cap insufficient so the caller can resize), -1 on miss/error.
#[no_mangle]
pub extern "C" fn yp_abstract_decode(id: u64, out: *mut u8, cap: usize) -> i64 {
    if out.is_null() && cap > 0 {
        return -1;
    }
    let g = match store().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let s = match g.as_ref() {
        Some(s) => s,
        None => return -1,
    };
    match s.decode(id) {
        Some(text) => {
            let bytes = text.as_bytes();
            if bytes.len() > cap {
                return -1; // caller should pass a bigger buffer
            }
            if !bytes.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
                }
            }
            bytes.len() as i64
        }
        None => -1,
    }
}

/// Batch decode: one lock acquisition, n decodes. See module header for the
/// lens/cap contract. `ids` must hold n u64s; `lens` must hold n i64s.
#[no_mangle]
pub extern "C" fn yp_abstract_decode_batch(
    ids: *const u64,
    n: usize,
    lens: *mut i64,
    out: *mut u8,
    cap: usize,
) -> i32 {
    if ids.is_null() || (n > 0 && (lens.is_null() || out.is_null())) {
        return -1;
    }
    let g = match store().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    let s = match g.as_ref() {
        Some(s) => s,
        None => return -1,
    };
    for i in 0..n {
        let id = unsafe { *ids.add(i) };
        let slot = unsafe { lens.add(i) };
        match s.decode(id) {
            Some(text) => {
                let bytes = text.as_bytes();
                if bytes.len() > cap {
                    unsafe { *slot = -1; }
                    continue;
                }
                if !bytes.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(), out.add(i * cap), bytes.len());
                    }
                }
                unsafe { *slot = bytes.len() as i64; }
            }
            None => unsafe { *slot = -1; },
        }
    }
    0
}
