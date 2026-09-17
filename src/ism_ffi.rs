// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

#[cfg(feature = "intelligent_shard_manager")]
#[cfg(feature = "intelligent_shard_manager")]
use crate::intelligent_shard_manager::IntelligentShardManager;
use crate::ism_error::ISMError;
use crate::shard::IsmPap;
use lazy_static::lazy_static;
use std::convert::TryInto;
use std::ffi::{c_char, c_int, c_uint, CStr, CString};
use std::sync::{Mutex, RwLock};

const RECORD_BYTES: usize = 8 + 32; // u64 id + 32-byte hash

lazy_static! {
    static ref ISM: RwLock<Option<IntelligentShardManager>> = RwLock::new(None);
    static ref LAST_WATCHDOG_ADVICE: Mutex<Option<String>> = Mutex::new(None);
}

/// Initialise (or reset) the global ISM singleton.
#[no_mangle]
pub extern "C" fn yp_ism_init() -> c_int {
    let mut guard = ISM.write().unwrap();
    *guard = Some(IntelligentShardManager::new());
    0
}

/// Build the ISM from parallel arrays of ids and 32-byte hashes.
///
/// `ids` points to `count` little-endian u64 ids.
/// `hashes` points to `count * 32` bytes of hash data.
#[no_mangle]
pub extern "C" fn yp_ism_build_flat(
    ids: *const u64,
    hashes: *const u8,
    count: c_uint,
) -> c_int {
    if ids.is_null() || hashes.is_null() || count == 0 {
        return -1;
    }
    let count = count as usize;
    let ids = unsafe { std::slice::from_raw_parts(ids, count) };
    let hashes = unsafe { std::slice::from_raw_parts(hashes, count * 32) };

    let mut guard = ISM.write().unwrap();
    let mut mgr = guard.take().unwrap_or_else(IntelligentShardManager::new);
    match mgr.build_parallel_flat(ids, hashes, 32) {
        Ok(()) => {
            *guard = Some(mgr);
            0
        }
        Err(_) => {
            *guard = Some(mgr);
            -2
        }
    }
}

/// Build the ISM from parallel arrays with the intelligent watchdog enabled.
/// Returns 0 on success, -3 if the watchdog killed the build.
#[no_mangle]
pub extern "C" fn yp_ism_build_with_watchdog_flat(
    ids: *const u64,
    hashes: *const u8,
    count: c_uint,
) -> c_int {
    if ids.is_null() || hashes.is_null() || count == 0 {
        return -1;
    }
    let count = count as usize;
    let ids = unsafe { std::slice::from_raw_parts(ids, count) };
    let hashes = unsafe { std::slice::from_raw_parts(hashes, count * 32) };

    let mut papers = Vec::with_capacity(count);
    for i in 0..count {
        let id = ids[i];
        let off = i * 32;
        let pap: IsmPap = hashes[off..off + 32].try_into().unwrap_or([0u8; 32]);
        papers.push((id, pap));
    }

    let mut guard = ISM.write().unwrap();
    let mut mgr = guard.take().unwrap_or_else(IntelligentShardManager::new);
    match mgr.build_parallel_with_watchdog(papers) {
        Ok(()) => {
            *guard = Some(mgr);
            0
        }
        Err(ISMError::KilledByWatchdog { reason, advice }) => {
            *LAST_WATCHDOG_ADVICE.lock().unwrap() = Some(format!("{}\n{}", reason, advice));
            *guard = Some(mgr);
            -3
        }
        Err(_) => {
            *guard = Some(mgr);
            -2
        }
    }
}

/// Build the ISM from a memory-mapped file of fixed-size records.
///
/// `path` is a null-terminated UTF-8 file path. The file must contain exactly
/// `n` records of `8 + hash_len` bytes: a little-endian u64 id followed by the
/// hash bytes. `sequential` selects single-threaded shard construction (set to
/// 1 for deterministic, low-memory builds) vs parallel shard construction.
#[no_mangle]
pub extern "C" fn yp_ism_build_from_file(
    path: *const c_char,
    n: c_uint,
    hash_len: c_uint,
    sequential: c_int,
) -> c_int {
    if path.is_null() || n == 0 || hash_len == 0 {
        return -1;
    }
    let path = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    let mut guard = ISM.write().unwrap();
    let mut mgr = guard.take().unwrap_or_else(IntelligentShardManager::new);
    let result = if sequential != 0 {
        mgr.build_flat_from_file_sequential(path, n as usize, hash_len as usize)
    } else {
        mgr.build_parallel_flat_from_file(path, n as usize, hash_len as usize)
    };
    match result {
        Ok(()) => {
            *guard = Some(mgr);
            0
        }
        Err(_) => {
            *guard = Some(mgr);
            -2
        }
    }
}

/// Build the ISM from a packed byte buffer.
///
/// Layout: for each of `count` papers, 8 little-endian id bytes followed by
/// 32 hash bytes (40 bytes total per paper).
#[no_mangle]
pub extern "C" fn yp_ism_build(paps: *const u8, count: c_uint) -> c_int {
    if paps.is_null() || count == 0 {
        return -1;
    }
    let count = count as usize;
    let total = count.checked_mul(RECORD_BYTES).unwrap_or(0);
    let bytes = unsafe { std::slice::from_raw_parts(paps, total) };

    let mut papers = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * RECORD_BYTES;
        let id = u64::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
            bytes[off + 4],
            bytes[off + 5],
            bytes[off + 6],
            bytes[off + 7],
        ]);
        let hash: IsmPap = bytes[off + 8..off + RECORD_BYTES]
            .try_into()
            .unwrap_or([0u8; 32]);
        papers.push((id, hash));
    }

    let mut guard = ISM.write().unwrap();
    let mut mgr = guard.take().unwrap_or_else(IntelligentShardManager::new);
    match mgr.build_parallel(papers) {
        Ok(()) => {
            *guard = Some(mgr);
            0
        }
        Err(_) => {
            *guard = Some(mgr);
            -2
        }
    }
}

/// Query the ISM by cell id. Returns a JSON array of matches; caller frees
/// the returned string with `yp_ism_free_string`.
#[no_mangle]
pub extern "C" fn yp_ism_query(cell_id: u64) -> *mut c_char {
    let guard = ISM.read().unwrap();
    let json = match guard.as_ref() {
        Some(mgr) => {
            let results = mgr.query_parallel(cell_id);
            if results.is_empty() {
                "[]".to_string()
            } else {
                let arr: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "cell_id": r.cell_id,
                            "shard_id": r.shard_id,
                            "data": r.data,
                        })
                    })
                    .collect();
                serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
            }
        }
        None => "{\"error\":\"not initialized\"}".to_string(),
    };
    CString::new(json).unwrap_or_default().into_raw()
}

/// Return JSON status: shard count, total papers, build time, learned params.
#[no_mangle]
pub extern "C" fn yp_ism_status() -> *mut c_char {
    let guard = ISM.read().unwrap();
    let json = match guard.as_ref() {
        Some(mgr) => mgr.status_json(),
        None => "{\"error\":\"not initialized\"}".to_string(),
    };
    CString::new(json).unwrap_or_default().into_raw()
}

/// Return the last watchdog advice if the build was killed, otherwise empty.
#[no_mangle]
pub extern "C" fn yp_ism_watchdog_status() -> *mut c_char {
    let json = match LAST_WATCHDOG_ADVICE.lock().unwrap().as_ref() {
        Some(advice) => serde_json::json!({
            "triggered": true,
            "advice": advice,
        }),
        None => serde_json::json!({
            "triggered": false,
            "advice": "",
        }),
    }
    .to_string();
    CString::new(json).unwrap_or_default().into_raw()
}

/// Free a string returned by this FFI module.
#[no_mangle]
pub extern "C" fn yp_ism_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}
