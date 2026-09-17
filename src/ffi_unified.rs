// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Unified FFI exports for the iOS 12-feature completion manifest.
//! These functions are thin placeholders / adapters where the underlying
//! engine modules already exist; they provide the exact C ABI the Swift
//! layer expects.

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_uint, c_uchar};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

// ═══════════════════════════════════════════════════════════════
// PHASE 2A — Domain detection (Item 9)
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
#[derive(Clone, Copy)]
pub enum YPDomainShard {
    General = 0,
    CS = 1,
    Medical = 2,
    Legal = 3,
}

#[no_mangle]
pub extern "C" fn yp_detect_domain(query_text: *const c_char) -> YPDomainShard {
    if query_text.is_null() {
        return YPDomainShard::General;
    }
    let query = unsafe { CStr::from_ptr(query_text).to_string_lossy() };
    let q = query.to_lowercase();
    if q.contains("medical") || q.contains("patient") || q.contains("clinical") {
        YPDomainShard::Medical
    } else if q.contains("legal") || q.contains("law") || q.contains("court") {
        YPDomainShard::Legal
    } else if q.contains("neural") || q.contains("algorithm") || q.contains("network") {
        YPDomainShard::CS
    } else {
        YPDomainShard::General
    }
}

#[no_mangle]
pub extern "C" fn yp_load_shard_index(
    _shard: YPDomainShard,
    _hnsw_path: *const c_char,
    _ism_path: *const c_char,
) -> i32 {
    // WIRE TO: unload current shard, load only active shard.
    // Returning 0 lets the iOS flow continue for now.
    0
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2B — SAH beacon index (Item 10)
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn yp_sah_search(
    query_text: *const c_char,
    beacon_hits_out: *mut c_uint,
    _beacon_scores_out: *mut c_float,
    max_beacons: usize,
) -> c_uint {
    if query_text.is_null() || beacon_hits_out.is_null() || max_beacons == 0 {
        return 0;
    }
    // WIRE TO: sah_beacon_index.bin (507 beacons)
    // Returning 0 means "no beacon match", so iOS falls back to HNSW.
    0
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2C — Adaptive cascade prefix filter (Item 8)
// ═══════════════════════════════════════════════════════════════

#[repr(C)]
pub struct YPPrefixResult {
    pub candidate_count: c_uint,
    pub recall_estimate: c_float,
}

#[no_mangle]
pub extern "C" fn yp_cascade_prefix_search(
    query_hash: *const c_uchar,
    _prefix_bits: c_uint,
    _top_percentile: c_float,
    _candidates_out: *mut c_uint,
    _candidates_cap: usize,
) -> YPPrefixResult {
    if query_hash.is_null() {
        return YPPrefixResult {
            candidate_count: 0,
            recall_estimate: 0.0,
        };
    }
    // WIRE TO: Aug 3 adaptive cascade experiments
    YPPrefixResult {
        candidate_count: 0,
        recall_estimate: 0.0,
    }
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2D — PQ/OPQ quantized fallback (Item 11)
// ═══════════════════════════════════════════════════════════════

static DISCOVERY_MODE: AtomicBool = AtomicBool::new(false);

/// Tiered PQ search that does not collide with the existing `yp_pq_search`
/// symbol in `ffi_pq.rs`.
#[no_mangle]
pub extern "C" fn yp_pq_search_tiered(
    _query_vec: *const c_float,
    _dim: usize,
    _tier: c_uchar,
    _top_k: usize,
    _ids_out: *mut c_uint,
    _scores_out: *mut c_float,
) -> c_uint {
    // WIRE TO: opq_rotation_20260727.npz + pq_opq_8x48_256_20260727.npz
    // Returning 0 means "no PQ results", so iOS falls back to HNSW.
    0
}

#[no_mangle]
pub extern "C" fn yp_set_discovery_mode(enabled: bool) {
    DISCOVERY_MODE.store(enabled, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn yp_get_discovery_mode() -> bool {
    DISCOVERY_MODE.load(Ordering::Relaxed)
}

// ═══════════════════════════════════════════════════════════════
// PHASE 2E — Thermal throttling (Item 12)
// ═══════════════════════════════════════════════════════════════

static THERMAL_STATE: AtomicI32 = AtomicI32::new(0); // 0=nominal, 1=fair, 2=serious, 3=critical

#[no_mangle]
pub extern "C" fn yp_set_thermal_state(state: i32) {
    THERMAL_STATE.store(state, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn yp_get_thermal_state() -> i32 {
    THERMAL_STATE.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn yp_thermal_throttle_ef(default_ef: i32) -> i32 {
    match THERMAL_STATE.load(Ordering::Relaxed) {
        2 => (default_ef as f32 * 0.7) as i32, // serious: reduce 30%
        3 => (default_ef as f32 * 0.5) as i32, // critical: reduce 50%
        _ => default_ef,
    }
}

// ═══════════════════════════════════════════════════════════════
// Beacon index stubs (referenced by YellowPhoenix.h / iOS native engine)
// ═══════════════════════════════════════════════════════════════

#[no_mangle]
pub extern "C" fn yp_beacon_index_new(_capacity: u32) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn yp_beacon_index_insert(
    _hash_ptr: *const u8,
    _hash_len: usize,
    _id: u32,
    _tag: u8,
) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn yp_beacon_index_search(
    _hash_ptr: *const u8,
    _hash_len: usize,
    _k: u32,
    _out_ids: *mut u32,
    _out_dists: *mut u32,
    _out_tags: *mut u8,
    _out_count: *mut u32,
) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn yp_beacon_index_count() -> u32 {
    0
}

#[no_mangle]
pub extern "C" fn yp_beacon_index_get_hash(_node_id: u32, _out_hash: *mut u8) -> i32 {
    -1
}

#[no_mangle]
pub extern "C" fn yp_beacon_index_save(_path_ptr: *const u8, _path_len: usize) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn yp_beacon_index_load(_path_ptr: *const u8, _path_len: usize) -> i32 {
    0
}
