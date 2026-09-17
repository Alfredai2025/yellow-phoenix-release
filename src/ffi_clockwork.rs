// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Constitutional Clockwork FFI Bridge
//!
//! Exports the Golden Hash, ITF, EXM, and sentinel subsystems to Python via
//! a C ABI. All singletons are lazily initialized on first use and protected
//! by a mutex.

use std::sync::{Mutex, OnceLock};

use libc::{c_float, c_int, c_void};
use std::os::raw::c_ulonglong;

use crate::exm::ExamBoundary;
use crate::golden_hash::{GoldenHash, TelemetrySnapshot};
use crate::itf::IntelligentTelemetryFilter;
use crate::sentinels::MemorySentinel;

// =============================================================================
// SINGLETONS
// =============================================================================

static GOLDEN_HASH: OnceLock<Mutex<GoldenHash>> = OnceLock::new();
static ITF: OnceLock<Mutex<IntelligentTelemetryFilter>> = OnceLock::new();
static EXM: OnceLock<Mutex<ExamBoundary>> = OnceLock::new();
static MEMORY_SENTINEL: OnceLock<Mutex<MemorySentinel>> = OnceLock::new();

fn golden_hash() -> Option<&'static Mutex<GoldenHash>> {
    GOLDEN_HASH.get_or_init(|| Mutex::new(GoldenHash::new()));
    GOLDEN_HASH.get()
}

fn itf() -> Option<&'static Mutex<IntelligentTelemetryFilter>> {
    ITF.get_or_init(|| Mutex::new(IntelligentTelemetryFilter::new()));
    ITF.get()
}

fn exm() -> Option<&'static Mutex<ExamBoundary>> {
    EXM.get_or_init(|| Mutex::new(ExamBoundary::new()));
    EXM.get()
}

fn memory_sentinel() -> Option<&'static Mutex<MemorySentinel>> {
    MEMORY_SENTINEL.get_or_init(|| Mutex::new(MemorySentinel::new(90.0)));
    MEMORY_SENTINEL.get()
}

// =============================================================================
// GOLDEN HASH
// =============================================================================

/// Initialize all clockwork singletons. Safe to call multiple times.
#[no_mangle]
pub extern "C" fn yp_clockwork_init() -> c_int {
    let _ = golden_hash();
    let _ = itf();
    let _ = exm();
    let _ = memory_sentinel();
    0
}

/// Advance the Golden Hash sovereign cycle and return the current RTC epoch.
#[no_mangle]
pub extern "C" fn yp_golden_hash_tick() -> c_ulonglong {
    match golden_hash() {
        Some(g) => g.lock().map(|gh| gh.tick()).unwrap_or(0),
        None => 0,
    }
}

/// Run constitutional adjudication on the supplied telemetry.
///
/// On success, writes the override level (0..3) to `out_level` and the target
/// gear index (0..9, or -1 for none) to `out_gear`. Returns 0.
/// On failure, returns -1.
#[no_mangle]
pub extern "C" fn yp_golden_hash_adjudicate(
    p50_ms: c_float,
    p99_ms: c_float,
    r1: c_float,
    r5: c_float,
    cpu_temp: c_float,
    memory_pct: c_float,
    autopoiesis: c_float,
    exam_violation: c_int,
    git_violations: c_int,
    circuit_trips: c_int,
    out_level: *mut c_int,
    out_gear: *mut c_int,
) -> c_int {
    if out_level.is_null() || out_gear.is_null() {
        return -1;
    }

    let telemetry = TelemetrySnapshot {
        p50_ms,
        p99_ms,
        r1,
        r5,
        cpu_temp,
        memory_pct,
        autopoiesis,
        exam_violation: exam_violation != 0,
        git_violations: git_violations.max(0) as u32,
        circuit_trips: circuit_trips.max(0) as u32,
    };

    let gh = match golden_hash() {
        Some(g) => g,
        None => return -1,
    };

    let (level, gear) = gh.lock().map(|g| g.adjudicate(&telemetry)).unwrap_or((crate::golden_hash::AdjudicationLevel::None, None));

    unsafe {
        *out_level = level as c_int;
        *out_gear = gear.map(|g| g.index() as c_int).unwrap_or(-1);
    }
    0
}

// =============================================================================
// ITF
// =============================================================================

/// Evaluate telemetry against ITF anchors. Writes the number of alert frames
/// to `out_count` and a compressed 32-bit alert word to `out_word`.
///
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn yp_itf_filter(
    tick: c_ulonglong,
    p50_ms: c_float,
    p99_ms: c_float,
    r1: c_float,
    r5: c_float,
    cpu_temp: c_float,
    memory_pct: c_float,
    autopoiesis: c_float,
    exam_violation: c_int,
    git_violations: c_int,
    circuit_trips: c_int,
    out_count: *mut c_int,
    out_word: *mut c_ulonglong,
) -> c_int {
    if out_count.is_null() {
        return -1;
    }

    let telemetry = TelemetrySnapshot {
        p50_ms,
        p99_ms,
        r1,
        r5,
        cpu_temp,
        memory_pct,
        autopoiesis,
        exam_violation: exam_violation != 0,
        git_violations: git_violations.max(0) as u32,
        circuit_trips: circuit_trips.max(0) as u32,
    };

    let filter = match itf() {
        Some(f) => f,
        None => return -1,
    };

    let frames = filter.lock().map(|f| f.filter(tick as u64, &telemetry)).unwrap_or_default();
    let word = filter.lock().map(|f| f.compress(&frames)).unwrap_or(0);

    unsafe {
        *out_count = frames.len() as c_int;
        if !out_word.is_null() {
            *out_word = word as c_ulonglong;
        }
    }
    0
}

// =============================================================================
// EXM
// =============================================================================

/// Check whether `content` is in the exam set. Returns 1 if blocked, 0 if
/// allowed, -1 if the EXM singleton is unavailable.
#[no_mangle]
pub extern "C" fn yp_exm_check(content: *const c_void, len: usize) -> c_int {
    if content.is_null() {
        return 0;
    }
    let content = unsafe { std::slice::from_raw_parts(content as *const u8, len) };
    match exm() {
        Some(e) => {
            let blocked = e.lock().map(|exm| exm.block_ingestion(content)).unwrap_or(false);
            if blocked { 1 } else { 0 }
        }
        None => -1,
    }
}

/// Load a list of exam fingerprints into the EXM index.
/// `fingerprints` points to an array of `count` little-endian `u64` values.
/// Returns 0 on success, -1 on failure.
#[no_mangle]
pub extern "C" fn yp_exm_load(fingerprints: *const c_ulonglong, count: usize) -> c_int {
    if fingerprints.is_null() {
        return -1;
    }
    let fps = unsafe { std::slice::from_raw_parts(fingerprints, count) };
    match exm() {
        Some(e) => {
            if let Ok(mut exm) = e.lock() {
                exm.load(fps);
            }
            0
        }
        None => -1,
    }
}

/// Return EXM statistics. Writes block count to `out_blocks` and false-positive
/// count to `out_fp`. Returns 0 on success.
#[no_mangle]
pub extern "C" fn yp_exm_stats(out_blocks: *mut c_ulonglong, out_fp: *mut c_ulonglong) -> c_int {
    match exm() {
        Some(e) => {
            let (blocks, fp) = e.lock().map(|exm| exm.stats()).unwrap_or((0, 0));
            unsafe {
                if !out_blocks.is_null() {
                    *out_blocks = blocks;
                }
                if !out_fp.is_null() {
                    *out_fp = fp;
                }
            }
            0
        }
        None => -1,
    }
}

// =============================================================================
// SENTINELS
// =============================================================================

/// Return 1 if the memory sentinel is distressed, 0 otherwise, -1 on error.
#[no_mangle]
pub extern "C" fn yp_memory_distressed() -> c_int {
    match memory_sentinel() {
        Some(s) => {
            let distressed = s.lock().map(|m| m.is_distressed()).unwrap_or(false);
            if distressed { 1 } else { 0 }
        }
        None => -1,
    }
}

/// Hard query timeout limit in milliseconds.
#[no_mangle]
pub extern "C" fn yp_query_timeout_ms() -> c_ulonglong {
    5000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_init() {
        assert_eq!(yp_clockwork_init(), 0);
        assert!(yp_golden_hash_tick() > 0);
    }

    #[test]
    fn test_ffi_adjudicate_normal() {
        yp_clockwork_init();
        let mut level: c_int = -1;
        let mut gear: c_int = -1;
        let rc = yp_golden_hash_adjudicate(
            4.0, 6.0, 0.92, 0.98, 45.0, 50.0, 0.95, 0, 0, 0,
            &mut level, &mut gear,
        );
        assert_eq!(rc, 0);
        assert_eq!(level, 0);
        assert_eq!(gear, -1);
    }

    #[test]
    fn test_ffi_adjudicate_thermal() {
        yp_clockwork_init();
        let mut level: c_int = -1;
        let mut gear: c_int = -1;
        let rc = yp_golden_hash_adjudicate(
            4.0, 6.0, 0.92, 0.98, 80.0, 50.0, 0.95, 0, 0, 0,
            &mut level, &mut gear,
        );
        assert_eq!(rc, 0);
        assert_eq!(level, 3);
        assert_eq!(gear, 8);
    }

    #[test]
    fn test_ffi_exm() {
        yp_clockwork_init();
        let exam = b"held-out exam content";
        let normal = b"ordinary training content";

        // Seed EXM with the exam fingerprint via the load helper.
        let fp = crate::exm::hash_content(exam);
        let rc = yp_exm_load(&fp, 1);
        assert_eq!(rc, 0);

        assert_eq!(yp_exm_check(exam.as_ptr() as *const c_void, exam.len()), 1);
        assert_eq!(yp_exm_check(normal.as_ptr() as *const c_void, normal.len()), 0);
    }

    #[test]
    fn test_ffi_memory_and_timeout() {
        assert!(yp_query_timeout_ms() > 0);
        let distressed = yp_memory_distressed();
        assert!(distressed == 0 || distressed == 1);
    }
}
