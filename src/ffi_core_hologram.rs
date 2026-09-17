// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::c_void;
use crate::core::hologram::StandingHologram;
use crate::types::multivector::BinaryMultivector;

/// Create a new 128-dim wave-field hologram.
#[no_mangle]
pub extern "C" fn yp_wave_new() -> *mut c_void {
    let h = Box::new(StandingHologram::<128>::new());
    Box::into_raw(h) as *mut c_void
}

/// Inject a 128-bit paper into the wave field.
/// `bits` must point to 16 bytes (128 bits).
#[no_mangle]
pub extern "C" fn yp_wave_inject(ptr: *mut c_void, bits: *const u8, len: usize) -> i32 {
    if ptr.is_null() || bits.is_null() || len != 16 {
        return -1;
    }
    let h = unsafe { &mut *(ptr as *mut StandingHologram<128>) };
    let bytes = unsafe { std::slice::from_raw_parts(bits, len) };

    let mut mv = BinaryMultivector::new();
    for byte_idx in 0..16 {
        for bit_idx in 0..8 {
            let global_bit = byte_idx * 8 + bit_idx;
            if (bytes[byte_idx] >> (7 - bit_idx)) & 1 == 1 {
                mv.set_bit(global_bit, true);
            }
        }
    }
    h.inject(&mv);
    0
}

/// Query the wave field with a 128-bit pattern.
/// Returns the signed resonance score (clamped to non-negative).
#[no_mangle]
pub extern "C" fn yp_wave_query(ptr: *mut c_void, bits: *const u8, len: usize) -> i32 {
    if ptr.is_null() || bits.is_null() || len != 16 {
        return -1;
    }
    let h = unsafe { &mut *(ptr as *mut StandingHologram<128>) };
    let bytes = unsafe { std::slice::from_raw_parts(bits, len) };

    let mut mv = BinaryMultivector::new();
    for byte_idx in 0..16 {
        for bit_idx in 0..8 {
            let global_bit = byte_idx * 8 + bit_idx;
            if (bytes[byte_idx] >> (7 - bit_idx)) & 1 == 1 {
                mv.set_bit(global_bit, true);
            }
        }
    }
    h.query(&mv) as i32
}

/// Free the hologram.
#[no_mangle]
pub extern "C" fn yp_wave_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut StandingHologram<128>);
        }
    }
}

/// Return number of injected papers (count from wave amplitude stats).
#[no_mangle]
pub extern "C" fn yp_wave_count(ptr: *mut c_void) -> usize {
    if ptr.is_null() {
        return 0;
    }
    let h = unsafe { &*(ptr as *mut StandingHologram<128>) };
    // StandingHologram doesn't expose count directly; return 0 for now
    0
}
