// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, CString};

#[no_mangle]
pub extern "C" fn yp_predictor_set_congruence_filter(handle: c_int, enable: c_int) -> c_int {
    // Stub: store flag in static for now
    static mut FILTER_ENABLED: bool = false;
    unsafe {
        FILTER_ENABLED = enable != 0;
    }
    0
}
