// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Self-tuner state machine — shadow-mode autopoiesis.
//!
//! Mirrors the Python `AutopoiesisLoop` logic in Rust. It receives observation
//! JSON, maintains a history, and returns what it *would* do. It never touches
//! the index; Python evaluates and applies any proposal.

use alloc::{string::String, vec::Vec};
use core::ffi::c_void;
use serde::{Deserialize, Serialize};

/// One observation from the Python side.
#[derive(Debug, Clone, Deserialize)]
pub struct Observation {
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub avg_us: f64,
    pub degraded: bool,
    pub ef: i32,
}

/// Diagnosis returned to Python.
#[derive(Debug, Clone, Serialize)]
pub struct Diagnosis {
    pub phase: &'static str,
    pub status: &'static str,
    pub action: Option<&'static str>,
    pub proposed_ef: Option<i32>,
    pub delta_us: Option<f64>,
    pub delta_pct: Option<f64>,
}

/// Proposal returned to Python.
#[derive(Debug, Clone, Serialize)]
pub struct Proposal {
    pub phase: &'static str,
    pub action: Option<&'static str>,
    pub proposed_ef: Option<i32>,
}

/// Evaluation acknowledgment.
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationAck {
    pub phase: &'static str,
    pub applied: bool,
    pub current_ef: i32,
}

/// Evaluation input from Python.
#[derive(Debug, Clone, Deserialize)]
pub struct EvaluationInput {
    pub applied: bool,
    pub new_ef: i32,
}

pub struct SelfTuner {
    threshold_us: f64,
    history: Vec<Observation>,
    current_ef: i32,
}

impl SelfTuner {
    pub fn new(threshold_us: f64) -> Self {
        Self {
            threshold_us,
            history: Vec::new(),
            current_ef: 50,
        }
    }

    pub fn submit_observation(&mut self, obs: Observation) -> Diagnosis {
        self.current_ef = obs.ef;
        self.history.push(obs.clone());

        if self.history.len() < 2 {
            return Diagnosis {
                phase: "diagnose",
                status: "baseline",
                action: None,
                proposed_ef: None,
                delta_us: None,
                delta_pct: None,
            };
        }

        let prev = &self.history[self.history.len() - 2];
        let curr = &self.history[self.history.len() - 1];
        let delta = curr.p50_us - prev.p50_us;
        let delta_pct = if prev.p50_us > 0.0 {
            (delta / prev.p50_us) * 100.0
        } else {
            0.0
        };

        if curr.degraded {
            Diagnosis {
                phase: "diagnose",
                status: "degraded",
                action: Some("increase_ef"),
                proposed_ef: Some(curr.ef + 25),
                delta_us: Some(delta),
                delta_pct: Some(delta_pct),
            }
        } else if delta_pct > 10.0 {
            Diagnosis {
                phase: "diagnose",
                status: "warming",
                action: Some("monitor"),
                proposed_ef: None,
                delta_us: Some(delta),
                delta_pct: Some(delta_pct),
            }
        } else {
            Diagnosis {
                phase: "diagnose",
                status: "healthy",
                action: None,
                proposed_ef: None,
                delta_us: Some(delta),
                delta_pct: Some(delta_pct),
            }
        }
    }

    pub fn propose(&self) -> Proposal {
        if self.history.len() < 2 {
            return Proposal {
                phase: "propose",
                action: None,
                proposed_ef: None,
            };
        }

        let prev = &self.history[self.history.len() - 2];
        let curr = &self.history[self.history.len() - 1];
        let delta = curr.p50_us - prev.p50_us;
        let delta_pct = if prev.p50_us > 0.0 {
            (delta / prev.p50_us) * 100.0
        } else {
            0.0
        };

        if curr.degraded {
            Proposal {
                phase: "propose",
                action: Some("increase_ef"),
                proposed_ef: Some(curr.ef + 25),
            }
        } else if delta_pct > 10.0 {
            Proposal {
                phase: "propose",
                action: Some("monitor"),
                proposed_ef: None,
            }
        } else {
            Proposal {
                phase: "propose",
                action: None,
                proposed_ef: None,
            }
        }
    }

    pub fn submit_evaluation(&mut self, eval: EvaluationInput) -> EvaluationAck {
        if eval.applied {
            self.current_ef = eval.new_ef;
        }
        EvaluationAck {
            phase: "consolidate",
            applied: eval.applied,
            current_ef: self.current_ef,
        }
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

unsafe fn c_str_to_str(ptr: *const std::ffi::c_char) -> Option<&'static str> {
    if ptr.is_null() {
        return None;
    }
    let len = libc::strlen(ptr);
    let slice = core::slice::from_raw_parts(ptr as *const u8, len);
    core::str::from_utf8(slice).ok()
}

fn json_to_c_string<T: Serialize>(value: &T) -> *mut std::ffi::c_char {
    match serde_json::to_string(value) {
        Ok(s) => {
            let c = match std::ffi::CString::new(s) {
                Ok(c) => c,
                Err(_) => return std::ptr::null_mut(),
            };
            c.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn yp_tuner_new(threshold_us: f64) -> *mut c_void {
    let tuner = Box::new(SelfTuner::new(threshold_us));
    Box::into_raw(tuner) as *mut c_void
}

#[no_mangle]
pub extern "C" fn yp_tuner_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr as *mut SelfTuner);
        }
    }
}

#[no_mangle]
pub extern "C" fn yp_tuner_submit_observation(
    ptr: *mut c_void,
    json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    if ptr.is_null() || json.is_null() {
        return json_to_c_string(&Diagnosis {
            phase: "diagnose",
            status: "error",
            action: None,
            proposed_ef: None,
            delta_us: None,
            delta_pct: None,
        });
    }
    let tuner = unsafe { &mut *(ptr as *mut SelfTuner) };
    let obs: Observation = match unsafe { c_str_to_str(json) }
        .and_then(|s| serde_json::from_str(s).ok())
    {
        Some(o) => o,
        None => {
            return json_to_c_string(&Diagnosis {
                phase: "diagnose",
                status: "error",
                action: None,
                proposed_ef: None,
                delta_us: None,
                delta_pct: None,
            });
        }
    };
    let diag = tuner.submit_observation(obs);
    json_to_c_string(&diag)
}

#[no_mangle]
pub extern "C" fn yp_tuner_propose(ptr: *mut c_void) -> *mut std::ffi::c_char {
    if ptr.is_null() {
        return json_to_c_string(&Proposal {
            phase: "propose",
            action: None,
            proposed_ef: None,
        });
    }
    let tuner = unsafe { &*(ptr as *mut SelfTuner) };
    let prop = tuner.propose();
    json_to_c_string(&prop)
}

#[no_mangle]
pub extern "C" fn yp_tuner_submit_evaluation(
    ptr: *mut c_void,
    json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    if ptr.is_null() || json.is_null() {
        return json_to_c_string(&EvaluationAck {
            phase: "consolidate",
            applied: false,
            current_ef: 50,
        });
    }
    let tuner = unsafe { &mut *(ptr as *mut SelfTuner) };
    let eval: EvaluationInput = match unsafe { c_str_to_str(json) }
        .and_then(|s| serde_json::from_str(s).ok())
    {
        Some(e) => e,
        None => {
            return json_to_c_string(&EvaluationAck {
                phase: "consolidate",
                applied: false,
                current_ef: tuner.current_ef,
            });
        }
    };
    let ack = tuner.submit_evaluation(eval);
    json_to_c_string(&ack)
}

#[no_mangle]
pub extern "C" fn yp_tuner_free_string(ptr: *mut std::ffi::c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(ptr);
        }
    }
}
