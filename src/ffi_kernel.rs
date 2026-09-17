// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! FFI bridge that exposes the GeometricAutopoiesis kernel to Python.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;
use core::slice;

use crate::core::geometric_autopoiesis::{
    Diagnosis, Evaluation, GeometricAutopoiesis, GeometricState,
};
use crate::core::hologram_index::StandingHologram;
use crate::types::multivector::BinaryMultivector;
use crate::types::ternary::TernaryMultivector;

// FFI-safe type alias (kept for spec compliance).
#[allow(dead_code)]
type size_t = usize;

// ---------------------------------------------------------------------------
// FFI-safe structs
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct FfiBinaryMultivector {
    pub chunk0: u64,
    pub chunk1: u64,
}

#[repr(C)]
pub struct FfiTernaryMultivector {
    pub pos0: u64,
    pub pos1: u64,
    pub neg0: u64,
    pub neg1: u64,
}

#[repr(C)]
pub struct FfiDiagnosis {
    pub stuck: bool,
    pub novelty_score: f32,
    pub blind_spot: f32,
    pub exploration_radius: i16,
}

#[repr(C)]
pub struct FfiProposal {
    pub bridge_idx: usize,
    pub bridge_score: f32,
    pub conformal_distance: f32,
    pub superposition: FfiTernaryMultivector,
    pub ok: bool,
}

#[repr(C)]
pub struct FfiEvaluation {
    pub consensus: f32,
    pub coverage: f32,
    pub novelty: f32,
    pub approved: bool,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn to_ffi_binary(bm: &BinaryMultivector) -> FfiBinaryMultivector {
    FfiBinaryMultivector {
        chunk0: bm.0[0],
        chunk1: bm.0[1],
    }
}

fn from_ffi_binary(ffi: *const FfiBinaryMultivector) -> BinaryMultivector {
    if ffi.is_null() {
        return BinaryMultivector([0; 2]);
    }
    unsafe {
        let raw = ptr::read(ffi);
        BinaryMultivector([raw.chunk0, raw.chunk1])
    }
}

fn to_ffi_ternary(tm: &TernaryMultivector) -> FfiTernaryMultivector {
    FfiTernaryMultivector {
        pos0: tm.pos[0],
        pos1: tm.pos[1],
        neg0: tm.neg[0],
        neg1: tm.neg[1],
    }
}

#[allow(dead_code)]
fn from_ffi_ternary(ffi: *const FfiTernaryMultivector) -> TernaryMultivector {
    if ffi.is_null() {
        return TernaryMultivector::new();
    }
    unsafe {
        let raw = ptr::read(ffi);
        TernaryMultivector {
            pos: [raw.pos0, raw.pos1],
            neg: [raw.neg0, raw.neg1],
        }
    }
}

fn to_ffi_diagnosis(d: &Diagnosis) -> FfiDiagnosis {
    FfiDiagnosis {
        stuck: d.stuck,
        novelty_score: d.novelty_score,
        blind_spot: d.blind_spot,
        exploration_radius: d.exploration_radius,
    }
}

fn from_ffi_diagnosis(ffi: *const FfiDiagnosis) -> Diagnosis {
    if ffi.is_null() {
        return Diagnosis {
            stuck: false,
            novelty_score: 0.0,
            blind_spot: 0.0,
            exploration_radius: 0,
        };
    }
    unsafe {
        let d = ptr::read(ffi);
        Diagnosis {
            stuck: d.stuck,
            novelty_score: d.novelty_score,
            blind_spot: d.blind_spot,
            exploration_radius: d.exploration_radius,
        }
    }
}

fn to_ffi_evaluation(e: &Evaluation) -> FfiEvaluation {
    FfiEvaluation {
        consensus: e.consensus,
        coverage: e.coverage,
        novelty: e.novelty,
        approved: e.approved,
    }
}

#[allow(dead_code)]
fn from_ffi_evaluation(ffi: *const FfiEvaluation) -> Evaluation {
    if ffi.is_null() {
        return Evaluation {
            consensus: 0.0,
            coverage: 0.0,
            novelty: 0.0,
            approved: false,
        };
    }
    unsafe {
        let e = ptr::read(ffi);
        Evaluation {
            consensus: e.consensus,
            coverage: e.coverage,
            novelty: e.novelty,
            approved: e.approved,
        }
    }
}

// ---------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------

/// Create an opaque kernel handle with an empty geometric state.
#[no_mangle]
pub unsafe extern "C" fn kernel_new() -> *mut c_void {
    let state = GeometricState {
        scalar: BinaryMultivector([0; 2]),
        vector: BinaryMultivector([0; 2]),
        bivector: BinaryMultivector([0; 2]),
        history: Vec::new(),
    };
    let kernel = Box::new(GeometricAutopoiesis::new(state));
    Box::into_raw(kernel) as *mut c_void
}

/// Free a kernel created by `kernel_new`.  Null-safe.
#[no_mangle]
pub unsafe extern "C" fn kernel_drop(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut GeometricAutopoiesis);
    }
}

/// Overwrite the three grades of the current geometric state.
/// Null argument pointers leave the corresponding grade unchanged.
#[no_mangle]
pub unsafe extern "C" fn kernel_set_state(
    ptr: *mut c_void,
    scalar: *const FfiBinaryMultivector,
    vector: *const FfiBinaryMultivector,
    bivector: *const FfiBinaryMultivector,
) {
    if ptr.is_null() {
        return;
    }
    let kernel = &mut *(ptr as *mut GeometricAutopoiesis);
    if !scalar.is_null() {
        kernel.state.scalar = from_ffi_binary(scalar);
    }
    if !vector.is_null() {
        kernel.state.vector = from_ffi_binary(vector);
    }
    if !bivector.is_null() {
        kernel.state.bivector = from_ffi_binary(bivector);
    }
}

/// Record a snapshot of the current state into the internal history.
#[no_mangle]
pub unsafe extern "C" fn kernel_snapshot(ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }
    let kernel = &mut *(ptr as *mut GeometricAutopoiesis);
    kernel.state.snapshot();
}

/// Static observation: compute a superposition from `health` and `alarm`.
#[no_mangle]
pub unsafe extern "C" fn kernel_observe(
    health: *const FfiBinaryMultivector,
    alarm: *const FfiBinaryMultivector,
) -> FfiTernaryMultivector {
    if health.is_null() || alarm.is_null() {
        return to_ffi_ternary(&TernaryMultivector::new());
    }
    let h = from_ffi_binary(health);
    let a = from_ffi_binary(alarm);
    let state = GeometricState {
        scalar: BinaryMultivector([0; 2]),
        vector: BinaryMultivector([0; 2]),
        bivector: BinaryMultivector([0; 2]),
        history: Vec::new(),
    };
    let tmp = GeometricAutopoiesis::new(state);
    to_ffi_ternary(&tmp.observe(&h, &a))
}

/// Return a diagnostic view of the current kernel.
#[no_mangle]
pub unsafe extern "C" fn kernel_diagnose(ptr: *mut c_void) -> FfiDiagnosis {
    if ptr.is_null() {
        return FfiDiagnosis {
            stuck: false,
            novelty_score: 0.0,
            blind_spot: 0.0,
            exploration_radius: 0,
        };
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    to_ffi_diagnosis(&kernel.diagnose())
}

/// Propose a candidate bridge from `desired` using the list of `candidates`.
/// Returns `ok = false` when there is no valid bridge.
#[no_mangle]
pub unsafe extern "C" fn kernel_propose(
    ptr: *mut c_void,
    desired: *const FfiBinaryMultivector,
    candidates_ptr: *const FfiBinaryMultivector,
    candidates_len: usize,
) -> FfiProposal {
    if ptr.is_null() || desired.is_null() || candidates_ptr.is_null() || candidates_len == 0 {
        return FfiProposal {
            bridge_idx: 0,
            bridge_score: 0.0,
            conformal_distance: 0.0,
            superposition: to_ffi_ternary(&TernaryMultivector::new()),
            ok: false,
        };
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    let des = from_ffi_binary(desired);
    let cands = slice::from_raw_parts(candidates_ptr, candidates_len)
        .iter()
        .map(|ffi| from_ffi_binary(ffi))
        .collect::<Vec<_>>();

    match kernel.propose(&des, &cands) {
        Some(p) => FfiProposal {
            bridge_idx: p.bridge_idx,
            bridge_score: p.bridge_score,
            conformal_distance: p.conformal_distance,
            superposition: to_ffi_ternary(&p.superposition),
            ok: true,
        },
        None => FfiProposal {
            bridge_idx: 0,
            bridge_score: 0.0,
            conformal_distance: 0.0,
            superposition: to_ffi_ternary(&TernaryMultivector::new()),
            ok: false,
        },
    }
}

/// Evaluate a set of results against the current kernel state.
#[no_mangle]
pub unsafe extern "C" fn kernel_evaluate(
    ptr: *mut c_void,
    results_ptr: *const FfiBinaryMultivector,
    results_len: usize,
) -> FfiEvaluation {
    if ptr.is_null() || results_ptr.is_null() || results_len == 0 {
        return FfiEvaluation {
            consensus: 0.0,
            coverage: 0.0,
            novelty: 0.0,
            approved: false,
        };
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    let results = slice::from_raw_parts(results_ptr, results_len)
        .iter()
        .map(|ffi| from_ffi_binary(ffi))
        .collect::<Vec<_>>();
    to_ffi_evaluation(&kernel.evaluate(&results))
}

/// Predict a new multivector given a `proposed` point and time `t`.
#[no_mangle]
pub unsafe extern "C" fn kernel_predict(
    ptr: *mut c_void,
    proposed: *const FfiBinaryMultivector,
    t: f32,
) -> FfiBinaryMultivector {
    if ptr.is_null() || proposed.is_null() {
        return FfiBinaryMultivector { chunk0: 0, chunk1: 0 };
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    let prop = from_ffi_binary(proposed);
    to_ffi_binary(&kernel.predict(&prop, t))
}

/// Consolidate bridge hypotheses, writing indices of kept hypotheses
/// into `out_ptr`. Returns the number of written indices.
#[no_mangle]
pub unsafe extern "C" fn kernel_consolidate(
    ptr: *mut c_void,
    hbs_ptr: *const FfiBinaryMultivector,
    hbs_len: usize,
    out_ptr: *mut u64,
    out_capacity: usize,
) -> usize {
    if ptr.is_null() || hbs_ptr.is_null() || out_ptr.is_null() || hbs_len == 0 || out_capacity == 0 {
        return 0;
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    let hbs = slice::from_raw_parts(hbs_ptr, hbs_len)
        .iter()
        .map(|ffi| from_ffi_binary(ffi))
        .collect::<Vec<_>>();
    let kept = kernel.consolidate(&hbs);
    let count = kept.len().min(out_capacity);
    for i in 0..count {
        *out_ptr.add(i) = kept[i] as u64;
    }
    count
}

/// Safety check: returns `true` if the proposed movement from `before`
/// to `after` is within the safety radius.
#[no_mangle]
pub unsafe extern "C" fn kernel_safety_check(
    ptr: *mut c_void,
    before: *const FfiBinaryMultivector,
    after: *const FfiBinaryMultivector,
) -> bool {
    if ptr.is_null() || before.is_null() || after.is_null() {
        return false;
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    let bef = from_ffi_binary(before);
    let aft = from_ffi_binary(after);
    kernel.safety_check(&bef, &aft)
}

/// Update internal parameters after an evaluation cycle.
#[no_mangle]
pub unsafe extern "C" fn kernel_update(
    ptr: *mut c_void,
    diag: *const FfiDiagnosis,
    eval: *const FfiEvaluation,
    feedback: f32,
) {
    if ptr.is_null() || diag.is_null() || eval.is_null() {
        return;
    }
    let kernel = &mut *(ptr as *mut GeometricAutopoiesis);
    let d = from_ffi_diagnosis(diag);
    let e = from_ffi_evaluation(eval);
    kernel.update(&d, &e, feedback);
}

/// Return the current novelty threshold.
#[no_mangle]
pub unsafe extern "C" fn kernel_get_novelty_threshold(ptr: *mut c_void) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    kernel.novelty_threshold
}

/// Return the current safety radius.
#[no_mangle]
pub unsafe extern "C" fn kernel_get_safety_radius(ptr: *mut c_void) -> f32 {
    if ptr.is_null() {
        return 0.0;
    }
    let kernel = &*(ptr as *mut GeometricAutopoiesis);
    kernel.safety_radius
}

// ---------------------------------------------------------------------------
// StandingHologram FFI
// ---------------------------------------------------------------------------

/// Create an opaque standing hologram handle.
#[no_mangle]
pub unsafe extern "C" fn hologram_new() -> *mut c_void {
    let holo = Box::new(StandingHologram::new());
    Box::into_raw(holo) as *mut c_void
}

/// Free a hologram created by `hologram_new`.
#[no_mangle]
pub unsafe extern "C" fn hologram_drop(ptr: *mut c_void) {
    if !ptr.is_null() {
        let _ = Box::from_raw(ptr as *mut StandingHologram);
    }
}

/// Ingest a batch of binary multivectors into the hologram.
#[no_mangle]
pub unsafe extern "C" fn hologram_batch_inject(
    ptr: *mut c_void,
    vecs_ptr: *const FfiBinaryMultivector,
    vecs_len: usize,
) {
    if ptr.is_null() || vecs_ptr.is_null() || vecs_len == 0 {
        return;
    }
    let holo = &mut *(ptr as *mut StandingHologram);
    let slice = slice::from_raw_parts(vecs_ptr, vecs_len);
    for ffi in slice {
        holo.inject(&from_ffi_binary(ffi));
    }
}

/// Query the hologram for the `top_k` nearest prototypes.
/// Writes indices into `out_indices` (capacity `out_capacity`) and returns
/// the number of indices written.
#[no_mangle]
pub unsafe extern "C" fn hologram_query(
    ptr: *mut c_void,
    query: *const FfiBinaryMultivector,
    top_k: usize,
    out_indices: *mut usize,
    out_capacity: usize,
) -> usize {
    if ptr.is_null() || query.is_null() || out_indices.is_null() || out_capacity == 0 {
        return 0;
    }
    let holo = &*(ptr as *mut StandingHologram);
    let q = from_ffi_binary(query);
    let results = holo.resonance_query(&q, top_k.min(out_capacity));
    let count = results.len();
    for (i, (idx, _dist)) in results.iter().enumerate() {
        *out_indices.add(i) = *idx;
    }
    count
}
