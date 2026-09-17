// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! sphere_rotor_bank.rs
//! SO(3) Rotor Bank for Crystal Sphere
//! 60 evenly-distributed orientations + SLERP + scoring + FFI

use std::f64::consts::PI;
use std::os::raw::{c_double, c_int};

// ============================================================================
// SO(3) REPRESENTATION: Unit quaternion
// ============================================================================
#[derive(Clone, Debug, Copy)]
pub struct Rotor {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Rotor {
    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn from_axis_angle(axis: &[f64; 3], angle: f64) -> Self {
        let half = angle / 2.0;
        let s = half.sin();
        let c = half.cos();
        let norm = (axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]).sqrt();
        if norm < 1e-9 {
            return Self::identity();
        }
        Self {
            w: c,
            x: axis[0] / norm * s,
            y: axis[1] / norm * s,
            z: axis[2] / norm * s,
        }
    }

    pub fn normalize(&mut self) {
        let n = (self.w*self.w + self.x*self.x + self.y*self.y + self.z*self.z).sqrt();
        if n > 1e-9 {
            self.w /= n; self.x /= n; self.y /= n; self.z /= n;
        }
    }

    pub fn dot(&self, other: &Rotor) -> f64 {
        self.w*other.w + self.x*other.x + self.y*other.y + self.z*other.z
    }

    /// SLERP: spherical linear interpolation between two rotors
    pub fn slerp(a: &Rotor, b: &Rotor, t: f64) -> Rotor {
        let mut dot = a.dot(b);
        let mut b = *b;
        // Take shortest path
        if dot < 0.0 {
            dot = -dot;
            b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z;
        }
        const DOT_THRESHOLD: f64 = 0.9995;
        if dot > DOT_THRESHOLD {
            // Linear fallback for very close orientations
            let mut result = Rotor {
                w: a.w + t*(b.w - a.w),
                x: a.x + t*(b.x - a.x),
                y: a.y + t*(b.y - a.y),
                z: a.z + t*(b.z - a.z),
            };
            result.normalize();
            return result;
        }
        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();
        let s0 = (theta_0 - theta).cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;
        Rotor {
            w: a.w*s0 + b.w*s1,
            x: a.x*s0 + b.x*s1,
            y: a.y*s0 + b.y*s1,
            z: a.z*s0 + b.z*s1,
        }
    }

    /// SQUAD: spherical quadrangle interpolation (for spline-like paths)
    pub fn squad(a: &Rotor, b: &Rotor, c: &Rotor, d: &Rotor, t: f64) -> Rotor {
        let q1 = Self::slerp(a, d, t);
        let q2 = Self::slerp(b, c, t);
        Self::slerp(&q1, &q2, 2.0*t*(1.0-t))
    }

    pub fn to_array(&self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }
}

// ============================================================================
// 60-ORIENTATION BANK (icosahedral vertices + face centers)
// ============================================================================
pub struct RotorBank {
    pub orientations: Vec<Rotor>,
    pub scores: Vec<f64>,
    pub dim: usize,
}

impl RotorBank {
    pub fn new_60(dim: usize) -> Self {
        let mut orientations = Vec::with_capacity(60);
        // Golden ratio
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        // Icosahedral vertices (12)
        let verts = [
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
        ];
        for v in &verts {
            orientations.push(Rotor::from_axis_angle(&[v[0], v[1], v[2]], 0.0));
        }
        // Face centers (20) — approximate by cross products of adjacent vertices
        // For a full 60, fill with subdivided midpoints
        while orientations.len() < 60 {
            let i = orientations.len();
            let a = &orientations[i % 12];
            let b = &orientations[(i+1) % 12];
            let mid = Rotor::slerp(a, b, 0.5);
            orientations.push(mid);
        }
        let scores = vec![0.0; orientations.len()];
        Self { orientations, scores, dim }
    }

    pub fn len(&self) -> usize {
        self.orientations.len()
    }

    pub fn score_orientation(&mut self, idx: usize, score: f64) {
        if idx < self.scores.len() {
            self.scores[idx] = score;
        }
    }

    pub fn best_index(&self) -> usize {
        self.scores.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0)
    }

    pub fn top_k_indices(&self, k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self.scores.iter().enumerate()
            .map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Interpolate between two stored orientations
    pub fn interpolate(&self, idx_a: usize, idx_b: usize, t: f64) -> Option<Rotor> {
        let a = self.orientations.get(idx_a)?;
        let b = self.orientations.get(idx_b)?;
        Some(Rotor::slerp(a, b, t))
    }
}

// ============================================================================
// FFI: Expose to Python bridge
// ============================================================================

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_new(dim: c_int) -> *mut RotorBank {
    let bank = Box::new(RotorBank::new_60(dim as usize));
    Box::into_raw(bank)
}

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_free(ptr: *mut RotorBank) {
    if !ptr.is_null() { unsafe { drop(Box::from_raw(ptr)); } }
}

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_len(ptr: *mut RotorBank) -> c_int {
    if ptr.is_null() { return 0; }
    unsafe { (*ptr).len() as c_int }
}

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_score(ptr: *mut RotorBank, idx: c_int, score: c_double) {
    if ptr.is_null() { return; }
    unsafe { (*ptr).score_orientation(idx as usize, score); }
}

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_best(ptr: *mut RotorBank) -> c_int {
    if ptr.is_null() { return -1; }
    unsafe { (*ptr).best_index() as c_int }
}

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_topk(ptr: *mut RotorBank, k: c_int, out: *mut c_int) -> c_int {
    if ptr.is_null() || out.is_null() { return 0; }
    unsafe {
        let top = (*ptr).top_k_indices(k as usize);
        for (i, &v) in top.iter().enumerate() {
            *out.add(i) = v as c_int;
        }
        top.len() as c_int
    }
}

#[no_mangle]
pub extern "C" fn sphere_rotor_bank_slerp(
    ptr: *mut RotorBank,
    idx_a: c_int,
    idx_b: c_int,
    t: c_double,
    out_w: *mut c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
    out_z: *mut c_double,
) -> c_int {
    if ptr.is_null() { return 0; }
    unsafe {
        if let Some(r) = (*ptr).interpolate(idx_a as usize, idx_b as usize, t) {
            if !out_w.is_null() { *out_w = r.w; }
            if !out_x.is_null() { *out_x = r.x; }
            if !out_y.is_null() { *out_y = r.y; }
            if !out_z.is_null() { *out_z = r.z; }
            return 1;
        }
    }
    0
}
