// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Conformal coordinates for `ConformalPoint`.

/// A point in the conformal model:
/// * Cartesian coordinates are doubled relative to the original `u16` lattice points.
/// * `inf` is the squared lattice distance: `cx² + cy² + cz²`.
/// * `origin` is always `1.0`.
#[derive(Debug, Clone, Copy)]
pub struct ConformalPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub inf: f32,
    pub origin: f32,
}

impl ConformalPoint {
    /// Construct from lattice indices `(cx, cy, cz)`.
    ///
    /// `x = 2·cx`, `y = 2·cy`, `z = 2·cz`,
    /// `inf = cx² + cy² + cz²`, `origin = 1.0`.
    pub fn from_lsh(cx: u16, cy: u16, cz: u16) -> Self {
        let cx_f = cx as f32;
        let cy_f = cy as f32;
        let cz_f = cz as f32;

        Self {
            x: 2.0 * cx_f,
            y: 2.0 * cy_f,
            z: 2.0 * cz_f,
            inf: cx_f * cx_f + cy_f * cy_f + cz_f * cz_f,
            origin: 1.0,
        }
    }

    /// Conformal inner product between two points:
    ///
    /// `-0.5 * ((x1 - x2)² + (y1 - y2)² + (z1 - z2)²)`
    pub fn dot(&self, other: &Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;

        -0.5 * (dx * dx + dy * dy + dz * dz)
    }

    /// Conformal distance: `√(-2 · dot(self, other))`.
    pub fn conformal_distance(&self, other: &Self) -> f32 {
        let neg_two_dot = -2.0 * self.dot(other);
        neg_two_dot.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_points_distance_zero() {
        let a = ConformalPoint::from_lsh(0, 0, 0);
        let b = ConformalPoint::from_lsh(0, 0, 0);
        let dist = a.conformal_distance(&b);
        core::assert_eq!(dist, 0.0);
    }

    #[test]
    fn zero_and_one_x_distance_two() {
        let a = ConformalPoint::from_lsh(0, 0, 0);
        let b = ConformalPoint::from_lsh(1, 0, 0);
        let dist = a.conformal_distance(&b);
        core::assert_eq!(dist, 2.0);
    }
}
