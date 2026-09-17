// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Phyllotactic entry point navigator for BinaryHNSW.
//! Uses golden-angle (137.5°) geometry on the 2D spectral plane.
//! Zero-risk: does not modify HNSW graph structure.

use std::collections::HashMap;

const PHI: f64 = 1.618033988749895;
const GOLDEN_ANGLE_RAD: f64 = 2.39996322972865332; // 2π/φ² ≈ 137.5°
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Pre-computed complex rotor: e^(i * golden_angle)
/// Every step is 4 FLOPs: z' = z * c  (no trig calls at runtime)
const C_RE: f64 = -0.7373688780783199; // cos(137.5°)
const C_IM: f64 = 0.6754902942624241;  // sin(137.5°)

#[inline]
fn normalize_angle(a: f64) -> f64 {
    let mut a = a % TWO_PI;
    if a < 0.0 {
        a += TWO_PI;
    }
    a
}

#[inline]
fn angular_distance(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs();
    diff.min(TWO_PI - diff)
}

#[derive(Clone, Copy, Debug)]
pub struct PhyllotacticEntry {
    pub node_idx: usize,
    pub angle: f64,
}

pub struct PhyllotacticNavigator {
    entries: Vec<PhyllotacticEntry>,
}

impl PhyllotacticNavigator {
    /// Build navigator from pre-computed (node_index, spectral_angle) pairs.
    pub fn build(node_angles: &[(usize, f64)]) -> Self {
        if node_angles.is_empty() {
            return Self { entries: Vec::new() };
        }

        // Normalize angles to [0, 2π) and sort for binary search
        let mut nodes: Vec<(usize, f64)> = node_angles
            .iter()
            .map(|&(idx, a)| (idx, normalize_angle(a)))
            .collect();
        nodes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // √N entry points, clamped to [1, min(N, 4096)]
        let n_entries = ((nodes.len() as f64).sqrt() as usize)
            .clamp(1, 4096)
            .min(nodes.len());
        let mut entries = Vec::with_capacity(n_entries);

        // Walk unit circle in golden-angle steps via complex multiplication
        let mut z_re: f64 = 1.0;
        let mut z_im: f64 = 0.0;

        for _ in 0..n_entries {
            let target_angle = normalize_angle(z_im.atan2(z_re));

            // Binary search nearest angle
            let pos = nodes.partition_point(|&(_, a)| a < target_angle);
            let idx = if pos == 0 {
                0
            } else if pos >= nodes.len() {
                nodes.len() - 1
            } else {
                let d_next = (nodes[pos].1 - target_angle).abs();
                let d_prev = (nodes[pos - 1].1 - target_angle).abs();
                if d_next < d_prev { pos } else { pos - 1 }
            };

            entries.push(PhyllotacticEntry {
                node_idx: nodes[idx].0,
                angle: nodes[idx].1,
            });

            // Rotate: z *= e^(i*137.5°)  — 4 FLOPs, branchless
            let new_re = z_re * C_RE - z_im * C_IM;
            let new_im = z_re * C_IM + z_im * C_RE;
            z_re = new_re;
            z_im = new_im;
        }

        Self { entries }
    }

    /// Build from spectral coordinate map.
    /// `node_idx_to_id`: HNSW internal index → user-facing ID
    pub fn build_from_coords(
        hnsw_node_count: usize,
        node_idx_to_id: impl Fn(usize) -> Option<u64>,
        id_to_spectral: &HashMap<u64, [f32; 2]>,
    ) -> Self {
        let mut angles = Vec::with_capacity(hnsw_node_count);
        for idx in 0..hnsw_node_count {
            if let Some(id) = node_idx_to_id(idx) {
                if let Some(&[x, y]) = id_to_spectral.get(&id) {
                    angles.push((idx, normalize_angle((y as f64).atan2(x as f64))));
                }
            }
        }
        Self::build(&angles)
    }

    /// Return the HNSW node index of the nearest phyllotactic entry point.
    pub fn nearest_entry(&self, query_spectral: [f32; 2]) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        let q_angle = normalize_angle((query_spectral[1] as f64).atan2(query_spectral[0] as f64));

        let pos = self.entries.partition_point(|e| e.angle < q_angle);
        let prev_pos = if pos == 0 {
            self.entries.len() - 1
        } else {
            pos - 1
        };
        let next_pos = if pos >= self.entries.len() { 0 } else { pos };

        let prev = &self.entries[prev_pos];
        let next = &self.entries[next_pos];

        if angular_distance(prev.angle, q_angle) <= angular_distance(next.angle, q_angle) {
            Some(prev.node_idx)
        } else {
            Some(next.node_idx)
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_rotation_produces_distinct_angles() {
        let mut z_re: f64 = 1.0;
        let mut z_im: f64 = 0.0;
        let mut angles = Vec::new();
        for _ in 0..20 {
            angles.push(z_im.atan2(z_re));
            let nr = z_re * C_RE - z_im * C_IM;
            let ni = z_re * C_IM + z_im * C_RE;
            z_re = nr;
            z_im = ni;
        }
        // Golden angle is irrational; no two of 20 steps should collide
        for i in 0..angles.len() {
            for j in (i + 1)..angles.len() {
                assert!((angles[i] - angles[j]).abs() > 0.001);
            }
        }
    }

    #[test]
    fn nearest_entry_returns_valid_close_angle() {
        // 64 entries spaced every 5.625° in [0, 2π).
        let entries: Vec<(usize, f64)> = (0..64)
            .map(|i| (i, (i as f64) * TWO_PI / 64.0))
            .collect();
        let nav = PhyllotacticNavigator::build(&entries);
        assert!(nav.len() > 0);

        let queries: [([f32; 2], f64); 4] = [
            ([1.0, 0.1], 0.0),                       // ~0°
            ([-0.1, 1.0], std::f64::consts::PI / 2.0), // ~90°
            ([-1.0, -0.1], std::f64::consts::PI),     // ~180°
            ([0.1, -1.0], 3.0 * std::f64::consts::PI / 2.0), // ~270°
        ];

        for (q, expected_angle) in queries {
            let got_idx = nav.nearest_entry(q).expect("should return an entry");
            let got_angle = entries[got_idx].1;
            let diff = angular_distance(got_angle, expected_angle);
            // With √64 = 8 sampled entries, max angular gap is ~45°.
            assert!(diff <= std::f64::consts::PI / 4.0,
                "query {:?} got angle {} rad, expected near {} rad",
                q, got_angle, expected_angle);
        }
    }
}
