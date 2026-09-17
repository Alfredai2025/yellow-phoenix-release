// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use alloc::vec::Vec;
use core::cmp::{Ordering, Reverse};

use crate::distance::pap_distance;
use crate::types::multivector::BinaryMultivector;

const CHUNKS: usize = 2;
const COORD_BITS: u32 = 5;
const COORD_MASK: u64 = (1u64 << COORD_BITS) - 1;
const NUM_CELLS: usize = 1 << (COORD_BITS * 3);

/// Locality-preserving geometric reflection: popcount projection.
/// Similar bit-density vectors cluster together.
#[inline(always)]
fn geometric_reflect(v: &BinaryMultivector) -> (u16, u16, u16) {
    (
        (v.0[0].count_ones() as u16).min(COORD_MASK as u16),
        (v.0[1].count_ones() as u16).min(COORD_MASK as u16),
        ((v.0[0] & v.0[1]).count_ones() as u16).min(COORD_MASK as u16),
    )
}

#[inline(always)]
fn pack_coord(c: (u16, u16, u16)) -> usize {
    (((c.0 as u64) << (COORD_BITS * 2)) | ((c.1 as u64) << COORD_BITS) | (c.2 as u64)) as usize
}

#[inline(always)]
fn add_signed_u16(a: u16, b: i16) -> u16 {
    if b < 0 {
        a.saturating_sub((-b) as u16)
    } else {
        a.saturating_add(b as u16)
    }
}

pub struct LSHIndex {
    cells: Vec<Vec<u32>>,
    pub(crate) density: Vec<u32>,
    prototypes: Vec<u64>,
    prototype_count: usize,
}

impl LSHIndex {
    pub fn new(_num_buckets: usize, _num_hashes: usize) -> Self {
        let num_cells = NUM_CELLS;
        let mut cells: Vec<Vec<u32>> = Vec::with_capacity(num_cells);
        cells.resize_with(num_cells, Vec::new);
        let mut density = Vec::new();
        density.resize(num_cells, 0u32);
        Self {
            cells,
            density,
            prototypes: Vec::new(),
            prototype_count: 0,
        }
    }

    /// Access a stored prototype by index. pub(crate) so cascade.rs can use it.
    #[inline(always)]
    pub(crate) fn get_prototype(&self, idx: usize) -> BinaryMultivector {
        let start = idx * CHUNKS;
        let mut arr = [0u64; CHUNKS];
        arr.copy_from_slice(&self.prototypes[start..start + CHUNKS]);
        BinaryMultivector(arr)
    }

    /// Collect ALL candidate indices from cells within Manhattan radius of query.
    /// Used by cascading retrieval Stage 1.
    pub fn collect_candidates(&self, q: &BinaryMultivector, max_radius: i16) -> Vec<usize> {
        let coord = geometric_reflect(q);
        let (cx, cy, cz) = coord;
        let mut cell_keys = Vec::new();

        for radius in 0..=max_radius {
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        if dx.abs() + dy.abs() + dz.abs() != radius { continue; }

                        let nx = add_signed_u16(cx, dx);
                        let ny = add_signed_u16(cy, dy);
                        let nz = add_signed_u16(cz, dz);
                        let nkey = pack_coord((nx, ny, nz));

                        if nkey >= self.cells.len() {
                            continue;
                        }

                        if self.density[nkey] == 0 {
                            continue;
                        }
                        cell_keys.push(nkey);
                    }
                }
            }
        }

        cell_keys.sort_by_key(|&k| Reverse(self.density[k]));

        let mut candidates = Vec::new();
        for key in cell_keys {
            if let Some(cell) = self.cells.get(key) {
                candidates.extend(cell.iter().map(|&x| x as usize));
            }
        }
        candidates
    }

    pub fn add(&mut self, paper: BinaryMultivector) {
        let idx = self.prototype_count;
        let coord = geometric_reflect(&paper);
        let key = pack_coord(coord);
        self.cells[key].push(idx as u32);
        self.density[key] = self.cells[key].len() as u32;
        self.prototypes.extend_from_slice(&paper.0);
        self.prototype_count += 1;
    }

    /// Compute the density gradient around a coordinate.
    /// Returns the 6 directions [(+X,0,0), (-X,0,0), (0,+Y,0), (0,-Y,0), (0,0,+Z), (0,0,-Z)]
    /// sorted by neighbor density (highest first).
    pub fn density_gradient(&self, coord: (u16, u16, u16)) -> [(i16, i16, i16); 6] {
        let (cx, cy, cz) = coord;
        let mut dirs = [
            ((1i16, 0i16, 0i16), 0u32),
            ((-1i16, 0i16, 0i16), 0u32),
            ((0i16, 1i16, 0i16), 0u32),
            ((0i16, -1i16, 0i16), 0u32),
            ((0i16, 0i16, 1i16), 0u32),
            ((0i16, 0i16, -1i16), 0u32),
        ];
        
        for i in 0..6 {
            let (dx, dy, dz) = dirs[i].0;
            let nx = add_signed_u16(cx, dx);
            let ny = add_signed_u16(cy, dy);
            let nz = add_signed_u16(cz, dz);
            let key = pack_coord((nx, ny, nz));
            dirs[i].1 = self.density.get(key).copied().unwrap_or(0);
        }
        
        dirs.sort_by_key(|&(_, d)| Reverse(d));
        
        [dirs[0].0, dirs[1].0, dirs[2].0, dirs[3].0, dirs[4].0, dirs[5].0]
    }

    pub fn query(&self, q: &BinaryMultivector, top_k: usize, fast_threshold: f32) -> Vec<(usize, f32)> {
        let coord = geometric_reflect(q);
        let key = pack_coord(coord);

        let mut best = Vec::with_capacity(top_k);
        let mut worst_best = f32::MAX;
        let mut checked = 0usize;

        for &idx in self.cells[key].iter() {
            let idx = idx as usize;
            let d = pap_distance(q, &self.get_prototype(idx));
            checked += 1;
            if d < worst_best || best.len() < top_k {
                best.push((idx, d));
                best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                if best.len() > top_k { best.pop(); }
                worst_best = best.last().map(|(_, d)| *d).unwrap_or(f32::MAX);
            }
        }

        if !best.is_empty() && worst_best <= fast_threshold && checked >= top_k {
            return best;
        }

        let (cx, cy, cz) = coord;
        let max_radius: i16 = if self.prototype_count < 100_000 { 3 } else { 1 };

        // Gradient expansion: radius 1 high-density first
        let gradient_dirs = self.density_gradient(coord);
        for &(dx, dy, dz) in &gradient_dirs {
            let nx = add_signed_u16(cx, dx);
            let ny = add_signed_u16(cy, dy);
            let nz = add_signed_u16(cz, dz);
            let nkey = pack_coord((nx, ny, nz));

            if nkey >= self.cells.len() {
                continue;
            }

            if self.density[nkey] == 0 {
                continue;
            }
            for &idx in self.cells[nkey].iter() {
                let idx = idx as usize;
                let d = pap_distance(q, &self.get_prototype(idx));
                checked += 1;
                if d < worst_best || best.len() < top_k {
                    best.push((idx, d));
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    if best.len() > top_k { best.pop(); }
                    worst_best = best.last().map(|(_, d)| *d).unwrap_or(f32::MAX);
                }
            }
            if best.len() >= top_k && worst_best <= fast_threshold {
                return best;
            }
        }

        // Uniform radius expansion for remaining cells (radius 2+)
        for radius in 2..=max_radius {
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        if dx.abs() + dy.abs() + dz.abs() != radius { continue; }

                        let nx = add_signed_u16(cx, dx);
                        let ny = add_signed_u16(cy, dy);
                        let nz = add_signed_u16(cz, dz);
                        let nkey = pack_coord((nx, ny, nz));

                        if nkey >= self.cells.len() {
                            continue;
                        }

                        if self.density[nkey] == 0 {
                            continue;
                        }

                        for &idx in self.cells[nkey].iter() {
                            let idx = idx as usize;
                            let d = pap_distance(q, &self.get_prototype(idx));
                            checked += 1;
                            if d < worst_best || best.len() < top_k {
                                best.push((idx, d));
                                best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                                if best.len() > top_k { best.pop(); }
                                worst_best = best.last().map(|(_, d)| *d).unwrap_or(f32::MAX);
                            }
                        }
                    }
                }
            }

            if best.len() >= top_k && worst_best <= fast_threshold {
                break;
            }
        }

        best
    }

    pub fn len(&self) -> usize { self.prototype_count }
    pub fn is_empty(&self) -> bool { self.prototype_count == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(indices: &[usize]) -> BinaryMultivector {
        let mut chunks = [0u64; CHUNKS];
        for &i in indices {
            chunks[i / 64] |= 1u64 << (i % 64);
        }
        BinaryMultivector(chunks)
    }

    #[test]
    fn add_and_len() {
        let mut idx = LSHIndex::new(16, 1);
        idx.add(make_vec(&[0, 1, 2]));
        idx.add(make_vec(&[3, 4, 5]));
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn query_finds_similar() {
        let mut idx = LSHIndex::new(64, 1);
        let q = make_vec(&[0, 1, 2, 3, 4]);
        idx.add(make_vec(&[0, 1, 2, 3, 5]));      // same popcount (5), same cell
        idx.add(make_vec(&[100, 101, 102, 103])); // popcount 4, different cell
        let results = idx.query(&q, 1, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.5);
    }

    #[test]
    fn early_termination_fast_path() {
        let mut idx = LSHIndex::new(64, 1);
        let q = make_vec(&[0, 1, 2]);
        idx.add(make_vec(&[0, 1, 2]));
        let results = idx.query(&q, 1, 0.0);
        assert_eq!(results.len(), 1);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn fallback_when_primary_empty() {
        let mut idx = LSHIndex::new(2, 1);
        let q = make_vec(&[0, 1, 2]); // popcount 3
        // Papers in adjacent cells (popcount 4 and 5)
        idx.add(make_vec(&[0, 1, 2, 3]));   // popcount 4
        idx.add(make_vec(&[0, 1, 2, 3, 4])); // popcount 5
        let results = idx.query(&q, 2, 0.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn top_k_ordering() {
        let mut idx = LSHIndex::new(16, 1);
        let q = make_vec(&[0, 1, 2, 3, 4]);
        idx.add(make_vec(&[0, 1, 2, 3, 4]));
        idx.add(make_vec(&[0, 1, 2, 3]));
        idx.add(make_vec(&[100, 101, 102]));
        let results = idx.query(&q, 2, 1.0);
        assert_eq!(results.len(), 2);
        assert!(results[0].1 <= results[1].1);
        assert!(results[0].1 < 1e-6);
    }

    #[test]
    fn density_tracks_cell_size() {
        let mut idx = LSHIndex::new(32, 1);
        let v = make_vec(&[0, 1, 2, 3, 4]);
        idx.add(v.clone());
        idx.add(v.clone());
        let coord = geometric_reflect(&v);
        let key = pack_coord(coord);
        assert_eq!(idx.density[key], 2);
    }

    #[test]
    fn empty_cells_skipped() {
        let mut idx = LSHIndex::new(32, 1);
        let v1 = make_vec(&[0, 1, 2, 3, 4]);
        let v2 = make_vec(&[100, 101, 102, 103, 104]);
        idx.add(v1);
        idx.add(v2);
        let q = make_vec(&[0, 1, 2, 3, 4]);
        let results = idx.query(&q, 2, 1.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn density_zero_for_empty_cell() {
        let idx = LSHIndex::new(32, 1);
        assert_eq!(idx.density[0], 0);
        assert_eq!(idx.density[1000], 0);
    }

    #[test]
    fn gradient_points_to_dense_region() {
        let mut idx = LSHIndex::new(32, 1);
        // Fill neighbor cell (+X) with many vectors
        let v = make_vec(&(0..10).collect::<Vec<_>>()); // (10,0,0)
        let v_neighbor = make_vec(&(0..11).collect::<Vec<_>>()); // (11,0,0)
        for _ in 0..100 {
            idx.add(v_neighbor);
        }
        let coord = geometric_reflect(&v); // (10,0,0)
        let dirs = idx.density_gradient(coord);
        let has_density = dirs.iter().any(|&(dx, dy, dz)| {
            let nx = add_signed_u16(coord.0, dx);
            let ny = add_signed_u16(coord.1, dy);
            let nz = add_signed_u16(coord.2, dz);
            let key = pack_coord((nx, ny, nz));
            idx.density.get(key).copied().unwrap_or(0) > 0
        });
        assert!(has_density);
    }

    #[test]
    fn gradient_empty_field() {
        let idx = LSHIndex::new(32, 1);
        let coord = (16u16, 16u16, 16u16);
        let dirs = idx.density_gradient(coord);
        // All directions should be (0,0,0) density, but method still returns 6 directions
        assert_eq!(dirs.len(), 6);
    }

    #[test]
    fn gradient_sorted_by_density() {
        let mut idx = LSHIndex::new(32, 1);
        let v = make_vec(&(0..10).collect::<Vec<_>>()); // (10,0,0)
        let v_neighbor = make_vec(&(0..11).collect::<Vec<_>>()); // (11,0,0), +X direction
        for _ in 0..50 {
            idx.add(v_neighbor);
        }
        let coord = geometric_reflect(&v); // (10,0,0)
        let dirs = idx.density_gradient(coord);
        // The +X neighbor cell has the highest density, so it should be first
        assert_eq!(dirs[0], (1, 0, 0));
    }
}
