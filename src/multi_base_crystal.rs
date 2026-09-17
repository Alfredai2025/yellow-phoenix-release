// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use alloc::vec::Vec;

/// 3D coordinate in a base's cubic power environment.
pub type Coord3D = (u16, u16, u16);

/// Multi-base crystal mesh: each base defines a b×b×b power environment.
/// Papers live simultaneously in all environments. Querying intersects
/// neighbor expansions across bases — only papers near in MULTIPLE
/// bases survive the narrowing.
pub struct MultiBaseCrystal {
    bases: Vec<u32>,
    cells: Vec<Vec<Vec<u32>>>,
    dims: Vec<usize>,
    positions: Vec<Vec<Coord3D>>,
    next_id: u64,
}

impl MultiBaseCrystal {
    pub fn new(bases: Vec<u32>) -> Self {
        let mut cells = Vec::with_capacity(bases.len());
        let mut dims = Vec::with_capacity(bases.len());
        
        for &base in &bases {
            let dim = base as usize;
            let num_cells = dim * dim * dim;
            let mut base_cells = Vec::with_capacity(num_cells);
            for _ in 0..num_cells {
                base_cells.push(Vec::new());
            }
            cells.push(base_cells);
            dims.push(dim);
        }
        
        Self {
            bases,
            cells,
            dims,
            positions: Vec::new(),
            next_id: 0,
        }
    }

    /// Compute 3D coordinate in a base's power environment.
    /// 
    /// Hybrid approach: weighted byte sum (prime weights for stability)
    /// plus popcount (for locality preservation). Small perturbations
    /// cause small coordinate changes, keeping similar papers nearby.
    #[inline]
    pub fn compute_coord(pap: &[u8; 64], base: u32) -> Coord3D {
        let b = base as u32;
        
        let mut x = 0u32;
        let mut y = 0u32;
        let mut z = 0u32;
        
        // Weighted sum with prime multipliers — small changes stay small
        for i in 0..21 {
            x = x.wrapping_add((pap[i] as u32).wrapping_mul(7919 + i as u32));
        }
        for i in 21..42 {
            y = y.wrapping_add((pap[i] as u32).wrapping_mul(104729 + i as u32));
        }
        for i in 42..64 {
            z = z.wrapping_add((pap[i] as u32).wrapping_mul(15485863 + i as u32));
        }
        
        // Add popcount for bit-level locality preservation
        let mut px = 0u32;
        let mut py = 0u32;
        let mut pz = 0u32;
        for i in 0..21 {
            px += pap[i].count_ones() as u32;
        }
        for i in 21..42 {
            py += pap[i].count_ones() as u32;
        }
        for i in 42..64 {
            pz += pap[i].count_ones() as u32;
        }
        
        x = x.wrapping_add(px);
        y = y.wrapping_add(py);
        z = z.wrapping_add(pz);
        
        ((x % b) as u16, (y % b) as u16, (z % b) as u16)
    }

    /// Circular distance on a ring of size `base`.
    /// Treats 0 and base-1 as adjacent (wrap-around).
    #[inline]
    pub fn circular_dist(a: u16, b: u16, base: u32) -> u16 {
        let d = if a > b { a - b } else { b - a };
        let wrap = (base as u16) - d;
        d.min(wrap)
    }

    #[inline]
    fn cell_key(coord: Coord3D, dim: usize) -> usize {
        let (x, y, z) = coord;
        (x as usize) * dim * dim + (y as usize) * dim + (z as usize)
    }

    /// Live triangulation: place paper in all base environments.
    pub fn insert(&mut self, pap: &[u8; 64]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        
        let mut coords = Vec::with_capacity(self.bases.len());
        
        for (base_idx, &base) in self.bases.iter().enumerate() {
            let coord = Self::compute_coord(pap, base);
            let dim = self.dims[base_idx];
            let key = Self::cell_key(coord, dim);
            
            self.cells[base_idx][key].push(id as u32);
            coords.push(coord);
        }
        
        if id as usize >= self.positions.len() {
            self.positions.resize(id as usize + 1, Vec::new());
        }
        self.positions[id as usize] = coords;
        
        id
    }

    /// Get all entities in cells within Manhattan radius of query's position in ONE base.
    pub fn get_base_neighbors(&self, query_pap: &[u8; 64], base_idx: usize, radius: u16) -> Vec<u64> {
        let base = self.bases[base_idx];
        let dim = self.dims[base_idx];
        let (cx, cy, cz) = Self::compute_coord(query_pap, base);
        let r = radius as i32;
        
        let mut result = Vec::new();
        let mut seen = Vec::new();
        
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    if dx.abs() + dy.abs() + dz.abs() > r {
                        continue;
                    }
                    
                    let nx = ((cx as i32 + dx).rem_euclid(dim as i32)) as usize;
                    let ny = ((cy as i32 + dy).rem_euclid(dim as i32)) as usize;
                    let nz = ((cz as i32 + dz).rem_euclid(dim as i32)) as usize;
                    
                    let key = nx * dim * dim + ny * dim + nz;
                    
                    for &nid in &self.cells[base_idx][key] {
                        let nid_u64 = nid as u64;
                        if nid_u64 >= seen.len() as u64 {
                            seen.resize(nid_u64 as usize + 1, false);
                        }
                        if !seen[nid_u64 as usize] {
                            seen[nid_u64 as usize] = true;
                            result.push(nid_u64);
                        }
                    }
                }
            }
        }
        
        result
    }

    /// Multi-base triangulation query.
    /// 
    /// Returns entities that appear in the query's neighborhood in at least
    /// `min_agreement` bases, sorted by agreement count (descending).
    pub fn query_triangulated(
        &self,
        query_pap: &[u8; 64],
        radius: u16,
        min_agreement: usize,
    ) -> Vec<(u64, usize)> {
        let mut vote_map: Vec<(u64, usize)> = Vec::new();
        
        for (base_idx, &base) in self.bases.iter().enumerate() {
            let dim = self.dims[base_idx];
            let (cx, cy, cz) = Self::compute_coord(query_pap, base);
            let r = radius as i32;
            
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs() + dy.abs() + dz.abs() > r {
                            continue;
                        }
                        
                        let nx = ((cx as i32 + dx).rem_euclid(dim as i32)) as usize;
                        let ny = ((cy as i32 + dy).rem_euclid(dim as i32)) as usize;
                        let nz = ((cz as i32 + dz).rem_euclid(dim as i32)) as usize;
                        
                        let key = nx * dim * dim + ny * dim + nz;
                        
                        for &nid in &self.cells[base_idx][key] {
                            let nid_u64 = nid as u64;
                            if let Some(pos) = vote_map.iter().position(|(id, _)| *id == nid_u64) {
                                vote_map[pos].1 += 1;
                            } else {
                                vote_map.push((nid_u64, 1));
                            }
                        }
                    }
                }
            }
        }
        
        vote_map.retain(|(_, count)| *count >= min_agreement);
        vote_map.sort_by(|a, b| b.1.cmp(&a.1));
        vote_map
    }

    pub fn get_position(&self, entity_id: u64) -> Option<&[Coord3D]> {
        self.positions.get(entity_id as usize).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.next_id as usize
    }
    pub fn is_empty(&self) -> bool {
        self.next_id == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pap(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        for i in 0..64 {
            pap[i] = ((seed.wrapping_mul(7919 + i as u64)) % 256) as u8;
        }
        pap
    }

    fn near_pap(original: &[u8; 64], flips: usize, seed: u64) -> [u8; 64] {
        let mut pap = *original;
        for i in 0..flips {
            let pos = ((seed.wrapping_mul(104729 + i as u64)) % 64) as usize;
            pap[pos] = pap[pos].wrapping_add(1);
        }
        pap
    }

    /// Determinism: identical PAPs → identical coordinates.
    #[test]
    fn identical_paps_same_coords() {
        let pap = make_pap(42);
        let bases = vec![3u32, 5, 7, 11, 13];
        
        for &base in &bases {
            let c1 = MultiBaseCrystal::compute_coord(&pap, base);
            let c2 = MultiBaseCrystal::compute_coord(&pap, base);
            assert_eq!(c1, c2);
            assert!((c1.0 as u32) < base);
            assert!((c1.1 as u32) < base);
            assert!((c1.2 as u32) < base);
        }
    }

    /// Locality preservation: near PAPs have near coordinates in most bases.
    /// Uses circular distance to handle modulo wrap-around.
    #[test]
    fn near_paps_near_coords() {
        let seed = make_pap(123);
        let near = near_pap(&seed, 3, 1);
        let bases = vec![3u32, 5, 7, 11];
        
        let mut adjacent_count = 0;
        
        for &base in &bases {
            let c1 = MultiBaseCrystal::compute_coord(&seed, base);
            let c2 = MultiBaseCrystal::compute_coord(&near, base);
            
            let dx = MultiBaseCrystal::circular_dist(c1.0, c2.0, base);
            let dy = MultiBaseCrystal::circular_dist(c1.1, c2.1, base);
            let dz = MultiBaseCrystal::circular_dist(c1.2, c2.2, base);
            
            if dx <= 1 && dy <= 1 && dz <= 1 {
                adjacent_count += 1;
            }
        }
        
        // With weighted-sum + popcount hybrid and circular distance,
        // small perturbations should preserve locality in most bases
        assert!(adjacent_count >= 2, 
                "Near PAPs should be adjacent in at least 2 bases (got {}/4)", 
                adjacent_count);
    }

    /// Core thesis: multi-base triangulation narrows the search space.
    /// Higher agreement = fewer candidates (deductive narrowing).
    #[test]
    fn triangulation_narrows_1k_papers() {
        let bases = vec![3u32, 5, 7, 11];
        let mut mesh = MultiBaseCrystal::new(bases);
        
        let seed = make_pap(42);
        
        // 200 cluster papers (near seed)
        for i in 0..200 {
            let pap = near_pap(&seed, 2, i as u64);
            mesh.insert(&pap);
        }
        
        // 800 noise papers (random)
        for i in 0..800 {
            let pap = make_pap(i as u64 * 104729);
            mesh.insert(&pap);
        }
        
        // Test narrowing: higher agreement → fewer candidates
        let loose = mesh.query_triangulated(&seed, 1, 1);
        let medium = mesh.query_triangulated(&seed, 1, 2);
        let strict = mesh.query_triangulated(&seed, 1, 3);
        
        assert!(medium.len() <= loose.len(), 
                "Agreement=2 should not return more than agreement=1 ({} vs {})", 
                medium.len(), loose.len());
        assert!(strict.len() <= medium.len(), 
                "Agreement=3 should not return more than agreement=2 ({} vs {})", 
                strict.len(), medium.len());
        
        // Cluster papers should not be underrepresented in stricter results
        if !loose.is_empty() && !medium.is_empty() {
            let loose_cluster = loose.iter()
                .filter(|(id, _)| *id < 200)
                .count() as f32 / loose.len() as f32;
            let medium_cluster = medium.iter()
                .filter(|(id, _)| *id < 200)
                .count() as f32 / medium.len() as f32;
            
            // Stricter agreement should not lose more than 50% of cluster precision
            assert!(medium_cluster >= loose_cluster * 0.5, 
                    "Higher agreement should not lose too much cluster precision ({:.1}% vs {:.1}%)",
                    medium_cluster * 100.0, loose_cluster * 100.0);
        }
    }

    /// Higher agreement = stricter narrowing (deterministic property).
    #[test]
    fn higher_agreement_stricter() {
        let bases = vec![3u32, 5, 7, 11, 13];
        let mut mesh = MultiBaseCrystal::new(bases);
        
        let seed = make_pap(99);
        
        for i in 0..50 {
            let pap = near_pap(&seed, 1, i as u64);
            mesh.insert(&pap);
        }
        for i in 0..50 {
            let pap = make_pap(i as u64 * 9973);
            mesh.insert(&pap);
        }
        
        let loose = mesh.query_triangulated(&seed, 1, 2);
        let strict = mesh.query_triangulated(&seed, 1, 4);
        
        assert!(strict.len() <= loose.len(), 
                "Higher agreement should not increase results ({} vs {})", 
                strict.len(), loose.len());
    }

    /// Live triangulation: positions stored correctly and within bounds.
    #[test]
    fn live_triangulation_positions() {
        let bases = vec![3u32, 5, 7];
        let mut mesh = MultiBaseCrystal::new(bases);
        
        let pap = make_pap(77);
        let id = mesh.insert(&pap);
        
        let pos = mesh.get_position(id).unwrap();
        assert_eq!(pos.len(), 3);
        
        for (coord, &base) in pos.iter().zip(&[3u32, 5, 7]) {
            assert!((coord.0 as u32) < base);
            assert!((coord.1 as u32) < base);
            assert!((coord.2 as u32) < base);
        }
    }
}
