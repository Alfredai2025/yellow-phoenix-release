// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::{BinaryHeap, HashSet};
use std::cmp::{Ordering, Reverse};
use rand::Rng;

/// f32 wrapper implementing total ordering for BinaryHeap use.
#[derive(Clone, Copy, Debug, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct FloatNode {
    pub id: u64,
    pub vec: Vec<f32>,
}

impl FloatNode {
    pub fn new(id: u64, vec: Vec<f32>) -> Self {
        FloatNode { id, vec }
    }
}

#[derive(Clone)]
pub struct FloatEdge {
    pub to: usize,
    pub dist: f32,
}

pub struct FloatHNSW {
    nodes: Vec<FloatNode>,
    layers: Vec<Vec<Vec<FloatEdge>>>,
    enter_point: Option<usize>,
    m: usize,
    m_l: f64,
    ef_construction: usize,
    ef_search: usize,
    dim: usize,
}

impl FloatHNSW {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let m = m.max(2);
        let m_l = 1.0 / (m as f64).ln();
        FloatHNSW {
            nodes: Vec::new(),
            layers: Vec::new(),
            enter_point: None,
            m,
            m_l,
            ef_construction,
            ef_search,
            dim: 0,
        }
    }

    #[inline(always)]
    fn max_degree_for_layer(&self, layer: usize) -> usize {
        if layer == 0 { self.m * 2 } else { self.m }
    }

    fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
        }
        1.0 - dot
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let mut level = 0;
        const MAX_LAYERS: usize = 16;
        while level < MAX_LAYERS - 1 && rng.random_bool(self.m_l) {
            level += 1;
        }
        level
    }

    pub fn insert(&mut self, id: u64, vec: Vec<f32>) {
        assert!(!vec.is_empty(), "empty vector");
        if self.dim == 0 {
            self.dim = vec.len();
        } else {
            assert_eq!(vec.len(), self.dim, "vector dimension mismatch");
        }

        let level = self.random_level();
        while self.layers.len() <= level {
            self.layers.push(Vec::new());
        }
        let new_idx = self.nodes.len();
        self.nodes.push(FloatNode::new(id, vec));
        // Extend every existing layer so all nodes have an entry; empty entries
        // represent nodes that do not participate in that layer.
        for l in 0..self.layers.len() {
            while self.layers[l].len() <= new_idx {
                self.layers[l].push(Vec::new());
            }
        }

        if let Some(ep) = self.enter_point {
            let mut curr_ep = ep;
            let q = self.nodes[new_idx].vec.clone();
            for l in (level + 1..self.layers.len()).rev() {
                curr_ep = self.greedy_closest(curr_ep, l, &q);
            }
            for l in (0..=level).rev() {
                let neighbors = self.search_layer(curr_ep, l, &q, self.ef_construction);
                let pruned = self.select_neighbors(&neighbors, self.m);
                for &(nidx, dist) in &pruned {
                    self.layers[l][new_idx].push(FloatEdge { to: nidx, dist });
                    self.layers[l][nidx].push(FloatEdge { to: new_idx, dist });
                    let cap = self.max_degree_for_layer(l);
                    if self.layers[l][nidx].len() > cap {
                        self.prune_neighbors(l, nidx, cap);
                    }
                }
                if let Some(&(best_idx, _)) = pruned.first() {
                    curr_ep = best_idx;
                }
            }
        } else {
            self.enter_point = Some(new_idx);
        }
    }

    fn greedy_closest(&self, ep: usize, level: usize, q: &[f32]) -> usize {
        let mut curr = ep;
        let mut changed = true;
        while changed {
            changed = false;
            let dist = Self::cosine_dist(&self.nodes[curr].vec, q);
            for edge in &self.layers[level][curr] {
                let d2 = Self::cosine_dist(&self.nodes[edge.to].vec, q);
                if d2 < dist {
                    curr = edge.to;
                    changed = true;
                    break;
                }
            }
        }
        curr
    }

    fn search_layer(&self, ep: usize, level: usize, q: &[f32], ef: usize) -> Vec<(usize, f32)> {
        let ef = ef.max(1);
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Reverse<(OrdF32, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrdF32, usize)> = BinaryHeap::new();
        let dist = OrdF32(Self::cosine_dist(&self.nodes[ep].vec, q));
        candidates.push(Reverse((dist, ep)));
        results.push((dist, ep));
        visited.insert(ep);

        while let Some(Reverse((cdist, cidx))) = candidates.pop() {
            if results.len() >= ef {
                let worst = results.peek().unwrap().0;
                if cdist > worst {
                    break;
                }
            }
            for edge in &self.layers[level][cidx] {
                if visited.insert(edge.to) {
                    let d = OrdF32(Self::cosine_dist(&self.nodes[edge.to].vec, q));
                    let should_add = results.len() < ef || d < results.peek().unwrap().0;
                    if should_add {
                        candidates.push(Reverse((d, edge.to)));
                        results.push((d, edge.to));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter().map(|(d, i)| (i, d.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }

    fn select_neighbors(&self, candidates: &[(usize, f32)], m: usize) -> Vec<(usize, f32)> {
        if candidates.len() <= m {
            return candidates.to_vec();
        }

        let mut selected: Vec<(usize, f32)> = Vec::with_capacity(m);

        for &(cidx, cdist) in candidates {
            if selected.len() >= m {
                break;
            }

            // hnswlib-style diversity heuristic: skip candidates that are closer
            // to an already-selected neighbor than to the query.
            let mut skip = false;
            for &(sidx, _) in &selected {
                let d_cn = Self::cosine_dist(&self.nodes[cidx].vec, &self.nodes[sidx].vec);
                if d_cn < cdist {
                    skip = true;
                    break;
                }
            }

            if !skip {
                selected.push((cidx, cdist));
            }
        }

        selected
    }

    fn prune_neighbors(&mut self, level: usize, idx: usize, cap: usize) {
        let mut edges = self.layers[level][idx].clone();
        edges.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        self.layers[level][idx] = edges.into_iter().take(cap).collect();
    }

    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        if k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        let ef = self.ef_search.max(k);
        let ep = match self.enter_point {
            Some(e) => e,
            None => return Vec::new(),
        };
        let mut curr = ep;
        for l in (1..self.layers.len()).rev() {
            curr = self.greedy_closest(curr, l, q);
        }
        let mut res = self.search_layer(curr, 0, q, ef);
        res.truncate(k);
        res.into_iter().map(|(idx, dist)| (self.nodes[idx].id, dist)).collect()
    }

    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef.max(1);
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Batch insert (vectors are concatenated: ids.len() * dim floats).
    pub fn insert_batch(&mut self, ids: &[u64], vecs: &[f32], dim: usize) {
        assert_eq!(ids.len() * dim, vecs.len());
        for (i, &id) in ids.iter().enumerate() {
            let v = vecs[i * dim..(i + 1) * dim].to_vec();
            self.insert(id, v);
        }
    }

    /// Save index to disk in compact binary format (version 1).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        const MAGIC: &[u8] = b"YPFL";
        file.write_all(MAGIC)?;
        file.write_all(&[1u8])?; // version
        file.write_all(&(self.dim as u64).to_le_bytes())?;
        file.write_all(&(self.m as u64).to_le_bytes())?;
        file.write_all(&(self.ef_construction as u64).to_le_bytes())?;
        file.write_all(&(self.ef_search as u64).to_le_bytes())?;
        file.write_all(&(self.layers.len() as u64).to_le_bytes())?;
        let ep = self.enter_point.unwrap_or(usize::MAX);
        file.write_all(&(ep as u64).to_le_bytes())?;
        file.write_all(&(self.nodes.len() as u64).to_le_bytes())?;

        let num_layers = self.layers.len();
        for (idx, node) in self.nodes.iter().enumerate() {
            file.write_all(&node.id.to_le_bytes())?;
            file.write_all(&(node.vec.len() as u64).to_le_bytes())?;
            for v in &node.vec {
                file.write_all(&v.to_le_bytes())?;
            }
            for layer in 0..num_layers {
                let count = self.layers[layer][idx].len() as u32;
                file.write_all(&count.to_le_bytes())?;
                for edge in &self.layers[layer][idx] {
                    file.write_all(&(edge.to as u32).to_le_bytes())?;
                    file.write_all(&edge.dist.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Load index from disk.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"YPFL" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut version = [0u8; 1];
        file.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported version"));
        }
        file.read_exact(&mut buf8)?; let dim = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?; let m = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?; let ef_construction = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?; let ef_search = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?; let num_layers = u64::from_le_bytes(buf8) as usize;
        file.read_exact(&mut buf8)?; let ep_raw = u64::from_le_bytes(buf8) as usize;
        let enter_point = if ep_raw == usize::MAX { None } else { Some(ep_raw) };
        file.read_exact(&mut buf8)?; let node_count = u64::from_le_bytes(buf8) as usize;

        let mut nodes = Vec::with_capacity(node_count);
        let mut layers: Vec<Vec<Vec<FloatEdge>>> = (0..num_layers)
            .map(|_| vec![Vec::new(); node_count])
            .collect();

        for _ in 0..node_count {
            file.read_exact(&mut buf8)?; let id = u64::from_le_bytes(buf8);
            file.read_exact(&mut buf8)?; let vec_len = u64::from_le_bytes(buf8) as usize;
            let mut vec = Vec::with_capacity(vec_len);
            for _ in 0..vec_len {
                file.read_exact(&mut buf4)?;
                vec.push(f32::from_le_bytes(buf4));
            }
            let idx = nodes.len();
            for layer in 0..num_layers {
                file.read_exact(&mut buf4)?;
                let count = u32::from_le_bytes(buf4) as usize;
                for _ in 0..count {
                    file.read_exact(&mut buf4)?; let to = u32::from_le_bytes(buf4) as usize;
                    file.read_exact(&mut buf4)?; let dist = f32::from_le_bytes(buf4);
                    layers[layer][idx].push(FloatEdge { to, dist });
                }
            }
            nodes.push(FloatNode { id, vec });
        }

        let m_l = 1.0 / (m as f64).ln();
        Ok(Self {
            nodes,
            layers,
            enter_point,
            m,
            m_l,
            ef_construction,
            ef_search,
            dim,
        })
    }
}
