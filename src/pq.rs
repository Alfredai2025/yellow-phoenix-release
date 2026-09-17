// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

// GHOST MODULE — NOT WIRED INTO src/lib.rs
// This file exists on disk but is NOT declared in src/lib.rs.
// It does NOT compile into the library and is NOT reachable from Python.
//
// AUDIT DATE: 2026-07-27
// ACTION: Preserved per "Wire First, Delete Never" policy.
//         Do not modify unless wiring into the build.
//
use alloc::vec::Vec;
use core::cmp::Ordering;

pub struct PQIndex {
    m: usize,
    subspace_dim: usize,
    num_centroids: usize,
    centroids: Vec<f32>,
    codes: Vec<u8>,
    num_vectors: usize,
    dim: usize,
}

impl PQIndex {
    pub fn new(m: usize, subspace_dim: usize) -> Self {
        let dim = m * subspace_dim;
        Self {
            m,
            subspace_dim,
            num_centroids: 256,
            centroids: Vec::new(),
            codes: Vec::new(),
            num_vectors: 0,
            dim,
        }
    }

    pub fn is_trained(&self) -> bool {
        !self.centroids.is_empty()
    }

    pub fn train(&mut self, vectors: &[f32], num_vectors: usize) {
        assert_eq!(vectors.len(), num_vectors * self.dim);
        assert!(num_vectors >= self.num_centroids);

        self.centroids
            .resize(self.m * self.num_centroids * self.subspace_dim, 0.0);

        for m in 0..self.m {
            let offset = m * self.subspace_dim;
            for k in 0..self.num_centroids {
                let src_idx = (k * 7919) % num_vectors;
                let src_start = src_idx * self.dim + offset;
                let dst_start = (m * self.num_centroids + k) * self.subspace_dim;
                for d in 0..self.subspace_dim {
                    self.centroids[dst_start + d] = vectors[src_start + d];
                }
            }

            let mut assignments = Vec::new();
            assignments.resize(num_vectors, 0usize);

            for _ in 0..10 {
                for v in 0..num_vectors {
                    let v_start = v * self.dim + offset;
                    let mut best_k = 0;
                    let mut best_dist = f32::MAX;
                    for k in 0..self.num_centroids {
                        let c_start = (m * self.num_centroids + k) * self.subspace_dim;
                        let mut dist = 0.0f32;
                        for d in 0..self.subspace_dim {
                            let diff = vectors[v_start + d] - self.centroids[c_start + d];
                            dist += diff * diff;
                        }
                        if dist < best_dist {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                    assignments[v] = best_k;
                }

                let mut new_centroids = Vec::new();
                new_centroids.resize(self.num_centroids * self.subspace_dim, 0.0f32);
                let mut counts = Vec::new();
                counts.resize(self.num_centroids, 0usize);

                for v in 0..num_vectors {
                    let k = assignments[v];
                    let v_start = v * self.dim + offset;
                    let c_start = k * self.subspace_dim;
                    for d in 0..self.subspace_dim {
                        new_centroids[c_start + d] += vectors[v_start + d];
                    }
                    counts[k] += 1;
                }

                for k in 0..self.num_centroids {
                    if counts[k] > 0 {
                        let c_start = (m * self.num_centroids + k) * self.subspace_dim;
                        for d in 0..self.subspace_dim {
                            self.centroids[c_start + d] =
                                new_centroids[k * self.subspace_dim + d] / counts[k] as f32;
                        }
                    }
                }
            }
        }
    }

    pub fn add(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dim);
        assert!(self.is_trained(), "train() before add()");

        let mut code = Vec::with_capacity(self.m);
        for m in 0..self.m {
            let offset = m * self.subspace_dim;
            let mut best_k = 0;
            let mut best_dist = f32::MAX;
            for k in 0..self.num_centroids {
                let c_start = (m * self.num_centroids + k) * self.subspace_dim;
                let mut dist = 0.0f32;
                for d in 0..self.subspace_dim {
                    let diff = vector[offset + d] - self.centroids[c_start + d];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_k = k;
                }
            }
            code.push(best_k as u8);
        }
        self.codes.extend_from_slice(&code);
        self.num_vectors += 1;
    }

    pub fn query(&self, query: &[f32], candidates: &[usize], top_k: usize) -> Vec<(usize, f32)> {
        assert_eq!(query.len(), self.dim);
        assert!(self.is_trained());

        let mut tables = Vec::new();
        tables.resize(self.m * self.num_centroids, 0.0f32);

        for m in 0..self.m {
            let q_offset = m * self.subspace_dim;
            for k in 0..self.num_centroids {
                let c_start = (m * self.num_centroids + k) * self.subspace_dim;
                let mut dist = 0.0f32;
                for d in 0..self.subspace_dim {
                    let diff = query[q_offset + d] - self.centroids[c_start + d];
                    dist += diff * diff;
                }
                tables[m * self.num_centroids + k] = dist;
            }
        }

        let mut results = Vec::with_capacity(top_k.min(candidates.len()));
        let mut worst_best = f32::MAX;

        for &idx in candidates {
            let code_start = idx * self.m;
            let mut dist = 0.0f32;
            for m in 0..self.m {
                let k = self.codes[code_start + m] as usize;
                dist += tables[m * self.num_centroids + k];
            }

            if dist < worst_best || results.len() < top_k {
                results.push((idx, dist));
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                if results.len() > top_k {
                    results.pop();
                }
                worst_best = results.last().map(|(_, d)| *d).unwrap_or(f32::MAX);
            }
        }

        results
    }

    pub fn len(&self) -> usize {
        self.num_vectors
    }
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn train_vectors(n: usize, dim: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                v.push(((i * 17 + d * 31) % 100) as f32 / 100.0);
            }
        }
        v
    }

    #[test]
    fn pq_train_query_ordering() {
        let mut pq = PQIndex::new(4, 2);
        let train = train_vectors(1000, 8);
        pq.train(&train, 1000);

        for i in 0..20 {
            let start = i * 8;
            pq.add(&train[start..start + 8]);
        }

        let query = vec![0.5f32; 8];
        let candidates: Vec<usize> = (0..20).collect();
        let results = pq.query(&query, &candidates, 5);

        assert_eq!(results.len(), 5);
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
    }

    #[test]
    fn pq_untrained_panics() {
        let pq = PQIndex::new(2, 4);
        assert!(!pq.is_trained());
    }
}
