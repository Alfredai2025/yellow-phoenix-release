// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! OPQ + Product Quantization index for compact semantic search.
//!
//! Supports generic dimensions: the OPQ rotation maps d_in (384) to d_out,
//! then PQ splits d_out into M subspaces of dsub = d_out / M dims each,
//! with 2^nbits centroids per subspace.

use alloc::vec::Vec;
use core::cmp::Ordering;

/// One OPQ+PQ tier (e.g. 8-byte or 16-byte).
pub struct OpqPqTier {
    /// Rotation matrix, row-major: d_out rows × d_in cols.
    rotation: Vec<f32>,
    pub d_in: usize,
    pub d_out: usize,
    /// Flat codebooks: M * ksub * dsub floats.
    codebooks: Vec<f32>,
    /// Flat PQ codes: n * M bytes.
    codes: Vec<u8>,
    pub n: usize,
    pub m: usize,
    pub ksub: usize,
    pub dsub: usize,
}

impl OpqPqTier {
    /// Construct a tier from already-flattened buffers.
    pub fn new(
        rotation: Vec<f32>,
        d_in: usize,
        d_out: usize,
        codebooks: Vec<f32>,
        codes: Vec<u8>,
        n: usize,
        m: usize,
        ksub: usize,
    ) -> Self {
        assert_eq!(rotation.len(), d_in * d_out);
        assert_eq!(d_out % m, 0);
        let dsub = d_out / m;
        assert_eq!(codebooks.len(), m * ksub * dsub);
        assert_eq!(codes.len(), n * m);
        Self {
            rotation,
            d_in,
            d_out,
            codebooks,
            codes,
            n,
            m,
            ksub,
            dsub,
        }
    }

    /// Rotate an embedding through the learned OPQ matrix.
    fn rotate(&self, emb: &[f32]) -> Vec<f32> {
        assert_eq!(emb.len(), self.d_in);
        let mut out = vec![0.0_f32; self.d_out];
        for i in 0..self.d_out {
            let mut acc = 0.0_f32;
            let row = i * self.d_in;
            for j in 0..self.d_in {
                acc += self.rotation[row + j] * emb[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Quantize a vector into an M-byte PQ code.
    pub fn encode(&self, emb: &[f32]) -> Vec<u8> {
        let rot = self.rotate(emb);
        let mut code = vec![0_u8; self.m];
        for m in 0..self.m {
            let sub_start = m * self.dsub;
            let mut best_c = 0_usize;
            let mut best_dist = f32::MAX;
            for c in 0..self.ksub {
                let mut dist = 0.0_f32;
                let base = (m * self.ksub + c) * self.dsub;
                for j in 0..self.dsub {
                    let diff = rot[sub_start + j] - self.codebooks[base + j];
                    dist += diff * diff;
                }
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            code[m] = best_c as u8;
        }
        code
    }

    /// Build the asymmetric distance table for a query embedding.
    fn distance_table(&self, emb: &[f32]) -> Vec<f32> {
        let rot = self.rotate(emb);
        let mut table = vec![0.0_f32; self.m * self.ksub];
        for m in 0..self.m {
            let sub_start = m * self.dsub;
            for c in 0..self.ksub {
                let mut dist = 0.0_f32;
                let base = (m * self.ksub + c) * self.dsub;
                for j in 0..self.dsub {
                    let diff = rot[sub_start + j] - self.codebooks[base + j];
                    dist += diff * diff;
                }
                table[m * self.ksub + c] = dist;
            }
        }
        table
    }

    /// Asymmetric distance: compute ADC distance to every stored code and return top-k.
    pub fn query(&self, emb: &[f32], top_k: usize) -> Vec<(u32, f32)> {
        let table = self.distance_table(emb);
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut d = 0.0_f32;
            let base = i * self.m;
            for m in 0..self.m {
                let c = self.codes[base + m] as usize;
                d += table[m * self.ksub + c];
            }
            dists.push((i as u32, d));
        }

        dists.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        dists.truncate(top_k);
        dists
    }

    /// Re-rank a candidate set using this tier's ADC distances.
    pub fn rerank(&self, emb: &[f32], candidates: &[u32]) -> Vec<(u32, f32)> {
        let table = self.distance_table(emb);
        let mut out: Vec<(u32, f32)> = Vec::with_capacity(candidates.len());
        for &id in candidates {
            let i = id as usize;
            if i >= self.n {
                continue;
            }
            let mut d = 0.0_f32;
            let base = i * self.m;
            for m in 0..self.m {
                let c = self.codes[base + m] as usize;
                d += table[m * self.ksub + c];
            }
            out.push((id, d));
        }

        out.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }
}
