// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! 512-bit spectral index for YP Functor Audit.
//!
//! Power-iteration eigendecomposition on 512-dimensional binary vectors.
//! This is separate from the existing 128-bit `TensorSpectralIndex` so that
//! F1/F2 can be measured directly on the same 512-bit representation used by
//! `BinaryHNSW`.

const PAP_512_BYTES: usize = 64;
const DIM: usize = PAP_512_BYTES * 8; // 512

#[derive(Clone, Debug)]
pub struct SpectralVector512 {
    pub id: u64,
    pub spectral_coords: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TensorSpectralIndex512 {
    vectors: Vec<SpectralVector512>,
    eigenvectors: Vec<Vec<f32>>,
    k: usize,
}

impl TensorSpectralIndex512 {
    /// Build a spectral index from a flat byte slice of N 64-byte PAPs.
    /// `k` is the number of eigenvectors (dimensions) to retain.
    pub fn build(pap_512_bytes: &[u8], k: usize) -> Self {
        let n = pap_512_bytes.len() / PAP_512_BYTES;
        assert!(n > 0, "tensor_spectral_512: empty input");
        assert!(
            pap_512_bytes.len() % PAP_512_BYTES == 0,
            "tensor_spectral_512: input length must be a multiple of 64"
        );
        let k = k.min(DIM).min(n);

        eprintln!("[tensor_spectral_512] Building {n} vectors, dim={DIM}, k={k}...");

        // Convert hashes to 512-bit 0/1 vectors.
        let mut vectors: Vec<SpectralVector512> = Vec::with_capacity(n);
        for i in 0..n {
            let slice = &pap_512_bytes[i * PAP_512_BYTES..(i + 1) * PAP_512_BYTES];
            let bits: Vec<f32> = slice
                .iter()
                .flat_map(|&b| (0..8).map(move |bit| ((b >> bit) & 1) as f32))
                .collect();
            vectors.push(SpectralVector512 {
                id: i as u64,
                spectral_coords: bits,
            });
        }

        // Build correlation matrix (DIM x DIM).
        eprintln!("[tensor_spectral_512] Computing {DIM}x{DIM} correlation matrix...");
        let mut corr = vec![vec![0.0f32; DIM]; DIM];
        for v in &vectors {
            for i in 0..DIM {
                let vi = v.spectral_coords[i];
                let row = &mut corr[i];
                for j in 0..DIM {
                    row[j] += vi * v.spectral_coords[j];
                }
            }
        }
        let nf = n as f32;
        for row in &mut corr {
            for x in row.iter_mut() {
                *x /= nf;
            }
        }

        // Power iteration for top k eigenvectors with deflation.
        eprintln!("[tensor_spectral_512] Power iteration...");
        let mut eigenvectors: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut working_corr = corr;

        for _ in 0..k {
            let mut vec = vec![1.0f32; DIM];
            for _ in 0..100 {
                let mut new_vec = vec![0.0f32; DIM];
                for i in 0..DIM {
                    let mut sum = 0.0f32;
                    let row = &working_corr[i];
                    for j in 0..DIM {
                        sum += row[j] * vec[j];
                    }
                    new_vec[i] = sum;
                }
                let norm = new_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for x in &mut new_vec {
                        *x /= norm;
                    }
                }
                vec = new_vec;
            }

            // Deflate.
            for i in 0..DIM {
                for j in 0..DIM {
                    working_corr[i][j] -= vec[i] * vec[j];
                }
            }
            eigenvectors.push(vec);
        }

        // Project original vectors onto eigenbasis.
        eprintln!("[tensor_spectral_512] Projecting to {k} dims...");
        for v in &mut vectors {
            let mut projected = vec![0.0f32; k];
            for i in 0..k {
                let mut sum = 0.0f32;
                let evec = &eigenvectors[i];
                for j in 0..DIM {
                    sum += v.spectral_coords[j] * evec[j];
                }
                projected[i] = sum;
            }
            v.spectral_coords = projected;
        }

        eprintln!("[tensor_spectral_512] Done.");
        Self {
            vectors,
            eigenvectors,
            k,
        }
    }

    /// Find the `top_k` nearest spectral neighbors for a 512-bit PAP.
    pub fn query(&self, pap_512: &[u8; PAP_512_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        let query_vec: Vec<f32> = pap_512
            .iter()
            .flat_map(|&b| (0..8).map(move |bit| ((b >> bit) & 1) as f32))
            .collect();

        // Project query onto eigenbasis.
        let mut projected = vec![0.0f32; self.k];
        for i in 0..self.k {
            let mut sum = 0.0f32;
            let evec = &self.eigenvectors[i];
            for j in 0..DIM {
                sum += query_vec[j] * evec[j];
            }
            projected[i] = sum;
        }

        let mut scores: Vec<(f32, u64)> = self
            .vectors
            .iter()
            .map(|v| {
                let dist = euclidean_sq(&projected, &v.spectral_coords);
                (-dist, v.id)
            })
            .collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scores.truncate(top_k);
        scores
    }

    /// Override the vector ids after building (e.g. with caller-supplied ids).
    pub fn set_ids(&mut self, ids: &[u64]) {
        for (v, &id) in self.vectors.iter_mut().zip(ids.iter()) {
            v.id = id;
        }
    }

    /// Look up the precomputed spectral coordinates for a document id.
    pub fn vector_by_id(&self, id: u64) -> Option<&[f32]> {
        self.vectors
            .iter()
            .find(|v| v.id == id)
            .map(|v| v.spectral_coords.as_slice())
    }
}

fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
