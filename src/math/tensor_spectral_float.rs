// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Float-embedding spectral index for YP Functor Audit.
//! Power-iteration eigendecomposition on real-valued embeddings (e.g. MiniLM).

#[derive(Clone, Debug)]
pub struct SpectralVectorFloat {
    pub id: u64,
    pub coords: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TensorSpectralIndexFloat {
    vectors: Vec<SpectralVectorFloat>,
    eigenvectors: Vec<Vec<f32>>,
    k: usize,
    dim: usize,
}

impl TensorSpectralIndexFloat {
    pub fn build(embeddings: &[f32], dim: usize, k: usize) -> Self {
        let n = embeddings.len() / dim;
        let k = k.min(dim).min(n.max(1));
        eprintln!("[tensor_spectral_float] Building {n} vectors, dim={dim}, k={k}...");

        // Correlation matrix
        let mut corr = vec![vec![0.0f32; dim]; dim];
        for i in 0..n {
            let vec = &embeddings[i * dim..(i + 1) * dim];
            for a in 0..dim {
                let va = vec[a];
                let row = &mut corr[a];
                for b in 0..dim {
                    row[b] += va * vec[b];
                }
            }
        }
        let nf = n as f32;
        for row in &mut corr {
            for x in row.iter_mut() {
                *x /= nf;
            }
        }

        // Power iteration with deflation
        let mut eigenvectors = Vec::with_capacity(k);
        let mut working_corr = corr;
        for _ in 0..k {
            let mut vec = vec![1.0f32; dim];
            for _ in 0..100 {
                let mut new_vec = vec![0.0f32; dim];
                for a in 0..dim {
                    let mut sum = 0.0f32;
                    let row = &working_corr[a];
                    for b in 0..dim {
                        sum += row[b] * vec[b];
                    }
                    new_vec[a] = sum;
                }
                let norm = new_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    for x in &mut new_vec {
                        *x /= norm;
                    }
                }
                vec = new_vec;
            }
            for a in 0..dim {
                for b in 0..dim {
                    working_corr[a][b] -= vec[a] * vec[b];
                }
            }
            eigenvectors.push(vec);
        }

        // Project embeddings onto eigenbasis
        eprintln!("[tensor_spectral_float] Projecting...");
        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            let emb = &embeddings[i * dim..(i + 1) * dim];
            let mut projected = vec![0.0f32; k];
            for j in 0..k {
                let mut sum = 0.0f32;
                let evec = &eigenvectors[j];
                for d in 0..dim {
                    sum += emb[d] * evec[d];
                }
                projected[j] = sum;
            }
            vectors.push(SpectralVectorFloat { id: i as u64, coords: projected });
        }

        eprintln!("[tensor_spectral_float] Done.");
        Self { vectors, eigenvectors, k, dim }
    }

    pub fn query(&self, query: &[f32], top_k: usize) -> Vec<(f32, u64)> {
        let mut projected = vec![0.0f32; self.k];
        for j in 0..self.k {
            let mut sum = 0.0f32;
            let evec = &self.eigenvectors[j];
            for d in 0..self.dim {
                sum += query[d] * evec[d];
            }
            projected[j] = sum;
        }
        let mut scores: Vec<(f32, u64)> = self.vectors.iter().map(|v| {
            let dist = euclidean_sq(&projected, &v.coords);
            (-dist, v.id)
        }).collect();
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scores.truncate(top_k);
        scores
    }

    pub fn vector_by_id(&self, id: u64) -> Option<&[f32]> {
        self.vectors.iter().find(|v| v.id == id).map(|v| v.coords.as_slice())
    }
}

fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
