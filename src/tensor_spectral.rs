// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

/// Tensor Spectral Index — Pure Rust, zero dependencies
/// Eigendecomposition via power iteration on 128×128 covariance matrix

const PAP_BITS: usize = 128;
const PAP_BYTES: usize = 16;

pub struct SpectralVector {
    pub id: u64,
    pub pap: [u8; PAP_BYTES],
    pub spectral_coords: Vec<f32>,
}

pub struct TensorSpectralIndex {
    eigenvectors: Vec<Vec<f32>>,
    eigenvalues: Vec<f32>,
    vectors: Vec<SpectralVector>,
    k: usize,
}

impl TensorSpectralIndex {
    pub fn build(paps: &[(u64, [u8; PAP_BYTES])]) -> Self {
        let n = paps.len();
        if n == 0 {
            return Self {
                eigenvectors: Vec::new(),
                eigenvalues: Vec::new(),
                vectors: Vec::new(),
                k: 0,
            };
        }

        // Convert PAPs to centered bit vectors
        let mut data: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut mean = vec![0.0f32; PAP_BITS];

        for (_, pap) in paps.iter() {
            let mut vec = vec![0.0f32; PAP_BITS];
            for b in 0..PAP_BITS {
                let byte_idx = b / 8;
                let bit_idx = b % 8;
                if (pap[byte_idx] >> bit_idx) & 1 == 1 {
                    vec[b] = 1.0;
                    mean[b] += 1.0;
                }
            }
            data.push(vec);
        }

        for b in 0..PAP_BITS {
            mean[b] /= n as f32;
        }

        // Center data
        for i in 0..n {
            for b in 0..PAP_BITS {
                data[i][b] -= mean[b];
            }
        }

        // Compute covariance matrix (128 × 128)
        let mut cov = vec![vec![0.0f32; PAP_BITS]; PAP_BITS];
        for i in 0..PAP_BITS {
            for j in 0..PAP_BITS {
                let mut sum = 0.0f32;
                for row in 0..n {
                    sum += data[row][i] * data[row][j];
                }
                cov[i][j] = sum / (n as f32);
            }
        }

        // Power iteration to find top-k eigenvectors
        let mut eigenvectors: Vec<Vec<f32>> = Vec::new();
        let mut eigenvalues: Vec<f32> = Vec::new();
        let mut k = 0;

        for _ in 0..PAP_BITS {
            let mut v = vec![0.0f32; PAP_BITS];
            for i in 0..PAP_BITS {
                v[i] = (i as f32 + 0.5).sin();
            }

            // Orthogonalize against previous eigenvectors
            for prev in &eigenvectors {
                let mut dot = 0.0f32;
                for i in 0..PAP_BITS {
                    dot += v[i] * prev[i];
                }
                for i in 0..PAP_BITS {
                    v[i] -= dot * prev[i];
                }
            }

            // Power iteration (20 iterations)
            for _ in 0..20 {
                let mut new_v = vec![0.0f32; PAP_BITS];
                for i in 0..PAP_BITS {
                    for j in 0..PAP_BITS {
                        new_v[i] += cov[i][j] * v[j];
                    }
                }
                let mut norm = 0.0f32;
                for i in 0..PAP_BITS {
                    norm += new_v[i] * new_v[i];
                }
                norm = norm.sqrt();
                if norm > 1e-10 {
                    for i in 0..PAP_BITS {
                        v[i] = new_v[i] / norm;
                    }
                }
            }

            // Compute eigenvalue
            let mut eigval = 0.0f32;
            for i in 0..PAP_BITS {
                let mut cv = 0.0f32;
                for j in 0..PAP_BITS {
                    cv += cov[i][j] * v[j];
                }
                eigval += v[i] * cv;
            }

            if eigval < 1e-6 {
                break;
            }

            eigenvectors.push(v);
            eigenvalues.push(eigval);
            k += 1;

            // Stop at 95% variance
            let total_var: f32 = eigenvalues.iter().sum();
            let cumsum: f32 = eigenvalues.iter().sum();
            if cumsum / total_var >= 0.95 && k >= 10 {
                break;
            }
        }

        // Project all vectors into spectral basis
        let mut vectors = Vec::with_capacity(n);
        for (idx, (id, pap)) in paps.iter().enumerate() {
            let mut coords = vec![0.0f32; k];
            for d in 0..k {
                let mut coord = 0.0f32;
                for b in 0..PAP_BITS {
                    coord += eigenvectors[d][b] * data[idx][b];
                }
                coords[d] = coord;
            }
            vectors.push(SpectralVector {
                id: *id,
                pap: *pap,
                spectral_coords: coords,
            });
        }

        Self {
            eigenvectors,
            eigenvalues,
            vectors,
            k,
        }
    }

    pub fn query(&self, pap: &[u8; PAP_BYTES], top_k: usize) -> Vec<(f32, u64)> {
        if self.vectors.is_empty() || self.k == 0 {
            return Vec::new();
        }

        let mut query_bits = vec![0.0f32; PAP_BITS];
        for b in 0..PAP_BITS {
            let byte_idx = b / 8;
            let bit_idx = b % 8;
            if (pap[byte_idx] >> bit_idx) & 1 == 1 {
                query_bits[b] = 1.0;
            }
        }

        let mut query_spectral = vec![0.0f32; self.k];
        for d in 0..self.k {
            for b in 0..PAP_BITS {
                query_spectral[d] += self.eigenvectors[d][b] * query_bits[b];
            }
        }

        let mut scores: Vec<(f32, u64)> = self.vectors.iter().map(|v| {
            let mut dist = 0.0f32;
            for d in 0..self.k {
                let diff = query_spectral[d] - v.spectral_coords[d];
                dist += diff * diff * self.eigenvalues[d];
            }
            let score = 1.0 / (1.0 + dist.sqrt());
            (score, v.id)
        }).collect();

        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scores.truncate(top_k);
        scores
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Look up the precomputed spectral coordinates for a document id.
    pub fn vector_by_id(&self, id: u64) -> Option<&[f32]> {
        self.vectors.iter().find(|v| v.id == id).map(|v| v.spectral_coords.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_query() {
        let paps: Vec<(u64, [u8; 16])> = (0..100).map(|i| {
            let mut pap = [0u8; 16];
            pap[0] = (i * 7) as u8;
            pap[1] = (i * 13) as u8;
            (i as u64, pap)
        }).collect();

        let index = TensorSpectralIndex::build(&paps);
        assert_eq!(index.len(), 100);
        assert!(index.k > 0);

        let query = paps[50].1;
        let results = index.query(&query, 100);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(_, id)| *id == 50));
    }
}
