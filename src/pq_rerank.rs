// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! OPQ-PQ asymmetric-distance re-ranker for 384-d MiniLM embeddings.
//!
//! Loads a compact PQ index (rotation + mean, codebooks, codes) and re-ranks
//! a short candidate list — typically the top-500 IDs returned by a binary
//! Hamming HNSW index — using fast per-subquantizer distance lookup tables.
//!
//! File layout expected (all little-endian):
//!
//!   `{prefix}_opq.bin`:
//!       [384*384 f32 rotation R][384 f32 mean]
//!
//!   `{prefix}_codebooks.bin`:
//!       [8 * 256 * 48 f32 centroids]
//!
//!   `{prefix}_index.bin`:
//!       [u64 N][N * 8 bytes PQ codes]
//!
//!   `{prefix}_ids.bin` (optional):
//!       [N u64 IDs]  -- if omitted, IDs are assumed to be 0..N-1.

use alloc::vec::Vec;
use core::cmp::Ordering;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use rustc_hash::FxHashMap;

const DIM: usize = 384;
const SUBQ: usize = 8;
const SUB_DIM: usize = 48;
const CENTROIDS: usize = 256;

/// An OPQ-PQ index suitable for re-ranking candidate IDs.
pub struct PqRerankIndex {
    mean: Vec<f32>,
    /// Row-major rotation matrix R (384 x 384).  A query is rotated as
    /// `(query - mean) @ R.T`, which in this layout is the dot product of
    /// the centered query with each row of R.
    rotation: Vec<f32>,
    /// Per-subquantizer centroid tables: `subq * 256 * sub_dim` floats.
    codebooks: Vec<Vec<f32>>,
    /// Flattened PQ codes: `[code_0_0, ..., code_0_7, code_1_0, ...]`.
    codes: Vec<u8>,
    len: usize,
    /// Optional mapping from record ID to code-row position.
    id_to_pos: Option<FxHashMap<u64, u32>>,
}

impl PqRerankIndex {
    /// Load the three index files sharing `prefix`.
    pub fn load<P: AsRef<Path>>(prefix: P) -> Result<Self, String> {
        let prefix = prefix.as_ref();
        let opq = read_f32_file(&path_with_suffix(prefix, "_opq.bin"))?;
        if opq.len() != DIM * DIM + DIM {
            return Err(format!(
                "opq file size mismatch: expected {} floats, got {}",
                DIM * DIM + DIM,
                opq.len()
            ));
        }
        let rotation = opq[..DIM * DIM].to_vec();
        let mean = opq[DIM * DIM..].to_vec();

        let cb = read_f32_file(&path_with_suffix(prefix, "_codebooks.bin"))?;
        if cb.len() != SUBQ * CENTROIDS * SUB_DIM {
            return Err(format!(
                "codebooks size mismatch: expected {} floats, got {}",
                SUBQ * CENTROIDS * SUB_DIM,
                cb.len()
            ));
        }
        let mut codebooks = Vec::with_capacity(SUBQ);
        for s in 0..SUBQ {
            let start = s * CENTROIDS * SUB_DIM;
            codebooks.push(cb[start..start + CENTROIDS * SUB_DIM].to_vec());
        }

        let (codes, len) = read_index_file(&path_with_suffix(prefix, "_index.bin"))?;

        let ids_path = path_with_suffix(prefix, "_ids.bin");
        let id_to_pos = if ids_path.exists() {
            let ids = read_u64_file(&ids_path)?;
            if ids.len() != len {
                return Err(format!(
                    "ids length mismatch: expected {}, got {}",
                    len,
                    ids.len()
                ));
            }
            let mut map = FxHashMap::default();
            map.reserve(ids.len());
            for (pos, &id) in ids.iter().enumerate() {
                map.insert(id, pos as u32);
            }
            Some(map)
        } else {
            None
        };

        Ok(Self {
            mean,
            rotation,
            codebooks,
            codes,
            len,
            id_to_pos,
        })
    }

    /// Number of indexed vectors.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Re-rank `candidates` by asymmetric PQ distance and return the top `k`.
    ///
    /// `query` must be a 384-dim float vector.  Returned scores are cosine
    /// approximations in `[0, 1]`, assuming unit-normalized embeddings.
    pub fn rerank(&self, query: &[f32], candidates: &[u64], k: usize) -> Vec<(u64, f32)> {
        if self.len == 0 || candidates.is_empty() || k == 0 {
            return Vec::new();
        }
        let query_rot = self.rotate_query(query);
        let lut = self.build_lut(&query_rot);

        let k = k.min(candidates.len());
        let mut heap: Vec<(u64, f32)> = Vec::with_capacity(k);

        for &id in candidates {
            let Some(pos) = self.pos_for_id(id) else { continue };
            let dist = self.distance_from_lut(&lut, pos);
            // Asymmetric PQ distance is approximately ||q - d||^2 in the
            // original (rotated) space.  For unit-normalized vectors,
            // cos(q, d) ≈ 1 - ||q - d||^2 / 2.
            let score = (1.0 - dist * 0.5).max(0.0).min(1.0);

            if heap.len() < k {
                heap.push((id, score));
                if heap.len() == k {
                    heap.sort_by(|a, b| cmp_f32(b.1, a.1));
                }
            } else if score > heap[k - 1].1 {
                heap[k - 1] = (id, score);
                heap.sort_by(|a, b| cmp_f32(b.1, a.1));
            }
        }

        heap.sort_by(|a, b| cmp_f32(b.1, a.1));
        heap
    }

    /// Rotate and center a query vector.
    fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        let mut centered = Vec::with_capacity(DIM);
        for i in 0..DIM {
            centered.push(query.get(i).copied().unwrap_or(0.0) - self.mean[i]);
        }

        let mut rotated = vec![0.0f32; DIM];
        for i in 0..DIM {
            let row = &self.rotation[i * DIM..(i + 1) * DIM];
            let mut acc = 0.0f32;
            for j in 0..DIM {
                acc += centered[j] * row[j];
            }
            rotated[i] = acc;
        }
        rotated
    }

    /// Build per-subquantizer distance lookup tables for a rotated query.
    fn build_lut(&self, query_rot: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0f32; SUBQ * CENTROIDS];
        for s in 0..SUBQ {
            let q_sub = &query_rot[s * SUB_DIM..(s + 1) * SUB_DIM];
            let centroids = &self.codebooks[s];
            for c in 0..CENTROIDS {
                let base = c * SUB_DIM;
                let mut dist = 0.0f32;
                for d in 0..SUB_DIM {
                    let diff = q_sub[d] - centroids[base + d];
                    dist += diff * diff;
                }
                lut[s * CENTROIDS + c] = dist;
            }
        }
        lut
    }

    #[inline]
    fn distance_from_lut(&self, lut: &[f32], pos: usize) -> f32 {
        let base = pos * SUBQ;
        let mut dist = 0.0f32;
        for s in 0..SUBQ {
            let code = self.codes[base + s] as usize;
            dist += lut[s * CENTROIDS + code];
        }
        dist
    }

    #[inline]
    fn pos_for_id(&self, id: u64) -> Option<usize> {
        if let Some(map) = &self.id_to_pos {
            map.get(&id).copied().map(|p| p as usize)
        } else if (id as usize) < self.len {
            Some(id as usize)
        } else {
            None
        }
    }
}

fn path_with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut p = prefix.to_path_buf();
    let name = p.file_name().unwrap_or_default().to_os_string();
    p.set_file_name(format!("{}{}", name.to_string_lossy(), suffix));
    p
}

fn read_f32_file(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: f32 file length not a multiple of 4", path.display()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn read_u64_file(path: &Path) -> Result<Vec<u64>, String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    if bytes.len() % 8 != 0 {
        return Err(format!("{}: u64 file length not a multiple of 8", path.display()));
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        out.push(u64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3],
                                      chunk[4], chunk[5], chunk[6], chunk[7]]));
    }
    Ok(out)
}

fn read_index_file(path: &Path) -> Result<(Vec<u8>, usize), String> {
    let mut file = File::open(path).map_err(|e| format!("failed to open {}: {}", path.display(), e))?;
    let mut n_buf = [0u8; 8];
    file.read_exact(&mut n_buf)
        .map_err(|e| format!("failed to read header from {}: {}", path.display(), e))?;
    let n = u64::from_le_bytes(n_buf) as usize;

    let expected = n.checked_mul(SUBQ).ok_or_else(|| format!("index too large: {}", n))?;
    let mut codes = vec![0u8; expected];
    file.read_exact(&mut codes)
        .map_err(|e| format!("failed to read codes from {}: expected {} bytes, err {}", path.display(), expected, e))?;
    Ok((codes, n))
}

#[inline]
fn cmp_f32(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_opq(dir: &TempDir, prefix: &str) {
        let mut buf: Vec<u8> = Vec::new();
        // Identity rotation.
        for i in 0..DIM {
            for j in 0..DIM {
                let v: f32 = if i == j { 1.0 } else { 0.0 };
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for _ in 0..DIM {
            buf.extend_from_slice(&0.0f32.to_le_bytes());
        }
        let path = dir.path().join(format!("{}_opq.bin", prefix));
        let mut f = File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
    }

    fn write_codebooks(dir: &TempDir, prefix: &str) {
        let mut buf: Vec<u8> = Vec::new();
        for s in 0..SUBQ {
            for c in 0..CENTROIDS {
                for d in 0..SUB_DIM {
                    let v = (s * 10 + c) as f32 + d as f32 * 0.01;
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        let path = dir.path().join(format!("{}_codebooks.bin", prefix));
        let mut f = File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
    }

    fn write_index(dir: &TempDir, prefix: &str, n: usize) {
        let path = dir.path().join(format!("{}_index.bin", prefix));
        let mut f = File::create(&path).unwrap();
        f.write_all(&(n as u64).to_le_bytes()).unwrap();
        let mut codes = vec![0u8; n * SUBQ];
        for i in 0..n {
            for s in 0..SUBQ {
                codes[i * SUBQ + s] = ((i + s) % 256) as u8;
            }
        }
        f.write_all(&codes).unwrap();
    }

    #[test]
    fn smoke_load_and_rerank() {
        let dir = TempDir::new().unwrap();
        write_opq(&dir, "pq");
        write_codebooks(&dir, "pq");
        write_index(&dir, "pq", 100);

        let idx = PqRerankIndex::load(dir.path().join("pq")).unwrap();
        assert_eq!(idx.len(), 100);

        let query = vec![0.0f32; DIM];
        let candidates: Vec<u64> = (0..100).collect();
        let top = idx.rerank(&query, &candidates, 5);
        assert_eq!(top.len(), 5);
    }
}
