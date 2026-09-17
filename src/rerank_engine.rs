// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::fs::File;
use std::slice;
use memmap2::Mmap;

/// Zero-copy memory-mapped embedding matrix for HNSW candidate re-ranking.
///
/// Maps a float32 .npy file (shape N × D) into process address space.
/// Row i corresponds to the embedding of paper with rust_id == i (arxiv1m sequential load).
pub struct RerankEngine {
    data: *const f32,
    rows: usize,
    cols: usize,
    _mmap: Mmap,
}

impl RerankEngine {
    /// Load a float32 .npy file via read-only memory map.
    ///
    /// # Safety
    /// The file must remain valid and unmodified for the lifetime of the engine.
    pub unsafe fn new(npy_path: &str) -> Result<Self, String> {
        let file = File::open(npy_path).map_err(|e| format!("open: {}", e))?;
        let mmap = Mmap::map(&file).map_err(|e| format!("mmap: {}", e))?;
        let bytes = mmap.as_ref();

        let (data_offset, rows, cols) = parse_npy_f32_header(bytes)?;
        if data_offset + rows * cols * 4 > bytes.len() {
            return Err("npy data truncated".into());
        }

        let data = bytes.as_ptr().add(data_offset) as *const f32;
        Ok(Self { data, rows, cols, _mmap: mmap })
    }

    /// Zero-copy access to a single embedding row.
    /// Used to obtain the float query vector for re-ranking (row == doc id).
    pub fn embedding(&self, id: u64) -> Option<&[f32]> {
        let idx = id as usize;
        if idx >= self.rows {
            return None;
        }
        Some(unsafe { slice::from_raw_parts(self.data.add(idx * self.cols), self.cols) })
    }

    /// Re-rank HNSW candidates by cosine similarity to a query embedding.
    ///
    /// # Arguments
    /// * `candidates` — HNSW node indices (uint64). For arxiv1m, node_id == rust_id == row index.
    /// * `query` — query embedding (f32, already L2-normalized by Python caller).
    /// * `k` — return top-k results.
    ///
    /// # Returns
    /// Vec of (node_id, cosine_score), sorted descending.
    pub fn rerank(&self, candidates: &[u64], query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.cols, "query dimension mismatch");
        let mut scored = Vec::with_capacity(candidates.len());
        let cols = self.cols;

        for &node_id in candidates {
            let idx = node_id as usize;
            if idx >= self.rows {
                continue;
            }

            // Zero-copy fetch: pointer arithmetic into the mmap.
            let emb = unsafe { slice::from_raw_parts(self.data.add(idx * cols), cols) };
            let score = dot_product(query, emb);
            scored.push((node_id, score));
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

/// Parse standard NumPy .npy header for float32 C-contiguous 2D array.
/// Returns (data_offset, rows, cols).
unsafe fn parse_npy_f32_header(bytes: &[u8]) -> Result<(usize, usize, usize), String> {
    if bytes.len() < 10 {
        return Err("file too short".into());
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err("not a .npy file".into());
    }

    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = if major == 1 && minor == 0 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else if major == 2 && minor == 0 {
        if bytes.len() < 12 {
            return Err("file too short for v2".into());
        }
        (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
    } else {
        return Err(format!("unsupported npy version {}.{}", major, minor));
    };

    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err("header exceeds file".into());
    }

    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| "invalid utf-8 in header")?;

    let shape_start = header.find('(').ok_or("no shape")?;
    let shape_end = header.find(')').ok_or("unclosed shape")?;
    let dims: Vec<usize> = header[shape_start + 1..shape_end]
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "invalid shape")?;

    if dims.len() != 2 {
        return Err(format!("expected 2D, got {}D", dims.len()));
    }
    let rows = dims[0];
    let cols = dims[1];

    if !header.contains("<f4") && !header.contains("|f4") {
        return Err("expected float32 dtype".into());
    }

    Ok((header_end, rows, cols))
}

/// Portable SIMD-friendly dot product with 8× unrolled accumulation.
/// Embeddings are pre-normalized, so dot product == cosine similarity.
#[inline(always)]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    let len = a.len();
    let mut i = 0;
    while i + 8 <= len {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
        i += 8;
    }
    while i < len {
        s0 += a[i] * b[i];
        i += 1;
    }
    s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
}
