// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, CStr};
use std::sync::Mutex;
use once_cell::sync::Lazy;

const N_SUBCHUNKS: usize = 32;
const DIM: usize = 16;
const K: usize = 16;

struct CodebookBank {
    centers: Vec<Vec<f32>>, // [subchunk][k * dim]
}

static BANK: Lazy<Mutex<Option<CodebookBank>>> = Lazy::new(|| Mutex::new(None));

#[no_mangle]
pub extern "C" fn yp_hierarchical_load(data_dir: *const c_char) -> i32 {
    if data_dir.is_null() { return -1; }
    let dir = unsafe { CStr::from_ptr(data_dir).to_string_lossy() };
    let mut centers = Vec::with_capacity(N_SUBCHUNKS);
    for sub in 0..N_SUBCHUNKS {
        let path = format!("{}/soft_centers_{}.f32", dir, sub);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => return -2,
        };
        let expected = K * DIM * 4;
        if bytes.len() != expected { return -3; }
        let f32s: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        centers.push(f32s);
    }
    let mut guard = BANK.lock().unwrap();
    *guard = Some(CodebookBank { centers });
    0
}

fn token_emb(tok: &str) -> Vec<f32> {
    // Simple hash-based embedding for tokens (deterministic)
    let hash = blake3::hash(tok.as_bytes());
    let mut v = vec![0.0f32; DIM];
    let bytes = hash.as_bytes();
    for i in 0..DIM {
        let idx = i * 2;
        v[i] = ((u16::from_le_bytes([bytes[idx], bytes[idx+1]]) as f32) / 65535.0) * 2.0 - 1.0;
    }
    v
}

fn find_nearest(q: &[f32], centers: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k);
    for ci in 0..(centers.len() / DIM) {
        let mut dist = 0.0f32;
        for d in 0..DIM {
            let diff = q[d] - centers[ci * DIM + d];
            dist += diff * diff;
        }
        best.push((ci, dist));
    }
    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    best.truncate(k);
    best
}

#[no_mangle]
pub extern "C" fn yp_hierarchical_encode(text: *const c_char, out_hash: *mut u8, out_len: usize) -> i32 {
    if text.is_null() || out_hash.is_null() || out_len != 64 { return -1; }
    let text = unsafe { CStr::from_ptr(text).to_string_lossy() };
    let guard = BANK.lock().unwrap();
    let bank = match guard.as_ref() { Some(b) => b, None => return -4 };
    let mut hash512 = [0u8; 64];

    for sub in 0..N_SUBCHUNKS {
        let mut acc = [0.0f32; DIM];
        let mut tok_count = 0usize;
        for tok in text.split_whitespace() {
            let q = token_emb(tok);
            let nearest = find_nearest(&q, &bank.centers[sub], 3);
            let total_dist: f32 = nearest.iter().map(|(_, d)| 1.0 / (d + 0.001)).sum();
            for (ci, dist) in nearest {
                let weight = (1.0 / (dist + 0.001)) / total_dist;
                for d in 0..DIM {
                    acc[d] += bank.centers[sub][ci * DIM + d] * weight;
                }
            }
            tok_count += 1;
        }
        if tok_count == 0 { continue; }

        let mut b0 = 0u8; let mut b1 = 0u8;
        for d in 0..8 { if acc[d] > 0.0 { b0 |= 1 << d; } }
        for d in 0..8 { if acc[d + 8] > 0.0 { b1 |= 1 << d; } }
        let idx = sub * 2;
        hash512[idx] = b0;
        hash512[idx + 1] = b1;
    }

    unsafe { std::ptr::copy_nonoverlapping(hash512.as_ptr(), out_hash, 64); }
    0
}
