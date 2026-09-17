// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! BipolarEncoder FFI — reverse-engineered from ITQ.
use std::ffi::{c_char, CStr};
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use crate::core::encoder::BipolarEncoder;

const CHUNK_FLOATS: usize = 128 * 128;
const BIAS_FLOATS: usize = 128;

struct BipolarBank {
    encoders: [BipolarEncoder; 4],
}

static BANK: Mutex<Option<BipolarBank>> = Mutex::new(None);

fn load_f32s(path: &Path, expected: usize) -> Option<Vec<f32>> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() != expected * 4 { return None; }
    let mut v = Vec::with_capacity(expected);
    for chunk in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(v)
}

fn build_encoder(proj: &[f32], bias: &[f32]) -> Option<BipolarEncoder> {
    if proj.len() != CHUNK_FLOATS || bias.len() != BIAS_FLOATS { return None; }
    let mut projection = [[0.0f32; 128]; 128];
    for r in 0..128 {
        for c in 0..128 {
            projection[r][c] = proj[r * 128 + c];
        }
    }
    Some(BipolarEncoder::from_pretrained(&projection, bias))
}

fn hash_token(seed: u64, token: &str) -> u64 {
    let mut h = seed;
    for b in token.bytes() {
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(b as u64);
    }
    h
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn tokenize_to_rows(text: &str, seed: u64) -> Vec<usize> {
    let mut rows = Vec::new();
    for tok in text.split_whitespace() {
        let h = hash_token(seed, tok);
        let h2 = splitmix64(h);
        rows.push((h2 % 128) as usize);
    }
    rows
}

#[no_mangle]
pub extern "C" fn yp_bipolar_load(dir_path: *const c_char) -> i32 {
    if dir_path.is_null() { return -1; }
    let path = unsafe { CStr::from_ptr(dir_path).to_string_lossy() };
    let dir = Path::new(&*path);
    let mut encoders = Vec::with_capacity(4);
    for i in 0..4 {
        let proj = match load_f32s(&dir.join(format!("bipolar_proj_{}.f32", i)), CHUNK_FLOATS) {
            Some(v) => v, None => return -2 - i,
        };
        let bias = match load_f32s(&dir.join(format!("bipolar_bias_{}.f32", i)), BIAS_FLOATS) {
            Some(v) => v, None => return -6 - i,
        };
        let enc = match build_encoder(&proj, &bias) {
            Some(e) => e, None => return -10 - i,
        };
        encoders.push(enc);
    }
    let bank = BipolarBank {
        encoders: [encoders.remove(0), encoders.remove(0), encoders.remove(0), encoders.remove(0)],
    };
    if let Ok(mut guard) = BANK.lock() {
        *guard = Some(bank); 0
    } else { -20 }
}

#[no_mangle]
pub extern "C" fn yp_bipolar_encode(text: *const c_char, out_hash: *mut u8, out_len: usize) -> i32 {
    if text.is_null() || out_hash.is_null() || out_len != 64 { return -1; }
    let text = unsafe { CStr::from_ptr(text).to_string_lossy() };
    let rows = tokenize_to_rows(&text, 42);
    if rows.is_empty() { return -2; }
    let guard = match BANK.lock() { Ok(g) => g, Err(_) => return -3 };
    let bank = match guard.as_ref() { Some(b) => b, None => return -4 };
    let mut hash512 = [0u8; 64];
    for chunk_i in 0..4 {
        let bits128 = bank.encoders[chunk_i].encode_tokens_u64(&rows);
        let bytes = bits128.to_le_bytes();
        let start = chunk_i * 16;
        hash512[start..start + 16].copy_from_slice(&bytes);
    }
    unsafe { std::ptr::copy_nonoverlapping(hash512.as_ptr(), out_hash, 64); }
    0
}
