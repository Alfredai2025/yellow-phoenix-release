// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, CStr};
use std::collections::HashMap;
use std::sync::Mutex;
use std::io::Read;

const N_SUBCHUNKS: usize = 32;
const DIM: usize = 16;
const K: usize = 16;

struct SoftTable {
    lookup: HashMap<String, [u8; 16]>,
    centers: Vec<Vec<f32>>,
}

static TABLE: Mutex<Option<SoftTable>> = Mutex::new(None);

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; f.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u16(f: &mut std::fs::File) -> std::io::Result<u16> {
    let mut b = [0u8; 2]; f.read_exact(&mut b)?; Ok(u16::from_le_bytes(b))
}

#[no_mangle]
pub extern "C" fn yp_soft_load(dir_path: *const c_char) -> i32 {
    if dir_path.is_null() { return -1; }
    let dir = unsafe { CStr::from_ptr(dir_path).to_string_lossy() };

    // Load centers
    let mut centers = Vec::with_capacity(N_SUBCHUNKS);
    for i in 0..N_SUBCHUNKS {
        let path = format!("{}/soft_centers_{}.f32", dir, i);
        let bytes = match std::fs::read(&path) {
            Ok(b) => b, Err(_) => return -(2 + i as i32),
        };
        if bytes.len() != K * DIM * 4 { return -(3 + i as i32); }
        let f32s: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        centers.push(f32s);
    }

    // Load lookup table
    let lut_path = format!("{}/soft_token_lookup.bin", dir);
    let mut file = match std::fs::File::open(&lut_path) {
        Ok(f) => f, Err(_) => return -10,
    };
    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() || &magic != b"YPST" { return -11; }
    if read_u32(&mut file).unwrap_or(0) != 1 { return -12; }
    let num_tokens = read_u32(&mut file).unwrap_or(0) as usize;
    let _ = read_u32(&mut file); let _ = read_u32(&mut file);

    let mut lookup = HashMap::with_capacity(num_tokens);
    for _ in 0..num_tokens {
        let tok_len = read_u16(&mut file).unwrap_or(0) as usize;
        let mut tok_buf = vec![0u8; tok_len];
        if file.read_exact(&mut tok_buf).is_err() { return -13; }
        let token = String::from_utf8(tok_buf).unwrap_or_default();
        let mut packed = [0u8; 16];
        if file.read_exact(&mut packed).is_err() { return -14; }
        lookup.insert(token, packed);
    }

    let mut guard = TABLE.lock().unwrap();
    *guard = Some(SoftTable { lookup, centers });
    0
}

#[no_mangle]
pub extern "C" fn yp_soft_encode(text: *const c_char, out_hash: *mut u8, out_len: usize) -> i32 {
    if text.is_null() || out_hash.is_null() || out_len != 64 { return -1; }
    let text = unsafe { CStr::from_ptr(text).to_string_lossy() };
    let guard = TABLE.lock().unwrap();
    let table = match guard.as_ref() { Some(t) => t, None => return -4 };

    let mut hash512 = [0u8; 64];
    for sub in 0..N_SUBCHUNKS {
        let mut acc = [0.0f32; DIM];
        let mut tok_count = 0usize;
        for tok in text.split_whitespace() {
            let packed = match table.lookup.get(tok.to_lowercase().as_str()) {
                Some(p) => p, None => continue,
            };
            let byte_idx = sub / 2;
            let idx = if sub % 2 == 0 {
                packed[byte_idx] & 0x0F
            } else {
                (packed[byte_idx] >> 4) & 0x0F
            } as usize;
            for d in 0..DIM {
                acc[d] += table.centers[sub][idx * DIM + d];
            }
            tok_count += 1;
        }
        if tok_count == 0 { continue; }
        let inv = 1.0 / (tok_count as f32);
        let mut b0 = 0u8; let mut b1 = 0u8;
        for d in 0..8 { if acc[d] * inv > 0.0 { b0 |= 1 << d; } }
        for d in 0..8 { if acc[d + 8] * inv > 0.0 { b1 |= 1 << d; } }
        let idx = sub * 2;
        hash512[idx] = b0;
        hash512[idx + 1] = b1;
    }
    unsafe { std::ptr::copy_nonoverlapping(hash512.as_ptr(), out_hash, 64); }
    0
}

#[no_mangle]
pub extern "C" fn yp_hierarchical_load(dir_path: *const c_char) -> i32 { yp_soft_load(dir_path) }
#[no_mangle]
pub extern "C" fn yp_hierarchical_encode(text: *const c_char, out_hash: *mut u8, out_len: usize) -> i32 {
    yp_soft_encode(text, out_hash, out_len)
}
