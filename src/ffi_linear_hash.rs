// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::io::Read;
use std::sync::Mutex;

const DIM: usize = 512;

struct LinearHashModel {
    vocab: HashMap<String, Vec<f32>>, // token -> 512 weights
    bias: Vec<f32>,
}

static MODEL: Mutex<Option<LinearHashModel>> = Mutex::new(None);

fn read_u32(f: &mut std::fs::File) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u16(f: &mut std::fs::File) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    f.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

#[no_mangle]
pub extern "C" fn yp_linear_hash_load(path: *const c_char) -> i32 {
    if path.is_null() {
        return -1;
    }
    let p = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let mut file = match std::fs::File::open(&*p) {
        Ok(f) => f,
        Err(_) => return -2,
    };

    let mut magic = [0u8; 4];
    if file.read_exact(&mut magic).is_err() || &magic != b"YPLH" {
        return -3;
    }
    if read_u32(&mut file).unwrap_or(0) != 1 {
        return -4;
    }
    let vocab_size = read_u32(&mut file).unwrap_or(0) as usize;
    let dim = read_u32(&mut file).unwrap_or(0) as usize;
    if dim != DIM {
        return -5;
    }

    // Bias
    let mut bias = vec![0.0f32; DIM];
    for d in 0..DIM {
        let mut b = [0u8; 4];
        if file.read_exact(&mut b).is_err() {
            return -6;
        }
        bias[d] = f32::from_le_bytes(b);
    }

    // Vocab weights
    let mut vocab = HashMap::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        let tok_len = read_u16(&mut file).unwrap_or(0) as usize;
        let mut tok_buf = vec![0u8; tok_len];
        if file.read_exact(&mut tok_buf).is_err() {
            return -7;
        }
        let token = String::from_utf8(tok_buf).unwrap_or_default();

        let mut w = vec![0.0f32; DIM];
        for d in 0..DIM {
            let mut b = [0u8; 4];
            if file.read_exact(&mut b).is_err() {
                return -8;
            }
            w[d] = f32::from_le_bytes(b);
        }
        vocab.insert(token, w);
    }

    let mut guard = MODEL.lock().unwrap();
    *guard = Some(LinearHashModel { vocab, bias });
    0
}

#[no_mangle]
pub extern "C" fn yp_linear_hash_encode(
    text: *const c_char,
    out_hash: *mut u8,
    out_len: usize,
) -> i32 {
    if text.is_null() || out_hash.is_null() || out_len != 64 {
        return -1;
    }
    let text = unsafe { CStr::from_ptr(text).to_string_lossy() };
    let guard = MODEL.lock().unwrap();
    let model = match guard.as_ref() {
        Some(m) => m,
        None => return -9,
    };

    let mut acc = model.bias.clone();
    let mut tok_count = 0usize;

    for tok in text.split_whitespace() {
        let t = tok.to_lowercase();
        if let Some(w) = model.vocab.get(&t) {
            for d in 0..DIM {
                acc[d] += w[d];
            }
            tok_count += 1;
        }
    }

    if tok_count == 0 {
        return -10;
    }

    let mut hash512 = [0u8; 64];
    for sub in 0..32 {
        let mut b0 = 0u8;
        let mut b1 = 0u8;
        for d in 0..8 {
            if acc[sub * 16 + d] > 0.0 {
                b0 |= 1 << d;
            }
        }
        for d in 0..8 {
            if acc[sub * 16 + d + 8] > 0.0 {
                b1 |= 1 << d;
            }
        }
        hash512[sub * 2] = b0;
        hash512[sub * 2 + 1] = b1;
    }

    unsafe {
        std::ptr::copy_nonoverlapping(hash512.as_ptr(), out_hash, 64);
    }
    0
}
