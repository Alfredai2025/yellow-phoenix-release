// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, c_void, CStr};
use std::fs::File;
use std::io::Read;
use std::sync::OnceLock;

const ITQ_MAGIC: &[u8] = b"YPITQ512";
const ITQ_HEADER_LEN: usize = 12; // magic(8) + n_dims(u32) + n_bits(u32)

struct ItqModel {
    n_dims: usize,
    n_bits: usize,
    mean: Vec<f64>,
    proj: Vec<f64>, // row-major: n_dims * n_bits
}

static ITQ_MODEL: OnceLock<ItqModel> = OnceLock::new();

fn load_itq_model(path: &str) -> Result<ItqModel, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {}", path, e))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| format!("read: {}", e))?;

    if buf.len() < ITQ_HEADER_LEN {
        return Err("file too small for header".into());
    }

    if &buf[0..8] != ITQ_MAGIC {
        return Err(format!("bad magic: expected {:?}", ITQ_MAGIC));
    }

    let n_dims = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
    let n_bits = 512usize;
    let expected_body = (n_dims * 4) + (n_dims * n_bits * 4);
    if buf.len() < ITQ_HEADER_LEN + expected_body {
        return Err(format!(
            "file too small: need {} bytes, have {}",
            ITQ_HEADER_LEN + expected_body,
            buf.len()
        ));
    }

    let mut mean = Vec::with_capacity(n_dims);
    let mut offset = ITQ_HEADER_LEN;
    for _ in 0..n_dims {
        let val = f32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]]);
        mean.push(val as f64);
        offset += 4;
    }

    let mut proj = Vec::with_capacity(n_dims * n_bits);
    for _ in 0..(n_dims * n_bits) {
        let val = f32::from_le_bytes([buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3]]);
        proj.push(val as f64);
        offset += 4;
    }

    Ok(ItqModel {
        n_dims,
        n_bits,
        mean,
        proj,
    })
}

/// Load an ITQ model from a binary file.
/// Format: "YPITQ512" magic (8 bytes), n_dims u32, then n_dims f32 mean values,
/// then n_dims * 512 f32 projection values (row-major).
/// Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn yp_itq_init(path: *const c_char) -> c_int {
    if path.is_null() {
        eprintln!("yp_itq_init: NULL path");
        return -1;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("yp_itq_init: invalid UTF-8");
            return -2;
        }
    };

    // DEFENSIVE: check file exists before trying to load
    if !std::path::Path::new(path_str).exists() {
        eprintln!("yp_itq_init: file not found: {}", path_str);
        return -4;
    }

    let meta = match std::fs::metadata(path_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("yp_itq_init: metadata failed: {}", e);
            return -5;
        }
    };
    eprintln!("yp_itq_init: loading {} ({} bytes)", path_str, meta.len());

    // Idempotent: if a model is already loaded, do not fail.
    if ITQ_MODEL.get().is_some() {
        eprintln!("yp_itq_init: already initialized, returning 0");
        return 0;
    }

    let model = match load_itq_model(path_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("yp_itq_init: load_itq_model failed: {}", e);
            return -3;
        }
    };

    match ITQ_MODEL.set(model) {
        Ok(_) => {
            eprintln!("yp_itq_init: success");
            0
        }
        Err(_) => {
            eprintln!("yp_itq_init: model already set (race)");
            0
        }
    }
}

/// Encode a 384-d float embedding into a 512-bit ITQ hash.
/// `vec` must point to `dim` f32 values.
/// `out_hash` must point to at least n_bits/8 (64) bytes.
/// Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn yp_itq_encode(vec: *const f32, dim: usize, out_hash: *mut u8) -> c_int {
    if vec.is_null() || out_hash.is_null() {
        return -1;
    }
    let model = match ITQ_MODEL.get() {
        Some(m) => m,
        None => return -5,
    };
    if dim != model.n_dims {
        return -6;
    }

    let emb = unsafe { std::slice::from_raw_parts(vec, dim) };
    let hash_out = unsafe { std::slice::from_raw_parts_mut(out_hash, model.n_bits / 8) };
    hash_out.fill(0);

    for bit in 0..model.n_bits {
        let mut acc: f64 = 0.0;
        for d in 0..model.n_dims {
            let centered = (emb[d] as f64) - model.mean[d];
            let w = model.proj[d * model.n_bits + bit];
            acc += centered * w;
        }
        // MSB-first within each byte (numpy packbits convention) — the index
        // files (YISM/YPH5) and the Python exporter all pack bit 0 into the
        // HIGH bit of byte 0. LSB-first here bit-reverses every byte, which
        // reads as ~256/512 hamming distance against every stored hash and
        // silently broke all semantic queries (found 2026-09-09).
        if acc >= 0.0 {
            hash_out[bit / 8] |= 1 << (7 - (bit % 8));
        }
    }

    0
}

/// Free the loaded ITQ model. Currently a no-op because the model is held in a
/// global OnceLock for the process lifetime. Provided for API symmetry.
#[no_mangle]
pub extern "C" fn yp_itq_free(_handle: *mut c_void) {}

/// Benchmark/debug helper: encode the all-zeros vector and return the first
/// 8 hash bytes as a little-endian u64. Lets harnesses verify WHICH model is
/// resident (and that the bit-packing convention is intact) before trusting
/// any downstream numbers — independent of error paths. Returns the
/// yp_itq_encode rc; -5 if no model is loaded.
#[no_mangle]
pub extern "C" fn yp_itq_zero_fingerprint(out: *mut u64) -> c_int {
    if out.is_null() {
        return -1;
    }
    let n_dims = match ITQ_MODEL.get() {
        Some(m) => m.n_dims,
        None => return -5,
    };
    let zeros = vec![0f32; n_dims];
    let mut hash = [0u8; 64];
    let rc = yp_itq_encode(zeros.as_ptr(), n_dims, hash.as_mut_ptr());
    if rc != 0 {
        return rc;
    }
    unsafe { *out = u64::from_le_bytes(hash[0..8].try_into().unwrap()) };
    0
}
