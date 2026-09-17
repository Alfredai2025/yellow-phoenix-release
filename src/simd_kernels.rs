// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! simd_kernels.rs — M4.2: Runtime-dispatched SIMD kernels.
//!
//! Gated behind the `simd` Cargo feature.  When `simd` is enabled the library
//! detects the best available vector path at runtime:
//!
//!   * AVX-512 + VPOPCNTDQ on x86_64
//!   * AVX2 on x86_64 (byte-extraction popcount)
//!   * NEON on aarch64
//!   * Pure Rust fallback everywhere else
//!
//! The default build (without `simd`) keeps the original scalar implementations.

use std::sync::OnceLock;

/// Selected SIMD execution path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdPath {
    Avx512,
    Avx2,
    Neon,
    Fallback,
}

/// Detect the best SIMD path once and cache the result.
pub fn detect_simd_path() -> SimdPath {
    static PATH: OnceLock<SimdPath> = OnceLock::new();
    *PATH.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
                return SimdPath::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdPath::Avx2;
            }
            SimdPath::Fallback
        }
        #[cfg(target_arch = "aarch64")]
        {
            SimdPath::Neon
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdPath::Fallback
        }
    })
}

/// 128-bit PAP distance: returns `(popcount(a & b), popcount(a), popcount(b))`.
pub fn simd_pap_distance_128(a: &[u8; 16], b: &[u8; 16]) -> (u32, u32, u32) {
    if cfg!(feature = "simd") {
        match detect_simd_path() {
            #[cfg(target_arch = "x86_64")]
            SimdPath::Avx512 => unsafe { avx512_popcount_128(a, b) },
            #[cfg(target_arch = "x86_64")]
            SimdPath::Avx2 => unsafe { avx2_popcount_128(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdPath::Neon => unsafe { neon_popcount_128(a, b) },
            _ => scalar_popcount_128(a, b),
        }
    } else {
        scalar_popcount_128(a, b)
    }
}

/// 512-bit PAP distance: returns `(popcount(a & b), popcount(a), popcount(b))`.
pub fn simd_pap_distance_512(a: &[u8; 64], b: &[u8; 64]) -> (u32, u32, u32) {
    if cfg!(feature = "simd") {
        match detect_simd_path() {
            #[cfg(target_arch = "x86_64")]
            SimdPath::Avx512 => unsafe { avx512_popcount_512(a, b) },
            #[cfg(target_arch = "x86_64")]
            SimdPath::Avx2 => unsafe { avx2_popcount_512(a, b) },
            #[cfg(target_arch = "aarch64")]
            SimdPath::Neon => unsafe { neon_popcount_512(a, b) },
            _ => scalar_popcount_512(a, b),
        }
    } else {
        scalar_popcount_512(a, b)
    }
}

/// Project a 64-byte hash into three spectral coordinates.
pub fn simd_spectral_project(hash: &[u8; 64]) -> [f32; 3] {
    if cfg!(feature = "simd") {
        match detect_simd_path() {
            #[cfg(target_arch = "aarch64")]
            SimdPath::Neon => unsafe { neon_spectral_project(hash) },
            _ => scalar_spectral_project(hash),
        }
    } else {
        scalar_spectral_project(hash)
    }
}

// =============================================================================
// Scalar fallbacks
// =============================================================================

fn scalar_popcount_128(a: &[u8; 16], b: &[u8; 16]) -> (u32, u32, u32) {
    let mut x = 0u32;
    let mut ac = 0u32;
    let mut bc = 0u32;
    for i in 0..16 {
        x += (a[i] & b[i]).count_ones();
        ac += a[i].count_ones();
        bc += b[i].count_ones();
    }
    (x, ac, bc)
}

fn scalar_popcount_512(a: &[u8; 64], b: &[u8; 64]) -> (u32, u32, u32) {
    let mut x = 0u32;
    let mut ac = 0u32;
    let mut bc = 0u32;
    for i in 0..64 {
        x += (a[i] & b[i]).count_ones();
        ac += a[i].count_ones();
        bc += b[i].count_ones();
    }
    (x, ac, bc)
}

fn scalar_spectral_project(hash: &[u8; 64]) -> [f32; 3] {
    let w0 = weights_c0();
    let w1 = weights_c1();
    let w2 = weights_c2();
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    for i in 0..64 {
        let x = hash[i] as f32 / 255.0;
        c0 += x * w0[i];
        c1 += x * w1[i];
        c2 += x * w2[i];
    }
    [c0, c1, c2]
}

// =============================================================================
// aarch64 NEON paths
// =============================================================================

#[cfg(target_arch = "aarch64")]
unsafe fn neon_popcount_128(a: &[u8; 16], b: &[u8; 16]) -> (u32, u32, u32) {
    use std::arch::aarch64::*;
    let table = vld1q_u8([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4].as_ptr());
    let va = vld1q_u8(a.as_ptr());
    let vb = vld1q_u8(b.as_ptr());
    let vand = vandq_u8(va, vb);

    fn popcnt_tbl(v: uint8x16_t, table: uint8x16_t) -> u32 {
        unsafe {
            let lo = vandq_u8(v, vdupq_n_u8(0x0F));
            let hi = vshrq_n_u8(v, 4);
            let pc_lo = vqtbl1q_u8(table, lo);
            let pc_hi = vqtbl1q_u8(table, hi);
            let sum8 = vaddq_u8(pc_lo, pc_hi);
            vaddvq_u8(sum8) as u32
        }
    }

    (
        popcnt_tbl(vand, table),
        popcnt_tbl(va, table),
        popcnt_tbl(vb, table),
    )
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_popcount_512(a: &[u8; 64], b: &[u8; 64]) -> (u32, u32, u32) {
    let mut x = 0u32;
    let mut ac = 0u32;
    let mut bc = 0u32;
    for chunk in 0..4 {
        let off = chunk * 16;
        let a16: [u8; 16] = a[off..off + 16].try_into().unwrap();
        let b16: [u8; 16] = b[off..off + 16].try_into().unwrap();
        let (cx, cac, cbc) = neon_popcount_128(&a16, &b16);
        x += cx;
        ac += cac;
        bc += cbc;
    }
    (x, ac, bc)
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_spectral_project(hash: &[u8; 64]) -> [f32; 3] {
    use std::arch::aarch64::*;

    let w0 = weights_c0();
    let w1 = weights_c1();
    let w2 = weights_c2();

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);

    for i in (0..64).step_by(4) {
        let vals = [hash[i] as f32 / 255.0, hash[i + 1] as f32 / 255.0, hash[i + 2] as f32 / 255.0, hash[i + 3] as f32 / 255.0];
        let f = vld1q_f32(vals.as_ptr());
        acc0 = vfmaq_f32(acc0, f, vld1q_f32(w0.as_ptr().add(i)));
        acc1 = vfmaq_f32(acc1, f, vld1q_f32(w1.as_ptr().add(i)));
        acc2 = vfmaq_f32(acc2, f, vld1q_f32(w2.as_ptr().add(i)));
    }

    [vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2)]
}

// =============================================================================
// x86_64 AVX-512 paths
// =============================================================================

#[cfg(target_arch = "x86_64")]
unsafe fn avx512_popcount_128(a: &[u8; 16], b: &[u8; 16]) -> (u32, u32, u32) {
    use std::arch::x86_64::*;
    let va = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let vb = _mm_loadu_si128(b.as_ptr() as *const __m128i);
    let vand = _mm_and_si128(va, vb);

    let widen = |v: __m128i| _mm512_cvtepu8_epi64(v);
    let pop = |v: __m128i| _mm512_reduce_add_epi64(_mm512_popcnt_epi64(widen(v))) as u32;

    (pop(vand), pop(va), pop(vb))
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx512_popcount_512(a: &[u8; 64], b: &[u8; 64]) -> (u32, u32, u32) {
    use std::arch::x86_64::*;
    let va = _mm512_loadu_si512(a.as_ptr() as *const __m512i);
    let vb = _mm512_loadu_si512(b.as_ptr() as *const __m512i);
    let vand = _mm512_and_si512(va, vb);
    let pop = |v: __m512i| _mm512_reduce_add_epi64(_mm512_popcnt_epi64(v)) as u32;
    (pop(vand), pop(va), pop(vb))
}

// =============================================================================
// x86_64 AVX2 paths
// =============================================================================

#[cfg(target_arch = "x86_64")]
unsafe fn avx2_popcount_128(a: &[u8; 16], b: &[u8; 16]) -> (u32, u32, u32) {
    use std::arch::x86_64::*;
    let va = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let vb = _mm_loadu_si128(b.as_ptr() as *const __m128i);
    let vand = _mm_and_si128(va, vb);

    let mut bytes = [0u8; 16];
    let mut pop = |v: __m128i| -> u32 {
        _mm_storeu_si128(bytes.as_mut_ptr() as *mut __m128i, v);
        bytes.iter().map(|x| x.count_ones()).sum()
    };

    (pop(vand), pop(va), pop(vb))
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx2_popcount_512(a: &[u8; 64], b: &[u8; 64]) -> (u32, u32, u32) {
    use std::arch::x86_64::*;
    let mut x = 0u32;
    let mut ac = 0u32;
    let mut bc = 0u32;
    let mut bytes = [0u8; 32];

    for chunk in 0..2 {
        let off = chunk * 32;
        let va = _mm256_loadu_si256(a[off..].as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b[off..].as_ptr() as *const __m256i);
        let vand = _mm256_and_si256(va, vb);

        let mut pop = |v: __m256i| -> u32 {
            _mm256_storeu_si256(bytes.as_mut_ptr() as *mut __m256i, v);
            bytes.iter().map(|byte| byte.count_ones()).sum()
        };

        x += pop(vand);
        ac += pop(va);
        bc += pop(vb);
    }

    (x, ac, bc)
}

// =============================================================================
// Shared projection weights
// =============================================================================

fn weights_c0() -> &'static [f32; 64] {
    static W: OnceLock<[f32; 64]> = OnceLock::new();
    W.get_or_init(|| {
        let mut arr = [0.0f32; 64];
        for i in 0..64 {
            arr[i] = (((i * 7 + 3) % 64) as f32 / 32.0) - 1.0;
        }
        arr
    })
}

fn weights_c1() -> &'static [f32; 64] {
    static W: OnceLock<[f32; 64]> = OnceLock::new();
    W.get_or_init(|| {
        let mut arr = [0.0f32; 64];
        for i in 0..64 {
            arr[i] = (((i * 13 + 5) % 64) as f32 / 32.0) - 1.0;
        }
        arr
    })
}

fn weights_c2() -> &'static [f32; 64] {
    static W: OnceLock<[f32; 64]> = OnceLock::new();
    W.get_or_init(|| {
        let mut arr = [0.0f32; 64];
        for i in 0..64 {
            arr[i] = (((i * 31 + 11) % 64) as f32 / 32.0) - 1.0;
        }
        arr
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_does_not_panic() {
        let path = detect_simd_path();
        println!("detected SIMD path: {:?}", path);
    }

    #[test]
    fn simd_128_matches_scalar() {
        let a = [0xABu8; 16];
        let b = [0xCDu8; 16];
        assert_eq!(simd_pap_distance_128(&a, &b), scalar_popcount_128(&a, &b));
    }

    #[test]
    fn simd_512_matches_scalar() {
        let a = [0xABu8; 64];
        let b = [0xCDu8; 64];
        assert_eq!(simd_pap_distance_512(&a, &b), scalar_popcount_512(&a, &b));
    }

    #[test]
    fn simd_spectral_matches_scalar() {
        let h = [0x42u8; 64];
        let simd = simd_spectral_project(&h);
        let scalar = scalar_spectral_project(&h);
        assert!((simd[0] - scalar[0]).abs() < 0.001);
        assert!((simd[1] - scalar[1]).abs() < 0.001);
        assert!((simd[2] - scalar[2]).abs() < 0.001);
    }
}
