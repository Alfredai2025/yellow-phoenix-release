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

// ---------------------------------------------------------------------------
// ANN-benchmark two-stage kernels (added 2026-10-04, crown campaign Step 1).
// AVX2+FMA via runtime detection; scalar fallback elsewhere. Standard
// vector-instruction techniques; no third-party code.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn have_avx2_fma() -> bool {
    use std::sync::OnceLock;
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        std::arch::is_x86_64_feature_detected!("avx2")
            && std::arch::is_x86_64_feature_detected!("fma")
    })
}

#[inline]
pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2_fma() {
            // SAFETY: features detected above.
            return unsafe { l2_sq_avx2(a, b) };
        }
    }
    l2_sq_scalar(a, b)
}

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if have_avx2_fma() {
            // SAFETY: features detected above.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        let d0 = _mm256_sub_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
        );
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        let d1 = _mm256_sub_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
        );
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        i += 16;
    }
    let mut s = ann_hsum256(acc0) + ann_hsum256(acc1);
    while i < n {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        s += d * d;
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 16 <= n {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
        ));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(
            _mm256_loadu_ps(a.as_ptr().add(i + 8)),
            _mm256_loadu_ps(b.as_ptr().add(i + 8)),
        ));
        i += 16;
    }
    let mut s = ann_hsum256(acc0) + ann_hsum256(acc1);
    while i < n {
        s += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ann_hsum256(v: core::arch::x86_64::__m256) -> f32 {
    use core::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s4 = _mm_add_ps(lo, hi);
    let sh = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, sh);
    let s1 = _mm_add_ss(s2, _mm_movehdup_ps(s2));
    _mm_cvtss_f32(s1)
}

/// Encode 512 bits: bit b = sign( dot(qm, wt_row_b) ); wt is 512x128 row-major
/// (transposed at server load for cache/SIMD friendliness).
pub fn encode512(qm: &[f32; 128], wt: &[[f32; 128]; 512]) -> [u8; 64] {
    let mut code = [0u8; 64];
    let mut b = 0;
    while b < 512 {
        if dot_f32(qm, &wt[b]) >= 0.0 {
            code[b / 8] |= 1 << (7 - (b % 8));
        }
        b += 1;
    }
    code
}

// ---------------------------------------------------------------------------
// Int8 dot-product kernels for the Int8Hnsw engine (T-int8 SIMD pass).
// Computes sum(a[i] * b[i]) over 128 i8 lanes, exact in i32.
// Codes are clipped to [-127,127] so _mm256_maddubs_epi16 pair-sums never
// saturate (max pair sum 2*127*127 = 32258 < 32767). x86_64 + NEON + scalar.
// ---------------------------------------------------------------------------

/// Runtime-dispatched int8 dot product over 128 lanes.
pub fn i8_dot_128(a: &[i8; 128], b: &[i8; 128]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { avx2_i8_dot_128(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_i8_dot_128(a, b) };
    }
    i8_dot_128_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx2_i8_dot_128(a: &[i8; 128], b: &[i8; 128]) -> i32 {
    use std::arch::x86_64::*;
    let ones = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_si256();
    // 4 chunks of 32 bytes; maddubs -> i16 pair sums (no saturation at [-127,127]),
    // then madd_epi16 with ones widens pairs to i32 and accumulates.
    for chunk in 0..4 {
        let off = chunk * 32;
        let va = _mm256_loadu_si256(a[off..].as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b[off..].as_ptr() as *const __m256i);
        let prod = _mm256_maddubs_epi16(va, vb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_i8_dot_128(a: &[i8; 128], b: &[i8; 128]) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    // 8 chunks of 16 bytes: widen to i16, multiply, pairwise-add to i32.
    for chunk in 0..8 {
        let off = chunk * 16;
        let va = vld1q_s8(a[off..].as_ptr());
        let vb = vld1q_s8(b[off..].as_ptr());
        let wa = vmovl_s8(vget_low_s8(va));
        let wb = vmovl_s8(vget_low_s8(vb));
        let hi_a = vmovl_s8(vget_high_s8(va));
        let hi_b = vmovl_s8(vget_high_s8(vb));
        let lo_p = vmulq_s16(wa, wb);
        let hi_p = vmulq_s16(hi_a, hi_b);
        acc = vaddq_s32(acc, vpaddlq_s16(lo_p));
        acc = vaddq_s32(acc, vpaddlq_s16(hi_p));
    }
    vaddvq_s32(acc)
}

fn i8_dot_128_scalar(a: &[i8; 128], b: &[i8; 128]) -> i32 {
    let mut s = 0i32;
    for i in 0..128 {
        s += a[i] as i32 * b[i] as i32;
    }
    s
}

/// Exact int8 L2 via the identity sum((a-b)^2) = sum(a^2) + sum(b^2) - 2*dot(a,b).
/// All integer, order-independent (associative i32 adds, no overflow: max ~8.2M).
pub fn i8_l2_128(a: &[i8; 128], b: &[i8; 128], a_sq: i32, b_sq: i32) -> i32 {
    a_sq + b_sq - 2 * i8_dot_128(a, b)
}

/// Squared norm of a 128-lane int8 vector.
pub fn i8_sq_128(a: &[i8; 128]) -> i32 {
    i8_dot_128(a, a)
}

// ---------------------------------------------------------------------------
// Mixed f32 x i8 dot product for reconstruction-L2 distance (Int8Hnsw v2).
// sum(v[i] * c[i]) with v f32, c i8 -> f32. Used with the norm identity:
//   ||v - x_hat||^2 = ||v||^2 + ||x_hat||^2 - 2*v.x_hat,  x_hat = C[cell]+code
// so per candidate only this dot + precomputed norms are needed.
// ---------------------------------------------------------------------------

pub fn f32_i8_dot_128(v: &[f32; 128], c: &[i8; 128]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            return unsafe { avx2_f32_i8_dot_128(v, c) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon_f32_i8_dot_128(v, c) };
    }
    f32_i8_dot_128_scalar(v, c)
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx2_f32_i8_dot_128(v: &[f32; 128], c: &[i8; 128]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    for chunk in 0..4 {
        let off = chunk * 32;
        let vv = _mm256_loadu_ps(v[off..].as_ptr());
        let cc8 = _mm_loadu_si128(c[off..].as_ptr() as *const __m128i);
        let cc32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(cc8));
        acc = _mm256_fmadd_ps(vv, cc32, acc);
    }
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr() as *mut __m256i, acc);
    lanes.iter().sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn neon_f32_i8_dot_128(v: &[f32; 128], c: &[i8; 128]) -> f32 {
    use std::arch::aarch64::*;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    for i in 0..16 {
        let off = i * 8;
        let v0 = vld1q_f32(v[off..].as_ptr());
        let v1 = vld1q_f32(v[off + 4..].as_ptr());
        let c8 = vld1_s8(c[off..].as_ptr());
        let c16 = vmovl_s8(c8);
        let c32_lo = vmovl_s16(vget_low_s16(c16));
        let c32_hi = vmovl_s16(vget_high_s16(c16));
        acc0 = vmlaq_f32(acc0, v0, vcvtq_f32_s32(c32_lo));
        acc1 = vmlaq_f32(acc1, v1, vcvtq_f32_s32(c32_hi));
    }
    vaddvq_f32(vaddq_f32(acc0, acc1))
}

#[cfg(target_arch = "x86_64")]
unsafe fn avx2_f32_i8_dot_128(v: &[f32; 128], c: &[i8; 128]) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    for i in 0..16 {
        let off = i * 8;
        let vv = _mm256_loadu_ps(v[off..].as_ptr());
        let c_raw = _mm_loadu_si128(c[off..].as_ptr() as *const __m128i);
        let c_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c_raw));
        acc = _mm256_fmadd_ps(vv, c_f32, acc);
    }
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi, lo);
    let tmp1 = _mm_hadd_ps(sum128, sum128);
    let tmp2 = _mm_hadd_ps(tmp1, tmp1);
    _mm_cvtss_f32(tmp2)
}

fn f32_i8_dot_128_scalar(v: &[f32; 128], c: &[i8; 128]) -> f32 {
    let mut s = 0f32;
    for i in 0..128 {
        s += v[i] * c[i] as f32;
    }
    s
}
