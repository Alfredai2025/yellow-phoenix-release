// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! SIMD Hamming distance for 512-bit (64-byte) hashes.
//!
//! Architecture support:
//! - AArch64 (Apple Silicon): NEON vector popcount via vcntq_u8
//! - x86_64: AVX2 (256-bit) with 4× u64 popcount fallback
//! - Generic: byte LUT popcount (still ~2× faster than bit-by-bit)

// =============================================================================
// PUBLIC API
// =============================================================================

/// Compute Hamming distance between two 64-byte hashes.
/// Dispatches to the best available implementation at runtime.
#[inline]
pub fn hamming_distance_512(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    {
        // Apple Silicon always has NEON
        unsafe { neon::hamming_distance_512(a, b) }
    }

    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    {
        if avx2::is_available() {
            unsafe { avx2::hamming_distance_512(a, b) }
        } else {
            lut::hamming_distance_512(a, b)
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        lut::hamming_distance_512(a, b)
    }
}

/// Verify all implementations return identical results.
/// Call once at startup (e.g., from yp_proof_of_life or yp_binary_hnsw_new).
pub fn verify_all_implementations() -> bool {
    let mut rng_a = [0u8; 64];
    let mut rng_b = [0u8; 64];
    for i in 0..64 {
        rng_a[i] = (i * 7 + 3) as u8;
        rng_b[i] = (i * 11 + 5) as u8;
    }
    let expected_random: u32 = rng_a
        .iter()
        .zip(rng_b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();

    let test_cases: [([u8; 64], [u8; 64], u32); 5] = [
        ([0x00; 64], [0x00; 64], 0),   // identical
        ([0x00; 64], [0xFF; 64], 512), // opposite
        ([0x55; 64], [0xAA; 64], 512), // checkerboard
        ([0x55; 64], [0x55; 64], 0),   // identical checkerboard
        (rng_a, rng_b, expected_random),
    ];

    for (i, (a, b, expected)) in test_cases.iter().enumerate() {
        let scalar = scalar::hamming_distance_512(a, b);
        let lut = lut::hamming_distance_512(a, b);

        if scalar != *expected {
            eprintln!(
                "SIMD verify: scalar case {} returned {}, expected {}",
                i, scalar, expected
            );
            return false;
        }
        if lut != scalar {
            eprintln!(
                "SIMD verify: LUT mismatch case {}: scalar={}, lut={}",
                i, scalar, lut
            );
            return false;
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon = unsafe { neon::hamming_distance_512(a, b) };
            if neon != scalar {
                eprintln!(
                    "SIMD verify: NEON mismatch case {}: scalar={}, neon={}",
                    i, scalar, neon
                );
                return false;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            if avx2::is_available() {
                let avx = unsafe { avx2::hamming_distance_512(a, b) };
                if avx != scalar {
                    eprintln!(
                        "SIMD verify: AVX2 mismatch case {}: scalar={}, avx={}",
                        i, scalar, avx
                    );
                    return false;
                }
            }
        }
    }

    true
}

// =============================================================================
// AArch64 NEON (Apple Silicon M1/M2/M3)
// =============================================================================

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// 512-bit Hamming distance using NEON.
    /// Processes 16 bytes per iteration (128-bit vector).
    /// Uses vcntq_u8 (vector popcount per byte) + vaddvq_u8 (horizontal sum).
    #[target_feature(enable = "neon")]
    pub unsafe fn hamming_distance_512(a: &[u8; 64], b: &[u8; 64]) -> u32 {
        let a0 = vld1q_u8(a.as_ptr().add(0));
        let a1 = vld1q_u8(a.as_ptr().add(16));
        let a2 = vld1q_u8(a.as_ptr().add(32));
        let a3 = vld1q_u8(a.as_ptr().add(48));

        let b0 = vld1q_u8(b.as_ptr().add(0));
        let b1 = vld1q_u8(b.as_ptr().add(16));
        let b2 = vld1q_u8(b.as_ptr().add(32));
        let b3 = vld1q_u8(b.as_ptr().add(48));

        let x0 = veorq_u8(a0, b0);
        let x1 = veorq_u8(a1, b1);
        let x2 = veorq_u8(a2, b2);
        let x3 = veorq_u8(a3, b3);

        let c0 = vcntq_u8(x0);
        let c1 = vcntq_u8(x1);
        let c2 = vcntq_u8(x2);
        let c3 = vcntq_u8(x3);

        vaddvq_u8(c0) as u32
            + vaddvq_u8(c1) as u32
            + vaddvq_u8(c2) as u32
            + vaddvq_u8(c3) as u32
    }
}

// =============================================================================
// x86_64 AVX2
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Check AVX2 availability at runtime.
    pub fn is_available() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// 512-bit Hamming distance using AVX2.
    /// AVX2 lacks vector popcount, so we process 32 bytes per iteration
    /// and use 4× u64::count_ones() within each 32-byte chunk.
    #[target_feature(enable = "avx2")]
    pub unsafe fn hamming_distance_512(a: &[u8; 64], b: &[u8; 64]) -> u32 {
        let mut dist: u32 = 0;

        for chunk in 0..2 {
            let offset = chunk * 32;

            let av = _mm256_loadu_si256(a.as_ptr().add(offset) as *const __m256i);
            let bv = _mm256_loadu_si256(b.as_ptr().add(offset) as *const __m256i);
            let xv = _mm256_xor_si256(av, bv);

            let lanes = [
                _mm256_extract_epi64(xv, 0) as u64,
                _mm256_extract_epi64(xv, 1) as u64,
                _mm256_extract_epi64(xv, 2) as u64,
                _mm256_extract_epi64(xv, 3) as u64,
            ];

            for lane in lanes {
                dist += lane.count_ones();
            }
        }

        dist
    }
}

// =============================================================================
// LUT-based fast path (no SIMD intrinsics, still ~2× faster than bit-by-bit)
// Works on all architectures.
// =============================================================================

mod lut {
    /// 256-byte lookup table: POPCOUNT_LUT[x] = number of set bits in x.
    static POPCOUNT_LUT: [u8; 256] = {
        let mut table = [0u8; 256];
        let mut i: usize = 0;
        while i < 256 {
            table[i] = i.count_ones() as u8;
            i += 1;
        }
        table
    };

    #[inline]
    pub fn hamming_distance_512(a: &[u8; 64], b: &[u8; 64]) -> u32 {
        let mut dist: u32 = 0;
        for i in 0..64 {
            dist += POPCOUNT_LUT[(a[i] ^ b[i]) as usize] as u32;
        }
        dist
    }
}

// =============================================================================
// Scalar reference (byte-by-byte, guaranteed correct)
// =============================================================================

mod scalar {
    /// The simplest correct implementation. Used for verification only.
    pub fn hamming_distance_512(a: &[u8; 64], b: &[u8; 64]) -> u32 {
        let mut dist: u32 = 0;
        for i in 0..64 {
            dist += (a[i] ^ b[i]).count_ones();
        }
        dist
    }
}
