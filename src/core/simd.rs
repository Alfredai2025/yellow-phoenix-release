// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

extern crate alloc;

use crate::types::multivector::BinaryMultivector;

/// Trait for computing SIMD resonance scores between binary multivectors.
pub trait ResonanceKernel {
    fn resonance_score(a: &BinaryMultivector, b: &BinaryMultivector) -> u32;
    fn batch_resonance(
        query: &BinaryMultivector,
        targets: &[BinaryMultivector],
    ) -> alloc::vec::Vec<u32>;
}

/// Fallback kernel using software XOR + popcount.
pub struct FallbackKernel;

impl ResonanceKernel for FallbackKernel {
    #[inline]
    fn resonance_score(a: &BinaryMultivector, b: &BinaryMultivector) -> u32 {
        let xor0 = a.0[0] ^ b.0[0];
        let xor1 = a.0[1] ^ b.0[1];
        xor0.count_ones() + xor1.count_ones()
    }

    #[inline]
    fn batch_resonance(
        query: &BinaryMultivector,
        targets: &[BinaryMultivector],
    ) -> alloc::vec::Vec<u32> {
        targets
            .iter()
            .map(|t| Self::resonance_score(query, t))
            .collect()
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_impl {
    use ::core::arch::aarch64::*;
    use super::{BinaryMultivector, ResonanceKernel};

    pub struct NeonKernel;

    impl ResonanceKernel for NeonKernel {
        #[inline]
        fn resonance_score(a: &BinaryMultivector, b: &BinaryMultivector) -> u32 {
            unsafe {
                let va = vld1q_u8(a.to_bytes().as_ptr());
                let vb = vld1q_u8(b.to_bytes().as_ptr());
                let vxor = veorq_u8(va, vb);
                let vcnt = vcntq_u8(vxor);
                vaddvq_u8(vcnt) as u32
            }
        }

        #[inline]
        fn batch_resonance(
            query: &BinaryMultivector,
            targets: &[BinaryMultivector],
        ) -> alloc::vec::Vec<u32> {
            targets
                .iter()
                .map(|t| Self::resonance_score(query, t))
                .collect()
        }
    }
}

/// Compile-time selected kernel.
pub struct Kernel;

#[cfg(target_arch = "aarch64")]
impl ResonanceKernel for Kernel {
    #[inline]
    fn resonance_score(a: &BinaryMultivector, b: &BinaryMultivector) -> u32 {
        neon_impl::NeonKernel::resonance_score(a, b)
    }

    #[inline]
    fn batch_resonance(
        query: &BinaryMultivector,
        targets: &[BinaryMultivector],
    ) -> alloc::vec::Vec<u32> {
        neon_impl::NeonKernel::batch_resonance(query, targets)
    }
}

#[cfg(not(target_arch = "aarch64"))]
impl ResonanceKernel for Kernel {
    #[inline]
    fn resonance_score(a: &BinaryMultivector, b: &BinaryMultivector) -> u32 {
        FallbackKernel::resonance_score(a, b)
    }

    #[inline]
    fn batch_resonance(
        query: &BinaryMultivector,
        targets: &[BinaryMultivector],
    ) -> alloc::vec::Vec<u32> {
        FallbackKernel::batch_resonance(query, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn compare_kernels() {
        let a = BinaryMultivector::from_seed_text("phase two kernel test");
        let b = BinaryMultivector::from_seed_text("different seed text");
        let c = BinaryMultivector::from_seed_text("phase two kernel test");

        let neon_score = neon_impl::NeonKernel::resonance_score(&a, &b);
        let fallback_score = FallbackKernel::resonance_score(&a, &b);
        assert_eq!(neon_score, fallback_score);

        assert_eq!(
            neon_impl::NeonKernel::resonance_score(&a, &c),
            FallbackKernel::resonance_score(&a, &c)
        );

        let targets = &[a, b, c];
        let neon_batch = neon_impl::NeonKernel::batch_resonance(&a, targets);
        let fallback_batch = FallbackKernel::batch_resonance(&a, targets);
        assert_eq!(neon_batch, fallback_batch);
    }
}
