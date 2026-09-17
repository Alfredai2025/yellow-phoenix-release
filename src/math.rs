// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Multi-base residue math primitives.

pub mod functor_bounds;
pub mod tensor_spectral_512;

use xxhash_rust::xxh3::xxh3_128;

/// Extract a single residue for base `b` and dimension `k` from a 64-byte PAP.
///
/// Implements the exact formula:
/// ```text
/// acc = b
/// for i in 0..64:
///     coeff = (i * (k + 1) + b) % 256
///     acc += pap[i] * coeff
/// return acc % b
/// ```
pub fn extract_residue(pap: &[u8; 64], base: u8, k: usize) -> u8 {
    let mut acc: u64 = base as u64;
    let base_usize = base as usize;
    for (i, &byte) in pap.iter().enumerate() {
        let coeff = (i.wrapping_mul(k + 1) + base_usize) % 256;
        acc = acc.wrapping_add((byte as u64).wrapping_mul(coeff as u64));
    }
    (acc % base as u64) as u8
}

/// Full 4-base composite signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompositeSignature {
    pub triangular: [u8; 22],   // base 3
    pub tetrahedral: [u8; 16],  // base 4
    pub cubic: [u8; 11],        // base 6
    pub octahedral: [u8; 8],    // base 8
}

impl CompositeSignature {
    /// Compute the full signature from a 64-byte PAP.
    pub fn from_pap(pap: &[u8; 64]) -> Self {
        let mut triangular = [0u8; 22];
        let mut tetrahedral = [0u8; 16];
        let mut cubic = [0u8; 11];
        let mut octahedral = [0u8; 8];

        for k in 0..22 {
            triangular[k] = extract_residue(pap, 3, k);
        }
        for k in 0..16 {
            tetrahedral[k] = extract_residue(pap, 4, k);
        }
        for k in 0..11 {
            cubic[k] = extract_residue(pap, 6, k);
        }
        for k in 0..8 {
            octahedral[k] = extract_residue(pap, 8, k);
        }

        Self {
            triangular,
            tetrahedral,
            cubic,
            octahedral,
        }
    }

    /// Primary CRT key combining base-3 and base-8 first residues.
    pub fn primary_key(&self) -> u8 {
        crt(self.triangular[0] as u64, 3, self.octahedral[0] as u64, 8) as u8
    }

    /// 128-bit composite hash of all 57 residues.
    pub fn composite_hash(&self) -> u128 {
        let mut buf = [0u8; 57];
        buf[0..22].copy_from_slice(&self.triangular);
        buf[22..38].copy_from_slice(&self.tetrahedral);
        buf[38..49].copy_from_slice(&self.cubic);
        buf[49..57].copy_from_slice(&self.octahedral);
        xxh3_128(&buf)
    }

    /// Cross-base consistency check.
    pub fn is_consistent(&self) -> bool {
        self.cubic[0] % 3 == self.triangular[0]
            && self.tetrahedral[0] % 4 == self.octahedral[0] % 4
    }

    /// Corrected total entropy of the residue signature in bits.
    pub fn total_entropy_bits() -> f32 {
        119.0f32
    }
}

/// Chinese Remainder Theorem for two coprime moduli.
/// Returns the unique x in [0, m*n) such that:
///   x ≡ a (mod m)
///   x ≡ b (mod n)
pub fn crt(a: u64, m: u64, b: u64, n: u64) -> u64 {
    // moduli must be coprime
    assert_eq!(gcd(m, n), 1, "crt requires coprime moduli");
    let inv_m = mod_inverse(m, n);
    let diff = (b + n - (a % n)) % n;
    let t = (diff * inv_m) % n;
    (a + t * m) % (m * n)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let tmp = b;
        b = a % b;
        a = tmp;
    }
    a
}

fn mod_inverse(a: u64, m: u64) -> u64 {
    let (mut t, mut new_t) = (0i64, 1i64);
    let (mut r, mut new_r) = (m as i64, a as i64);
    while new_r != 0 {
        let quotient = r / new_r;
        let tmp_t = t - quotient * new_t;
        t = new_t;
        new_t = tmp_t;
        let tmp_r = r - quotient * new_r;
        r = new_r;
        new_r = tmp_r;
    }
    assert_eq!(r, 1, "mod_inverse: a and m are not coprime");
    ((t % m as i64 + m as i64) % m as i64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pap_from_seed(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        let bytes = seed.to_le_bytes();
        for chunk in pap.chunks_exact_mut(8) {
            chunk.copy_from_slice(&bytes);
        }
        pap
    }

    #[test]
    fn test_extract_deterministic() {
        let pap = pap_from_seed(123);
        let r1 = extract_residue(&pap, 3, 0);
        let r2 = extract_residue(&pap, 3, 0);
        assert_eq!(r1, r2);
        assert!(r1 < 3);
    }

    #[test]
    fn test_signature_deterministic() {
        let pap = pap_from_seed(456);
        let s1 = CompositeSignature::from_pap(&pap);
        let s2 = CompositeSignature::from_pap(&pap);
        assert_eq!(s1, s2);
        assert_eq!(s1.composite_hash(), s2.composite_hash());
    }

    #[test]
    fn test_consistency_valid() {
        let pap = pap_from_seed(789);
        let sig = CompositeSignature::from_pap(&pap);
        assert!(sig.is_consistent());
    }

    #[test]
    fn test_primary_key_range() {
        for seed in 0..1000u64 {
            let pap = pap_from_seed(seed);
            let sig = CompositeSignature::from_pap(&pap);
            let key = sig.primary_key();
            assert!(key < 24, "primary key {} out of range", key);
        }
    }

    #[test]
    fn test_different_paps() {
        let pap_a = pap_from_seed(100);
        let pap_b = pap_from_seed(200);
        let sig_a = CompositeSignature::from_pap(&pap_a);
        let sig_b = CompositeSignature::from_pap(&pap_b);
        assert_ne!(sig_a, sig_b);
        assert_ne!(sig_a.composite_hash(), sig_b.composite_hash());
    }
}
pub mod tensor_spectral_float;
pub mod functor_bounds_float;
