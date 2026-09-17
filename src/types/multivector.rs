// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Pure geometric binary multivector (128‑bit) with XOR + popcount operations.

/// 128‑bit binary multivector stored in two 64‑bit limbs.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BinaryMultivector(pub [u64; 2]);

#[inline]
#[cfg(target_arch = "aarch64")]
fn popcount_u64(x: u64) -> u32 {
    unsafe {
        use core::arch::aarch64::{vld1_u8, vcnt_u8, vaddlv_u8};
        let bytes = x.to_ne_bytes();
        let v = vld1_u8(bytes.as_ptr());
        let cnt = vcnt_u8(v);
        let sum = vaddlv_u8(cnt);
        sum as u32
    }
}

#[inline]
#[cfg(not(target_arch = "aarch64"))]
fn popcount_u64(x: u64) -> u32 {
    x.count_ones() as u32
}

impl BinaryMultivector {
    /// All‑zero multivector.
    #[inline]
    pub fn new() -> Self {
        Self([0u64; 2])
    }

    /// Read bit `index` (0‑indexed, `index < 128`).
    #[inline]
    pub fn bit(&self, index: usize) -> bool {
        assert!(index < 128, "bit index {} out of range", index);
        if index < 64 {
            (self.0[0] >> index) & 1 != 0
        } else {
            (self.0[1] >> (index - 64)) & 1 != 0
        }
    }

    /// Set bit `index` to `value` (`index < 128`).
    #[inline]
    pub fn set_bit(&mut self, index: usize, value: bool) {
        assert!(index < 128, "bit index {} out of range", index);
        if index < 64 {
            let mask = 1u64 << index;
            if value {
                self.0[0] |= mask;
            } else {
                self.0[0] &= !mask;
            }
        } else {
            let j = index - 64;
            let mask = 1u64 << j;
            if value {
                self.0[1] |= mask;
            } else {
                self.0[1] &= !mask;
            }
        }
    }

    /// Compute the geometric product (XOR + popcount) between two multivectors.
    #[inline]
    pub fn geometric_product(&self, other: &Self) -> u32 {
        let a = self.0[0] ^ other.0[0];
        let b = self.0[1] ^ other.0[1];
        popcount_u64(a) + popcount_u64(b)
    }

    /// Hamming distance (popcount of XOR) between two multivectors.
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        popcount_u64(self.0[0] ^ other.0[0]) + popcount_u64(self.0[1] ^ other.0[1])
    }

    /// Deterministically generate a `BinaryMultivector` from a seed text.
    ///
    /// Uses a splitmix64‑based PRNG seeded by a FNV‑1a hash of the input.
    #[inline]
    pub fn from_seed_text(text: &str) -> Self {
        // FNV‑1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for b in text.bytes() {
            hash = hash.wrapping_mul(0x100000001b3) ^ (b as u64);
        }

        let mut state = hash;
        let a = splitmix64(&mut state);
        let b = splitmix64(&mut state);
        BinaryMultivector([a, b])
    }

    /// Serialize the multivector to a 16‑byte array (little‑endian).
    #[inline]
    pub const fn to_bytes(&self) -> [u8; 16] {
        let a = self.0[0].to_le_bytes();
        let b = self.0[1].to_le_bytes();
        let mut bytes = [0u8; 16];
        let mut i = 0;
        while i < 8 {
            bytes[i] = a[i];
            bytes[i + 8] = b[i];
            i += 1;
        }
        bytes
    }

    /// Build a multivector from a little‑endian 16‑byte array.
    #[inline]
    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        let a = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let b = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self([a, b])
    }

    /// Store an index in the low 32 bits of the second limb, preserving the
    /// upper 32 bits of that limb and the entire first limb.
    #[inline]
    pub fn with_index(mut self, index: usize) -> Self {
        self.0[1] = (self.0[1] & !0xFFFF_FFFFu64) | ((index as u64) & 0xFFFF_FFFF);
        self
    }

    /// Extract the index stored in the low 32 bits of the second limb.
    #[inline]
    pub fn extract_index(&self) -> usize {
        (self.0[1] & 0xFFFF_FFFF) as usize
    }
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z = z ^ (z >> 31);
    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::TryInto;

    #[test]
    fn geometric_product_same() {
        let a = BinaryMultivector([0b1010, 0b1100]);
        assert_eq!(a.geometric_product(&a), 0);
    }

    #[test]
    fn geometric_product_different() {
        let a = BinaryMultivector([0b1010, 0b1100]);
        let b = BinaryMultivector([0b1111, 0b0000]);
        let expected = popcount_u64(0b0101) + popcount_u64(0b1100);
        assert_eq!(a.geometric_product(&b), expected);
    }

    #[test]
    fn from_seed_text_deterministic() {
        let a = BinaryMultivector::from_seed_text("hello");
        let b = BinaryMultivector::from_seed_text("hello");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn to_bytes_roundtrip() {
        let mv = BinaryMultivector([0x0123456789abcdef, 0xfedcba9876543210]);
        let bytes = mv.to_bytes();
        let a = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let b = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(mv.0[0], a);
        assert_eq!(mv.0[1], b);
    }

    #[test]
    fn with_index_roundtrip() {
        let v = BinaryMultivector([0x1234_5678_9ABC_DEF0, 0xFEDC_BA98_7654_3210]);
        let indexed = v.with_index(42);
        assert_eq!(indexed.extract_index(), 42);
        // Upper 32 bits of chunk 1 preserved
        assert_eq!(indexed.0[1] & 0xFFFF_FFFF_0000_0000, 0xFEDC_BA98_0000_0000);
    }

    #[test]
    fn with_index_max_u32() {
        let v = BinaryMultivector([0, 0]);
        let indexed = v.with_index(0xFFFF_FFFF);
        assert_eq!(indexed.extract_index(), 0xFFFF_FFFF);
    }
}
