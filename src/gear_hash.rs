// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! 512-bit helical hash — the gear's persistent constitutional state.
//!
//! Layout (64 bytes fixed + 16 adaptive = 80 bytes of 64 available):
//! Actually: 512 bits = 64 bytes total.
//! 8 windows × 6 bytes = 48 bytes.
//! 16 bytes remaining = adaptive breathing room.
//!
//! The 48 fixed bytes are scattered via Lucas-masked layout.

use crate::genesis::GENESIS_SEED;

/// A 512-bit hash (64 bytes) that acts as the gear's constitutional identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GearHash {
    pub bytes: [u8; 64],
}

/// Constitutional windows within the hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstitutionalWindow {
    Goals = 0,        // Pinnacle targets (latency, R@1, thermal)
    Commands = 1,     // Action directives (pre-warm, re-rank, throttle)
    Laws = 2,         // Hard constraints (max temp, exam boundary)
    Checks = 3,       // Integrity verification (checksum, phase alignment)
    Autopoiesis = 4,  // Self-repair directives
    Growth = 5,       // Expansion directives
    Intelligence = 6, // Learning & retuning
    Reserve = 7,      // Scars, evolution history, generation count
}

/// Computes the byte offset for a given window in a gear's hash.
/// The offset depends on gear_id, genesis seed, and lucas_number.
/// Without the seed, a dump of the 64 bytes reveals nothing.
pub fn window_offset(gear_id: u8, window: ConstitutionalWindow, lucas_number: u8) -> usize {
    let seed = GENESIS_SEED;
    let window_idx = window as u8;

    // LCG-style permutation within 0..58 (64 - 6 = 58 possible offsets)
    let mult = (seed.wrapping_mul(0x9e3779b97f4a7c15) | 1) as u32;
    let inc = (seed.wrapping_mul(0x85ebca6b).wrapping_add(gear_id as u64) | 1) as u32;
    let base = (window_idx as u32).wrapping_mul(mult).wrapping_add(inc);
    let mix = ((base >> 24) ^ (base >> 16) ^ (base >> 8) ^ base) as usize;

    // Lucas number adds secondary perturbation
    let lucas_perturb = ((lucas_number as u64).wrapping_mul(seed) >> 32) as usize;

    ((mix.wrapping_add(lucas_perturb)) % 58) as usize
}

/// Read a 6-byte window from the hash at its scattered offset.
pub fn read_window(
    hash: &GearHash,
    gear_id: u8,
    window: ConstitutionalWindow,
    lucas: u8,
) -> [u8; 6] {
    let offset = window_offset(gear_id, window, lucas);
    let mut result = [0u8; 6];
    result.copy_from_slice(&hash.bytes[offset..offset + 6]);
    result
}

/// Write a 6-byte window to the hash at its scattered offset.
pub fn write_window(
    hash: &mut GearHash,
    gear_id: u8,
    window: ConstitutionalWindow,
    lucas: u8,
    value: [u8; 6],
) {
    let offset = window_offset(gear_id, window, lucas);
    hash.bytes[offset..offset + 6].copy_from_slice(&value);
}

/// Initialize a gear hash from identity, seed, and lucas number.
pub fn init_gear_hash(gear_id: u8, lucas: u8) -> GearHash {
    let seed = GENESIS_SEED;
    let mut bytes = [0u8; 64];

    // Deterministic initialization: hash of (gear_id, seed, lucas)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    gear_id.hash(&mut hasher);
    seed.hash(&mut hasher);
    lucas.hash(&mut hasher);
    let init_hash = hasher.finish();

    // Spread 64-bit hash across 64 bytes with LCG expansion
    let mut state = init_hash;
    for i in 0..64 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        bytes[i] = (state >> 56) as u8;
    }

    GearHash { bytes }
}

impl GearHash {
    pub fn zero() -> Self {
        Self { bytes: [0u8; 64] }
    }

    /// Compute hash of the full 64 bytes (for integrity checks).
    pub fn checksum(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.bytes.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_offset_deterministic() {
        let o1 = window_offset(0, ConstitutionalWindow::Goals, 2);
        let o2 = window_offset(0, ConstitutionalWindow::Goals, 2);
        assert_eq!(o1, o2);
        assert!(o1 < 58);
    }

    #[test]
    fn test_window_offset_different_gears() {
        let o1 = window_offset(0, ConstitutionalWindow::Goals, 2);
        let o2 = window_offset(1, ConstitutionalWindow::Goals, 2);
        assert_ne!(o1, o2, "Different gears should have different offsets");
    }

    #[test]
    fn test_window_offset_different_windows() {
        let o1 = window_offset(0, ConstitutionalWindow::Goals, 2);
        let o2 = window_offset(0, ConstitutionalWindow::Laws, 2);
        assert_ne!(o1, o2, "Different windows should have different offsets");
    }

    #[test]
    fn test_read_write_roundtrip() {
        let mut hash = init_gear_hash(0, 2);
        let value = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        write_window(&mut hash, 0, ConstitutionalWindow::Goals, 2, value);
        let read = read_window(&hash, 0, ConstitutionalWindow::Goals, 2);
        assert_eq!(read, value);
    }

    #[test]
    fn test_init_deterministic() {
        let h1 = init_gear_hash(5, 13);
        let h2 = init_gear_hash(5, 13);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_checksum_changes_on_mutation() {
        let mut hash = init_gear_hash(0, 2);
        let c1 = hash.checksum();
        hash.bytes[0] ^= 0xFF;
        let c2 = hash.checksum();
        assert_ne!(c1, c2);
    }
}
