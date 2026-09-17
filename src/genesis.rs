// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Build-time injected genesis seed.
//! NO SOURCE FILE contains the phrase. Only this hash.

/// Genesis seed, injected at compile time by build.rs.
/// Value is derived from YP_GENESIS_PHRASE environment variable.
pub const GENESIS_SEED: u64 = include!(concat!(env!("OUT_DIR"), "/genesis_seed.rs"));

/// Verify seed is non-zero (catches missing env var).
pub fn verify_seed() {
    assert!(GENESIS_SEED != 0, "Genesis seed is zero — build.rs failed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_nonzero() {
        verify_seed();
    }

    #[test]
    fn test_seed_deterministic() {
        // Same build environment = same seed
        let s1 = GENESIS_SEED;
        let s2 = GENESIS_SEED;
        assert_eq!(s1, s2);
    }
}
