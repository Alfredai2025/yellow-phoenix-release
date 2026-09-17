// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::types::multivector::BinaryMultivector;

/// Accumulator field that stores signed wave amplitudes.
///
/// Each element is a count of superposed +1/-1 contributions.
#[derive(Clone)]
pub struct WaveField<const D: usize> {
    /// Signed per‑position sums, enough headroom for up to 1M superpositions.
    pub data: [i32; D],
}

impl<const D: usize> WaveField<D> {
    /// All‑zero field.
    pub fn new() -> Self {
        Self { data: [0i32; D] }
    }
}

/// A live holographic field together with a cached binary threshold snapshot.
///
/// The snapshot is used for ultra‑fast queries. The structure is intentionally
/// single‑threaded (`&mut self` for all mutation) – no locks, no atomics, no
/// cache‑coherency overhead.
pub struct StandingHologram<const D: usize> {
    /// Live signed wave accumulation field.
    wave: WaveField<D>,
    /// Thresholded binary copy used for geometric queries.
    snapshot: BinaryMultivector,
    /// `true` when the snapshot is out of date with respect to `wave`.
    dirty: bool,
}

impl<const D: usize> StandingHologram<D> {
    /// Zero‑initialised hologram with a zeroed snapshot.
    pub fn new() -> Self {
        assert!(D <= 128, "StandingHologram dimension {} exceeds 128", D);
        Self {
            wave: WaveField::new(),
            snapshot: BinaryMultivector::new(),
            dirty: true,
        }
    }

    /// Superpose a paper onto the wave field (add +1 for active bits only).
    pub fn inject(&mut self, paper: &BinaryMultivector) {
        for i in 0..D {
            if paper.bit(i) {
                self.wave.data[i] += 1;
            }
        }
        self.dirty = true;
    }

    /// Remove a paper’s contribution from the wave field.
    pub fn eject(&mut self, paper: &BinaryMultivector) {
        for i in 0..D {
            if paper.bit(i) {
                self.wave.data[i] -= 1;
            }
        }
        self.dirty = true;
    }

    /// Re‑compute the thresholded snapshot if it is out of date.
    ///
    /// Returns a shared reference to the fresh (or already fresh) snapshot.
    pub fn snapshot(&mut self) -> &BinaryMultivector {
        if self.dirty {
            for i in 0..D {
                self.snapshot.set_bit(i, self.wave.data[i] > 0);
            }
            self.dirty = false;
        }
        &self.snapshot
    }

    /// Wave-field resonance: dot product of the live wave field with the
    /// active bits of the query. Rewards query bits that are strongly
    /// supported by the accumulated superposition; ignores inactive bits.
    pub fn query(&mut self, q: &BinaryMultivector) -> u32 {
        let mut score: i32 = 0;
        for i in 0..D {
            if q.bit(i) {
                score += self.wave.data[i];
            }
        }
        score.max(0) as u32
    }
}

/// Coarse 32‑dimension hologram.
pub type HologramTier0 = StandingHologram<32>;

/// Base 128‑dimension hologram (512 bytes maximum).
pub type HologramTier1 = StandingHologram<128>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_10_papers_returns_nonzero() {
        let mut h = StandingHologram::<128>::new();
        for _ in 0..10 {
            let mut p = BinaryMultivector::new();
            for i in 0..128 {
                p.set_bit(i, (i % 3) == 0);
            }
            h.inject(&p);
        }

        let mut q = BinaryMultivector::new();
        for i in 0..128 {
            q.set_bit(i, i % 2 == 0);
        }

        let score = h.query(&q);
        assert!(score != 0, "Expected non‑zero resonance after 10 injections");
    }

    #[test]
    fn eject_changes_score() {
        let mut h = StandingHologram::<32>::new();
        let mut p = BinaryMultivector::new();
        for i in 0..32 {
            p.set_bit(i, i % 2 == 0);
        }
        h.inject(&p);

        let mut q = BinaryMultivector::new();
        for i in 0..32 {
            q.set_bit(i, i % 3 == 0);
        }

        let before = h.query(&q);
        h.eject(&p);
        let after = h.query(&q);

        assert!(
            before != after,
            "Score must change after eject"
        );
    }

    #[test]
    fn snapshot_threshold() {
        // Use small D so we can manually check all bits.
        let mut h = StandingHologram::<4>::new();
        let mut p = BinaryMultivector::new();
        // pattern: [true, false, true, false]
        p.set_bit(0, true);
        p.set_bit(1, false);
        p.set_bit(2, true);
        p.set_bit(3, false);

        // Inject twice, so wave becomes [+2, -2, +2, -2]
        h.inject(&p);
        h.inject(&p);

        let snap = h.snapshot();
        assert_eq!(snap.bit(0), true, "bit 0 should be 1 because +2 > 0");
        assert_eq!(snap.bit(1), false, "bit 1 should be 0 because -2 < 0");
        assert_eq!(snap.bit(2), true);
        assert_eq!(snap.bit(3), false);
    }

    #[test]
    fn tier1_wave_field_size_does_not_exceed_512_bytes() {
        // The live wave field itself is the L1‑resident working set.
        let size = core::mem::size_of::<WaveField<128>>();
        assert!(
            size <= 512,
            "Tier‑1 wave field occupies {} bytes, exceeding 512 bytes",
            size
        );
    }
}
