// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

extern crate alloc;

use alloc::vec::Vec;
use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use core::cmp::Ordering;

use crate::types::multivector::BinaryMultivector;
use crate::distance::pap_distance;
use super::hologram_index::StandingHologram;

// ---------------------------------------------------------------------------
// helper to fold a 512‑bit wave into the 128‑bit binary representation
// ---------------------------------------------------------------------------
pub fn hologram_wave_as_binary(wave: &[u64; 8]) -> BinaryMultivector {
    let chunk0 = wave[0] ^ wave[2] ^ wave[4] ^ wave[6];
    let chunk1 = wave[1] ^ wave[3] ^ wave[5] ^ wave[7];
    BinaryMultivector([chunk0, chunk1])
}

// ---------------------------------------------------------------------------
// 3‑tier query pipeline
// ---------------------------------------------------------------------------
pub struct TieredQuery {
    pub holograms: Vec<StandingHologram>,
    pub exact_fallback_threshold: usize,
}

impl TieredQuery {
    /// Create a new pipeline with the default threshold (1000 prototypes).
    pub fn new(holograms: Vec<StandingHologram>) -> Self {
        Self {
            holograms,
            exact_fallback_threshold: 1000,
        }
    }

    /// Run the 3‑tier query and return up to `top_k` (index, score) pairs.
    ///
    /// **Tier 1** – For each hologram compute its resonance PAP and scan
    /// the prototype store, retaining at most 100 candidates per hologram
    /// whose PAP distance to the query is < 0.5.
    ///
    /// **Tier 2** – Ensemble vote: rank by the number of holograms that
    /// voted for a candidate, break ties using
    /// `score = 1.0 / (1.0 + avg(pap_resonances))`.
    ///
    /// **Tier 3** – Exact fallback: if the result set is still too small
    /// and the total number of prototypes across all holograms is less
    /// than `exact_fallback_threshold`, perform a brute‑force PAP scan
    /// over every unique prototype and back‑fill the remaining slots.
    pub fn query(&self, q: &BinaryMultivector, top_k: usize) -> Vec<(usize, f32)> {
        // ––– Tier 1 & Tier 2 –––
        let mut votes: BTreeMap<usize, (u32, f32)> = BTreeMap::new(); // idx -> (count, sum_resonance)
        let mut total_prototypes: usize = 0;

        for h in &self.holograms {
            let wave_bin = hologram_wave_as_binary(&h.wave);
            let resonance = pap_distance(q, &wave_bin);
            let mut taken = 0usize;

            for &(idx, ref vec) in &h.prototypes {
                let pap = pap_distance(q, vec);
                if pap < 0.5 {
                    let entry = votes.entry(idx).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += resonance;
                    taken += 1;
                    if taken >= 100 {
                        break;
                    }
                }
            }
            total_prototypes += h.prototypes.len();
        }

        // build a candidate list sorted by vote count (desc) then by score (desc)
        let mut candidate_vec: Vec<(usize, u32, f32)> = votes
            .into_iter()
            .map(|(idx, (cnt, sum_r))| {
                let avg = sum_r / cnt as f32;
                (idx, cnt, 1.0 / (1.0 + avg))
            })
            .collect();

        candidate_vec.sort_by(|a, b| {
            match b.1.cmp(&a.1) {
                Ordering::Equal => b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal),
                other => other,
            }
        });

        let mut top_result: Vec<(usize, f32)> = candidate_vec
            .into_iter()
            .take(top_k)
            .map(|(idx, _cnt, score)| (idx, score))
            .collect();

        // ––– Tier 3 (fallback) –––
        if top_result.len() < top_k && total_prototypes < self.exact_fallback_threshold {
            // gather every unique prototype across all holograms
            let mut unique = BTreeMap::new();
            for h in &self.holograms {
                for &(idx, ref vec) in &h.prototypes {
                    unique.entry(idx).or_insert(vec);
                }
            }

            // brute‑force PAP scan sorted by exact distance (lower is better)
            let mut full_scan: Vec<(usize, f32, f32)> = unique
                .into_iter()
                .map(|(idx, vec)| {
                    let pap = pap_distance(q, vec);
                    (idx, pap, 1.0 / (1.0 + pap))
                })
                .collect();

            full_scan.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let existing: BTreeSet<usize> = top_result.iter().map(|&(i, _)| i).collect();
            let needed = top_k - top_result.len();

            let mut added = 0;
            for (idx, _pap, score) in full_scan {
                if !existing.contains(&idx) {
                    top_result.push((idx, score));
                    added += 1;
                    if added >= needed {
                        break;
                    }
                }
            }
        }

        top_result
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::types::multivector::BinaryMultivector;
    use crate::distance::pap_distance;
    use alloc::vec;

    // Helper that builds a standing hologram with zeroed‑out “other” fields.
    // The exact layout of StandingHologram is defined in `hologram_index.rs`,
    // but for testing we only need to populate `wave` and `prototypes`.
    fn make_hologram(
        wave: [u64; 8],
        prototypes: Vec<(usize, BinaryMultivector)>,
    ) -> StandingHologram {
        StandingHologram::from_parts(wave, prototypes)
    }

    fn binv(lo: u64, hi: u64) -> BinaryMultivector {
        BinaryMultivector([lo, hi])
    }

    // Convenience: an identity wave that folds to zeros.
    fn zero_wave() -> [u64; 8] {
        [0u64; 8]
    }

    // ------------------------------------------------------------------
    // tier1_finds_candidates
    // ------------------------------------------------------------------
    #[test]
    fn tier1_finds_candidates() {
        let wave = zero_wave();
        let h1 = make_hologram(wave, vec![(0, binv(0b1, 0)), (1, binv(0b10, 0))]);
        let h2 = make_hologram(wave, vec![(2, binv(0b100, 0)), (0, binv(0b1, 0))]);
        let h3 = make_hologram(wave, vec![(1, binv(0b10, 0)), (3, binv(0b1000, 0))]);

        let pipeline = TieredQuery::new(vec![h1, h2, h3]);
        let q = binv(0b1, 0); // close to prototype 0

        let result = pipeline.query(&q, 3);
        let indices: Vec<usize> = result.into_iter().map(|(i, _)| i).collect();
        // all three candidates (0,1,2) should appear, maybe also 3
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
        // 0 should be the top candidate because it appears in two holograms
        assert_eq!(indices[0], 0);
    }

    // ------------------------------------------------------------------
    // tier2_ensemble_wins
    // ------------------------------------------------------------------
    #[test]
    fn tier2_ensemble_wins() {
        let wave = zero_wave();
        // Two prototype vectors both fairly close to query.
        let v_a = binv(0b1, 0);
        let v_b = binv(0b1, 0b1); // slightly farther
        let h1 = make_hologram(wave, vec![(0, v_a)]);
        let h2 = make_hologram(wave, vec![(0, v_a)]);
        let h3 = make_hologram(wave, vec![(1, v_b)]);

        let pipeline = TieredQuery::new(vec![h1, h2, h3]);
        let q = v_a; // exact match for prototype 0

        let result = pipeline.query(&q, 2);
        // candidate 0 appears in 2 holograms → should get first place
        assert_eq!(result[0].0, 0);
    }

    // ------------------------------------------------------------------
    // tier3_fallback
    // ------------------------------------------------------------------
    #[test]
    fn tier3_fallback() {
        let wave = zero_wave();

        // Only two prototypes exist in a single hologram — extremely sparse.
        let prototypes = vec![(0, binv(0b1, 0)), (1, binv(0b10, 0))];
        let h = make_hologram(wave, prototypes);
        let pipeline = TieredQuery::new(vec![h]);

        let q = binv(0b1, 0); // close to 0, far from 1

        // Ask for 5 results, but only 2 exist. With exact_fallback_threshold = 1000,
        // Tier 3 should fill the remaining 3 slots with the best (only) prototypes,
        // re‑ordering them by exact PAP distance.
        let result = pipeline.query(&q, 5);
        // We still have only 2 unique prototypes, so the vector length stays 2.
        assert_eq!(result.len(), 2);
        // The nearest prototype (0) must appear first.
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 1);
    }

    // ------------------------------------------------------------------
    // full_pipeline_1k
    // ------------------------------------------------------------------
    #[test]
    fn full_pipeline_1k() {
        // Create 1000 synthetic prototypes 0..999,
        // all distributed among 10 holograms.
        let mut prototypes = [binv(0, 0); 1000];
        for i in 0..1000u64 {
            // set a single bit in the lower half → increasing distance from zero.
            prototypes[i as usize] = binv(1 << (i % 64), 0);
        }

        let mut holograms: Vec<StandingHologram> = Vec::with_capacity(10);
        for bkt in 0..10 {
            let mut store: Vec<(usize, BinaryMultivector)> = Vec::with_capacity(100);
            for i in 0..1000 {
                if i % 10 == bkt {
                    store.push((i, prototypes[i]));
                }
            }
            holograms.push(make_hologram(zero_wave(), store));
        }

        let mut pipeline = TieredQuery::new(holograms);
        // Allow the exact fallback to run over the full 1000 prototypes.
        pipeline.exact_fallback_threshold = 2000;
        // Query with the prototype 0 (lowest bit set).
        let q = prototypes[0];
        let top = pipeline.query(&q, 3);

        // The nearest neighbour is prototype 0 itself (PAP = 0).
        assert_eq!(top[0].0, 0);
        // The fallback should have filled the remaining two slots.
        assert_eq!(top.len(), 3);
    }
}
