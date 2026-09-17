//! fuzz_tests.rs — M1.6 quick fuzz tests (2 of 5).
//!
//! These tests are intentionally coarse: they verify that the engine and its
//! stages remain stable and discriminating under synthetic load, not that the
//! model is accurate.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use pams::engine_feeder::EngineFeeder;
use pams::hash_stage::FeatureResult;
use pams::hybrid_mesh::PAP_512_BYTES;
use pams::wedge_stage::WedgeStage;

const RNG_SEED: u64 = 0xdeadbeefcafebabe;

/// Fuzz 1: random bytes into the cascade-miss feed must not crash the engine.
#[test]
fn fuzz_feed_pipe_corruption() {
    let mut feeder = EngineFeeder::new();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    for _ in 0..10_000 {
        let len = rng.random_range(1..=256);
        let garbage: Vec<u8> = (0..len).map(|_| rng.random::<u8>()).collect();
        // Must not panic on arbitrary bytes.
        feeder.inject_raw("cascade_miss", &garbage);
    }

    // Feeder must remain alive and able to process a valid message afterwards.
    assert!(feeder.is_alive());
    feeder.inject_raw("cascade_miss", b"{\"valid\":true}");
    assert!(feeder.poll("cascade_miss").is_some());
}

/// Fuzz 2: 1M random hashes through the wedge stage.
/// Wedge must discriminate: no two distinct random inputs should collapse to
/// the same top-1 id (probability of accidental collision is negligible).
#[test]
fn fuzz_hash_collision() {
    let wedge = WedgeStage::new();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let mut seen_top1 = std::collections::HashSet::new();

    for _ in 0..1_000_000 {
        let mut pap_512 = [0u8; PAP_512_BYTES];
        rng.fill(&mut pap_512);

        // Build a 5-candidate list whose scores are derived from the query hash.
        // This guarantees the candidate set is deterministic per query and rich
        // enough to stress the wedge re-ranker.
        let mut candidates: Vec<(u64, f32)> = (0..5)
            .map(|i| {
                let id = u64::from_le_bytes([
                    pap_512[i * 8],
                    pap_512[i * 8 + 1],
                    pap_512[i * 8 + 2],
                    pap_512[i * 8 + 3],
                    pap_512[i * 8 + 4],
                    pap_512[i * 8 + 5],
                    pap_512[i * 8 + 6],
                    pap_512[i * 8 + 7],
                ]);
                let score = 0.5 + 0.5 * ((id % 1000) as f32 / 1000.0);
                (id, score)
            })
            .collect();
        // Make the first two scores very close so the wedge has to discriminate.
        candidates[1].1 = candidates[0].1 - 1e-6;
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let spectral = FeatureResult {
            score: candidates[0].1,
            confidence: 0.6,
            candidates,
        };

        let result = wedge.query(&pap_512, &spectral);
        let top1 = result.candidates.first().map(|(id, _)| *id).unwrap_or(0);

        assert!(
            seen_top1.insert(top1),
            "wedge stage collapsed two distinct hashes to the same top-1 id {}",
            top1
        );
    }
}
