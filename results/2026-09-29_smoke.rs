//! Self-contained smoke test — no external data, no absolute paths.
//! Verifies the core engine end-to-end: build a small binary HNSW,
//! insert pseudo-random 512-bit hashes, search, and roundtrip save/load.
//! This is the test CI runs and the first thing a reviewer can execute.

use pams::binary_hnsw::{BinaryHNSW, Hash512};

/// Tiny deterministic PRNG (splitmix64) — no external crate needed.
struct Rng(u64);
impl Rng {
    fn next_bytes(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
        }
    }
}

fn build_index(n: usize, seed: u64) -> (BinaryHNSW, Hash512) {
    let mut rng = Rng(seed);
    let mut idx = BinaryHNSW::new();
    let mut last = [0u8; 64];
    for id in 0..n as u64 {
        let mut h: Hash512 = [0u8; 64];
        rng.next_bytes(&mut h);
        idx.insert(id, h);
        last = h;
    }
    assert!(!idx.is_empty());
    (idx, last)
}

#[test]
fn insert_and_exact_search() {
    let (idx, query_hash) = build_index(2_000, 0xDEAD_BEEF_CAFE_1234);
    let results = idx.search(&query_hash, 10);
    assert!(!results.is_empty(), "search returned nothing");
    assert!(results.len() <= 10);
    // The query IS an inserted document: an exact (distance 0) match must exist.
    assert!(
        results.iter().any(|(dist, _idx, _tag)| *dist == 0),
        "exact hash not found in top-k: {:?}",
        results
    );
}

#[test]
fn save_load_roundtrip() {
    let (idx, query_hash) = build_index(500, 0x5EED_5EED);
    let path = std::env::temp_dir().join("yp_smoke_hnsw.bin");
    let path_str = path.to_str().unwrap();
    idx.save(path_str).expect("save failed");

    let loaded = BinaryHNSW::load(path_str).expect("load failed");
    let results = loaded.search(&query_hash, 10);
    assert!(
        results.iter().any(|(dist, _idx, _tag)| *dist == 0),
        "exact hash not found after reload"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn empty_index_search_is_safe() {
    let idx = BinaryHNSW::new();
    let q: Hash512 = [7u8; 64];
    assert!(idx.search(&q, 10).is_empty());
}
