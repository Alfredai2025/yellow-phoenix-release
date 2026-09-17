#![cfg(feature = "intelligent-shard-manager")]

use pams::intelligent_shard_manager::{IntelligentShardManager, ISM_MAX_SHARD_CAPACITY};
use pams::shard::{IsmPap, Shard, ShardError};
use std::collections::HashMap;
use std::time::Instant;

fn make_papers(n: usize) -> Vec<(u64, IsmPap)> {
    let mut rng = fastrand::Rng::new();
    (0..n as u64)
        .map(|i| {
            let mut pap = [0u8; 32];
            rng.fill(&mut pap[..]);
            (i, pap)
        })
        .collect()
}

#[test]
fn test_shard_capacity_limit() {
    let mut shard = Shard::new(0, 2);
    shard.insert(1, [1u8; 32]).unwrap();
    shard.insert(2, [2u8; 32]).unwrap();
    assert!(matches!(
        shard.insert(3, [3u8; 32]),
        Err(ShardError::Full { capacity: 2 })
    ));
}

#[test]
fn test_single_shard_build_50m() {
    let n = 50_000_000usize;
    let papers = make_papers(n);
    let mut shard = Shard::new(0, ISM_MAX_SHARD_CAPACITY);

    let start = Instant::now();
    for (id, pap) in papers {
        shard.insert(id, pap).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();

    assert_eq!(shard.len(), n);
    println!("50M single shard build: {:.2}s ({:.0} inserts/s)", elapsed, n as f64 / elapsed);
    assert!(elapsed < 120.0, "50M single-shard build exceeded 120s");
}

#[test]
#[ignore = "takes ~2 minutes and needs ~6.5 GB RAM"]
fn test_four_shard_parallel_200m() {
    let n = 200_000_000usize;
    let papers = make_papers(n);

    let mut mgr = IntelligentShardManager::new();
    let start = Instant::now();
    mgr.build_parallel(papers.clone()).unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    assert_eq!(mgr.total_count, n);
    assert_eq!(mgr.shards.len(), 4);
    assert!(elapsed < 120.0, "200M parallel build exceeded 120s");
    println!("200M four-shard parallel build: {:.2}s ({:.0} inserts/s)", elapsed, n as f64 / elapsed);
}

#[test]
fn test_query_fan_out() {
    let mut mgr = IntelligentShardManager::new();
    let papers = make_papers(100_000);
    mgr.build_parallel(papers).unwrap();

    let results = mgr.query_parallel(42_000);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].cell_id, 42_000);
}

#[test]
fn test_learned_params() {
    let mut mgr = IntelligentShardManager::new();
    mgr.build_parallel(make_papers(10_000)).unwrap();

    let path = "/tmp/ism_learned_test.json";
    mgr.save_learned(path).unwrap();
    let loaded = IntelligentShardManager::load_learned(path);
    assert_eq!(loaded.learned.optimal_shard_size, ISM_MAX_SHARD_CAPACITY);
    assert!(loaded.learned.build_time_per_m > 0.0);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_parallel_correctness_vs_single_hashmap() {
    let n = 100_000usize;
    let papers = make_papers(n);

    // Ground truth: single std HashMap.
    let mut truth: HashMap<u64, IsmPap> = HashMap::with_capacity(n);
    for (id, pap) in papers.iter() {
        truth.insert(*id, pap.clone());
    }

    // ISM parallel build.
    let mut mgr = IntelligentShardManager::new();
    mgr.build_parallel(papers).unwrap();

    // Spot-check a sample.
    for id in [0, 1, n as u64 / 2, n as u64 - 1] {
        let ism_result = mgr.query_parallel(id);
        assert_eq!(ism_result.len(), 1);
        assert_eq!(ism_result[0].cell_id, id);
        assert_eq!(ism_result[0].data, truth[&id]);
    }
}
