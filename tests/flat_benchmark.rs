#![cfg(feature = "flat-array")]

use pams::flat_array::{FlatArray, FlatError, FLAT_HASH_LEN};
use pams::flat_query_engine::{FlatQueryEngine, HealthStatus};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn make_test_file(path: &str, n: usize) {
    let mut file = File::create(path).unwrap();
    for i in 0..n {
        file.write_all(&(i as u64).to_le_bytes()).unwrap();
        let hash: Vec<u8> = (0..32).map(|j| ((i + j) % 256) as u8).collect();
        file.write_all(&hash).unwrap();
    }
    file.flush().unwrap();
}

#[test]
fn test_flat_array_build_1m() {
    let path = "/tmp/yp_flat_test_1m.bin";
    make_test_file(path, 1_000_000);
    let arr = FlatArray::from_file(path, 1_000_000).unwrap();
    assert_eq!(arr.len(), 1_000_000);
    assert!(arr.verify_checksum());
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_query_correctness() {
    let path = "/tmp/yp_flat_test_correctness.bin";
    make_test_file(path, 1000);
    let arr = FlatArray::from_file(path, 1000).unwrap();
    let engine = FlatQueryEngine::new(arr).unwrap();
    for i in 0..1000u64 {
        let expected: Vec<u8> = (0..32).map(|j| ((i as usize + j) % 256) as u8).collect();
        let mut expected_arr = [0u8; FLAT_HASH_LEN];
        expected_arr.copy_from_slice(&expected);
        assert_eq!(engine.query(i), Some(expected_arr));
    }
    assert_eq!(engine.query(1000), None);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_query_engine_stats() {
    let path = "/tmp/yp_flat_test_stats.bin";
    make_test_file(path, 1000);
    let arr = FlatArray::from_file(path, 1000).unwrap();
    let engine = FlatQueryEngine::new(arr).unwrap();
    for i in 0..500 {
        engine.query(i as u64);
    }
    for _ in 0..10 {
        engine.query(9999); // misses
    }
    let s = engine.stats();
    assert_eq!(s.hits, 500);
    assert_eq!(s.misses, 10);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_batch_query() {
    let path = "/tmp/yp_flat_test_batch.bin";
    make_test_file(path, 1000);
    let arr = FlatArray::from_file(path, 1000).unwrap();
    let engine = FlatQueryEngine::new(arr).unwrap();
    let ids: Vec<u64> = (0..100).chain(1000..1010).collect();
    let results = engine.query_batch(&ids);
    assert_eq!(results.len(), 110);
    assert!(results[..100].iter().all(|r| r.is_some()));
    assert!(results[100..].iter().all(|r| r.is_none()));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_circuit_breaker() {
    let path = "/tmp/yp_flat_test_circuit.bin";
    make_test_file(path, 10);
    let arr = FlatArray::from_file(path, 10).unwrap();
    let engine = FlatQueryEngine::new(arr).unwrap();
    assert!(!engine.health().circuit_open);
    // Simulate corruption detection; circuit should open after threshold.
    engine.mark_corruption();
    engine.mark_corruption();
    engine.mark_corruption();
    assert!(engine.health().circuit_open);
    assert_eq!(engine.health().status, HealthStatus::Critical);
    // While circuit is open, queries return None.
    assert_eq!(engine.query(0), None);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_checksum_validation() {
    let path = "/tmp/yp_flat_test_checksum.bin";
    make_test_file(path, 100);
    let arr = FlatArray::from_file(path, 100).unwrap();
    let persist_path = "/tmp/yp_flat_test_checksum.flat";
    arr.persist(persist_path).unwrap();

    // Tamper with one byte of the persisted file to corrupt the checksum.
    const HEADER_BYTES: usize = 4 + 4 + 8 + 8;
    let mut bytes = std::fs::read(persist_path).unwrap();
    let data_offset = HEADER_BYTES + 5 * FLAT_HASH_LEN + 3;
    bytes[data_offset] = bytes[data_offset].wrapping_add(1);
    std::fs::write(persist_path, &bytes).unwrap();

    let loaded = FlatArray::load(persist_path);
    assert!(matches!(loaded, Err(FlatError::ChecksumMismatch)));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(persist_path).unwrap();
}

#[test]
fn test_flat_persist_and_reload() {
    let path = "/tmp/yp_flat_test_persist.bin";
    make_test_file(path, 1000);
    let arr = FlatArray::from_file(path, 1000).unwrap();
    let persist_path = "/tmp/yp_flat_test_persisted.flat";
    arr.persist(persist_path).unwrap();
    let loaded = FlatArray::load(persist_path).unwrap();
    assert_eq!(loaded.len(), 1000);
    assert!(loaded.verify_checksum());
    let engine = FlatQueryEngine::new(loaded).unwrap();
    assert!(engine.query(500).is_some());
    std::fs::remove_file(path).unwrap();
    std::fs::remove_file(persist_path).unwrap();
}

#[test]
fn test_flat_health_degradation() {
    let path = "/tmp/yp_flat_test_health.bin";
    make_test_file(path, 100);
    let arr = FlatArray::from_file(path, 100).unwrap();
    let engine = FlatQueryEngine::new(arr).unwrap();
    assert_eq!(engine.health().status, HealthStatus::Ok);
    engine.mark_memory_pressure(true);
    assert_eq!(engine.health().status, HealthStatus::Degraded);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_memory_under_quota() {
    let path = "/tmp/yp_flat_test_quota.bin";
    make_test_file(path, 10_000_000);
    let arr = FlatArray::from_file(path, 10_000_000).unwrap();
    let bytes = arr.memory_bytes();
    assert_eq!(bytes, 10_000_000 * FLAT_HASH_LEN);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn test_flat_benchmark_10m() {
    let path = "/tmp/yp_flat_test_10m.bin";
    make_test_file(path, 10_000_000);
    let t0 = Instant::now();
    let arr = FlatArray::from_file(path, 10_000_000).unwrap();
    let build_ms = t0.elapsed().as_millis();
    assert!(build_ms < 10_000, "10M build took {}ms", build_ms);
    let engine = FlatQueryEngine::new(arr).unwrap();
    let mut latencies = Vec::new();
    for i in 0..1000 {
        let tq = Instant::now();
        engine.query(i as u64);
        latencies.push(tq.elapsed().as_micros());
    }
    latencies.sort();
    let p50 = latencies[500];
    println!("10M P50: {}us", p50);
    assert!(p50 < 10, "P50 should be <10us, got {}us", p50);
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "takes ~1 minute and needs ~6.5 GB RAM"]
fn test_flat_benchmark_200m() {
    let path = "/tmp/yp_flat_test_200m.bin";
    make_test_file(path, 200_000_000);
    let t0 = Instant::now();
    let arr = FlatArray::from_file(path, 200_000_000).unwrap();
    let build_s = t0.elapsed().as_secs_f64();
    assert!(build_s < 10.0, "200M build took {}s", build_s);
    let engine = FlatQueryEngine::new(arr).unwrap();
    let mut latencies = Vec::new();
    for i in 0..1000 {
        let tq = Instant::now();
        engine.query(i as u64);
        latencies.push(tq.elapsed().as_micros());
    }
    latencies.sort();
    let p50 = latencies[500];
    let p99 = latencies[990];
    println!("200M build: {:.2}s, P50: {}us, P99: {}us", build_s, p50, p99);
    assert!(p50 < 10, "P50 should be <10us, got {}us", p50);
    assert!(p99 < 30, "P99 should be <30us, got {}us", p99);
    std::fs::remove_file(path).unwrap();
}
