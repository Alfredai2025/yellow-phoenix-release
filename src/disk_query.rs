// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

// GHOST MODULE — NOT WIRED INTO src/lib.rs
// This file exists on disk but is NOT declared in src/lib.rs.
// It does NOT compile into the library and is NOT reachable from Python.
//
// AUDIT DATE: 2026-07-27
// ACTION: Preserved per "Wire First, Delete Never" policy.
//         Do not modify unless wiring into the build.
//
//! disk_query.rs — Production-grade disk-backed PAP query engine
//! Reads single data file + index file. O(1) bucket lookup + linear scan.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

pub const PAP_BYTES: usize = 64;
pub const BUCKET_BITS: usize = 16;
pub const NUM_BUCKETS: usize = 1 << BUCKET_BITS;

#[derive(Clone, Debug)]
pub struct PAPResult {
    pub offset: u64,
    pub distance: f32,
    pub pap: [u8; PAP_BYTES],
}

pub struct DiskQueryEngine {
    data_path: String,
    index_path: String,
}

impl DiskQueryEngine {
    pub fn new(data_path: &str, index_path: &str) -> Self {
        Self { data_path: data_path.to_string(), index_path: index_path.to_string() }
    }

    pub fn query_bucket(&self, bucket_id: u32, query_pap: &[u8; PAP_BYTES], top_k: usize) -> Vec<PAPResult> {
        let (offset, count) = self.read_bucket_index(bucket_id);
        if count == 0 { return Vec::new(); }

        let paps = self.read_paps(offset, count);
        let mut results: Vec<PAPResult> = paps.into_iter().enumerate()
            .map(|(idx, pap)| PAPResult {
                offset: offset + idx as u64,
                distance: pap_distance(query_pap, &pap),
                pap,
            }).collect();

        results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
        results.truncate(top_k);
        results
    }

    fn read_bucket_index(&self, bucket_id: u32) -> (u64, u64) {
        let mut file = File::open(&self.index_path).expect("open index");
        file.seek(SeekFrom::Start(bucket_id as u64 * 16)).expect("seek");
        let mut buf = [0u8; 16];
        file.read_exact(&mut buf).expect("read");
        let offset = u64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]);
        let count = u64::from_le_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]);
        (offset, count)
    }

    fn read_paps(&self, offset: u64, count: u64) -> Vec<[u8; PAP_BYTES]> {
        let mut file = File::open(&self.data_path).expect("open data");
        file.seek(SeekFrom::Start(offset * PAP_BYTES as u64)).expect("seek");
        let total = count as usize * PAP_BYTES;
        let mut buffer = vec![0u8; total];
        file.read_exact(&mut buffer).expect("read");
        buffer.chunks_exact(PAP_BYTES).map(|c| { let mut p = [0u8; PAP_BYTES]; p.copy_from_slice(c); p }).collect()
    }
}

pub fn pap_distance(a: &[u8; PAP_BYTES], b: &[u8; PAP_BYTES]) -> f32 {
    let mut x = 0u32; let mut ac = 0u32; let mut bc = 0u32;
    for i in 0..PAP_BYTES { x += (a[i] & b[i]).count_ones(); ac += a[i].count_ones(); bc += b[i].count_ones(); }
    if ac == 0 || bc == 0 { return 0.0; }
    (x as f32) / ((ac as f32) * (bc as f32)).sqrt()
}

pub fn bucket_hash(pap: &[u8; PAP_BYTES]) -> u32 {
    (((pap[0] as u32) << 8) | (pap[1] as u32)) & ((NUM_BUCKETS - 1) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_disk_query_roundtrip() {
        let data_path = "/tmp/test_disk_data.bin";
        let index_path = "/tmp/test_disk_index.bin";
        let mut paps: Vec<[u8; PAP_BYTES]> = Vec::new();
        for i in 0..1000u64 {
            let mut pap = [0u8; PAP_BYTES];
            let mut s = i.wrapping_mul(0x9e3779b97f4a7c15);
            for j in 0..PAP_BYTES { s = s.wrapping_mul(0x2545f4914f6cdd1d).wrapping_add(1); pap[j] = (s >> 56) as u8; }
            paps.push(pap);
        }
        let mut index = vec![0u8; NUM_BUCKETS * 16];
        index[0..8].copy_from_slice(&0u64.to_le_bytes());
        index[8..16].copy_from_slice(&1000u64.to_le_bytes());
        { let mut f = File::create(data_path).unwrap(); for pap in &paps { f.write_all(pap).unwrap(); } }
        { let mut f = File::create(index_path).unwrap(); f.write_all(&index).unwrap(); }
        let engine = DiskQueryEngine::new(data_path, index_path);
        let results = engine.query_bucket(0, &paps[500], 10);
        assert!(!results.is_empty());
        assert!((results[0].distance - 1.0).abs() < 1e-6);
        let _ = std::fs::remove_file(data_path);
        let _ = std::fs::remove_file(index_path);
    }
}
