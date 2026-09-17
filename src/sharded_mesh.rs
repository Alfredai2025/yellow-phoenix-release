// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! sharded_mesh.rs — Memory-optimized shard using Yellow's existing math.
//!
//! Optimizations layered on top of the base mesh:
//!   * 16-byte hash truncation (4× smaller than 64-byte hashes)
//!   * f16 spectral coordinates (2× smaller than f32)
//!   * Zipf hot/cold separation (top 10% hot in RAM, rest memory-mapped)
//!   * Hash-bucketed cold storage so no per-paper id index is kept in RAM.
//!
//! Combined, these drop the per-paper footprint from ~76 bytes to ~30 bytes.

use crate::spectral_coords::{SpectralCoords, SpectralCoordsF16};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Per-paper entry in an optimized shard: 30 bytes on disk / in compressed form.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CompressedPaper {
    pub id: u64,              // 8 bytes
    pub hash_16: [u8; 16],    // 16 bytes (truncated 512-bit hash)
    pub spectral: SpectralCoordsF16, // 6 bytes
}

/// In-RAM hot tier: papers expected to receive most queries.
pub struct HotTier {
    papers: HashMap<[u8; 16], CompressedPaper>,
}

/// Memory-mapped cold tier: remaining papers grouped by 16-bit bucket prefix.
pub struct ColdTier {
    mmap: memmap2::Mmap,
    bucket_offsets: Vec<(usize, usize)>, // (start_index, end_index) per bucket
    records: usize,
}

/// A single optimized shard: hot RAM + cold mmap.
pub struct OptimizedShard {
    hot: HotTier,
    cold: ColdTier,
}

impl CompressedPaper {
    /// On-wire size in bytes: id(8) + hash_16(16) + spectral f16(6).
    pub const COMPRESSED_SIZE: usize = 30;

    /// Truncate a 64-byte PAP hash to its first 16 bytes.
    pub fn truncate_hash(hash_512: &[u8; 64]) -> [u8; 16] {
        let mut h = [0u8; 16];
        h.copy_from_slice(&hash_512[..16]);
        h
    }

    /// Bucket prefix used for cold grouping (first 2 bytes of the 16-byte hash).
    pub fn bucket_prefix(hash_16: &[u8; 16]) -> u16 {
        u16::from_le_bytes([hash_16[0], hash_16[1]])
    }
}

impl HotTier {
    pub fn new() -> Self {
        Self { papers: HashMap::new() }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.papers.is_empty()
    }

    pub fn insert(&mut self, paper: CompressedPaper) {
        self.papers.insert(paper.hash_16, paper);
    }

    pub fn query(&self, hash_16: &[u8; 16]) -> Option<&CompressedPaper> {
        self.papers.get(hash_16)
    }

    pub fn len(&self) -> usize {
        self.papers.len()
    }

    pub fn memory_bytes(&self) -> usize {
        // HashMap entry overhead is roughly 2× the stored size in practice.
        self.papers.len() * CompressedPaper::COMPRESSED_SIZE * 2
    }
}

impl ColdTier {
    const MAGIC: u32 = 0x4F50_5348; // "OPSH"
    const VERSION: u32 = 1;
    const RECORD_SIZE: usize = CompressedPaper::COMPRESSED_SIZE;

    /// Build a cold tier from a list of papers and memory-map it from `path`.
    pub fn build_and_mmap(path: &Path, papers: &[CompressedPaper]) -> io::Result<Self> {
        if papers.is_empty() {
            return Ok(Self {
                mmap: unsafe { memmap2::Mmap::map(&File::open("/dev/null")?)? },
                bucket_offsets: vec![(0, 0); 65536],
                records: 0,
            });
        }

        // Group by bucket prefix, keeping papers in stable order within each bucket.
        let mut by_bucket: Vec<Vec<CompressedPaper>> = vec![Vec::new(); 65536];
        for &p in papers {
            by_bucket[CompressedPaper::bucket_prefix(&p.hash_16) as usize].push(p);
        }

        let mut bucket_offsets = Vec::with_capacity(65536);
        let mut flat: Vec<CompressedPaper> = Vec::with_capacity(papers.len());
        for bucket in by_bucket {
            let start = flat.len();
            flat.extend(bucket);
            bucket_offsets.push((start, flat.len()));
        }

        // Write to file.
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(&Self::MAGIC.to_le_bytes())?;
        file.write_all(&Self::VERSION.to_le_bytes())?;
        file.write_all(&(flat.len() as u64).to_le_bytes())?;
        for p in &flat {
            let mut rec = [0u8; Self::RECORD_SIZE];
            rec[0..8].copy_from_slice(&p.id.to_le_bytes());
            rec[8..24].copy_from_slice(&p.hash_16);
            rec[24..26].copy_from_slice(&p.spectral.c0.to_le_bytes());
            rec[26..28].copy_from_slice(&p.spectral.c1.to_le_bytes());
            rec[28..30].copy_from_slice(&p.spectral.c2.to_le_bytes());
            file.write_all(&rec)?;
        }
        file.flush()?;
        drop(file);

        // Memory-map.
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        Ok(Self { mmap, bucket_offsets, records: flat.len() })
    }

    /// Number of cold records.
    pub fn len(&self) -> usize {
        self.records
    }

    /// Search the cold bucket matching `hash_16`.
    pub fn query(&self, hash_16: &[u8; 16]) -> Option<CompressedPaper> {
        if self.records == 0 {
            return None;
        }
        let (start, end) = self.bucket_offsets[CompressedPaper::bucket_prefix(hash_16) as usize];
        if start == end {
            return None;
        }
        // Header: magic(4) + version(4) + count(8) = 16 bytes.
        let base = 16 + start * Self::RECORD_SIZE;
        let bytes = &self.mmap[base..base + (end - start) * Self::RECORD_SIZE];
        for chunk in bytes.chunks_exact(Self::RECORD_SIZE) {
            let id = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let mut h = [0u8; 16];
            h.copy_from_slice(&chunk[8..24]);
            if &h == hash_16 {
                let c0 = u16::from_le_bytes(chunk[24..26].try_into().unwrap());
                let c1 = u16::from_le_bytes(chunk[26..28].try_into().unwrap());
                let c2 = u16::from_le_bytes(chunk[28..30].try_into().unwrap());
                return Some(CompressedPaper {
                    id,
                    hash_16: h,
                    spectral: SpectralCoordsF16 { c0, c1, c2 },
                });
            }
        }
        None
    }

    /// Touch the first byte of every page in the cold mmap so the kernel
    /// faults them in once. This eliminates page-fault jitter from P99.
    pub fn pre_fault(&self) -> usize {
        if self.records == 0 {
            return 0;
        }
        let page_size = 4096usize; // safe cross-platform lower bound
        let ptr = self.mmap.as_ptr();
        let len = self.mmap.len();
        let mut touched = 0;
        for offset in (0..len).step_by(page_size) {
            unsafe { std::ptr::read_volatile(ptr.add(offset)) };
            touched += 1;
        }
        touched
    }
}

impl OptimizedShard {
    /// Build an optimized shard from raw records.
    ///
    /// * `records` — iterator of (id, 64-byte hash, SpectralCoords).
    /// * `hot_ratio` — fraction of papers kept in RAM (default 0.10).
    /// * `cold_path` — path for the memory-mapped cold tier.
    ///
    /// Hot papers are chosen as the first `hot_ratio` of the input, which the
    /// Zipf generator already orders by popularity.
    pub fn build<I>(
        records: I,
        hot_ratio: f64,
        cold_path: &Path,
    ) -> io::Result<Self>
    where
        I: ExactSizeIterator<Item = (u64, [u8; 64], SpectralCoords)>,
    {
        let n = records.len();
        let hot_count = ((n as f64) * hot_ratio).ceil() as usize;
        let mut hot = HotTier::new();
        let mut cold_vec = Vec::with_capacity(n.saturating_sub(hot_count));

        for (idx, (id, hash_512, coords)) in records.enumerate() {
            let paper = CompressedPaper {
                id,
                hash_16: CompressedPaper::truncate_hash(&hash_512),
                spectral: SpectralCoordsF16::from_coords(&coords),
            };
            if idx < hot_count {
                hot.insert(paper);
            } else {
                cold_vec.push(paper);
            }
        }

        let cold = ColdTier::build_and_mmap(cold_path, &cold_vec)?;
        Ok(Self { hot, cold })
    }

    /// Query by a 64-byte hash; returns the matching paper id and spectral coords.
    pub fn query(&self, hash_512: &[u8; 64]) -> Option<(u64, SpectralCoords)> {
        let hash_16 = CompressedPaper::truncate_hash(hash_512);
        if let Some(p) = self.hot.query(&hash_16) {
            return Some((p.id, p.spectral.to_coords()));
        }
        self.cold.query(&hash_16).map(|p| (p.id, p.spectral.to_coords()))
    }

    /// Pre-fault the entire cold tier into kernel page cache. One-time cost at
    /// startup; removes page-fault jitter from the query path.
    pub fn pre_fault_cold_pages(&self) -> usize {
        self.cold.pre_fault()
    }

    /// RAM-resident memory usage (hot tier + negligible bucket offsets).
    pub fn memory_bytes(&self) -> usize {
        self.hot.memory_bytes() + self.cold.bucket_offsets.len() * std::mem::size_of::<(usize, usize)>()
    }

    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    pub fn cold_len(&self) -> usize {
        self.cold.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectral_coords::SpectralCoords;

    fn fake_record(id: u64) -> (u64, [u8; 64], SpectralCoords) {
        let mut h = [0u8; 64];
        h[0..8].copy_from_slice(&id.to_le_bytes());
        (id, h, SpectralCoords { c0: 0.1, c1: -0.2, c2: 0.3 })
    }

    #[test]
    fn roundtrip_compressed_paper() {
        let (_, hash_512, coords) = fake_record(42);
        let p = CompressedPaper {
            id: 42,
            hash_16: CompressedPaper::truncate_hash(&hash_512),
            spectral: SpectralCoordsF16::from_coords(&coords),
        };
        assert_eq!(p.id, 42);
        assert_eq!(p.hash_16, &hash_512[..16]);
        let back = p.spectral.to_coords();
        assert!((back.c0 - coords.c0).abs() < 0.001);
        assert!((back.c1 - coords.c1).abs() < 0.001);
        assert!((back.c2 - coords.c2).abs() < 0.001);
    }

    #[test]
    fn optimized_shard_hot_and_cold() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cold.bin");
        let records: Vec<_> = (0..1000).map(fake_record).collect();
        let shard = OptimizedShard::build(records.into_iter(), 0.10, &path).unwrap();
        assert_eq!(shard.hot_len(), 100);
        assert_eq!(shard.cold_len(), 900);

        // Query a hot record.
        let (_, h, _) = fake_record(5);
        let (id, _) = shard.query(&h).unwrap();
        assert_eq!(id, 5);

        // Query a cold record.
        let (_, h, _) = fake_record(500);
        let (id, _) = shard.query(&h).unwrap();
        assert_eq!(id, 500);
    }
}
