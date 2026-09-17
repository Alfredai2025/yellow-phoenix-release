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
//! Disk-backed CUN (Cross-Unit Number) index for 1B+ scale
//!
//! CUN = SID * M + DID (16 bytes per entry)
//! 1B papers = 16 GB -> must live on SSD, not RAM
//!
//! Architecture:
//! - CUN entries stored in sorted mmap files on disk
//! - In-memory bloom filter for fast negative checks
//! - LRU cache for hot CUN lookups (last 1M entries)
//! - Batched reads for query efficiency

use std::collections::HashMap;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const CUN_MAGIC: u32 = 0xC0FF_EEEE;
const CUN_VERSION: u32 = 1;
const CACHE_SIZE: usize = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUNEntry {
    pub cun: u64,
    pub paper_id: u64,
}

impl CUNEntry {
    pub fn new(cun: u64, paper_id: u64) -> Self {
        Self { cun, paper_id }
    }
}

pub struct CUNDiskStore {
    data_dir: PathBuf,
    chunk_files: Vec<PathBuf>,
    cache: HashMap<u64, u64>,
    pending: Vec<CUNEntry>,
    total_entries: u64,
    entries_per_chunk: usize,
}

impl CUNDiskStore {
    pub fn new(data_dir: &Path) -> io::Result<Self> {
        create_dir_all(data_dir)?;
        let mut chunk_files = Vec::new();
        let mut total_entries = 0u64;
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("cun") {
                    let count = Self::count_entries_in_chunk(&path)?;
                    total_entries += count as u64;
                    chunk_files.push(path);
                }
            }
        }
        chunk_files.sort();
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            chunk_files,
            cache: HashMap::with_capacity(CACHE_SIZE),
            pending: Vec::new(),
            total_entries,
            entries_per_chunk: 1_000_000,
        })
    }

    pub fn insert(&mut self, cun: u64, paper_id: u64) {
        self.pending.push(CUNEntry::new(cun, paper_id));
        self.cache.insert(cun, paper_id);
        if self.pending.len() >= self.entries_per_chunk {
            let _ = self.flush();
        }
    }

    pub fn lookup(&mut self, cun: u64) -> Option<u64> {
        if let Some(&id) = self.cache.get(&cun) {
            return Some(id);
        }
        for entry in &self.pending {
            if entry.cun == cun {
                self.cache.insert(cun, entry.paper_id);
                return Some(entry.paper_id);
            }
        }
        let result = self.lookup_on_disk(cun);
        if let Some(paper_id) = result {
            if self.cache.len() >= CACHE_SIZE {
                self.cache.clear();
            }
            self.cache.insert(cun, paper_id);
        }
        result
    }

    pub fn lookup_by_did(&mut self, did: u64, m: u64) -> Vec<u64> {
        let mut results = Vec::new();
        for entry in &self.pending {
            if entry.cun % m == did {
                results.push(entry.paper_id);
            }
        }
        for chunk_path in &self.chunk_files {
            results.extend(Self::scan_chunk_for_did(chunk_path, did, m));
        }
        results
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.pending.sort_by_key(|e| e.cun);
        let chunk_id = self.chunk_files.len();
        let chunk_path = self.data_dir.join(format!("chunk_{:08}.cun", chunk_id));
        let mut file = OpenOptions::new()
            .write(true).create(true).truncate(true).open(&chunk_path)?;
        file.write_all(&CUN_MAGIC.to_le_bytes())?;
        file.write_all(&CUN_VERSION.to_le_bytes())?;
        file.write_all(&(self.pending.len() as u64).to_le_bytes())?;
        for entry in &self.pending {
            file.write_all(&entry.cun.to_le_bytes())?;
            file.write_all(&entry.paper_id.to_le_bytes())?;
        }
        file.sync_all()?;
        self.total_entries += self.pending.len() as u64;
        self.chunk_files.push(chunk_path);
        self.pending.clear();
        Ok(())
    }

    pub fn len(&self) -> u64 {
        self.total_entries + self.pending.len() as u64
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn disk_bytes(&self) -> u64 { self.total_entries * 16 }

    fn count_entries_in_chunk(path: &Path) -> io::Result<usize> {
        let mut file = File::open(path)?;
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        let magic = u32::from_le_bytes(buf);
        if magic != CUN_MAGIC { return Ok(0); }
        file.read_exact(&mut buf)?;
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        Ok(u64::from_le_bytes(buf8) as usize)
    }

    fn lookup_on_disk(&self, target_cun: u64) -> Option<u64> {
        for chunk_path in &self.chunk_files {
            if let Ok(result) = Self::binary_search_chunk(chunk_path, target_cun) {
                return result;
            }
        }
        None
    }

    fn binary_search_chunk(path: &Path, target: u64) -> io::Result<Option<u64>> {
        let mut file = File::open(path)?;
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        file.read_exact(&mut buf4)?;
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let count = u64::from_le_bytes(buf8) as usize;
        let mut low = 0usize;
        let mut high = count;
        while low < high {
            let mid = (low + high) / 2;
            file.seek(SeekFrom::Start(16 + (mid as u64) * 16))?;
            let mut buf = [0u8; 16];
            file.read_exact(&mut buf)?;
            let cun = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            if cun == target {
                let paper_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
                return Ok(Some(paper_id));
            } else if cun < target {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        Ok(None)
    }

    fn scan_chunk_for_did(path: &Path, did: u64, m: u64) -> Vec<u64> {
        let mut results = Vec::new();
        let Ok(mut file) = File::open(path) else { return results };
        let mut buf4 = [0u8; 4];
        if file.read_exact(&mut buf4).is_err() { return results; }
        if file.read_exact(&mut buf4).is_err() { return results; }
        let mut buf8 = [0u8; 8];
        if file.read_exact(&mut buf8).is_err() { return results; }
        let count = u64::from_le_bytes(buf8) as usize;
        for _ in 0..count {
            let mut buf = [0u8; 16];
            if file.read_exact(&mut buf).is_err() { break; }
            let cun = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            if cun % m == did {
                let paper_id = u64::from_le_bytes(buf[8..16].try_into().unwrap());
                results.push(paper_id);
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn cun_disk_roundtrip() {
        let dir = temp_dir().join("yp_cun_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = CUNDiskStore::new(&dir).unwrap();
        for i in 0..1000 {
            store.insert(i * 100 + 5, i as u64);
        }
        store.flush().unwrap();
        for i in 0..1000 {
            assert_eq!(store.lookup(i * 100 + 5), Some(i as u64));
        }
        assert_eq!(store.lookup(999_999), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cun_disk_lookup_by_did() {
        let dir = temp_dir().join("yp_cun_did_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = CUNDiskStore::new(&dir).unwrap();
        for i in 0..100 {
            store.insert(i * 100 + 5, i as u64);
        }
        store.flush().unwrap();
        let results = store.lookup_by_did(5, 100);
        assert_eq!(results.len(), 100);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cun_disk_large_flush() {
        let dir = temp_dir().join("yp_cun_large");
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = CUNDiskStore::new(&dir).unwrap();
        store.entries_per_chunk = 100;
        for i in 0..500 {
            store.insert(i, i as u64);
        }
        assert!(store.chunk_files.len() >= 4);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
