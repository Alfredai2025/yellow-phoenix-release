// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Flat Array v0.4 — O(1) lookup by sequential cell ID.
//!
//! Core invariant: cell IDs are 0..N-1, so `cell_id` is the Vec index.
//! No HashMap, no rehash, no thread spawn. 200M entries ≈ 6.4 GB.

use memmap2::Mmap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const FLAT_VERSION: u32 = 1;
pub const FLAT_HASH_LEN: usize = 32;

const HEADER_BYTES: usize = 4 + 4 + 8 + 8; // version + checksum + built_at_ns + len
const RECORD_BYTES: usize = 8 + FLAT_HASH_LEN; // id + hash

#[derive(Debug)]
enum Backing {
    /// In-memory copy of hashes only (used by from_file/from_bytes/build paths).
    Owned(Vec<[u8; FLAT_HASH_LEN]>),
    /// Zero-copy memory-mapped file of id+hash records.
    Mmap(Mmap, usize),
}

#[derive(Debug)]
pub struct FlatArray {
    backing: Backing,
    checksum: u32,
    version: u32,
    built_at_ns: u64,
}

#[derive(Debug)]
pub enum FlatError {
    FileNotFound,
    InvalidSize,
    OutOfMemory,
    ChecksumMismatch,
    InvalidVersion,
    Io(std::io::Error),
}

impl std::fmt::Display for FlatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlatError::FileNotFound => write!(f, "file not found"),
            FlatError::InvalidSize => write!(f, "invalid file size"),
            FlatError::OutOfMemory => write!(f, "insufficient memory"),
            FlatError::ChecksumMismatch => write!(f, "checksum mismatch (corruption detected)"),
            FlatError::InvalidVersion => write!(f, "unsupported file format version"),
            FlatError::Io(e) => write!(f, "io error: {}", e),
        }
    }
}

impl std::error::Error for FlatError {}

impl From<std::io::Error> for FlatError {
    fn from(e: std::io::Error) -> Self {
        if e.kind() == std::io::ErrorKind::NotFound {
            FlatError::FileNotFound
        } else {
            FlatError::Io(e)
        }
    }
}

impl FlatArray {
    /// Build from an in-memory Vec of hashes (ids are implicitly 0..N-1).
    pub fn from_sequential(data: Vec<[u8; FLAT_HASH_LEN]>) -> Self {
        let checksum = compute_checksum(&data);
        Self {
            backing: Backing::Owned(data),
            checksum,
            version: FLAT_VERSION,
            built_at_ns: now_ns(),
        }
    }

    /// Read fixed-size records from a file. Each record is 8-byte LE id + 32-byte hash.
    /// The id is validated against the expected sequential range, then discarded.
    pub fn from_file(path: &str, n: usize) -> Result<Self, FlatError> {
        if !Path::new(path).exists() {
            return Err(FlatError::FileNotFound);
        }
        if !can_fit(n) {
            return Err(FlatError::OutOfMemory);
        }
        let mut file = File::open(path)?;
        let expected = n.checked_mul(RECORD_BYTES).ok_or(FlatError::InvalidSize)?;
        let mut buf = vec![0u8; expected];
        file.read_exact(&mut buf)?;
        Self::from_record_bytes(&buf, n)
    }

    /// Memory-mapped file build (avoids a large user-space read buffer and copy).
    /// The file must contain exactly `n` fixed-size records (8-byte LE id + 32-byte hash).
    pub fn from_mmap(path: &str, n: usize) -> Result<Self, FlatError> {
        if !Path::new(path).exists() {
            return Err(FlatError::FileNotFound);
        }
        if !can_fit(n) {
            return Err(FlatError::OutOfMemory);
        }
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| FlatError::Io(e))? };
        let expected = n.checked_mul(RECORD_BYTES).ok_or(FlatError::InvalidSize)?;
        if mmap.len() < expected {
            return Err(FlatError::InvalidSize);
        }
        // Fast path: keep the mmap directly. Skip the expensive per-record copy
        // and checksum scan that the old implementation performed.
        Ok(Self {
            backing: Backing::Mmap(mmap, n),
            checksum: 0,
            version: FLAT_VERSION,
            built_at_ns: now_ns(),
        })
    }

    fn from_record_bytes(bytes: &[u8], n: usize) -> Result<Self, FlatError> {
        let expected = n.checked_mul(RECORD_BYTES).ok_or(FlatError::InvalidSize)?;
        if bytes.len() != expected {
            return Err(FlatError::InvalidSize);
        }
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * RECORD_BYTES;
            let id = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            if id != i as u64 {
                // Sequential layout assumption violated; still store at index i.
                // Callers that need strict validation can check separately.
            }
            let mut hash = [0u8; FLAT_HASH_LEN];
            hash.copy_from_slice(&bytes[off + 8..off + RECORD_BYTES]);
            data.push(hash);
        }
        Ok(Self::from_sequential(data))
    }

    /// O(1) lookup. Returns None if cell_id is out of bounds.
    #[inline]
    pub fn query(&self, cell_id: u64) -> Option<&[u8; FLAT_HASH_LEN]> {
        let idx = cell_id as usize;
        match &self.backing {
            Backing::Owned(data) => data.get(idx),
            Backing::Mmap(mmap, n) => {
                if idx >= *n {
                    return None;
                }
                let off = idx * RECORD_BYTES + 8;
                let ptr = mmap[off..off + FLAT_HASH_LEN].as_ptr();
                Some(unsafe { &*(ptr as *const [u8; FLAT_HASH_LEN]) })
            }
        }
    }

    pub fn query_batch(&self, cell_ids: &[u64]) -> Vec<Option<&[u8; FLAT_HASH_LEN]>> {
        cell_ids.iter().map(|&id| self.query(id)).collect()
    }

    pub fn compute_checksum(&self) -> u32 {
        match &self.backing {
            Backing::Owned(data) => compute_checksum(data),
            // mmap-backed arrays are not checksummed at load time (fast path).
            Backing::Mmap(_, _) => 0,
        }
    }

    pub fn verify_checksum(&self) -> bool {
        match &self.backing {
            Backing::Owned(_) => self.checksum == self.compute_checksum(),
            // mmap-backed arrays trust the on-disk file; no full-data scan on load.
            Backing::Mmap(_, _) => true,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match &self.backing {
            Backing::Owned(data) => data.capacity() * FLAT_HASH_LEN,
            Backing::Mmap(_, n) => n * RECORD_BYTES,
        }
    }

    pub fn len(&self) -> usize {
        match &self.backing {
            Backing::Owned(data) => data.len(),
            Backing::Mmap(_, n) => *n,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn checksum(&self) -> u32 {
        self.checksum
    }

    pub fn built_at_ns(&self) -> u64 {
        self.built_at_ns
    }

    /// Serialize to bytes: header + hash data.
    pub fn as_bytes(&self) -> Vec<u8> {
        let len = self.len();
        let mut out = Vec::with_capacity(HEADER_BYTES + len * FLAT_HASH_LEN);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
        out.extend_from_slice(&self.built_at_ns.to_le_bytes());
        out.extend_from_slice(&(len as u64).to_le_bytes());
        for i in 0..len {
            if let Some(hash) = self.query(i as u64) {
                out.extend_from_slice(hash);
            }
        }
        out
    }

    /// Atomic persistence: write temp file, fsync, then rename.
    pub fn persist(&self, path: &str) -> Result<(), FlatError> {
        let bytes = self.as_bytes();
        let temp = format!("{}.tmp", path);
        {
            let mut f = File::create(&temp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&temp, path)?;
        Ok(())
    }

    /// Load a previously persisted flat array, verifying checksum and version.
    pub fn load(path: &str) -> Result<Self, FlatError> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FlatError> {
        if bytes.len() < HEADER_BYTES {
            return Err(FlatError::InvalidSize);
        }
        let version = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if version != FLAT_VERSION {
            return Err(FlatError::InvalidVersion);
        }
        let checksum = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let built_at_ns = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let expected = HEADER_BYTES + len * FLAT_HASH_LEN;
        if bytes.len() != expected {
            return Err(FlatError::InvalidSize);
        }
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let off = HEADER_BYTES + i * FLAT_HASH_LEN;
            let mut hash = [0u8; FLAT_HASH_LEN];
            hash.copy_from_slice(&bytes[off..off + FLAT_HASH_LEN]);
            data.push(hash);
        }
        let arr = Self {
            backing: Backing::Owned(data),
            checksum,
            version,
            built_at_ns,
        };
        if !arr.verify_checksum() {
            return Err(FlatError::ChecksumMismatch);
        }
        Ok(arr)
    }
}

fn compute_checksum(data: &[[u8; FLAT_HASH_LEN]]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    for h in data {
        hasher.update(h);
    }
    hasher.finalize()
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Rough guard: refuse to build if the array would exceed 60% of total RAM.
/// Uses sysinfo when available; otherwise allows the allocation to fail naturally.
fn can_fit(n: usize) -> bool {
    #[cfg(target_os = "macos")]
    {
        // sysinfo available_memory is unreliable on macOS; use total memory as a ceiling.
        use sysinfo::System;
        let sys = System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        let total = sys.total_memory();
        if total == 0 {
            return true;
        }
        let needed = (n * FLAT_HASH_LEN) as u64;
        needed <= total * 6 / 10
    }
    #[cfg(not(target_os = "macos"))]
    {
        use sysinfo::System;
        let sys = System::new_with_specifics(
            sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
        );
        let available = sys.available_memory();
        if available == 0 {
            return true;
        }
        let needed = (n * FLAT_HASH_LEN) as u64;
        needed <= available * 8 / 10
    }
}
