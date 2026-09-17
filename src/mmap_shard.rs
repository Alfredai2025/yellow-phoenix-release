// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! MMAP Shard — disk-backed storage for cells that do not fit in RAM.
//!
//! File format v0.1:
//!   [HEADER: 32 bytes]
//!     - magic: [u8; 8] = b"YPGRDv01"
//!     - num_cells: u64
//!     - created_at_ns: u64
//!     - header_crc: u32 (CRC32 of the first 28 bytes)
//!     - reserved: u32
//!   [OFFSET TABLE: num_cells × 24 bytes]
//!     - offset: u64
//!     - length: u64
//!     - crc32: u32 (checksum of cell data)
//!     - reserved: u32
//!   [CELL DATA: variable, 8-byte aligned]
//!
//! The offset table is small and is mlocked so it is never swapped out.
//! Cell data is mmap'd and managed by the OS page cache.
//!
//! HONEST v0.1 limitations:
//!   * Append-only: updated cells leave dead space.
//!   * No compaction or garbage collection.
//!   * Atomic writes are flush-based, not temp-file-rename.
//!   * Windows is not supported.

use std::fs::OpenOptions;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crc32fast::Hasher;
use memmap2::{MmapMut, MmapOptions};

const MAGIC: &[u8; 8] = b"YPGRDv01";
const HEADER_SIZE: usize = 32;
const OFFSET_RECORD_SIZE: usize = 24;

/// On-disk record for one cell.
#[derive(Clone, Copy, Debug, Default)]
struct OffsetRecord {
    offset: u64,
    length: u64,
    crc32: u32,
}

pub struct MmapShard {
    file_path: String,
    num_cells: u64,
    mmap: MmapMut,
    next_offset: u64,
}

impl MmapShard {
    /// Open or create a shard file at `path` prepared for `num_cells`.
    pub fn create_or_open(path: &str, num_cells: u64) -> io::Result<Self> {
        let file_path = path.to_string();
        let exists = Path::new(path).exists();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        // Pre-size: header + offset table + generous data region.
        // Estimate 256 bytes per cell average.
        let estimated_size = HEADER_SIZE as u64 + num_cells * OFFSET_RECORD_SIZE as u64 + num_cells * 256;
        file.set_len(estimated_size)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        if exists && Self::valid_header(&mmap) {
            let header = Self::read_header(&mmap);
            let next_offset = Self::compute_next_offset(&mmap, header.num_cells);
            return Ok(Self {
                file_path,
                num_cells: header.num_cells,
                mmap,
                next_offset,
            });
        }

        // Initialize fresh header.
        let created_at_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self::write_header(&mut mmap, num_cells, created_at_ns)?;

        // Mlock the offset table region (never swap).
        let table_size = (num_cells as usize) * OFFSET_RECORD_SIZE;
        unsafe {
            let table_ptr = mmap.as_ptr().add(HEADER_SIZE) as *const libc::c_void;
            libc::mlock(table_ptr, table_size);
        }

        Ok(Self {
            file_path,
            num_cells,
            mmap,
            next_offset: HEADER_SIZE as u64 + num_cells * OFFSET_RECORD_SIZE as u64,
        })
    }

    /// Write a cell's data to the shard. Returns the byte offset of the data.
    pub fn write_cell(&mut self, cell_id: u64, data: &[u8]) -> io::Result<u64> {
        if cell_id >= self.num_cells {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cell_id exceeds shard capacity",
            ));
        }

        let offset = self.next_offset;
        let aligned_len = Self::align8(data.len() as u64);

        // Ensure the mmap has room.
        let required = offset + aligned_len;
        if required > self.mmap.len() as u64 {
            self.grow(required)?;
        }

        // Write cell data.
        self.mmap[offset as usize..offset as usize + data.len()].copy_from_slice(data);
        // Zero pad to 8-byte alignment.
        for i in data.len()..aligned_len as usize {
            self.mmap[offset as usize + i] = 0;
        }

        // Update offset table.
        let crc = crc32(data);
        let rec = OffsetRecord {
            offset,
            length: data.len() as u64,
            crc32: crc,
        };
        Self::write_offset_record(&mut self.mmap, self.num_cells, cell_id, rec)?;

        // Flush data and offset table to disk.
        self.mmap.flush_range(offset as usize, aligned_len as usize)?;
        let table_start = HEADER_SIZE + (cell_id as usize) * OFFSET_RECORD_SIZE;
        self.mmap.flush_range(table_start, OFFSET_RECORD_SIZE)?;

        self.next_offset = offset + aligned_len;
        Ok(offset)
    }

    /// Read a cell's data. Verifies CRC32 and returns None on corruption.
    pub fn read_cell(&self, cell_id: u64) -> Option<&[u8]> {
        if cell_id >= self.num_cells {
            return None;
        }
        let rec = Self::read_offset_record(&self.mmap, self.num_cells, cell_id)?;
        if rec.length == 0 {
            return None;
        }
        let start = rec.offset as usize;
        let end = start + rec.length as usize;
        if end > self.mmap.len() {
            return None;
        }
        let data = &self.mmap[start..end];
        let actual_crc = crc32(data);
        if actual_crc != rec.crc32 {
            // Corruption detected. v0.1 logs via eprintln; v0.2 should trip circuit breaker.
            eprintln!(
                "[mmap_shard] CRC mismatch for cell {}: expected {:08x}, got {:08x}",
                cell_id, rec.crc32, actual_crc
            );
            return None;
        }
        Some(data)
    }

    /// Prefetch hints for hot cells.
    pub fn prefetch_cells(&self, cell_ids: &[u64]) {
        for &id in cell_ids {
            if let Some(rec) = Self::read_offset_record(&self.mmap, self.num_cells, id) {
                if rec.length > 0 {
                    unsafe {
                        libc::madvise(
                            self.mmap.as_ptr().add(rec.offset as usize) as *mut libc::c_void,
                            rec.length as usize,
                            libc::MADV_WILLNEED,
                        );
                    }
                }
            }
        }
    }

    pub fn cell_count(&self) -> u64 {
        self.num_cells
    }

    fn grow(&mut self, required: u64) -> io::Result<()> {
        let new_len = (required.max(self.mmap.len() as u64 * 2)).max(4096);
        // Drop old mmap, resize file, remap.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)?;
        file.set_len(new_len)?;
        // Need to re-mmap mutably.
        let new_mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        // Replace self.mmap. The old one is dropped here.
        // This is safe because we have no outstanding borrows.
        let _old = std::mem::replace(&mut self.mmap, new_mmap);
        Ok(())
    }

    fn align8(n: u64) -> u64 {
        (n + 7) & !7
    }

    fn valid_header(mmap: &[u8]) -> bool {
        if mmap.len() < HEADER_SIZE {
            return false;
        }
        if &mmap[0..8] != MAGIC {
            return false;
        }
        let stored_crc = u32::from_le_bytes([mmap[24], mmap[25], mmap[26], mmap[27]]);
        let actual_crc = crc32(&mmap[0..24]);
        stored_crc == actual_crc
    }

    fn read_header(mmap: &[u8]) -> Header {
        let num_cells = u64::from_le_bytes([
            mmap[8], mmap[9], mmap[10], mmap[11], mmap[12], mmap[13], mmap[14], mmap[15],
        ]);
        let created_at_ns = u64::from_le_bytes([
            mmap[16], mmap[17], mmap[18], mmap[19], mmap[20], mmap[21], mmap[22], mmap[23],
        ]);
        Header {
            num_cells,
            created_at_ns,
        }
    }

    fn write_header(mmap: &mut [u8], num_cells: u64, created_at_ns: u64) -> io::Result<()> {
        mmap[0..8].copy_from_slice(MAGIC);
        mmap[8..16].copy_from_slice(&num_cells.to_le_bytes());
        mmap[16..24].copy_from_slice(&created_at_ns.to_le_bytes());
        let crc = crc32(&mmap[0..24]);
        mmap[24..28].copy_from_slice(&crc.to_le_bytes());
        mmap[28..32].fill(0);
        Ok(())
    }

    fn read_offset_record(mmap: &[u8], num_cells: u64, cell_id: u64) -> Option<OffsetRecord> {
        if cell_id >= num_cells {
            return None;
        }
        let base = HEADER_SIZE + (cell_id as usize) * OFFSET_RECORD_SIZE;
        if base + OFFSET_RECORD_SIZE > mmap.len() {
            return None;
        }
        let offset = u64::from_le_bytes([
            mmap[base],
            mmap[base + 1],
            mmap[base + 2],
            mmap[base + 3],
            mmap[base + 4],
            mmap[base + 5],
            mmap[base + 6],
            mmap[base + 7],
        ]);
        let length = u64::from_le_bytes([
            mmap[base + 8],
            mmap[base + 9],
            mmap[base + 10],
            mmap[base + 11],
            mmap[base + 12],
            mmap[base + 13],
            mmap[base + 14],
            mmap[base + 15],
        ]);
        let crc32 = u32::from_le_bytes([mmap[base + 16], mmap[base + 17], mmap[base + 18], mmap[base + 19]]);
        Some(OffsetRecord {
            offset,
            length,
            crc32,
        })
    }

    fn write_offset_record(
        mmap: &mut [u8],
        num_cells: u64,
        cell_id: u64,
        rec: OffsetRecord,
    ) -> io::Result<()> {
        if cell_id >= num_cells {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cell_id exceeds shard capacity",
            ));
        }
        let base = HEADER_SIZE + (cell_id as usize) * OFFSET_RECORD_SIZE;
        mmap[base..base + 8].copy_from_slice(&rec.offset.to_le_bytes());
        mmap[base + 8..base + 16].copy_from_slice(&rec.length.to_le_bytes());
        mmap[base + 16..base + 20].copy_from_slice(&rec.crc32.to_le_bytes());
        mmap[base + 20..base + 24].fill(0);
        Ok(())
    }

    fn compute_next_offset(mmap: &[u8], num_cells: u64) -> u64 {
        let mut max_end = HEADER_SIZE as u64 + num_cells * OFFSET_RECORD_SIZE as u64;
        for id in 0..num_cells {
            if let Some(rec) = Self::read_offset_record(mmap, num_cells, id) {
                let end = rec.offset + rec.length;
                if end > max_end {
                    max_end = end;
                }
            }
        }
        max_end
    }
}

impl Drop for MmapShard {
    fn drop(&mut self) {
        let table_ptr = unsafe { self.mmap.as_ptr().add(HEADER_SIZE) as *const libc::c_void };
        let table_size = (self.num_cells as usize) * OFFSET_RECORD_SIZE;
        unsafe {
            libc::munlock(table_ptr, table_size);
        }
    }
}

struct Header {
    num_cells: u64,
    created_at_ns: u64,
}

fn crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let tmp = tempfile_path("test_roundtrip");
        {
            let mut shard = MmapShard::create_or_open(&tmp, 16).unwrap();
            let data = b"hello yellow phoenix";
            shard.write_cell(3, data).unwrap();
            let read = shard.read_cell(3).unwrap();
            assert_eq!(read, data);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_multiple_cells() {
        let tmp = tempfile_path("test_multiple_cells");
        {
            let mut shard = MmapShard::create_or_open(&tmp, 16).unwrap();
            for i in 0..8u64 {
                let data = format!("cell-{}", i).into_bytes();
                shard.write_cell(i, &data).unwrap();
            }
            for i in 0..8u64 {
                let expected = format!("cell-{}", i).into_bytes();
                assert_eq!(shard.read_cell(i).unwrap(), expected.as_slice());
            }
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_empty_cell_returns_none() {
        let tmp = tempfile_path("test_empty_cell");
        {
            let shard = MmapShard::create_or_open(&tmp, 4).unwrap();
            assert!(shard.read_cell(2).is_none());
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_out_of_range() {
        let tmp = tempfile_path("test_out_of_range");
        {
            let mut shard = MmapShard::create_or_open(&tmp, 4).unwrap();
            assert!(shard.write_cell(4, b"x").is_err());
            assert!(shard.read_cell(4).is_none());
        }
        std::fs::remove_file(&tmp).ok();
    }

    fn tempfile_path(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("{}_{}.bin", name, std::process::id()));
        p.to_string_lossy().to_string()
    }
}
