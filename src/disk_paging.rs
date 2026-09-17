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
//! Disk paging for cold cells — serialize HotCell to disk
//!
//! Architecture:
//! - HotCell (papers + threads) written to disk when memory pressure
//! - Promoted back to RAM after 3 accesses

use std::collections::HashMap;
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::dynamic_mesh::{HotCell, DynamicPaper};

const MAGIC: u32 = 0xC011_FACE;
const VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct ColdCellMeta {
    pub access_count: u32,
    pub last_access: Instant,
}

pub struct DiskPager {
    cold_dir: PathBuf,
    cold_meta: HashMap<u64, ColdCellMeta>,
}

impl DiskPager {
    pub fn new(cold_dir: &Path) -> io::Result<Self> {
        create_dir_all(cold_dir)?;
        Ok(Self {
            cold_dir: cold_dir.to_path_buf(),
            cold_meta: HashMap::new(),
        })
    }

    pub fn write_cold(&self, cell_id: u64, cell: &HotCell) -> io::Result<()> {
        let path = self.cold_dir.join(format!("cell_{}.bin", cell_id));
        let mut file = OpenOptions::new()
            .write(true).create(true).truncate(true).open(&path)?;

        file.write_all(&MAGIC.to_le_bytes())?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&cell_id.to_le_bytes())?;
        file.write_all(&cell.pulse_energy.to_le_bytes())?;

        let thread_count = cell.thread_ids.len() as u32;
        file.write_all(&thread_count.to_le_bytes())?;
        for &tid in &cell.thread_ids {
            file.write_all(&tid.to_le_bytes())?;
        }

        let paper_count = cell.papers.len() as u32;
        file.write_all(&paper_count.to_le_bytes())?;
        for paper in &cell.papers {
            file.write_all(&paper.entity_id.to_le_bytes())?;
            file.write_all(&paper.power.to_le_bytes())?;
            file.write_all(&paper.mesh_time.to_le_bytes())?;
            file.write_all(&paper.column_depth.to_le_bytes())?;
        }

        file.sync_all()?;
        Ok(())
    }

    pub fn read_cold(&self, cell_id: u64) -> io::Result<HotCell> {
        let path = self.cold_dir.join(format!("cell_{}.bin", cell_id));
        let mut file = File::open(&path)?;

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        file.read_exact(&mut buf4)?;
        if u32::from_le_bytes(buf4) != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        file.read_exact(&mut buf4)?;
        if u32::from_le_bytes(buf4) != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad version"));
        }
        file.read_exact(&mut buf8)?;
        let _stored_id = u64::from_le_bytes(buf8);

        file.read_exact(&mut buf4)?;
        let pulse_energy = f32::from_le_bytes(buf4);

        file.read_exact(&mut buf4)?;
        let thread_count = u32::from_le_bytes(buf4);
        let mut thread_ids = Vec::with_capacity(thread_count as usize);
        for _ in 0..thread_count {
            file.read_exact(&mut buf8)?;
            thread_ids.push(u64::from_le_bytes(buf8));
        }

        file.read_exact(&mut buf4)?;
        let paper_count = u32::from_le_bytes(buf4);
        let mut papers = Vec::with_capacity(paper_count as usize);
        for _ in 0..paper_count {
            file.read_exact(&mut buf8)?; let entity_id = u64::from_le_bytes(buf8);
            file.read_exact(&mut buf4)?; let power = f32::from_le_bytes(buf4);
            file.read_exact(&mut buf8)?; let mesh_time = u64::from_le_bytes(buf8);
            let mut buf2 = [0u8; 2];
            file.read_exact(&mut buf2)?; let column_depth = u16::from_le_bytes(buf2);
            papers.push(DynamicPaper { entity_id, power, mesh_time, column_depth });
        }

        Ok(HotCell { papers, pulse_energy, thread_ids })
    }

    pub fn mark_cold(&mut self, cell_id: u64) {
        self.cold_meta.insert(cell_id, ColdCellMeta {
            access_count: 0,
            last_access: Instant::now(),
        });
    }

    pub fn access(&mut self, cell_id: u64) -> bool {
        if let Some(meta) = self.cold_meta.get_mut(&cell_id) {
            meta.access_count += 1;
            meta.last_access = Instant::now();
            meta.access_count >= 3
        } else {
            false
        }
    }

    pub fn mark_hot(&mut self, cell_id: u64) {
        self.cold_meta.remove(&cell_id);
    }

    pub fn cold_count(&self) -> usize { self.cold_meta.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn roundtrip_cold_storage() {
        let dir = temp_dir().join("yp_test_cold");
        let _ = std::fs::remove_dir_all(&dir);
        let pager = DiskPager::new(&dir).unwrap();
        let cell = HotCell {
            papers: vec![DynamicPaper { entity_id: 1, power: 0.5, mesh_time: 100, column_depth: 5 }],
            pulse_energy: 0.1,
            thread_ids: vec![7, 8],
        };
        pager.write_cold(42, &cell).unwrap();
        let loaded = pager.read_cold(42).unwrap();
        assert_eq!(loaded.pulse_energy, 0.1);
        assert_eq!(loaded.papers.len(), 1);
        assert_eq!(loaded.papers[0].entity_id, 1);
        assert_eq!(loaded.thread_ids, vec![7, 8]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_cold_and_access() {
        let dir = temp_dir().join("yp_test_access");
        let _ = std::fs::remove_dir_all(&dir);
        let mut pager = DiskPager::new(&dir).unwrap();
        pager.mark_cold(99);
        assert!(!pager.access(99));
        assert!(!pager.access(99));
        assert!(pager.access(99));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
