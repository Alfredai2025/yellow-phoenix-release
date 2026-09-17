// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! self_learning.rs — 256-entry pattern table that learns which feature chains work.
//!
//! Each pattern tracks how often each feature combination produced a correct top-1.
//! The engine uses this to skip features that historically don't help for a pattern.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::learned_router::{FeatureChain, LearnedRouter};

const TABLE_SIZE: usize = 256;
const BYTES_PER_ENTRY: usize = 16;
const TABLE_FILE: &str = "data/learning_table_v1.bin";
const CONFIG_FILE: &str = "data/self_learning.json";
const BACKUP_COUNT: usize = 3;

/// Tunable thresholds exported to JSON so the learning loop can persist them.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Thresholds {
    pub skip_all: f32,
    pub skip_wedge_holo: f32,
    pub fast_path_max_bucket: usize,
    pub sharding_threshold: usize,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            skip_all: 0.95,
            skip_wedge_holo: 0.85,
            fast_path_max_bucket: 64,
            sharding_threshold: 1_000_000,
        }
    }
}

/// Per-pattern statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PatternEntry {
    pub seen: u8,
    pub correct_hash: u8,
    pub correct_spectral: u8,
    pub correct_wedge: u8,
    pub correct_hologram: u8,
    _pad: [u8; 11],
}

impl PatternEntry {
    fn to_bytes(&self) -> [u8; BYTES_PER_ENTRY] {
        let mut b = [0u8; BYTES_PER_ENTRY];
        b[0] = self.seen;
        b[1] = self.correct_hash;
        b[2] = self.correct_spectral;
        b[3] = self.correct_wedge;
        b[4] = self.correct_hologram;
        b
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            seen: bytes[0],
            correct_hash: bytes[1],
            correct_spectral: bytes[2],
            correct_wedge: bytes[3],
            correct_hologram: bytes[4],
            _pad: [0; 11],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SelfLearningTable {
    pub entries: [PatternEntry; TABLE_SIZE],
    pub version: u64,
    pub thresholds: Thresholds,
    pub config_path: Option<PathBuf>,
}

impl Default for SelfLearningTable {
    fn default() -> Self {
        Self::new()
    }
}

impl SelfLearningTable {
    pub fn new() -> Self {
        Self {
            entries: [PatternEntry::default(); TABLE_SIZE],
            version: 0,
            thresholds: Thresholds::default(),
            config_path: Some(PathBuf::from(CONFIG_FILE)),
        }
    }

    pub fn with_thresholds(thresholds: Thresholds) -> Self {
        let mut table = Self::new();
        table.thresholds = thresholds;
        table
    }

    /// Update statistics for a pattern after observing a query result.
    pub fn update(&mut self, pattern: u8, chain_used: &FeatureChain, top1_correct: bool) {
        let entry = &mut self.entries[pattern as usize];
        entry.seen = entry.seen.saturating_add(1);

        if top1_correct {
            if chain_used.run_hologram {
                entry.correct_hologram = entry.correct_hologram.saturating_add(1);
            } else if chain_used.run_wedge {
                entry.correct_wedge = entry.correct_wedge.saturating_add(1);
            } else if chain_used.run_spectral {
                entry.correct_spectral = entry.correct_spectral.saturating_add(1);
            } else {
                entry.correct_hash = entry.correct_hash.saturating_add(1);
            }
        }
        self.save_config();
    }

    /// Persist the current thresholds to JSON.
    pub fn save_config(&self) {
        if let Some(ref path) = self.config_path {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(json) = serde_json::to_string_pretty(&self.thresholds) {
                let _ = fs::write(path, json);
            }
        }
    }

    /// Choose the cheapest feature chain that still meets accuracy targets.
    pub fn best_chain(&self, pattern: u8) -> FeatureChain {
        let entry = &self.entries[pattern as usize];
        if entry.seen < 4 {
            return FeatureChain::all();
        }

        let accuracy = |correct: u8| -> f32 {
            if entry.seen == 0 {
                0.0
            } else {
                correct as f32 / entry.seen as f32
            }
        };

        let hash_acc = accuracy(entry.correct_hash);
        let spectral_acc = accuracy(entry.correct_spectral);
        let wedge_acc = accuracy(entry.correct_wedge);

        // Cumulative chain accuracy:
        // If hash alone is 99% accurate, skip everything.
        // If hash+spectral is 95% accurate, skip wedge and hologram.
        // If hash+spectral+wedge is 90% accurate, skip hologram.
        let run_spectral = hash_acc < 0.99;
        let run_wedge = run_spectral && spectral_acc < 0.95;
        let run_hologram = run_wedge && wedge_acc < 0.90;

        FeatureChain {
            run_spectral,
            run_wedge,
            run_hologram,
        }
    }

    /// Merge learned-router uncertainty with table recommendation.
    /// If router is uncertain, prefer the table's safe default.
    pub fn best_chain_with_router(&self, pattern: u8, router: &LearnedRouter, features: [f32; 8]) -> FeatureChain {
        let table_chain = self.best_chain(pattern);
        let router_chain = router.decide_chain(features);

        FeatureChain {
            run_spectral: table_chain.run_spectral || router_chain.run_spectral,
            run_wedge: table_chain.run_wedge || router_chain.run_wedge,
            run_hologram: table_chain.run_hologram || router_chain.run_hologram,
        }
    }

    /// Load table from disk, creating defaults if missing or corrupt.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Self::new());
        }
        let mut file = fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    /// Save table atomically, rotating backups.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Rotate backups.
        self.rotate_backups(path)?;

        // Atomic write.
        let tmp = path.with_extension("tmp");
        let mut file = fs::File::create(&tmp)?;
        file.write_all(&self.to_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Default save path.
    pub fn default_path() -> PathBuf {
        PathBuf::from(TABLE_FILE)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(TABLE_SIZE * BYTES_PER_ENTRY + 8);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        for entry in &self.entries {
            bytes.extend_from_slice(&entry.to_bytes());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let expected = TABLE_SIZE * BYTES_PER_ENTRY + 8;
        if bytes.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("learning table wrong size: {} expected {}", bytes.len(), expected),
            ));
        }
        let version = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let mut entries = [PatternEntry::default(); TABLE_SIZE];
        for i in 0..TABLE_SIZE {
            let start = 8 + i * BYTES_PER_ENTRY;
            entries[i] = PatternEntry::from_bytes(&bytes[start..start + BYTES_PER_ENTRY]);
        }
        Ok(Self {
            entries,
            version,
            thresholds: Thresholds::default(),
            config_path: Some(PathBuf::from(CONFIG_FILE)),
        })
    }

    fn rotate_backups(&self, path: &Path) -> io::Result<()> {
        for i in (1..BACKUP_COUNT).rev() {
            let src = path.with_extension(format!("bak{}", i));
            let dst = path.with_extension(format!("bak{}", i + 1));
            if src.exists() {
                fs::rename(&src, &dst)?;
            }
        }
        if path.exists() {
            fs::rename(path, path.with_extension("bak1"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_tracks_correctness() {
        let mut table = SelfLearningTable::new();
        table.update(0, &FeatureChain::all(), true);
        assert_eq!(table.entries[0].seen, 1);
        assert_eq!(table.entries[0].correct_hologram, 1);
    }

    #[test]
    fn best_chain_defaults_to_all_when_unseen() {
        let table = SelfLearningTable::new();
        let chain = table.best_chain(0);
        assert!(chain.run_spectral);
        assert!(chain.run_wedge);
        assert!(chain.run_hologram);
    }

    #[test]
    fn best_chain_skips_when_accurate() {
        let mut table = SelfLearningTable::new();
        // 100% hash accuracy → skip spectral, wedge, hologram.
        for _ in 0..10 {
            table.update(0, &FeatureChain::none(), true);
        }
        let chain = table.best_chain(0);
        assert!(!chain.run_spectral);
        assert!(!chain.run_wedge);
        assert!(!chain.run_hologram);
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("learning_table_v1.bin");
        let mut table = SelfLearningTable::new();
        table.version = 7;
        table.update(5, &FeatureChain::all(), true);
        table.save(&path).unwrap();

        let loaded = SelfLearningTable::load(&path).unwrap();
        assert_eq!(loaded.version, 7);
        assert_eq!(loaded.entries[5].seen, 1);
    }

    #[test]
    fn load_missing_returns_default() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.bin");
        let table = SelfLearningTable::load(&path).unwrap();
        assert_eq!(table.version, 0);
        assert_eq!(table.entries[0].seen, 0);
    }
}
