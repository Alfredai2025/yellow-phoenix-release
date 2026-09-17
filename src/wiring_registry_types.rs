// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! wiring_registry_types.rs — data structures for the Phoenix Wiring Registry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single tracked source file.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub mtime: String,
}

/// A tracked function / FFI export.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionEntry {
    pub name: String,
    pub file: String,
    pub line: u32,
    pub is_ffi: bool,
    pub python_binding: Option<String>,
}

/// A connection between two system parts.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConnectionEntry {
    pub source: String,
    pub target: String,
    pub conn_type: String,
    pub data_format: String,
}

/// A feed pipe state.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipeEntry {
    pub path: String,
    pub permissions: u32,
    pub last_message: String,
    pub healthy: bool,
}

/// A tracked table / data file.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TableEntry {
    pub path: String,
    pub size: u64,
    pub version: String,
    pub sha256: String,
}

/// Build artifact tracking.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub build_time: String,
}

/// Full registry snapshot.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegistryState {
    pub version: String,
    pub git_hash: String,
    pub scanned_at: String,
    pub files: Vec<FileEntry>,
    pub functions: Vec<FunctionEntry>,
    pub connections: Vec<ConnectionEntry>,
    pub feed_pipes: Vec<PipeEntry>,
    pub tables: Vec<TableEntry>,
    pub artifacts: Vec<ArtifactEntry>,
    pub ffi_exports: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// A single detected change between two registry states.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChangeType {
    Added,
    Removed,
    Modified,
    Moved,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegistryChange {
    pub change_type: ChangeType,
    pub category: String,
    pub key: String,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
}

/// Result of verifying current state against backups.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationResult {
    Clean,
    Damaged {
        current_hash: String,
        expected_hash: String,
        details: Vec<RegistryChange>,
    },
    Critical(String),
}

/// Health report produced by periodic scans.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HealthReport {
    pub timestamp: String,
    pub healthy: bool,
    pub issues: Vec<String>,
    pub changes: Vec<RegistryChange>,
}
