// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! wiring_registry_scanner.rs — scan project files and produce RegistryState.

use crate::wiring_registry_crypto::sha256_hex;
use crate::wiring_registry_types::{FileEntry, RegistryState};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Patterns to ignore during scanning.
const IGNORE_PATTERNS: &[&str] = &[
    ".git",
    "target",
    "__pycache__",
    ".venv",
    ".venv_pdf",
    ".aider.tags.cache.v4",
    ".hypothesis",
    ".pytest_cache",
    "_tmp_wiring",
    "src_old_scattered",
];

/// File extensions that matter to the registry.
const TRACKED_EXTENSIONS: &[&str] = &["rs", "py", "toml", "sh"];

/// Scan a directory tree and return a RegistryState.
pub fn scan_project(root: impl AsRef<Path>) -> std::io::Result<RegistryState> {
    let root = root.as_ref();
    let mut state = RegistryState {
        version: env!("CARGO_PKG_VERSION").to_string(),
        git_hash: git_hash(),
        scanned_at: iso_now(),
        ..RegistryState::default()
    };

    let mut files = Vec::new();
    scan_dir(root, root, &mut files)?;
    state.files = files;

    // Basic metadata.
    state.metadata.insert("root".into(), root.canonicalize()?.to_string_lossy().into_owned());
    state.metadata.insert("tracked_extensions".into(), TRACKED_EXTENSIONS.join(","));

    Ok(state)
}

fn scan_dir(base: &Path, dir: &Path, out: &mut Vec<FileEntry>) -> std::io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();

        if should_ignore(&name) {
            continue;
        }

        if path.is_dir() {
            scan_dir(base, &path, out)?;
        } else if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext = ext.to_string_lossy();
                if TRACKED_EXTENSIONS.contains(&ext.as_ref()) {
                    out.push(file_entry(base, &path)?);
                }
            }
        }
    }
    Ok(())
}

fn should_ignore(name: &str) -> bool {
    IGNORE_PATTERNS.contains(&name)
}

fn file_entry(base: &Path, path: &Path) -> std::io::Result<FileEntry> {
    let data = fs::read(path)?;
    let meta = fs::metadata(path)?;
    let rel = path.strip_prefix(base).unwrap_or(path).to_string_lossy().into_owned();
    let mtime = meta
        .modified()
        .unwrap_or_else(|_| SystemTime::UNIX_EPOCH)
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    Ok(FileEntry {
        path: rel,
        size: meta.len(),
        sha256: sha256_hex(&data),
        mtime: format!("{}", mtime),
    })
}

fn git_hash() -> String {
    let git_dir = PathBuf::from(".git");
    if !git_dir.exists() {
        return "unknown".into();
    }
    let head = fs::read_to_string(git_dir.join("HEAD")).unwrap_or_default();
    if head.starts_with("ref: ") {
        let ref_path = head.trim().strip_prefix("ref: ").unwrap_or("");
        fs::read_to_string(git_dir.join(ref_path))
            .unwrap_or_default()
            .trim()
            .to_string()
    } else {
        head.trim().to_string()
    }
}

fn iso_now() -> String {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}T{:02}:{:02}:{:02}Z", now / 86400, (now % 86400) / 3600, (now % 3600) / 60, now % 60)
}

/// Compute differences between two scanned states.
pub fn diff_states(old: &RegistryState, new: &RegistryState) -> Vec<crate::wiring_registry_types::RegistryChange> {
    use crate::wiring_registry_types::{ChangeType, RegistryChange};
    let mut changes = Vec::new();
    let old_map: HashMap<&str, &FileEntry> = old.files.iter().map(|f| (f.path.as_str(), f)).collect();
    let new_map: HashMap<&str, &FileEntry> = new.files.iter().map(|f| (f.path.as_str(), f)).collect();

    for (path, entry) in &new_map {
        match old_map.get(path) {
            None => changes.push(RegistryChange {
                change_type: ChangeType::Added,
                category: "file".into(),
                key: path.to_string(),
                old_value: None,
                new_value: Some(format!("{} {}", entry.size, &entry.sha256[..16])),
            }),
            Some(old_entry) if old_entry.sha256 != entry.sha256 => {
                let old_short = old_entry.sha256.chars().take(16).collect::<String>();
                let new_short = entry.sha256.chars().take(16).collect::<String>();
                changes.push(RegistryChange {
                    change_type: ChangeType::Modified,
                    category: "file".into(),
                    key: path.to_string(),
                    old_value: Some(format!("{} {}", old_entry.size, old_short)),
                    new_value: Some(format!("{} {}", entry.size, new_short)),
                })
            }
            _ => {}
        }
    }

    for path in old_map.keys() {
        if !new_map.contains_key(path) {
            changes.push(RegistryChange {
                change_type: ChangeType::Removed,
                category: "file".into(),
                key: path.to_string(),
                old_value: Some(path.to_string()),
                new_value: None,
            });
        }
    }

    changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn scan_detects_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("test.rs"), b"fn main() {}").unwrap();
        fs::write(root.join("test.py"), b"print('hi')").unwrap();
        fs::write(root.join("readme.md"), b"# ignored").unwrap();

        let state = scan_project(root).unwrap();
        let paths: Vec<_> = state.files.iter().map(|f| f.path.clone()).collect();
        assert!(paths.contains(&"test.rs".into()));
        assert!(paths.contains(&"test.py".into()));
        assert!(!paths.contains(&"readme.md".into()));
    }

    #[test]
    fn diff_detects_modification() {
        let mut old = RegistryState::default();
        old.files.push(FileEntry {
            path: "a.rs".into(),
            size: 10,
            sha256: "abc".into(),
            mtime: "0".into(),
        });
        let mut new = old.clone();
        new.files[0].sha256 = "def".into();

        let changes = diff_states(&old, &new);
        assert_eq!(changes.len(), 1);
        assert!(matches!(changes[0].change_type, crate::wiring_registry_types::ChangeType::Modified));
    }
}
