// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! wiring_registry.rs — Phoenix Wiring Registry: scan, backup, verify, restore.

use crate::wiring_registry_crypto::{decrypt_secure, derive_key, encrypt_secure, sha256_hex};
use crate::wiring_registry_scanner::{diff_states, scan_project};
use crate::wiring_registry_types::{HealthReport, RegistryChange, RegistryState, VerificationResult};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const REGISTRY_DIR: &str = "data/registry";
pub const CURRENT_FILE: &str = "registry_current.json";
pub const BACKUP_FILE: &str = "registry_backup.json";
pub const BACKUP_HASH_FILE: &str = "registry_backup.json.sha256";
pub const SECURE_FILE: &str = "registry_secure.bin";
pub const AUDIT_FILE: &str = "registry_audit.log";

pub struct WiringRegistry {
    pub current: RegistryState,
    pub root: PathBuf,
    pub key: [u8; 32],
}

impl WiringRegistry {
    /// Create a new registry for the given project root.
    pub fn new(root: impl AsRef<Path>) -> Self {
        let key_material = std::env::var("YP_REGISTRY_KEY").unwrap_or_else(|_| "yellow_phoenix_default_key".into());
        Self {
            current: RegistryState::default(),
            root: root.as_ref().to_path_buf(),
            key: derive_key(&key_material),
        }
    }

    /// Scan the project and update the current state.
    pub fn scan(&mut self) -> io::Result<()> {
        self.current = scan_project(&self.root)?;
        Ok(())
    }

    /// Save the current state as the standard JSON backup + SHA-256 checksum.
    pub fn save_backup(&self) -> io::Result<()> {
        let dir = self.registry_dir();
        fs::create_dir_all(&dir)?;
        let json = serde_json::to_string_pretty(&self.current).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let hash = sha256_hex(json.as_bytes());

        fs::write(dir.join(BACKUP_FILE), json)?;
        fs::write(dir.join(BACKUP_HASH_FILE), hash)?;
        Ok(())
    }

    /// Save the current state as the encrypted + HMAC secure backup.
    pub fn save_secure(&self) -> io::Result<()> {
        let dir = self.registry_dir();
        fs::create_dir_all(&dir)?;
        let json = serde_json::to_string(&self.current).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let sealed = encrypt_secure(json.as_bytes(), &self.key).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        fs::write(dir.join(SECURE_FILE), sealed)?;
        Ok(())
    }

    /// Save current state, backup, and secure backup in one call.
    pub fn save_all(&self) -> io::Result<()> {
        self.save_backup()?;
        self.save_secure()?;
        self.log_change("save_all", "Registry saved to backup and secure store")?;
        Ok(())
    }

    /// Verify the current state against the standard backup.
    pub fn verify(&self) -> io::Result<VerificationResult> {
        let dir = self.registry_dir();
        let backup_path = dir.join(BACKUP_FILE);
        let hash_path = dir.join(BACKUP_HASH_FILE);

        if !backup_path.exists() || !hash_path.exists() {
            return Ok(VerificationResult::Critical("No backup exists".into()));
        }

        let backup_json = fs::read_to_string(&backup_path)?;
        let expected_hash = fs::read_to_string(&hash_path)?.trim().to_string();
        let current_json = serde_json::to_string(&self.current).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let current_hash = sha256_hex(current_json.as_bytes());

        if current_hash == expected_hash {
            return Ok(VerificationResult::Clean);
        }

        let backup_state: RegistryState = serde_json::from_str(&backup_json).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let changes = diff_states(&backup_state, &self.current);

        // Also verify secure backup can still be decrypted (tamper check).
        let secure_path = dir.join(SECURE_FILE);
        if secure_path.exists() {
            let secure_data = fs::read(&secure_path)?;
            if decrypt_secure(&secure_data, &self.key).is_err() {
                return Ok(VerificationResult::Critical("Secure backup is corrupted or key is wrong".into()));
            }
        }

        Ok(VerificationResult::Damaged {
            current_hash,
            expected_hash,
            details: changes,
        })
    }

    /// Restore current state from the secure backup.
    pub fn restore_secure(&mut self) -> io::Result<()> {
        let dir = self.registry_dir();
        let secure_path = dir.join(SECURE_FILE);
        if !secure_path.exists() {
            return Err(io::Error::new(io::ErrorKind::NotFound, "secure backup not found"));
        }
        let secure_data = fs::read(&secure_path)?;
        let decrypted = decrypt_secure(&secure_data, &self.key).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.current = serde_json::from_slice(&decrypted).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        self.save_backup()?;
        self.log_change("restore_secure", "Registry restored from secure backup")?;
        Ok(())
    }

    /// Run a health check: scan current state, compare to backup, report changes.
    pub fn health_check(&mut self) -> io::Result<HealthReport> {
        let new_state = scan_project(&self.root)?;
        let changes = diff_states(&self.current, &new_state);
        self.current = new_state;

        let issues: Vec<String> = changes
            .iter()
            .map(|c| format!("{:?} {}: {}", c.change_type, c.category, c.key))
            .collect();

        Ok(HealthReport {
            timestamp: iso_now(),
            healthy: issues.is_empty(),
            issues,
            changes,
        })
    }

    /// Append a change to the audit log.
    pub fn log_change(&self, event_type: &str, details: &str) -> io::Result<()> {
        let dir = self.registry_dir();
        fs::create_dir_all(&dir)?;
        let path = dir.join(AUDIT_FILE);
        let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
        let entry = format!(
            "{{\"timestamp\":\"{}\",\"event_type\":\"{}\",\"details\":\"{}\",\"version\":\"{}\"}}\n",
            iso_now(),
            event_type,
            details.replace('"', "\\\""),
            env!("CARGO_PKG_VERSION")
        );
        file.write_all(entry.as_bytes())?;
        Ok(())
    }

    fn registry_dir(&self) -> PathBuf {
        self.root.join(REGISTRY_DIR)
    }
}

fn iso_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        1970 + now / 31_557_600,
        (now % 31_557_600) / 2_628_000 + 1,
        (now % 2_628_000) / 86_400 + 1,
        (now % 86_400) / 3600,
        (now % 3600) / 60,
        now % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backup_restore_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("YP_REGISTRY_KEY", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        let mut reg = WiringRegistry::new(tmp.path());
        reg.scan().unwrap();
        reg.save_all().unwrap();

        let mut reg2 = WiringRegistry::new(tmp.path());
        reg2.restore_secure().unwrap();
        assert_eq!(reg.current, reg2.current);
    }

    #[test]
    fn damage_detected() {
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("YP_REGISTRY_KEY", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        let mut reg = WiringRegistry::new(tmp.path());
        reg.scan().unwrap();
        reg.save_all().unwrap();

        // Modify the project.
        std::fs::write(tmp.path().join("new_file.rs"), b"// changed").unwrap();
        reg.scan().unwrap();

        let result = reg.verify().unwrap();
        assert!(matches!(result, VerificationResult::Damaged { .. }));
    }
}
