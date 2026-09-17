//! Integration tests for the Phoenix Wiring Registry.

use pams::wiring_registry::{WiringRegistry, REGISTRY_DIR};
use std::fs;

fn hex_key() {
    std::env::set_var(
        "YP_REGISTRY_KEY",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    );
}

#[test]
fn scan_finds_rust_and_python_files() {
    hex_key();
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("main.rs"), b"fn main() {}").unwrap();
    fs::write(tmp.path().join("bridge.py"), b"print('ok')").unwrap();

    let mut reg = WiringRegistry::new(tmp.path());
    reg.scan().unwrap();

    let paths: Vec<_> = reg.current.files.iter().map(|f| f.path.clone()).collect();
    assert!(paths.contains(&"main.rs".into()));
    assert!(paths.contains(&"bridge.py".into()));
}

#[test]
fn backup_and_restore_secure() {
    hex_key();
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("lib.rs"), b"pub mod x;").unwrap();

    let mut reg = WiringRegistry::new(tmp.path());
    reg.scan().unwrap();
    reg.save_all().unwrap();

    assert!(tmp.path().join(REGISTRY_DIR).join("registry_backup.json").exists());
    assert!(tmp.path().join(REGISTRY_DIR).join("registry_secure.bin").exists());

    let mut reg2 = WiringRegistry::new(tmp.path());
    reg2.restore_secure().unwrap();
    assert_eq!(reg.current, reg2.current);
}

#[test]
fn damage_detection_triggers_on_file_change() {
    hex_key();
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("tracked.rs"), b"v1").unwrap();

    let mut reg = WiringRegistry::new(tmp.path());
    reg.scan().unwrap();
    reg.save_all().unwrap();

    // Change file contents.
    fs::write(tmp.path().join("tracked.rs"), b"v2").unwrap();
    reg.scan().unwrap();

    let result = reg.verify().unwrap();
    assert!(
        matches!(result, pams::wiring_registry_types::VerificationResult::Damaged { .. }),
        "expected damaged verification result"
    );
}

#[test]
fn tampered_backup_detected() {
    hex_key();
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("x.rs"), b"x").unwrap();

    let mut reg = WiringRegistry::new(tmp.path());
    reg.scan().unwrap();
    reg.save_all().unwrap();

    // Tamper with standard backup while secure backup stays clean.
    let backup_path = tmp.path().join(REGISTRY_DIR).join("registry_backup.json");
    let mut content = fs::read_to_string(&backup_path).unwrap();
    content.push_str("\n");
    fs::write(&backup_path, content).unwrap();

    let result = reg.verify().unwrap();
    assert!(
        matches!(result, pams::wiring_registry_types::VerificationResult::Damaged { .. }),
        "expected damaged result after backup tamper"
    );
}

#[test]
fn health_check_reports_changes() {
    hex_key();
    let tmp = tempfile::tempdir().unwrap();
    fs::write(tmp.path().join("a.rs"), b"a").unwrap();

    let mut reg = WiringRegistry::new(tmp.path());
    reg.scan().unwrap();

    fs::write(tmp.path().join("b.rs"), b"b").unwrap();
    let report = reg.health_check().unwrap();

    assert!(!report.healthy);
    assert!(report.changes.iter().any(|c| c.key == "b.rs"));
}
