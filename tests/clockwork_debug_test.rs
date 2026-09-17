use pams::clockwork::*;
#[cfg(feature = "clockwork_debug")]
use pams::clockwork::clockwork_debug::*;
#[cfg(feature = "clockwork_debug")]
use std::sync::Mutex;

// Env vars are process-global; these tests must run serially.
#[cfg(feature = "clockwork_debug")]
static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[cfg(feature = "clockwork_debug")]
fn test_debug_ring_captures() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CLOCKWORK_DEBUG_MODE", "RING");
    let engine = ClockworkEngine::new();
    let _state = engine.tick();
    let ring = &engine.debug_ring;
    assert!(ring.is_active());
    assert_eq!(ring.mode(), DebugMode::Ring);
}

#[test]
#[cfg(feature = "clockwork_debug")]
fn test_anomaly_checksum_failure() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CLOCKWORK_DEBUG_MODE", "ALERT");
    let engine = ClockworkEngine::new();
    let bad_bytes = [0u8, 1, 2, 0, 0, 0xFF]; // corrupt checksum
    let state = engine.tick();

    capture_corruption(&state, &engine, &engine.debug_ring, bad_bytes, "test");

    let latest = engine.debug_ring.latest(1);
    assert!(!latest.is_empty());
    assert_eq!(
        latest[0].corruption_detected.as_ref().unwrap().actual_checksum,
        0xFF
    );
}

#[test]
#[cfg(feature = "clockwork_debug")]
fn test_debug_mode_explicit_off() {
    let _guard = ENV_LOCK.lock().unwrap();
    std::env::set_var("CLOCKWORK_DEBUG_MODE", "OFF");
    let mode = DebugMode::from_env();
    assert_eq!(mode, DebugMode::Off);
}
