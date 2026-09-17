//! Yellow Phoenix — Real Constitutional Feedback Loop Integration Test
//!
//! This test wires the organism-level modules together:
//!   ClockworkEngine + GoldenHash + ITF + CGT + Scar + SnapshotRing.
//!
//! It proves the organism learns from degradation, adjudicates overrides,
//! encodes scars, rolls back state, and prioritizes thermal interrupts.

use pams::clockwork::constitutional_loop::{ConstitutionalLoop, LoopAction};
use pams::clockwork::GearId;
use pams::golden_hash::{AdjudicationLevel, TelemetrySnapshot};

fn line() {
    println!("{}", "=".repeat(70));
}

fn banner(title: &str) {
    println!();
    line();
    println!("{}", title);
    line();
}

fn sub(title: &str) {
    println!();
    println!("{}", title);
}

fn normal_telemetry() -> TelemetrySnapshot {
    TelemetrySnapshot {
        p50_ms: 3.5,
        p99_ms: 5.0,
        r1: 0.92,
        r5: 0.98,
        cpu_temp: 45.0,
        memory_pct: 50.0,
        autopoiesis: 0.95,
        exam_violation: false,
        git_violations: 0,
        circuit_trips: 0,
    }
}

fn stress_telemetry() -> TelemetrySnapshot {
    TelemetrySnapshot {
        p50_ms: 12.0,
        p99_ms: 18.0,
        r1: 0.80,
        r5: 0.88,
        cpu_temp: 50.0,
        memory_pct: 50.0,
        autopoiesis: 0.60,
        ..normal_telemetry()
    }
}

fn recovery_telemetry() -> TelemetrySnapshot {
    TelemetrySnapshot {
        p50_ms: 4.0,
        p99_ms: 6.0,
        r1: 0.91,
        r5: 0.97,
        cpu_temp: 46.0,
        memory_pct: 52.0,
        autopoiesis: 0.94,
        ..normal_telemetry()
    }
}

fn second_stress_telemetry() -> TelemetrySnapshot {
    // Moderate stress — enough to trigger an alert when scar-biased.
    TelemetrySnapshot {
        p50_ms: 8.5,
        p99_ms: 11.5,
        r1: 0.86,
        r5: 0.94,
        cpu_temp: 45.0,
        memory_pct: 50.0,
        autopoiesis: 0.80,
        ..normal_telemetry()
    }
}

fn thermal_telemetry() -> TelemetrySnapshot {
    TelemetrySnapshot {
        cpu_temp: 80.0,
        ..normal_telemetry()
    }
}

fn memory_stress_telemetry() -> TelemetrySnapshot {
    TelemetrySnapshot {
        memory_pct: 95.0,
        ..normal_telemetry()
    }
}

#[test]
fn test_full_feedback_loop() {
    banner("FEEDBACK LOOP INTEGRATION TEST (REAL MODULES)");

    let mut organism = ConstitutionalLoop::new();
    let qry = GearId::G0;

    // -----------------------------------------------------------------
    // PHASE 1: Baseline
    // -----------------------------------------------------------------
    sub("[PHASE 1] Baseline — 50 ticks normal operation...");
    let mut baseline_overrides = 0;
    for _ in 0..50 {
        let out = organism.tick(&normal_telemetry());
        if out.override_level != AdjudicationLevel::None {
            baseline_overrides += 1;
        }
    }
    println!("  Baseline overrides: {}", baseline_overrides);
    assert_eq!(baseline_overrides, 0, "Baseline should produce no overrides");
    assert!(!organism.has_scar(qry), "No scar should exist after baseline");

    // -----------------------------------------------------------------
    // PHASE 2: Degradation
    // -----------------------------------------------------------------
    sub("[PHASE 2] Degradation — 100 ticks of stress...");
    let mut degrade_alerts = 0;
    let mut override_tick: Option<u64> = None;
    for _ in 0..100 {
        let out = organism.tick(&stress_telemetry());
        if out.alert_count > 0 {
            degrade_alerts += 1;
        }
        if out.override_level != AdjudicationLevel::None && override_tick.is_none() {
            override_tick = Some(out.tick);
            println!(
                "  Tick {}: {} override targeting {:?}",
                out.tick, out.override_level as i32, out.target_gear
            );
        }
    }
    println!("  Degradation alerts: {}", degrade_alerts);
    println!("  Override tick: {:?}", override_tick);
    assert!(
        override_tick.is_some(),
        "Golden Hash should issue an override during degradation"
    );
    assert_eq!(organism.gear(qry).pitch.load(std::sync::atomic::Ordering::Relaxed), 1, "QRY pitch should be tightened");
    assert!(organism.has_scar(qry), "A scar should be encoded in QRY");

    // -----------------------------------------------------------------
    // PHASE 3: Recovery
    // -----------------------------------------------------------------
    sub("[PHASE 3] Recovery — 100 ticks after override...");
    let mut recovery_overrides = 0;
    for i in 0..100 {
        let out = organism.tick(&recovery_telemetry());
        if out.override_level != AdjudicationLevel::None {
            if recovery_overrides < 3 {
                println!("  recovery tick {} override {:?}", i, out.override_level);
            }
            recovery_overrides += 1;
        }
    }
    println!("  Recovery overrides: {}", recovery_overrides);
    assert_eq!(
        recovery_overrides, 0,
        "No new overrides should fire during recovery"
    );
    assert!(organism.has_scar(qry), "Scar should persist into recovery");

    // -----------------------------------------------------------------
    // PHASE 4: Memory — scar-biased early alert
    // -----------------------------------------------------------------
    sub("[PHASE 4] Memory — proving scar biases future behavior...");
    let mut early_alert_tick: Option<u64> = None;
    for _ in 0..50 {
        let out = organism.tick(&second_stress_telemetry());
        if out.override_level != AdjudicationLevel::None && early_alert_tick.is_none() {
            early_alert_tick = Some(out.tick);
            println!(
                "  Tick {}: scar-biased early alert ({:?})",
                out.tick, out.action_taken
            );
        }
    }
    assert!(
        early_alert_tick.is_some(),
        "Second stress should trigger a scar-biased early override"
    );

    // -----------------------------------------------------------------
    // PHASE 5: Final verification
    // -----------------------------------------------------------------
    sub("[PHASE 5] Final verification...");
    println!("  Total ticks: {}", organism.gear(qry).engaged.load(std::sync::atomic::Ordering::Relaxed));
    println!("  QRY pitch: {}", organism.gear(qry).pitch.load(std::sync::atomic::Ordering::Relaxed));
    println!("  QRY scar present: {}", organism.has_scar(qry));

    banner("✅ FEEDBACK LOOP INTEGRATION TEST PASSED");
}

#[test]
fn test_multi_gear_degradation() {
    sub("[TEST] Multi-gear degradation scenario...");

    let mut organism = ConstitutionalLoop::new();
    let mut targeted = std::collections::HashSet::new();

    // Cycle through stress profiles that target different gears.
    for i in 0..60 {
        let telemetry = match i % 3 {
            0 => stress_telemetry(),       // latency -> QRY
            1 => thermal_telemetry(),      // thermal -> THR
            _ => memory_stress_telemetry(), // memory -> GRW
        };
        let out = organism.tick(&telemetry);
        if let Some(g) = out.target_gear {
            targeted.insert(g);
        }
    }

    println!("  Targeted gears: {:?}", targeted);
    assert!(targeted.contains(&GearId::G0), "QRY should be targeted");
    assert!(targeted.contains(&GearId::G8), "THR should be targeted");
    assert!(targeted.contains(&GearId::G6), "GRW should be targeted");
    println!("  ✅ Multi-gear degradation detected correctly");
}

#[test]
fn test_rollback_restores_state() {
    sub("[TEST] Rollback restores gear state...");

    let mut organism = ConstitutionalLoop::new();
    let qry = GearId::G0;

    // Normal operation to create a clean snapshot.
    for _ in 0..5 {
        organism.tick(&normal_telemetry());
    }

    // Degrade: override writes a scar.
    let out = organism.tick(&stress_telemetry());
    let tick = out.tick;
    assert!(organism.has_scar(qry), "Degradation should encode a scar");

    // Roll back to the snapshot taken just before the override.
    assert!(organism.rollback(qry, tick), "Rollback should succeed");
    assert!(!organism.has_scar(qry), "Rollback should remove the scar");

    println!("  ✅ Rollback restored state correctly");
}

#[test]
fn test_thermal_interrupt_priority() {
    sub("[TEST] Thermal interrupt priority...");

    let mut organism = ConstitutionalLoop::new();
    let out = organism.tick(&thermal_telemetry());

    assert_eq!(
        out.override_level,
        AdjudicationLevel::Severe,
        "Thermal emergency should trigger Level 3"
    );
    assert_eq!(
        out.target_gear,
        Some(GearId::G8),
        "Should target THR gear"
    );
    assert!(
        matches!(out.action_taken, LoopAction::ThermalCritical),
        "Action should be thermal critical"
    );
    assert!(organism.engine.thermal.is_critical(), "Engine should be in thermal critical");

    println!(
        "  Thermal: 80°C → Level {} override on gear {:?}",
        out.override_level as i32, out.target_gear
    );
    println!("  ✅ Thermal interrupt correctly prioritized");
}
