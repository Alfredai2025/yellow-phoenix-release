// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Clockwork Auto-Debugger
//! Forensic capture only. Never acts. Read-only camera into constitutional state.
//!
//! Modes: OFF | RING (circular buffer) | ALERT (RING + auto-dump on anomaly)
//! Runtime selectable via CLOCKWORK_DEBUG_MODE env var.

use crate::clockwork::*;
use crate::genesis::GENESIS_SEED;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// CONFIGURATION
// =============================================================================

const RING_CAPACITY: usize = 1024;
const DUMP_DIR: &str = "/tmp/yp_clockwork_incidents";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugMode {
    Off = 0,
    Ring = 1,
    Alert = 2,
}

impl DebugMode {
    pub fn from_env() -> Self {
        match std::env::var("CLOCKWORK_DEBUG_MODE").as_deref() {
            Ok("RING") | Ok("ring") | Ok("1") => DebugMode::Ring,
            Ok("ALERT") | Ok("alert") | Ok("2") => DebugMode::Alert,
            _ => DebugMode::Off,
        }
    }
}

// =============================================================================
// INCIDENT SNAPSHOT
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GearSnapshot {
    pub id: String,
    pub position: u64,
    pub pitch: u64,
    pub priority: [i16; 3],
    pub priority_magnitude: f32,
    pub engaged_tick: u64,
    pub domain_mask: u16,
    pub serves_domain_example: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusSnapshot {
    pub source: String,
    pub target: String,
    pub stimulus_type: String,
    pub affinity: f32,
    pub payload_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppointmentSnapshot {
    pub raw_opcode: Vec<u8>,
    pub decoded: DecodedOpcode,
    pub scheduled_tick: u64,
    pub executed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedOpcode {
    pub gear: String,
    pub domain_permuted_id: u8,
    pub domain_real_hint: String,
    pub stimulus: String,
    pub override_level: String,
    pub priority_delta: i8,
    pub checksum_valid: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorruptionReport {
    pub raw_bytes: Vec<u8>,
    pub expected_checksum: u8,
    pub actual_checksum: u8,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentSnapshot {
    pub incident_id: String,
    pub timestamp_sec: u64,
    pub tick: u64,
    pub sovereign_cycle: u64,
    pub thermal_state: String,
    pub active_gear: String,
    pub thermal_override: bool,
    pub genesis_seed_hex: String,
    pub debug_mode: String,
    pub gear_states: Vec<GearSnapshot>,
    pub stimuli_in_flight: Vec<StimulusSnapshot>,
    pub appointments_pending: Vec<AppointmentSnapshot>,
    pub appointments_executed_this_tick: Vec<AppointmentSnapshot>,
    pub corruption_detected: Option<CorruptionReport>,
    pub anomaly_reasons: Vec<String>,
}

// =============================================================================
// RING BUFFER
// =============================================================================

pub struct DebugRing {
    mode: DebugMode,
    buffer: Mutex<VecDeque<IncidentSnapshot>>,
    anomaly_count: AtomicU8,
}

impl DebugRing {
    pub fn new() -> Self {
        let mode = DebugMode::from_env();
        Self {
            mode,
            buffer: Mutex::new(VecDeque::with_capacity(RING_CAPACITY)),
            anomaly_count: AtomicU8::new(0),
        }
    }

    pub fn mode(&self) -> DebugMode {
        self.mode
    }

    pub fn is_active(&self) -> bool {
        self.mode != DebugMode::Off
    }

    pub fn push(&self, snapshot: IncidentSnapshot) {
        if !self.is_active() {
            return;
        }
        let mut buf = self.buffer.lock().unwrap();
        if buf.len() >= RING_CAPACITY {
            buf.pop_front();
        }
        buf.push_back(snapshot);
    }

    pub fn latest(&self, n: usize) -> Vec<IncidentSnapshot> {
        let buf = self.buffer.lock().unwrap();
        buf.iter().rev().take(n).cloned().collect()
    }

    pub fn dump_all(&self, path: &str) -> std::io::Result<()> {
        let buf = self.buffer.lock().unwrap();
        let json = serde_json::to_string_pretty(&buf.iter().collect::<Vec<_>>())?;
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn increment_anomaly(&self) -> u8 {
        self.anomaly_count.fetch_add(1, Ordering::Relaxed)
    }
}

// =============================================================================
// ANOMALY DETECTOR
// =============================================================================

pub struct AnomalyDetector;

impl AnomalyDetector {
    /// Scan for constitutional anomalies. Returns reasons if any found.
    pub fn scan(state: &ClockworkState, engine: &ClockworkEngine) -> Vec<String> {
        let mut reasons = Vec::new();

        // 1. Checksum failures in executed appointments
        for opcode in &state.appointments_executed {
            if Opcode::decode(opcode.bytes).is_none() {
                reasons.push(format!("checksum_fail:opcode:{:02x?}", opcode.bytes));
            }
        }

        // 2. Thermal state inconsistency
        let thermal = engine.thermal.current_state();
        if thermal == ThermalState::Critical && !state.thermal_override {
            reasons.push("thermal_critical_without_override".to_string());
        }

        // 3. Gear position regression (should never decrease)
        for gear in &engine.gears {
            let pos = gear.position.load(Ordering::Relaxed);
            let engaged = gear.engaged.load(Ordering::Relaxed);
            if engaged > state.tick {
                reasons.push(format!("gear_{:?}_engaged_in_future", gear.id));
            }
            if pos == 0 && state.tick > 100 {
                // Suspicious: gear stuck at zero after many ticks
                reasons.push(format!("gear_{:?}_stuck_at_zero", gear.id));
            }
        }

        // 4. Sovereign cycle drift
        let expected_cycle = engine.sovereign.rtc_epoch();
        if state.sovereign_cycle != expected_cycle {
            reasons.push(format!(
                "sovereign_drift:expected={}:actual={}",
                expected_cycle, state.sovereign_cycle
            ));
        }

        // 5. Null stimulus flood
        let null_count = state
            .stimulus_routed
            .iter()
            .filter(|s| s.stimulus_type == StimulusType::Null)
            .count();
        if null_count > 50 {
            reasons.push(format!("null_stimulus_flood:count={}", null_count));
        }

        reasons
    }
}

// =============================================================================
// SNAPSHOT BUILDER
// =============================================================================

pub fn build_snapshot(
    state: &ClockworkState,
    engine: &ClockworkEngine,
    corruption: Option<CorruptionReport>,
) -> IncidentSnapshot {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let seed_hex = format!("{:016x}", GENESIS_SEED);

    let gear_states: Vec<GearSnapshot> = engine
        .gears
        .iter()
        .map(|g| {
            let domains: Vec<u8> = (0..16).filter(|&d| g.serves_domain(d)).collect();
            GearSnapshot {
                id: format!("{:?}", g.id),
                position: g.position.load(Ordering::Relaxed),
                pitch: g.pitch.load(Ordering::Relaxed),
                priority: [g.priority.x, g.priority.y, g.priority.z],
                priority_magnitude: g.priority.magnitude(),
                engaged_tick: g.engaged.load(Ordering::Relaxed),
                domain_mask: g.domain_mask,
                serves_domain_example: domains.into_iter().take(4).collect(),
            }
        })
        .collect();

    let stimuli_in_flight: Vec<StimulusSnapshot> = state
        .stimulus_routed
        .iter()
        .map(|s| {
            let source = &engine.gears[s.source_gear.index()];
            let target = &engine.gears[s.target_gear.index()];
            let aff = source.affinity(target);
            StimulusSnapshot {
                source: format!("{:?}", s.source_gear),
                target: format!("{:?}", s.target_gear),
                stimulus_type: format!("{:?}", s.stimulus_type),
                affinity: aff,
                payload_len: s.payload.len(),
            }
        })
        .collect();

    let appointments_pending: Vec<AppointmentSnapshot> = engine
        .mesh
        .pending_for_gear(state.active_gear)
        .into_iter()
        .map(|op| decode_appointment_snapshot(&op))
        .collect();

    let appointments_executed: Vec<AppointmentSnapshot> = state
        .appointments_executed
        .iter()
        .map(|op| decode_appointment_snapshot(op))
        .collect();

    let anomaly_reasons = AnomalyDetector::scan(state, engine);

    IncidentSnapshot {
        incident_id: format!("cw-{}-{:08x}", now, state.tick),
        timestamp_sec: now,
        tick: state.tick,
        sovereign_cycle: state.sovereign_cycle,
        thermal_state: format!("{:?}", engine.thermal.current_state()),
        active_gear: format!("{:?}", state.active_gear),
        thermal_override: state.thermal_override,
        genesis_seed_hex: seed_hex,
        debug_mode: format!("{:?}", DebugMode::from_env()),
        gear_states,
        stimuli_in_flight,
        appointments_pending,
        appointments_executed_this_tick: appointments_executed,
        corruption_detected: corruption,
        anomaly_reasons,
    }
}

fn decode_appointment_snapshot(op: &Opcode) -> AppointmentSnapshot {
    let decoded = DecodedOpcode {
        gear: format!("{:?}", op.gear),
        domain_permuted_id: op.domain_id,
        domain_real_hint: format!("domain_{}", op.domain_id),
        stimulus: format!("{:?}", op.stimulus),
        override_level: format!("{:?}", op.override_level),
        priority_delta: op.priority_delta,
        checksum_valid: Opcode::decode(op.bytes).is_some(),
    };

    AppointmentSnapshot {
        raw_opcode: op.bytes.to_vec(),
        decoded,
        scheduled_tick: 0, // filled by caller if needed
        executed: false,
    }
}

// =============================================================================
// AUTO-DUMP
// =============================================================================

pub fn auto_dump_if_alert(snapshot: &IncidentSnapshot, ring: &DebugRing) {
    if ring.mode() != DebugMode::Alert {
        return;
    }
    if snapshot.anomaly_reasons.is_empty() && snapshot.corruption_detected.is_none() {
        return;
    }

    let _ = std::fs::create_dir_all(DUMP_DIR);
    let path = format!("{}/{}.json", DUMP_DIR, snapshot.incident_id);

    let json = match serde_json::to_string_pretty(snapshot) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("[clockwork_debug] JSON serialization failed: {}", e);
            return;
        }
    };

    match OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
    {
        Ok(mut file) => {
            let _ = file.write_all(json.as_bytes());
            eprintln!("[clockwork_debug] Incident dumped: {}", path);
        }
        Err(e) => {
            eprintln!("[clockwork_debug] Dump failed: {}", e);
        }
    }

    ring.increment_anomaly();
}

// =============================================================================
// PUBLIC API
// =============================================================================

pub fn capture_tick(state: &ClockworkState, engine: &ClockworkEngine, ring: &DebugRing) {
    if !ring.is_active() {
        return;
    }

    let snapshot = build_snapshot(state, engine, None);
    auto_dump_if_alert(&snapshot, ring);
    ring.push(snapshot);
}

pub fn capture_corruption(
    state: &ClockworkState,
    engine: &ClockworkEngine,
    ring: &DebugRing,
    raw_bytes: [u8; 6],
    context: &str,
) {
    if !ring.is_active() {
        return;
    }

    let expected = raw_bytes[0] ^ raw_bytes[1] ^ raw_bytes[2] ^ raw_bytes[3] ^ raw_bytes[4];
    let corruption = CorruptionReport {
        raw_bytes: raw_bytes.to_vec(),
        expected_checksum: expected,
        actual_checksum: raw_bytes[5],
        context: context.to_string(),
    };

    let mut snapshot = build_snapshot(state, engine, Some(corruption));
    snapshot
        .anomaly_reasons
        .push("opcode_checksum_failure".to_string());

    auto_dump_if_alert(&snapshot, ring);
    ring.push(snapshot);
}
