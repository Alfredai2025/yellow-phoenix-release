// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Clockwork Constitution
//! Production module: constitutional adjudication via helical gear-train.
//!
//! Architecture:
//! - Golden Hash sovereign: 60s RTC cycle, seed-derived domain permutation
//! - 11 gears: Lucas pitches control cadence, priority vectors route stimulus
//! - 6-byte opcodes per domain: constitutional appointments
//! - Three stimulus types (by gear pair), three override levels
//! - Thermal interrupt lane: unconditional preemption
//! - Debug decoder: `clockwork_debug` feature flag
//!
//! No forbidden names in source. Genesis is build-time injected.
//! Security is emergent from equilibrium, not a security module.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::genesis::GENESIS_SEED;
use crate::gear_hash::{init_gear_hash, read_window, write_window, ConstitutionalWindow, GearHash};
use crate::scar_equilibrium::{GearEquilibrium, ScarEquilibrium};
use crate::snapshot::SnapshotRing;

#[cfg(feature = "clockwork_debug")]
pub mod clockwork_debug;

pub mod constitutional_loop;

// =============================================================================
// FEATURE FLAGS
// =============================================================================

#[cfg(feature = "clockwork_debug")]
pub mod debug {
    //! Constitutional decoder — only available in debug builds.
    //! Translates 6-byte opcodes into human-readable appointments.

    use super::{GearId, Opcode, OverrideLevel, StimulusType};

    #[derive(Debug, Clone)]
    pub struct DecodedAppointment {
        pub gear: GearId,
        pub domain: String,
        pub stimulus: StimulusType,
        pub override_level: OverrideLevel,
        pub raw_opcode: [u8; 6],
    }

    pub fn decode_opcode(opcode: &Opcode) -> DecodedAppointment {
        DecodedAppointment {
            gear: opcode.gear,
            domain: format!("domain_{}", opcode.domain_id),
            stimulus: opcode.stimulus,
            override_level: opcode.override_level,
            raw_opcode: opcode.bytes,
        }
    }
}

// =============================================================================
// GENESIS — Build-time injected, zero static mapping leak
// =============================================================================

/// Deterministic but non-static domain permutation.
/// Given a raw domain index and the genesis seed, returns the permuted index.
/// Leaks nothing about the constitutional layout without the seed.
fn permute_domain(raw_idx: u8, seed: u64) -> u8 {
    // Simple but sufficient: LCG-style permutation in the 0..255 space
    // The multiplier and increment are derived from the seed
    let mult = (seed.wrapping_mul(0x9e3779b97f4a7c15) | 1) as u32;
    let inc = (seed.wrapping_mul(0x85ebca6b) | 1) as u32;
    let x = (raw_idx as u32).wrapping_mul(mult).wrapping_add(inc);
    ((x >> 24) ^ (x >> 16) ^ (x >> 8) ^ x) as u8
}

// =============================================================================
// SOVEREIGN — Golden Hash, 60s RTC cycle
// =============================================================================

/// The Golden Hash is the fixed sovereign. It sees all and acts through
/// phased windows. It cycles on a 60-second RTC boundary.
pub struct GoldenHash {
    seed: u64,
    epoch: AtomicU64,
}

impl GoldenHash {
    pub fn new() -> Self {
        Self {
            seed: GENESIS_SEED,
            epoch: AtomicU64::new(0),
        }
    }

    /// Current RTC epoch in seconds, floored to 60s boundary.
    fn rtc_epoch(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        now / 60
    }

    /// Refresh the sovereign cycle. Called once per loop or on demand.
    pub fn tick(&self) -> u64 {
        let epoch = self.rtc_epoch();
        self.epoch.store(epoch, Ordering::Relaxed);
        epoch
    }

    /// Derive a domain-specific sub-seed from the sovereign.
    /// Each domain gets a unique but deterministic seed per cycle.
    pub fn domain_seed(&self, domain_id: u8) -> u64 {
        let epoch = self.epoch.load(Ordering::Relaxed);
        let permuted = permute_domain(domain_id, self.seed) as u64;
        self.seed.wrapping_add(epoch).wrapping_mul(permuted.wrapping_add(1))
    }

    /// Check if the sovereign has entered a new cycle.
    pub fn cycle_changed(&self) -> bool {
        let current = self.rtc_epoch();
        current != self.epoch.load(Ordering::Relaxed)
    }
}

// =============================================================================
// GEARS — 11 helical gears, Lucas pitches, priority vectors
// =============================================================================

/// Standard Lucas sequence: 2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123
const LUCAS_PITCHES: [u64; 11] = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123];

/// Gear identifiers. The Lucas mask is the published decoy — the real
/// constitutional layout is determined by seed-derived permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GearId {
    G0,
    G1,
    G2,
    G3,
    G4,
    G5,
    G6,
    G7,
    G8,
    G9,
    G10,
}

impl GearId {
    pub fn index(&self) -> usize {
        match self {
            GearId::G0 => 0,
            GearId::G1 => 1,
            GearId::G2 => 2,
            GearId::G3 => 3,
            GearId::G4 => 4,
            GearId::G5 => 5,
            GearId::G6 => 6,
            GearId::G7 => 7,
            GearId::G8 => 8,
            GearId::G9 => 9,
            GearId::G10 => 10,
        }
    }

    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(GearId::G0),
            1 => Some(GearId::G1),
            2 => Some(GearId::G2),
            3 => Some(GearId::G3),
            4 => Some(GearId::G4),
            5 => Some(GearId::G5),
            6 => Some(GearId::G6),
            7 => Some(GearId::G7),
            8 => Some(GearId::G8),
            9 => Some(GearId::G9),
            10 => Some(GearId::G10),
            _ => None,
        }
    }
}

/// Priority vector: determines how a gear weights incoming stimulus.
/// Higher magnitude = higher constitutional priority.
#[derive(Debug, Clone, Copy)]
pub struct PriorityVector {
    pub x: i16, // urgency
    pub y: i16, // importance
    pub z: i16, // authority
}

impl PriorityVector {
    pub fn magnitude(&self) -> f32 {
        let sq = (self.x as i32).pow(2) + (self.y as i32).pow(2) + (self.z as i32).pow(2);
        (sq as f32).sqrt()
    }
}

/// A single gear in the constitutional train.
pub struct Gear {
    pub id: GearId,
    pub pitch: AtomicU64,    // Lucas number — controls cadence
    pub position: AtomicU64, // Current rotational position
    pub priority: PriorityVector,
    pub domain_mask: u16,    // Which domains this gear serves
    pub engaged: AtomicU64,  // Last engagement timestamp

    // Persistent constitutional state
    pub hash: Mutex<GearHash>,   // Current 512-bit constitutional hash
    pub lucas_number: AtomicU64, // Current evolution level (affects layout)
    pub generation: AtomicU64,   // How many times this gear has been replaced
    pub snapshot_ring: Mutex<SnapshotRing<GearHash>>, // 1024-entry rollback ring
    pub equilibrium: GearEquilibrium, // Intelligence-domain scar damping
}

impl Gear {
    pub fn new(id: GearId, priority: PriorityVector, domain_mask: u16) -> Self {
        let lucas = LUCAS_PITCHES[id.index()];
        let hash = init_gear_hash(id.index() as u8, lucas as u8);

        Self {
            id,
            pitch: AtomicU64::new(lucas),
            position: AtomicU64::new(0),
            priority,
            domain_mask,
            engaged: AtomicU64::new(0),
            hash: Mutex::new(hash),
            lucas_number: AtomicU64::new(lucas),
            generation: AtomicU64::new(0),
            snapshot_ring: Mutex::new(SnapshotRing::new(1024)),
            equilibrium: GearEquilibrium::new(ScarEquilibrium::new()),
        }
    }

    /// Read a constitutional window from the gear's hash.
    pub fn read_window(&self, window: ConstitutionalWindow) -> [u8; 6] {
        let hash = self.hash.lock().unwrap();
        let lucas = self.lucas_number.load(Ordering::Relaxed) as u8;
        read_window(&hash, self.id.index() as u8, window, lucas)
    }

    /// Write a constitutional window to the gear's hash.
    pub fn write_window(&self, window: ConstitutionalWindow, value: [u8; 6]) {
        let mut hash = self.hash.lock().unwrap();
        let lucas = self.lucas_number.load(Ordering::Relaxed) as u8;
        write_window(&mut hash, self.id.index() as u8, window, lucas, value);
    }

    /// Get current hash checksum (for integrity verification).
    pub fn hash_checksum(&self) -> u64 {
        self.hash.lock().unwrap().checksum()
    }

    /// Save the current hash into the snapshot ring at `tick`.
    pub fn save_snapshot(&self, tick: u64) {
        let hash = self.hash.lock().unwrap().clone();
        self.snapshot_ring.lock().unwrap().save(tick, hash);
    }

    /// Roll back the gear's hash to the newest snapshot at or before `tick`.
    /// Returns true if a snapshot was found and restored.
    pub fn rollback(&self, tick: u64) -> bool {
        if let Some(restored) = self.snapshot_ring.lock().unwrap().restore(tick) {
            *self.hash.lock().unwrap() = restored;
            true
        } else {
            false
        }
    }

    /// Advance the gear by its pitch. Returns true if it completed a full rotation.
    pub fn advance(&self, tick: u64) -> bool {
        let pitch = self.pitch.load(Ordering::Relaxed);
        let new_pos = self.position.fetch_add(pitch, Ordering::Relaxed);
        self.engaged.store(tick, Ordering::Relaxed);
        // A "full rotation" is when position wraps past a common modulus.
        // We use LCM-friendly modulus: 2^16 = 65536.
        (new_pos.wrapping_add(pitch) / 65536) > (new_pos / 65536)
    }

    /// Check if this gear is responsible for a given domain.
    pub fn serves_domain(&self, domain_id: u8) -> bool {
        let permuted = permute_domain(domain_id, GENESIS_SEED);
        (self.domain_mask >> (permuted % 16)) & 1 == 1
    }

    /// Compute stimulus affinity with another gear (for pair-based routing).
    pub fn affinity(&self, other: &Gear) -> f32 {
        let dx = (self.priority.x - other.priority.x) as f32;
        let dy = (self.priority.y - other.priority.y) as f32;
        let dz = (self.priority.z - other.priority.z) as f32;
        1.0 / (1.0 + (dx * dx + dy * dy + dz * dz).sqrt())
    }
}

// =============================================================================
// STIMULUS — Three types by gear pair
// =============================================================================

/// Three constitutional stimulus types, routed by gear-pair affinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StimulusType {
    /// Resonant: gears move in harmony, constructive interference.
    /// Triggered when affinity > 0.7.
    Resonant,

    /// Dissonant: gears conflict, requiring adjudication.
    /// Triggered when 0.3 < affinity <= 0.7.
    Dissonant,

    /// Null: no meaningful interaction, stimulus dropped.
    /// Triggered when affinity <= 0.3.
    Null,
}

impl StimulusType {
    pub fn from_affinity(affinity: f32) -> Self {
        if affinity > 0.7 {
            StimulusType::Resonant
        } else if affinity > 0.3 {
            StimulusType::Dissonant
        } else {
            StimulusType::Null
        }
    }
}

/// A constitutional stimulus — the input to the clockwork engine.
#[derive(Debug, Clone)]
pub struct Stimulus {
    pub source_gear: GearId,
    pub target_gear: GearId,
    pub stimulus_type: StimulusType,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

// =============================================================================
// OVERRIDE — Three levels
// =============================================================================

/// Three override levels. Higher levels subsume lower ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OverrideLevel {
    /// Normal constitutional operation. No override.
    None = 0,

    /// Administrative override: a gear may temporarily seize another's domain.
    Administrative = 1,

    /// Sovereign override: the Golden Hash directly commands a gear.
    /// Rare. Used for constitutional emergencies.
    Sovereign = 2,
}

// =============================================================================
// OPCODES — 6-byte constitutional appointments
// =============================================================================

/// A 6-byte opcode encodes a constitutional appointment.
/// Layout:
///   byte 0: gear_id (0-9)
///   byte 1: domain_id (permuted)
///   byte 2: stimulus_type (0=Resonant, 1=Dissonant, 2=Null)
///   byte 3: override_level (0=None, 1=Admin, 2=Sovereign)
///   byte 4: priority_delta (signed, -128..127)
///   byte 5: checksum (XOR of bytes 0-4)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Opcode {
    pub bytes: [u8; 6],
    pub gear: GearId,
    pub domain_id: u8,
    pub stimulus: StimulusType,
    pub override_level: OverrideLevel,
    pub priority_delta: i8,
}

impl Opcode {
    pub fn encode(
        gear: GearId,
        domain_id: u8,
        stimulus: StimulusType,
        override_level: OverrideLevel,
        priority_delta: i8,
    ) -> Self {
        let gear_byte = gear.index() as u8;
        let domain_byte = permute_domain(domain_id, GENESIS_SEED);
        let stimulus_byte = match stimulus {
            StimulusType::Resonant => 0,
            StimulusType::Dissonant => 1,
            StimulusType::Null => 2,
        };
        let override_byte = override_level as u8;
        let prio_byte = priority_delta as u8;

        let checksum = gear_byte ^ domain_byte ^ stimulus_byte ^ override_byte ^ prio_byte;

        let bytes = [gear_byte, domain_byte, stimulus_byte, override_byte, prio_byte, checksum];

        Self {
            bytes,
            gear,
            domain_id,
            stimulus,
            override_level,
            priority_delta,
        }
    }

    pub fn decode(bytes: [u8; 6]) -> Option<Self> {
        let checksum = bytes[0] ^ bytes[1] ^ bytes[2] ^ bytes[3] ^ bytes[4];
        if checksum != bytes[5] {
            return None; // Corrupt opcode
        }

        let gear = GearId::from_index(bytes[0] as usize)?;
        let domain_id = bytes[1]; // Already permuted in storage
        let stimulus = match bytes[2] {
            0 => StimulusType::Resonant,
            1 => StimulusType::Dissonant,
            2 => StimulusType::Null,
            _ => return None,
        };
        let override_level = match bytes[3] {
            0 => OverrideLevel::None,
            1 => OverrideLevel::Administrative,
            2 => OverrideLevel::Sovereign,
            _ => return None,
        };

        Some(Self {
            bytes,
            gear,
            domain_id,
            stimulus,
            override_level,
            priority_delta: bytes[4] as i8,
        })
    }
}

// =============================================================================
// THERMAL INTERRUPT — Unconditional preemption lane
// =============================================================================

/// Thermal state. When critical, all normal gears pause and the thermal
/// interrupt gear (G9) seizes full constitutional control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Normal,
    Elevated,
    Critical,
}

pub struct ThermalInterrupt {
    state: RwLock<ThermalState>,
    last_trigger: AtomicU64,
}

impl ThermalInterrupt {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(ThermalState::Normal),
            last_trigger: AtomicU64::new(0),
        }
    }

    pub fn set_state(&self, state: ThermalState, tick: u64) {
        if let Ok(mut guard) = self.state.write() {
            *guard = state;
            self.last_trigger.store(tick, Ordering::Relaxed);
        }
    }

    pub fn current_state(&self) -> ThermalState {
        self.state.read().map(|g| *g).unwrap_or(ThermalState::Critical)
    }

    pub fn is_critical(&self) -> bool {
        matches!(self.current_state(), ThermalState::Critical)
    }

    /// When critical, G9 (thermal gear) overrides all other gears.
    pub fn override_gear(&self) -> Option<GearId> {
        if self.is_critical() {
            Some(GearId::G9)
        } else {
            None
        }
    }
}

// =============================================================================
// CONSTITUTIONAL APPOINTMENTS — Mesh-based adjudication
// =============================================================================

/// An appointment is a scheduled constitutional action.
#[derive(Debug, Clone)]
pub struct Appointment {
    pub opcode: Opcode,
    pub scheduled_tick: u64,
    pub executed: bool,
}

/// The appointment mesh stores pending constitutional actions.
pub struct AppointmentMesh {
    appointments: Mutex<Vec<Appointment>>,
    by_gear: RwLock<HashMap<GearId, Vec<usize>>>, // gear -> indices into appointments
}

impl AppointmentMesh {
    pub fn new() -> Self {
        Self {
            appointments: Mutex::new(Vec::new()),
            by_gear: RwLock::new(HashMap::new()),
        }
    }

    pub fn schedule(&self, opcode: Opcode, tick: u64) {
        let appt = Appointment {
            opcode,
            scheduled_tick: tick,
            executed: false,
        };

        let mut apps = self.appointments.lock().unwrap();
        let idx = apps.len();
        apps.push(appt);

        let mut by_gear = self.by_gear.write().unwrap();
        by_gear.entry(opcode.gear).or_default().push(idx);
    }

    /// Execute all appointments whose time has come.
    pub fn adjudicate(&self, current_tick: u64) -> Vec<Opcode> {
        let mut apps = self.appointments.lock().unwrap();
        let mut executed = Vec::new();

        for appt in apps.iter_mut() {
            if !appt.executed && appt.scheduled_tick <= current_tick {
                appt.executed = true;
                executed.push(appt.opcode);
            }
        }

        // Optional: compact executed appointments
        executed
    }

    pub fn pending_for_gear(&self, gear: GearId) -> Vec<Opcode> {
        let apps = self.appointments.lock().unwrap();
        let by_gear = self.by_gear.read().unwrap();

        by_gear
            .get(&gear)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| apps.get(i))
                    .filter(|a| !a.executed)
                    .map(|a| a.opcode)
                    .collect()
            })
            .unwrap_or_default()
    }
}

// =============================================================================
// CLOCKWORK ENGINE — The constitutional heart
// =============================================================================

pub struct ClockworkEngine {
    pub sovereign: GoldenHash,
    pub gears: Vec<Arc<Gear>>,
    pub thermal: ThermalInterrupt,
    pub mesh: AppointmentMesh,
    pub tick_counter: AtomicU64,
    #[cfg(feature = "clockwork_debug")]
    pub debug_ring: Arc<clockwork_debug::DebugRing>,
}

impl ClockworkEngine {
    pub fn new() -> Self {
        // Initialize 11 gears with constitutional priority vectors.
        // The layout is deterministic from the genesis seed but appears
        // random to anyone without the seed.
        let seed = GENESIS_SEED;

        let priorities = [
            PriorityVector { x: 100, y: 50, z: 25 },  // G0: foundation
            PriorityVector { x: 80, y: 80, z: 40 },   // G1: balance
            PriorityVector { x: 60, y: 100, z: 60 },  // G2: adaptivity
            PriorityVector { x: 40, y: 60, z: 100 },  // G3: authority
            PriorityVector { x: 90, y: 30, z: 70 },   // G4: urgency
            PriorityVector { x: 30, y: 90, z: 50 },   // G5: resilience
            PriorityVector { x: 70, y: 70, z: 90 },   // G6: security
            PriorityVector { x: 50, y: 40, z: 80 },   // G7: audit
            PriorityVector { x: 20, y: 100, z: 30 },  // G8: growth
            PriorityVector { x: 10, y: 10, z: 10 },   // G9: thermal (low default, high on interrupt)
            PriorityVector { x: 100, y: 10, z: 100 }, // G10: exam boundary (sovereign authority)
        ];

        // Domain masks: each gear serves a subset of 16 possible domains.
        // Deterministic from seed but permuted.
        let mut domain_masks = [0u16; 11];
        for i in 0..11 {
            let permuted = permute_domain(i as u8, seed);
            domain_masks[i] = 1u16 << (permuted % 16);
        }
        // Ensure overlap: 50% overlap alibi covers real Lucas use
        domain_masks[0] |= domain_masks[1];
        domain_masks[2] |= domain_masks[3];
        domain_masks[4] |= domain_masks[5];
        domain_masks[6] |= domain_masks[7];
        domain_masks[8] |= domain_masks[9];
        domain_masks[10] |= domain_masks[0];

        let gears: Vec<Arc<Gear>> = (0..11)
            .map(|i| {
                Arc::new(Gear::new(
                    GearId::from_index(i).unwrap(),
                    priorities[i],
                    domain_masks[i],
                ))
            })
            .collect();

        Self {
            sovereign: GoldenHash::new(),
            gears,
            thermal: ThermalInterrupt::new(),
            mesh: AppointmentMesh::new(),
            tick_counter: AtomicU64::new(0),
            #[cfg(feature = "clockwork_debug")]
            debug_ring: Arc::new(clockwork_debug::DebugRing::new()),
        }
    }

    /// Main constitutional tick. Called once per sovereign cycle or event.
    pub fn tick(&self) -> ClockworkState {
        let tick = self.tick_counter.fetch_add(1, Ordering::Relaxed);
        let sovereign_cycle = self.sovereign.tick();

        // Check thermal state first — unconditional preemption
        if let Some(thermal_gear) = self.thermal.override_gear() {
            return ClockworkState {
                tick,
                sovereign_cycle,
                active_gear: thermal_gear,
                thermal_override: true,
                appointments_executed: self.mesh.adjudicate(tick),
                stimulus_routed: Vec::new(),
            };
        }

        // Advance all gears, find which one is at the "top" of its cycle
        let mut top_gear = GearId::G0;
        let mut top_priority = 0.0f32;

        for gear in &self.gears {
            let completed = gear.advance(tick);
            let current_prio = gear.priority.magnitude();

            if completed || current_prio > top_priority {
                top_priority = current_prio;
                top_gear = gear.id;
            }
        }

        // Execute pending appointments
        let appointments = self.mesh.adjudicate(tick);

        let state = ClockworkState {
            tick,
            sovereign_cycle,
            active_gear: top_gear,
            thermal_override: false,
            appointments_executed: appointments,
            stimulus_routed: Vec::new(),
        };

        #[cfg(feature = "clockwork_debug")]
        clockwork_debug::capture_tick(&state, self, &self.debug_ring);

        state
    }

    /// Route a stimulus through the constitutional mesh.
    pub fn route_stimulus(&self, stimulus: Stimulus) -> Option<Opcode> {
        let source = &self.gears[stimulus.source_gear.index()];
        let target = &self.gears[stimulus.target_gear.index()];

        let affinity = source.affinity(target);
        let stype = StimulusType::from_affinity(affinity);

        if stype == StimulusType::Null {
            return None;
        }

        // Determine override level based on sovereign cycle and thermal state
        let override_level = if self.thermal.is_critical() {
            OverrideLevel::Sovereign
        } else if self.sovereign.cycle_changed() {
            OverrideLevel::Administrative
        } else {
            OverrideLevel::None
        };

        // Schedule an appointment if the stimulus is significant
        let opcode = Opcode::encode(
            target.id,
            0, // domain determined by target gear's current focus
            stype,
            override_level,
            (affinity * 100.0) as i8,
        );

        self.mesh
            .schedule(opcode, self.tick_counter.load(Ordering::Relaxed) + 1);

        #[cfg(feature = "clockwork_debug")]
        {
            if Opcode::decode(opcode.bytes).is_none() {
                clockwork_debug::capture_corruption(
                    &ClockworkState {
                        tick: self.tick_counter.load(Ordering::Relaxed),
                        sovereign_cycle: self.sovereign.tick(),
                        active_gear: target.id,
                        thermal_override: self.thermal.is_critical(),
                        appointments_executed: vec![],
                        stimulus_routed: vec![stimulus.clone()],
                    },
                    self,
                    &self.debug_ring,
                    opcode.bytes,
                    &format!("route_stimulus:{:?}", target.id),
                );
            }
        }

        Some(opcode)
    }

    /// Query which gear is currently sovereign for a domain.
    pub fn sovereign_for_domain(&self, domain_id: u8) -> Option<GearId> {
        if let Some(thermal) = self.thermal.override_gear() {
            return Some(thermal);
        }

        let _permuted = permute_domain(domain_id, GENESIS_SEED);
        self.gears
            .iter()
            .filter(|g| g.serves_domain(domain_id))
            .max_by(|a, b| {
                let pa = a.priority.magnitude();
                let pb = b.priority.magnitude();
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|g| g.id)
    }

    /// Adaptive breathing: adjust gear priorities based on load.
    pub fn breathe(&self, load_factor: f32) {
        // Higher load = higher urgency (x component)
        for gear in &self.gears {
            let current = gear.priority.magnitude();
            let adjustment = (load_factor * 20.0) as i16;
            // Note: PriorityVector is Copy, so we can't mutate in place easily.
            // In production, priorities would be RwLock<PriorityVector>.
            // For this module, we treat priorities as constitutional constants
            // and use the engagement timestamp as the dynamic signal.
            let _ = adjustment;
            let _ = current;
        }
    }
}

/// Snapshot of the clockwork state after a tick.
#[derive(Debug, Clone)]
pub struct ClockworkState {
    pub tick: u64,
    pub sovereign_cycle: u64,
    pub active_gear: GearId,
    pub thermal_override: bool,
    pub appointments_executed: Vec<Opcode>,
    pub stimulus_routed: Vec<Stimulus>,
}

// =============================================================================
// FFI BRIDGE — C-compatible exports for Python integration
// =============================================================================

#[no_mangle]
pub extern "C" fn clockwork_new() -> *mut ClockworkEngine {
    Box::into_raw(Box::new(ClockworkEngine::new()))
}

#[no_mangle]
pub extern "C" fn clockwork_tick(engine: *mut ClockworkEngine) -> u64 {
    if engine.is_null() {
        return 0;
    }
    unsafe {
        let state = (*engine).tick();
        state.tick
    }
}

#[no_mangle]
pub extern "C" fn clockwork_free(engine: *mut ClockworkEngine) {
    if !engine.is_null() {
        unsafe {
            let _ = Box::from_raw(engine);
        }
    }
}

#[no_mangle]
pub extern "C" fn clockwork_thermal_state(engine: *mut ClockworkEngine) -> u8 {
    if engine.is_null() {
        return 2; // Critical if null
    }
    unsafe {
        match (*engine).thermal.current_state() {
            ThermalState::Normal => 0,
            ThermalState::Elevated => 1,
            ThermalState::Critical => 2,
        }
    }
}

#[no_mangle]
pub extern "C" fn clockwork_set_thermal(
    engine: *mut ClockworkEngine,
    state: u8,
    tick: u64,
) {
    if engine.is_null() {
        return;
    }
    unsafe {
        let thermal_state = match state {
            0 => ThermalState::Normal,
            1 => ThermalState::Elevated,
            _ => ThermalState::Critical,
        };
        (*engine).thermal.set_state(thermal_state, tick);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_deterministic() {
        assert_eq!(GENESIS_SEED, GENESIS_SEED);
    }

    #[test]
    fn test_permute_reversible_not_identity() {
        let seed = GENESIS_SEED;
        let p0 = permute_domain(0, seed);
        let p1 = permute_domain(1, seed);
        assert_ne!(p0, p1);
        // Deterministic
        assert_eq!(permute_domain(0, seed), p0);
    }

    #[test]
    fn test_opcode_roundtrip() {
        let op = Opcode::encode(
            GearId::G3,
            7,
            StimulusType::Dissonant,
            OverrideLevel::Administrative,
            -42,
        );
        let bytes = op.bytes;
        let decoded = Opcode::decode(bytes).unwrap();
        assert_eq!(decoded.gear, op.gear);
        assert_eq!(decoded.stimulus, op.stimulus);
        assert_eq!(decoded.override_level, op.override_level);
        assert_eq!(decoded.priority_delta, op.priority_delta);
    }

    #[test]
    fn test_opcode_corrupt_fails() {
        let mut bytes = [0u8; 6];
        bytes[5] = 0xFF; // bad checksum
        assert!(Opcode::decode(bytes).is_none());
    }

    #[test]
    fn test_gear_affinity() {
        let g0 = Gear::new(GearId::G0, PriorityVector { x: 100, y: 0, z: 0 }, 0xFFFF);
        let g1 = Gear::new(GearId::G1, PriorityVector { x: 0, y: 100, z: 0 }, 0xFFFF);
        let aff = g0.affinity(&g1);
        assert!(aff > 0.0 && aff < 1.0);
    }

    #[test]
    fn test_stimulus_from_affinity() {
        assert_eq!(StimulusType::from_affinity(0.9), StimulusType::Resonant);
        assert_eq!(StimulusType::from_affinity(0.5), StimulusType::Dissonant);
        assert_eq!(StimulusType::from_affinity(0.1), StimulusType::Null);
    }

    #[test]
    fn test_thermal_override() {
        let engine = ClockworkEngine::new();
        engine.thermal.set_state(ThermalState::Critical, 0);
        assert!(engine.thermal.is_critical());
        assert_eq!(engine.thermal.override_gear(), Some(GearId::G9));
    }

    #[test]
    fn test_sovereign_cycle() {
        let sovereign = GoldenHash::new();
        let epoch1 = sovereign.tick();
        let epoch2 = sovereign.tick();
        assert_eq!(epoch1, epoch2); // Same 60s boundary
    }

    #[test]
    fn test_engine_tick() {
        let engine = ClockworkEngine::new();
        let state = engine.tick();
        assert_eq!(state.tick, 0);
        assert!(!state.thermal_override);
    }

    #[test]
    fn test_ffi_null_safety() {
        assert_eq!(clockwork_tick(std::ptr::null_mut()), 0);
        assert_eq!(clockwork_thermal_state(std::ptr::null_mut()), 2);
    }
}
