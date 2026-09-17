// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Scar Encoding
//!
//! A scar is a compact memory of a past constitutional failure. It is encoded
//! into the 6-byte Reserve window of a gear's 512-bit hash. Future ticks can
//! read the scar and bias behavior (earlier alerts, tighter pitches).

use crate::clockwork::Gear;
use crate::gear_hash::{ConstitutionalWindow, read_window, write_window};

/// Classification of constitutional scars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ScarKind {
    /// Thermal stress override.
    Thermal = 0,
    /// Query latency override.
    Latency = 1,
    /// Recall/accuracy degradation.
    Recall = 2,
    /// Integrity (git/exam) violation.
    Integrity = 3,
    /// Memory or resource exhaustion.
    Resource = 4,
}

impl ScarKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(ScarKind::Thermal),
            1 => Some(ScarKind::Latency),
            2 => Some(ScarKind::Recall),
            3 => Some(ScarKind::Integrity),
            4 => Some(ScarKind::Resource),
            _ => None,
        }
    }
}

/// A scar encoded into the Reserve window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Scar {
    pub kind: ScarKind,
    /// Origin gear index (0-9).
    pub origin_gear: u8,
    /// Severity 0-15.
    pub severity: u8,
    /// 16-bit payload (e.g., override level, latency bucket).
    pub payload: u16,
}

impl Scar {
    /// Create a new scar.
    pub fn new(kind: ScarKind, origin_gear: u8, severity: u8, payload: u16) -> Self {
        Self {
            kind,
            origin_gear: origin_gear.min(9),
            severity: severity.min(15),
            payload,
        }
    }

    /// Encode the scar into the 6-byte Reserve window format.
    ///
    /// Layout:
    ///   byte 0: 0xA0 | kind  (magic nibble A + kind)
    ///   byte 1: origin_gear
    ///   byte 2: severity
    ///   byte 3: payload high byte
    ///   byte 4: payload low byte
    ///   byte 5: XOR checksum of bytes 0-4
    pub fn encode(&self) -> [u8; 6] {
        let mut bytes = [0u8; 6];
        bytes[0] = 0xA0 | (self.kind as u8);
        bytes[1] = self.origin_gear;
        bytes[2] = self.severity;
        bytes[3] = (self.payload >> 8) as u8;
        bytes[4] = self.payload as u8;
        bytes[5] = bytes[0] ^ bytes[1] ^ bytes[2] ^ bytes[3] ^ bytes[4];
        bytes
    }

    /// Decode a 6-byte window value into a scar.
    /// Returns `None` if no scar is present or the checksum is invalid.
    pub fn decode(bytes: [u8; 6]) -> Option<Self> {
        // No scar: all zeros (or any value with high nibble not 0xA).
        if bytes[0] & 0xF0 != 0xA0 {
            return None;
        }
        let checksum = bytes[0] ^ bytes[1] ^ bytes[2] ^ bytes[3] ^ bytes[4];
        if checksum != bytes[5] {
            return None;
        }
        let kind = ScarKind::from_u8(bytes[0] & 0x0F)?;
        Some(Self {
            kind,
            origin_gear: bytes[1].min(9),
            severity: bytes[2].min(15),
            payload: u16::from_be_bytes([bytes[3], bytes[4]]),
        })
    }

    /// Encode the scar into a gear's Reserve window.
    pub fn write_to_gear(&self, gear: &Gear) {
        let value = self.encode();
        gear.write_window(ConstitutionalWindow::Reserve, value);
    }

    /// Read a scar from a gear's Reserve window, if present.
    pub fn read_from_gear(gear: &Gear) -> Option<Self> {
        let bytes = gear.read_window(ConstitutionalWindow::Reserve);
        Self::decode(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clockwork::{GearId, PriorityVector};

    #[test]
    fn test_scar_roundtrip() {
        let scar = Scar::new(ScarKind::Latency, 0, 7, 0x1234);
        let bytes = scar.encode();
        let decoded = Scar::decode(bytes).unwrap();
        assert_eq!(scar, decoded);
    }

    #[test]
    fn test_no_scar_returns_none() {
        let bytes = [0u8; 6];
        assert!(Scar::decode(bytes).is_none());
    }

    #[test]
    fn test_type_check() {
        let thermal = Scar::new(ScarKind::Thermal, 8, 3, 0x0001);
        let latency = Scar::new(ScarKind::Latency, 0, 3, 0x0001);
        assert_ne!(thermal.kind, latency.kind);
        let thermal_bytes = thermal.encode();
        let decoded = Scar::decode(thermal_bytes).unwrap();
        assert_eq!(decoded.kind, ScarKind::Thermal);
        assert_eq!(decoded.origin_gear, 8);
    }

    #[test]
    fn test_corrupt_checksum_returns_none() {
        let scar = Scar::new(ScarKind::Recall, 7, 5, 0xABCD);
        let mut bytes = scar.encode();
        bytes[4] ^= 0xFF; // corrupt payload byte
        assert!(Scar::decode(bytes).is_none());
    }

    #[test]
    fn test_gear_roundtrip() {
        let gear = Gear::new(GearId::G0, PriorityVector { x: 1, y: 2, z: 3 }, 0xFFFF);
        let scar = Scar::new(ScarKind::Integrity, 0, 15, 0x00FF);
        scar.write_to_gear(&gear);
        let read = Scar::read_from_gear(&gear);
        assert_eq!(read, Some(scar));
    }
}
