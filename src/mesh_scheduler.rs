// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Mesh Scheduler — computes gear alignments and executes stimulus functions.
//!
//! When two gears align (their positions modulo LCM are equal), they mesh.
//! The stimulus type depends on the sender's identity and the gears involved.

use crate::clockwork::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Stimulus function types.
pub enum StimulusFunction {
    Xor,            // ~1 cycle/byte — QRY, MEM, THR, EXM
    InnerProduct,   // ~3 cycles/byte — ENC, AUT, DRF, GRW, CGT, LOG
    GeometricProduct, // ~50-200 cycles/byte — GEO, CGT, Golden Hash
}

/// Determines which stimulus function to use for a gear pair.
pub fn stimulus_function(sender: GearId, receiver: GearId) -> StimulusFunction {
    use GearId::*;
    match (sender, receiver) {
        // Fast gears use XOR
        (G0, G2) | (G2, G0) => StimulusFunction::Xor, // QRY ↔ MEM
        (G0, G8) | (G8, G0) => StimulusFunction::Xor, // QRY ↔ THR
        (G8, G3) | (G3, G8) => StimulusFunction::Xor, // THR ↔ GEO
        (G10, G6) | (G6, G10) => StimulusFunction::Xor, // EXM ↔ GRW
        (G10, G1) | (G1, G10) => StimulusFunction::Xor, // EXM ↔ ENC

        // GEO and CGT use Geometric Product
        (G3, G7) | (G7, G3) => StimulusFunction::GeometricProduct, // GEO ↔ CGT

        // Golden Hash uses Geometric Product with all gears
        (_, G9) | (G9, _) => StimulusFunction::GeometricProduct,

        // Everything else uses Inner Product
        _ => StimulusFunction::InnerProduct,
    }
}

/// Execute stimulus between two gears.
/// Returns the stimulus value (0.0 to 1.0) and the resulting state change.
pub fn execute_stimulus(
    sender: &Gear,
    receiver: &Gear,
    function: StimulusFunction,
) -> (f32, Vec<u8>) {
    match function {
        StimulusFunction::Xor => {
            let s_hash = sender.hash.lock().unwrap();
            let r_hash = receiver.hash.lock().unwrap();
            let mut result = vec![0u8; 64];
            let mut agreement = 0usize;
            for i in 0..64 {
                result[i] = s_hash.bytes[i] ^ r_hash.bytes[i];
                if result[i] == 0 {
                    agreement += 1;
                }
            }
            let affinity = agreement as f32 / 64.0;
            (affinity, result)
        }
        StimulusFunction::InnerProduct => {
            let affinity = sender.affinity(receiver);
            let payload = vec![(affinity * 255.0) as u8];
            (affinity, payload)
        }
        StimulusFunction::GeometricProduct => {
            // Placeholder: full Clifford G(4,1) product for Phase 3
            // Phase 2: use scalar approximation
            let affinity = sender.affinity(receiver);
            let payload = vec![(affinity * 255.0) as u8; 6];
            (affinity, payload)
        }
    }
}

/// Compute which gear pairs are aligned at a given tick.
pub fn compute_alignments(gears: &[Arc<Gear>], tick: u64) -> Vec<(GearId, GearId)> {
    let mut aligned = Vec::new();
    for i in 0..gears.len() {
        for j in (i + 1)..gears.len() {
            let g1 = &gears[i];
            let g2 = &gears[j];
            let p1 = g1.position.load(Ordering::Relaxed);
            let p2 = g2.position.load(Ordering::Relaxed);
            // Alignment: positions are close modulo LCM of pitches
            let pitch1 = g1.pitch.load(Ordering::Relaxed);
            let pitch2 = g2.pitch.load(Ordering::Relaxed);
            let lcm = pitch1 * pitch2 / gcd(pitch1, pitch2);
            if (p1 % lcm) == (p2 % lcm) {
                aligned.push((g1.id, g2.id));
            }
        }
    }
    aligned
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stimulus_function_xor() {
        assert!(matches!(
            stimulus_function(GearId::G0, GearId::G2),
            StimulusFunction::Xor
        ));
    }

    #[test]
    fn test_stimulus_function_geometric() {
        assert!(matches!(
            stimulus_function(GearId::G3, GearId::G7),
            StimulusFunction::GeometricProduct
        ));
    }

    #[test]
    fn test_xor_stimulus_range() {
        let g1 = Gear::new(GearId::G0, PriorityVector { x: 100, y: 0, z: 0 }, 0xFFFF);
        let g2 = Gear::new(GearId::G2, PriorityVector { x: 100, y: 0, z: 0 }, 0xFFFF);
        let (affinity, _) = execute_stimulus(&g1, &g2, StimulusFunction::Xor);
        assert!(affinity >= 0.0 && affinity <= 1.0);
    }
}
