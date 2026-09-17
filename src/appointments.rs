// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Yellow Phoenix — Constitutional Appointments
//!
//! Appointments are forced gear alignments scheduled for a specific tick.
//! They carry a 6-byte opcode payload and are resolved when their due tick
//! arrives.

use crate::clockwork::GearId;
use std::collections::BinaryHeap;

/// A scheduled constitutional appointment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Appointment {
    pub gear: GearId,
    pub due_tick: u64,
    pub opcode: [u8; 6],
}

// Reverse ordering so the earliest due_tick is at the top of the heap.
impl Ord for Appointment {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.due_tick.cmp(&self.due_tick)
    }
}

impl PartialOrd for Appointment {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A book of pending appointments.
pub struct AppointmentBook {
    appointments: BinaryHeap<Appointment>,
}

impl AppointmentBook {
    /// Create an empty appointment book.
    pub fn new() -> Self {
        Self {
            appointments: BinaryHeap::new(),
        }
    }

    /// Schedule a new appointment.
    pub fn schedule(&mut self, gear: GearId, due_tick: u64, opcode: [u8; 6]) {
        self.appointments.push(Appointment {
            gear,
            due_tick,
            opcode,
        });
    }

    /// Number of pending appointments.
    pub fn len(&self) -> usize {
        self.appointments.len()
    }

    /// True if no appointments are pending.
    pub fn is_empty(&self) -> bool {
        self.appointments.is_empty()
    }

    /// Return all appointments due at exactly `tick`, without removing them.
    pub fn due_at(&self, tick: u64) -> Vec<Appointment> {
        self.appointments
            .iter()
            .filter(|a| a.due_tick == tick)
            .copied()
            .collect()
    }

    /// Return all appointments due at or before `tick`, without removing them.
    pub fn overdue(&self, tick: u64) -> Vec<Appointment> {
        self.appointments
            .iter()
            .filter(|a| a.due_tick <= tick)
            .copied()
            .collect()
    }

    /// Remove and return all appointments due at or before `tick`.
    pub fn resolve_up_to(&mut self, tick: u64) -> Vec<Appointment> {
        let mut resolved = Vec::new();
        let mut remaining = BinaryHeap::new();
        while let Some(a) = self.appointments.pop() {
            if a.due_tick <= tick {
                resolved.push(a);
            } else {
                remaining.push(a);
            }
        }
        self.appointments = remaining;
        resolved
    }

    /// Peek at the next upcoming appointment, if any.
    pub fn next(&self) -> Option<Appointment> {
        self.appointments.peek().copied()
    }
}

impl Default for AppointmentBook {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_due_at_deterministic() {
        let mut book = AppointmentBook::new();
        book.schedule(GearId::G0, 10, [1, 2, 3, 4, 5, 6]);
        book.schedule(GearId::G2, 20, [6, 5, 4, 3, 2, 1]);
        book.schedule(GearId::G7, 10, [0; 6]);

        let due_at_10 = book.due_at(10);
        assert_eq!(due_at_10.len(), 2);
        assert!(due_at_10.iter().any(|a| a.gear == GearId::G0));
        assert!(due_at_10.iter().any(|a| a.gear == GearId::G7));

        let due_at_20 = book.due_at(20);
        assert_eq!(due_at_20.len(), 1);
        assert_eq!(due_at_20[0].gear, GearId::G2);
    }

    #[test]
    fn test_resolve_up_to() {
        let mut book = AppointmentBook::new();
        book.schedule(GearId::G0, 5, [1; 6]);
        book.schedule(GearId::G1, 10, [2; 6]);
        book.schedule(GearId::G2, 15, [3; 6]);

        let resolved = book.resolve_up_to(10);
        assert_eq!(resolved.len(), 2);
        assert_eq!(book.len(), 1);
        assert_eq!(book.next().unwrap().gear, GearId::G2);
    }

    #[test]
    fn test_overdue() {
        let mut book = AppointmentBook::new();
        book.schedule(GearId::G3, 7, [7; 6]);
        book.schedule(GearId::G4, 12, [8; 6]);

        let overdue = book.overdue(10);
        assert_eq!(overdue.len(), 1);
        assert_eq!(overdue[0].gear, GearId::G3);
    }
}
