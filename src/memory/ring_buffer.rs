// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use core::ptr;
use core::sync::atomic;

/// Lock‑free single‑producer single‑consumer ring buffer.
///
/// `N` slots, each slot is 32 bytes. Head/tail indices are accessed through
/// volatile reads/writes and separated by a `SeqCst` fence.
pub struct SpscRingBuffer<const N: usize> {
    head: u32,
    tail: u32,
    buffer: [[u8; 32]; N],
}

impl<const N: usize> SpscRingBuffer<N> {
    /// Create an empty ring buffer with all slots zeroed.
    pub fn new() -> Self {
        Self {
            head: 0,
            tail: 0,
            buffer: [[0u8; 32]; N],
        }
    }

    /// Try to push a 32‑byte slot.
    ///
    /// Returns `false` if the buffer is full.
    pub fn try_push(&mut self, slot: &[u8; 32]) -> bool {
        let head = unsafe { ptr::read_volatile(&self.head) };
        let tail = unsafe { ptr::read_volatile(&self.tail) };

        let next_head = head.wrapping_add(1) % (N as u32);
        if next_head == tail {
            return false;
        }

        let idx = (head % (N as u32)) as usize;
        unsafe {
            ptr::copy_nonoverlapping(
                slot.as_ptr(),
                self.buffer[idx].as_mut_ptr(),
                32,
            );
        }

        atomic::fence(atomic::Ordering::SeqCst);
        unsafe {
            ptr::write_volatile(&mut self.head, next_head);
        }
        true
    }

    /// Try to pop a 32‑byte slot.
    ///
    /// Returns `None` when the buffer is empty.
    pub fn try_pop(&mut self) -> Option<[u8; 32]> {
        let tail = unsafe { ptr::read_volatile(&self.tail) };
        let head = unsafe { ptr::read_volatile(&self.head) };

        if tail == head {
            return None;
        }

        atomic::fence(atomic::Ordering::SeqCst);

        let idx = (tail % (N as u32)) as usize;
        let mut slot: [u8; 32] = [0u8; 32];
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffer[idx].as_ptr(),
                slot.as_mut_ptr(),
                32,
            );
        }

        let next_tail = tail.wrapping_add(1) % (N as u32);
        unsafe {
            ptr::write_volatile(&mut self.tail, next_tail);
        }
        Some(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_1m_without_corruption() {
        let mut ring = SpscRingBuffer::<1024>::new();
        for i in 0..1_000_000u32 {
            let mut slot = [0u8; 32];
            slot[0..4].copy_from_slice(&i.to_le_bytes());
            assert!(ring.try_push(&slot), "push failed at {}", i);
            let popped = ring.try_pop().expect("pop should succeed after push");
            let found = u32::from_le_bytes([popped[0], popped[1], popped[2], popped[3]]);
            assert_eq!(found, i, "corruption at index {}", i);
        }
    }
}
