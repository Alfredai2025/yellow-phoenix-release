// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Cache‑line aware memory layout.

/// A cache‑line sized block of exactly 64 bytes.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug)]
pub struct CacheLine(pub [u8; 64]);

/// Marker trait for types that must fit in L1 cache.
pub trait L1Block {}

/// Compile‑time marker for cache‑hot data.
pub const HOT_PATH: bool = true;
