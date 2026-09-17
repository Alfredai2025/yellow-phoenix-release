// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

#[cfg(feature = "holographic-cascade")]
use pams::holographic_cascade::HolographicCascade;

#[cfg(feature = "holographic-cascade")]
fn main() {
    let h = HolographicCascade::new(8, 8, 8, 512);
    println!("len: {}", h.len());
    println!("vaults: {}", h.vaults.len());
    drop(h);
    println!("ok");
}

#[cfg(not(feature = "holographic-cascade"))]
fn main() {
    println!("skipped: build without the `holographic-cascade` feature");
}
