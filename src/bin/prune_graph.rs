// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Prune a binary HNSW graph with Vamana-style diverse-neighbor pruning.
//! Usage: prune_graph <in.bin> <out.bin> <alpha> [layer] [mode] [coords.f32]
//! mode: (none)=naive | adaptive | geo (needs coords = node PCA-2 f32)

use std::env;
use std::fs::File;
use std::io::Read;
use pams::binary_hnsw::BinaryHNSW;

fn load_f32(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let mut g = BinaryHNSW::load(&a[1]).expect("load graph");
    let alpha: f32 = a[3].parse().expect("alpha");
    let layer: usize = if a.len() > 4 { a[4].parse().expect("layer") } else { 0 };
    let mode = if a.len() > 5 { a[5].as_str() } else { "" };
    let removed = match mode {
        "adaptive" => g.prune_diverse_adaptive(alpha, layer, 8),
        "rev" => g.prune_diverse(alpha, layer, true),
        "geo" => {
            let coords = load_f32(&a[6]);
            let n2 = coords.len() / 2;
            let rms = (0..n2).map(|i| (coords[i*2]*coords[i*2] + coords[i*2+1]*coords[i*2+1]).sqrt()).sum::<f32>() / n2 as f32;
            g.prune_diverse_geo(alpha, layer, &coords, rms)
        }
        _ => g.prune_diverse(alpha, layer, false),
    };
    eprintln!("pruned layer {layer}: removed {removed} primary edges (alpha={alpha}{mode})");
    g.save(&a[2]).expect("save graph");
    println!("saved {}", a[2]);
}
