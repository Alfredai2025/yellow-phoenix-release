// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Bridge-edge overlay: after pruning freed arena slots, add long-range edges
//! from each node to its coarse PCA-2 cell anchor (highway/bridge overlay).
//! Usage: bridge_graph <graph.bin> <nodecoords.f32> <out.bin> [grid=8]

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
    let coords = load_f32(&a[2]);
    let n = coords.len() / 2;
    let grid: usize = if a.len() > 4 { a[4].parse().expect("grid") } else { 8 };
    // coarse cell anchor = node nearest each cell's coordinate mean
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for i in 0..n {
        xmin = xmin.min(coords[i * 2]); xmax = xmax.max(coords[i * 2]);
        ymin = ymin.min(coords[i * 2 + 1]); ymax = ymax.max(coords[i * 2 + 1]);
    }
    let cell = |x: f32, y: f32| -> usize {
        let cx = (((x - xmin) / (xmax - xmin + 1e-9)) * (grid as f32 - 0.001)) as usize;
        let cy = (((y - ymin) / (ymax - ymin + 1e-9)) * (grid as f32 - 0.001)) as usize;
        cy * grid + cx
    };
    let mut sum_x = vec![0f64; grid * grid];
    let mut sum_y = vec![0f64; grid * grid];
    let mut cnt_c = vec![0u64; grid * grid];
    for i in 0..n {
        let c = cell(coords[i * 2], coords[i * 2 + 1]);
        sum_x[c] += coords[i * 2] as f64;
        sum_y[c] += coords[i * 2 + 1] as f64;
        cnt_c[c] += 1;
    }
    let mut anchor = vec![u32::MAX; grid * grid];
    let mut best_d = vec![f32::MAX; grid * grid];
    for i in 0..n {
        let c = cell(coords[i * 2], coords[i * 2 + 1]);
        if cnt_c[c] == 0 { continue; }
        let mx = (sum_x[c] / cnt_c[c] as f64) as f32;
        let my = (sum_y[c] / cnt_c[c] as f64) as f32;
        let d = (coords[i * 2] - mx).hypot(coords[i * 2 + 1] - my);
        if d < best_d[c] { best_d[c] = d; anchor[c] = i as u32; }
    }
    let random_mode = a.len() > 5 && a[5] == "random";
    let added = if random_mode {
        // NEGATIVE CONTROL: bridge to a hash-random node (pure fn of index, no mutable capture)
        g.add_bridge_edges(n, |i| {
            let h = (i as u64).wrapping_mul(0x2545F4914F6CDD1D) ^ 0x9E3779B97F4A7C15;
            let h = h ^ (h >> 29);
            (h % n as u64) as u32
        })
    } else {
        g.add_bridge_edges(n, |i| anchor[cell(coords[i * 2], coords[i * 2 + 1])])
    };
    eprintln!("bridge overlay: {} cells, {} bridge edges added", grid * grid, added);
    g.save(&a[3]).expect("save");
    println!("saved {}", a[3]);
}
