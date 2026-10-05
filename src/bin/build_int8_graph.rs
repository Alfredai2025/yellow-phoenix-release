// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Build Int8HNSW from raw residual-int8 codes + coarse table.
//! Usage: build_int8_graph <codes.i8 (N x 128 int8)> <cells.u16 (N)> <coarse.f32 (KC x 128)> <out.bin> [m] [efC]

use std::env;
use std::fs::File;
use std::io::Read;
use pams::int8_hnsw::{I8Hnsw, D};

fn main() {
    let a: Vec<String> = env::args().collect();
    let mut codes_raw = Vec::new();
    File::open(&a[1]).expect("codes").read_to_end(&mut codes_raw).expect("read codes");
    let n = codes_raw.len() / D;
    let mut cells_raw = Vec::new();
    File::open(&a[2]).expect("cells").read_to_end(&mut cells_raw).expect("read cells");
    let mut coarse_raw = Vec::new();
    File::open(&a[3]).expect("coarse").read_to_end(&mut coarse_raw).expect("read coarse");
    let kc = coarse_raw.len() / 4 / D;
    let coarse: Vec<f32> = coarse_raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let m: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(24);
    let efc: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(400);
    let mut g = if std::path::Path::new(&a[4]).exists() {
        eprintln!("resuming from existing {}", a[4]);
        I8Hnsw::load(&a[4]).expect("resume load")
    } else {
        I8Hnsw::new(m, efc, coarse, kc)
    };
    eprintln!("building Int8HNSW: n={} kc={} m={} efC={} (resume at {})", n, kc, m, efc, g.nodes.len());
    let t0 = std::time::Instant::now();
    let start = g.nodes.len();
    for i in start..n {
        let mut code = [0i8; D];
        code.copy_from_slice(&codes_raw[i * D..(i + 1) * D].iter().map(|&b| b as i8).collect::<Vec<i8>>());
        let cell = u16::from_le_bytes([cells_raw[i * 2], cells_raw[i * 2 + 1]]);
        g.insert(i as u64, cell, code);
        if i % 100000 == 99999 { 
            eprintln!("  {}/{} ({:.0}/s)", i + 1, n, (i + 1 - start) as f64 / t0.elapsed().as_secs_f64().max(0.1));
            g.save(&a[4]).expect("checkpoint save");
        }
    }
    g.save(&a[4]).expect("save");
    eprintln!("saved {} in {:.0}s", a[4], t0.elapsed().as_secs_f64());
}
