// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::fs::File;
use std::io::Read;
use std::time::Instant;

use pams::cheat_sheet_cascade::{
    service_init, service_query, service_learn, BinaryVector,
};

fn load_vectors(path: &str) -> Vec<BinaryVector> {
    let mut file = File::open(path).expect("open vectors");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read vectors");
    let n = buf.len() / 64;
    (0..n).map(|i| {
        let mut v = [0u8; 64];
        v.copy_from_slice(&buf[i * 64..(i + 1) * 64]);
        v
    }).collect()
}

fn load_labels(path: &str) -> Vec<u64> {
    let mut file = File::open(path).expect("open labels");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read labels");
    let n = buf.len() / 8;
    (0..n).map(|i| {
        let mut b = [0u8; 8];
        b.copy_from_slice(&buf[i * 8..(i + 1) * 8]);
        u64::from_le_bytes(b)
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <vectors.bin> <labels.bin> [epochs]", args[0]);
        std::process::exit(1);
    }
    
    let vectors = load_vectors(&args[1]);
    let labels = load_labels(&args[2]);
    let epochs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    
    println!("Loaded {} vectors", vectors.len());
    service_init();
    
    for epoch in 0..epochs {
        let start = Instant::now();
        let mut hits = 0;
        let mut misses = 0;
        for i in 0..vectors.len() {
            let (_, layer, _) = service_query(&vectors[i]);
            if layer == 0 {
                misses += 1;
                service_learn(&vectors[i], labels[i]);
            } else {
                hits += 1;
            }
        }
        println!("Epoch {}: {} hits, {} misses ({:.1} ms)", 
            epoch + 1, hits, misses, start.elapsed().as_secs_f64() * 1000.0);
    }
    
    let start = Instant::now();
    let mut hits = 0;
    for v in &vectors {
        let (_, layer, _) = service_query(v);
        if layer != 0 { hits += 1; }
    }
    let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
    
    println!("\n==================================================");
    println!("  YP v3 PURE RUST — {} queries", vectors.len());
    println!("==================================================");
    println!("  Hits:       {} ({:.1}%)", hits, 100.0 * hits as f64 / vectors.len() as f64);
    println!("  Avg time:   {:.2} µs", elapsed / vectors.len() as f64);
    println!("  Total:      {:.2} ms", elapsed / 1000.0);
    println!("==================================================");
}
