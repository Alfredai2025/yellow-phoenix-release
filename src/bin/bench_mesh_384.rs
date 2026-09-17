// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use pams::crystal_mesh_384::{CrystalMesh384, PAP_384_BYTES};
use pams::hybrid_mesh::{HybridMesh, PAP_512_BYTES};
use rand::Rng;
use std::time::Instant;

fn random_pap_384(rng: &mut impl Rng) -> [u8; PAP_384_BYTES] {
    let mut p = [0u8; PAP_384_BYTES];
    rng.fill(&mut p[..]);
    p
}

fn random_pap_512(rng: &mut impl Rng) -> [u8; PAP_512_BYTES] {
    let mut p = [0u8; PAP_512_BYTES];
    rng.fill(&mut p[..]);
    p
}

fn main() {
    let n = 100_000;
    let queries = 10_000;
    let mut rng = rand::rng();

    // === 384-bit mesh ===
    let mut mesh384 = CrystalMesh384::new(n * 2);
    let t0 = Instant::now();
    for i in 0..n {
        mesh384.insert(i as u64, random_pap_384(&mut rng));
    }
    let insert_384 = t0.elapsed();

    let t0 = Instant::now();
    let mut q384 = [0u8; PAP_384_BYTES];
    for _ in 0..queries {
        rng.fill(&mut q384[..]);
        let _ = mesh384.query_k(&q384, 50);
    }
    let query_384 = t0.elapsed();

    // === 512-bit mesh (padded) ===
    let mut mesh512 = HybridMesh::new(n * 2 / 8, n * 2); // coarse/fine caps
    let t0 = Instant::now();
    for i in 0..n {
        let p512 = random_pap_512(&mut rng);
        let p128 = [0u8; 16]; // dummy coarse
        mesh512.insert_dual(i as u64, &p128, &p512);
    }
    let insert_512 = t0.elapsed();

    let t0 = Instant::now();
    let mut q512 = [0u8; PAP_512_BYTES];
    for _ in 0..queries {
        rng.fill(&mut q512[..]);
        let _ = mesh512.fine.query_k(&q512, 50);
    }
    let query_512 = t0.elapsed();

    println!("=== Mesh Benchmark: 384-bit vs 512-bit ===");
    println!("Records: {}", n);
    println!("Queries: {}", queries);
    println!("");
    println!("384-bit insert: {:?}  ({:.0} rec/s)", insert_384, n as f64 / insert_384.as_secs_f64());
    println!("512-bit insert: {:?}  ({:.0} rec/s)", insert_512, n as f64 / insert_512.as_secs_f64());
    println!("");
    println!("384-bit query:  {:?}  ({:.0} qps)", query_384, queries as f64 / query_384.as_secs_f64());
    println!("512-bit query:  {:?}  ({:.0} qps)", query_512, queries as f64 / query_512.as_secs_f64());
    println!("");
    println!("Speedup insert: {:.2}x", insert_512.as_secs_f64() / insert_384.as_secs_f64());
    println!("Speedup query:  {:.2}x", query_512.as_secs_f64() / query_384.as_secs_f64());
}
