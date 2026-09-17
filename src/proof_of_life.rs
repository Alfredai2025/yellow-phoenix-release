// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PAP_128_BYTES: usize = 16;
const PAP_512_BYTES: usize = 64;

fn pap_128_from_seed(seed: u64) -> [u8; PAP_128_BYTES] {
    let mut arr = [0u8; PAP_128_BYTES];
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for i in 0..PAP_128_BYTES {
        s = s.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
        arr[i] = (s >> 56) as u8;
    }
    arr
}

fn pap_512_from_seed(seed: u64) -> [u8; PAP_512_BYTES] {
    let mut arr = [0u8; PAP_512_BYTES];
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for i in 0..PAP_512_BYTES {
        s = s.wrapping_mul(0x2545F4914F6CDD1D).wrapping_add(1);
        arr[i] = (s >> 56) as u8;
    }
    arr
}

pub fn generate_proof_of_life() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let start = Instant::now();
    let mut mesh = crate::hybrid_mesh::HybridMesh::new(100_000, 100_000);
    for i in 0..100_000u64 {
        mesh.insert_dual(i, &pap_128_from_seed(i), &pap_512_from_seed(i + 1_000_000));
    }
    mesh.build_edges(8);
    let p128 = pap_128_from_seed(50_000);
    let p512 = pap_512_from_seed(1_050_000);
    let _ = mesh.query_auto(&p128, &p512, 10);
    let elapsed_ns = start.elapsed().as_nanos();

    let payload = format!("YP_POL|{}|{}", timestamp, elapsed_ns);
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in payload.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}
