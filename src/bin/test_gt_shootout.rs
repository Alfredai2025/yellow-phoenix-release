// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Strict argmin R@1 probe for the 10M HNSW graph using the audited
//! `yp_shootout_run_hnsw_gt_phase` (exact brute-force popcount ground truth,
//! argmin single-id comparison).
//!
//! Usage: test_gt_shootout <hnsw_path> [n] [ef_list_csv]
//!   e.g. test_gt_shootout data/papers_10m_hnsw_v5_m12.bin 100 1000,2000,4000
//!
//! For each ef: the gt phase runs TWICE — a warm-up pass (faults in mmap
//! pages, discarded) then a measured pass. Modes: 0=perturb_50 (50 random
//! bit flips ≈ 10% of 512 bits, the device "perturb" mode), 1=exclude_self,
//! 2=random_query.

use std::ffi::CString;
use pams::ffi_shootout::{
    yp_shootout_load_hnsw, yp_shootout_run_hnsw_gt_phase, yp_shootout_set_ef_search,
    YpModeMetrics, YpShootoutMetrics,
};

fn zero_mode() -> YpModeMetrics {
    YpModeMetrics {
        ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
        ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
        ism_r10: 0.0, hnsw_r10: 0.0,
    }
}

fn zero_metrics() -> YpShootoutMetrics {
    YpShootoutMetrics {
        perturb_50: zero_mode(),
        exclude_self: zero_mode(),
        random_query: zero_mode(),
        ism_mem_kb: 0,
        hnsw_mem_kb: 0,
        count: 0,
    }
}

fn print_mode(label: &str, m: &YpModeMetrics) {
    println!(
        "  {:<12} r1={:.4} r5={:.4} r10={:.4}  p50={}us p95={}us p99={}us",
        label, m.hnsw_r1, m.hnsw_r5, m.hnsw_r10, m.hnsw_p50_us, m.hnsw_p95_us, m.hnsw_p99_us
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <hnsw_path> [n] [ef_list_csv]", args[0]);
        std::process::exit(1);
    }
    let hnsw_path = CString::new(args[1].as_str()).unwrap();
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let ef_list: Vec<u32> = args.get(3)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![128]);

    unsafe {
        println!("Loading HNSW {} ...", args[1]);
        let r = yp_shootout_load_hnsw(hnsw_path.as_ptr());
        println!("load_hnsw -> {}", r);
        if r != 0 { std::process::exit(1); }

        for &ef in &ef_list {
            yp_shootout_set_ef_search(ef);

            // Warm-up pass (pages + branch predictors), discarded.
            let mut warm = zero_metrics();
            let t = std::time::Instant::now();
            let r = yp_shootout_run_hnsw_gt_phase(n, &mut warm);
            println!("\n=== ef={} warm-up pass -> rc={} (took {:?}, discarded) ===", ef, r, t.elapsed());
            if r != 0 { std::process::exit(1); }

            // Measured pass.
            let mut m = zero_metrics();
            let t = std::time::Instant::now();
            let r = yp_shootout_run_hnsw_gt_phase(n, &mut m);
            println!("=== ef={} measured pass -> rc={} (took {:?}) ===", ef, r, t.elapsed());
            if r != 0 { std::process::exit(1); }
            println!("count={} hnsw_mem_kb={}", m.count, m.hnsw_mem_kb);
            print_mode("perturb_50", &m.perturb_50);
            print_mode("exclude_self", &m.exclude_self);
            print_mode("random_query", &m.random_query);
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }
}
