// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::{c_char, c_int, CString};
use pams::ffi_shootout::{
    yp_shootout_load_hnsw, yp_shootout_load_ism,
    yp_shootout_run_hnsw_phase, yp_shootout_run_ism_phase,
    yp_shootout_set_ef_search, yp_shootout_unload_hnsw, yp_shootout_unload_ism,
    YpModeMetrics, YpShootoutMetrics,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <ism_path> <hnsw_path> [n] [ef]", args[0]);
        std::process::exit(1);
    }
    let ism_path = CString::new(args[1].as_str()).unwrap();
    let hnsw_path = CString::new(args[2].as_str()).unwrap();
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);
    let ef: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(128);

    unsafe {
        println!("Loading ISM ...");
        let r = yp_shootout_load_ism(ism_path.as_ptr());
        println!("load_ism -> {}", r);
        if r != 0 { std::process::exit(1); }

        let mut ism = YpShootoutMetrics {
            perturb_50: YpModeMetrics {
                ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
                ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
                ism_r10: 0.0, hnsw_r10: 0.0,
            },
            exclude_self: YpModeMetrics {
                ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
                ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
                ism_r10: 0.0, hnsw_r10: 0.0,
            },
            random_query: YpModeMetrics {
                ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
                ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
                ism_r10: 0.0, hnsw_r10: 0.0,
            },
            ism_mem_kb: 0,
            hnsw_mem_kb: 0,
            count: 0,
        };
        println!("Running ISM phase n={} ...", n);
        let r = yp_shootout_run_ism_phase(n, &mut ism);
        println!("run_ism_phase -> {}", r);
        println!("ISM perturb p50={}us p95={}us", ism.perturb_50.ism_p50_us, ism.perturb_50.ism_p95_us);
        println!("ISM exclude p50={}us", ism.exclude_self.ism_p50_us);
        println!("ISM random  p50={}us", ism.random_query.ism_p50_us);
        println!("ISM count={} mem={}KB", ism.count, ism.ism_mem_kb);
        if r != 0 { std::process::exit(1); }

        yp_shootout_unload_ism();

        println!("Loading HNSW ...");
        let r = yp_shootout_load_hnsw(hnsw_path.as_ptr());
        println!("load_hnsw -> {}", r);
        if r != 0 { std::process::exit(1); }

        yp_shootout_set_ef_search(ef);

        let mut hnsw = YpShootoutMetrics {
            perturb_50: YpModeMetrics {
                ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
                ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
                ism_r10: 0.0, hnsw_r10: 0.0,
            },
            exclude_self: YpModeMetrics {
                ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
                ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
                ism_r10: 0.0, hnsw_r10: 0.0,
            },
            random_query: YpModeMetrics {
                ism_p50_us: 0, ism_p95_us: 0, ism_p99_us: 0, hnsw_p50_us: 0, hnsw_p95_us: 0, hnsw_p99_us: 0,
                ism_r1: 0.0, hnsw_r1: 0.0, ism_r5: 0.0, hnsw_r5: 0.0,
                ism_r10: 0.0, hnsw_r10: 0.0,
            },
            ism_mem_kb: 0,
            hnsw_mem_kb: 0,
            count: 0,
        };
        println!("Running HNSW phase n={} ef={} ...", n, ef);
        let r = yp_shootout_run_hnsw_phase(n, &mut hnsw);
        println!("run_hnsw_phase -> {}", r);
        println!("HNSW perturb p50={}us r1={}", hnsw.perturb_50.hnsw_p50_us, hnsw.perturb_50.hnsw_r1);
        println!("HNSW exclude p50={}us r1={} r5={} r10={}",
                 hnsw.exclude_self.hnsw_p50_us, hnsw.exclude_self.hnsw_r1,
                 hnsw.exclude_self.hnsw_r5, hnsw.exclude_self.hnsw_r10);
        println!("HNSW random  p50={}us r1={}", hnsw.random_query.hnsw_p50_us, hnsw.random_query.hnsw_r1);
        println!("HNSW count={} mem={}KB", hnsw.count, hnsw.hnsw_mem_kb);

        yp_shootout_unload_hnsw();
    }
}
