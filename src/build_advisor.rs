// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use sysinfo::System;

pub struct BuildAdvisor;

impl BuildAdvisor {
    pub fn new() -> Self {
        BuildAdvisor
    }

    pub fn advice_for_failure(&self, total: u64) -> String {
        let mut system = System::new_all();
        system.refresh_memory();
        
        let available_mb = system.available_memory() / 1024 / 1024;
        let total_mb = system.total_memory() / 1024 / 1024;
        
        // Estimate: 8 bytes ID + 64 bytes hash + 16 bytes overhead = 88 bytes per paper
        let needed_mb = (total * 88) / 1024 / 1024;
        
        if needed_mb > available_mb {
            let suggested_shards = ((needed_mb as f64 / available_mb as f64).ceil() as usize).max(2).min(64);
            let gap_mb = needed_mb - available_mb;
            format!(
                "Build failed: {} papers need ~{}MB, only {}MB free ({}MB short). \
                 Suggest: {} shards, or free {}MB (close browsers/IDEs).",
                total, needed_mb, available_mb, gap_mb, suggested_shards, gap_mb
            )
        } else {
            format!(
                "Build failed: {} papers, memory OK ({}MB free of {}MB total). \
                 Check: disk space, file permissions, or hash_len mismatch.",
                total, available_mb, total_mb
            )
        }
    }
}
