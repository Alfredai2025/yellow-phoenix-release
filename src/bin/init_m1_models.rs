// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! init_m1_models.rs — Bootstrap default M1 component files for wiring scans.

use std::fs;
use std::path::Path;

use pams::learned_router::LearnedRouter;
use pams::self_learning::SelfLearningTable;

fn main() {
    let data_dir = Path::new("data");
    if !data_dir.exists() {
        fs::create_dir_all(data_dir).expect("create data dir");
    }

    // 1. Learned router default model (51 f32 values = 204 bytes).
    let router = LearnedRouter::new();
    router.save(data_dir.join("router_model_v1.bin"))
        .expect("save router model");
    println!("Saved data/router_model_v1.bin");

    // 2. Self-learning table (256 entries + version).
    // Save three times to populate backup rotation (.bak1, .bak2, .bak3).
    let table = SelfLearningTable::new();
    for i in 0..3 {
        let mut t = table.clone();
        t.version = i as u64;
        t.save(data_dir.join("learning_table_v1.bin"))
            .expect("save learning table");
    }
    println!("Saved data/learning_table_v1.bin with 3 backups");

    // 3. Result-cache placeholder (ResultCache is in-memory; file is a wiring-scan marker).
    let cache_marker = data_dir.join("cache_entries_v1.bin");
    if !cache_marker.exists() {
        fs::write(&cache_marker, b"").expect("create cache marker");
        println!("Created data/cache_entries_v1.bin placeholder");
    }

    // 4. Engine feeder named pipes.
    let pipes = [
        "/tmp/yp_feed_cascade",
        "/tmp/yp_feed_faiss",
        "/tmp/yp_feed_domain",
    ];
    for path in &pipes {
        let p = Path::new(path);
        if !p.exists() {
            #[cfg(unix)]
            {
                use std::process::Command;
                let _ = Command::new("mkfifo").arg(path).status();
            }
            println!("Created pipe {}", path);
        }
    }

    println!("M1 component files initialised.");
}
