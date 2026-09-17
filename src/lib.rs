// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(static_mut_refs)]

extern crate alloc;

#[cfg(all(not(test), not(feature = "std")))]
#[panic_handler]
fn panic(_info: &::core::panic::PanicInfo) -> ! {
    unsafe { libc::abort(); }
}

#[cfg(all(not(test), not(feature = "std")))]
#[no_mangle]
extern "C" fn rust_eh_personality() {}

// ---------- global allocator (libc-backed, required for the no_std prototype Vec) ----------

#[cfg(not(any(test, feature = "std")))]
mod alloc_fallback {
    use core::alloc::{GlobalAlloc, Layout};
    use core::mem;
    use core::ptr;
    use libc::{c_void, free, posix_memalign};

    pub struct LibcAllocator;

    unsafe impl GlobalAlloc for LibcAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let size = layout.size();

            // Zero-size allocation: return a non-null dangling pointer.
            if size == 0 {
                // For zero-sized allocations, Rust expects a non-null pointer.
                return ptr::NonNull::dangling().as_ptr();
            }

            // posix_memalign requires alignment ≥ sizeof(void*) and a power of two.
            let pointer_size = mem::size_of::<*const u8>();
            let align = layout.align().max(pointer_size);

            let mut memptr: *mut c_void = ptr::null_mut();
            let ret = unsafe { posix_memalign(&mut memptr, align, size) };
            if ret == 0 {
                memptr as *mut u8
            } else {
                ptr::null_mut()
            }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
            unsafe { free(ptr as *mut c_void) };
        }
    }

    #[global_allocator]
    static ALLOCATOR: LibcAllocator = LibcAllocator;
}

pub mod types;
pub mod memory;
pub mod core;
pub mod distance;
pub mod query;
pub mod direct_hash;
pub mod math;
pub mod solver;
pub mod crystal;
pub mod router;
pub mod lsh;

#[cfg(feature = "binary-hnsw")]
pub mod binary_hnsw;
pub mod float_hnsw;
pub mod ffi_float_hnsw;
pub mod sah_hash_bridge;

#[cfg(feature = "binary-hnsw")]
pub mod ffi_binary_hnsw;

#[cfg(feature = "binary-hnsw")]
pub mod ffi_engine;

#[cfg(feature = "binary-hnsw")]
pub mod ffi_itq;

#[cfg(feature = "binary-hnsw")]
pub mod ism_flat;

#[cfg(feature = "binary-hnsw")]
pub mod ffi_shootout;

#[cfg(feature = "binary-hnsw")]
pub mod ffi_pq;

#[cfg(feature = "binary-hnsw")]
pub mod pq_rerank;

#[cfg(feature = "binary-hnsw")]
pub mod ffi_pq_rerank;

#[cfg(feature = "holographic-cascade")]
pub mod holographic_cascade;

#[cfg(feature = "holographic-cascade")]
pub mod ffi_holographic;

#[cfg(feature = "holographic-cascade")]
pub mod pattern_keys;

#[cfg(feature = "holographic-cascade")]
pub mod spectral_hologram;

#[cfg(feature = "holographic-cascade")]
pub mod ffi_spectral_holo;

#[cfg(feature = "holographic-cascade")]
pub use holographic_cascade::{HolographicCascade, HologramTier1, HolographicVault};

pub mod genesis;
pub mod gear_hash;
pub mod mesh_scheduler;
pub mod scar_equilibrium;

#[cfg(feature = "std")]
pub mod golden_hash;

#[cfg(feature = "std")]
pub mod snapshot;

#[cfg(feature = "std")]
pub mod scar;

#[cfg(feature = "std")]
pub mod itf;

#[cfg(feature = "std")]
pub mod exm;

#[cfg(feature = "std")]
pub mod aut_bridge_health;

#[cfg(feature = "std")]
pub mod sentinels;

#[cfg(feature = "std")]
pub mod appointments;

#[cfg(feature = "std")]
pub mod sphere_rotor_bank;

#[cfg(feature = "std")]
pub mod ffi_clockwork;

#[cfg(feature = "std")]
pub mod clockwork;

#[cfg(feature = "std")]
pub mod rerank_engine;

pub mod ffi;
pub mod algebra;
#[cfg(feature = "std")]
pub mod phyllotactic;
pub mod ffi_kernel;

// ---------- FFI bridge ----------

use crate::core::resonance::QueryPipeline;
use crate::core::encoder::Encoder;
use crate::memory::ring_buffer::SpscRingBuffer;

static mut PIPELINE: Option<QueryPipeline> = None;
static mut RING: Option<SpscRingBuffer<1024>> = None;

/// Initialise the pipeline and the ring buffer with custom thresholds.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn mirror_mesh_init_with_thresholds(t0: u32, t1: u32, t2: u32) -> i32 {
    unsafe {
        PIPELINE = Some(QueryPipeline::with_thresholds(t0, t1, t2));
        RING = Some(SpscRingBuffer::<1024>::new());
    }
    0
}

/// Initialise the pipeline and the ring buffer.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn mirror_mesh_init() -> i32 {
    unsafe {
        PIPELINE = Some(QueryPipeline::new_default());
        RING = Some(SpscRingBuffer::<1024>::new());
    }
    0
}

/// Encode a paper from raw bytes and add it to every tier.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn mirror_mesh_add_paper(data: *const u8, len: usize) -> i32 {
    if data.is_null() || len == 0 {
        return -1;
    }
    let text = match unsafe { ::core::str::from_utf8(::core::slice::from_raw_parts(data, len)) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let encoder = Encoder::new(0);
    let bow = encoder.encode_paper_bow(text);
    let exact = encoder.encode_paper(text);
    unsafe {
        let pipeline = match PIPELINE.as_mut() {
            Some(p) => p,
            None => return -1,
        };
        pipeline.add_paper_bow(&bow, &exact);
    }
    0
}

#[no_mangle]
pub extern "C" fn mirror_mesh_query(
    data: *const u8,
    len: usize,
    result: *mut u8,
    result_len: usize,
) -> i32 {
    if data.is_null() || result.is_null() || result_len < 16 {
        return -11;
    }
    if len == 0 {
        return -1;
    }
    let text = match unsafe { ::core::str::from_utf8(::core::slice::from_raw_parts(data, len)) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let encoder = Encoder::new(0);
    let bow = encoder.encode_paper_bow(text);
    let exact = encoder.encode_paper(text);

    let pipeline = match unsafe { PIPELINE.as_mut() } {
        Some(p) => p,
        None => return -13,
    };
    let query_result = pipeline.query_bow(&bow, &exact);

    let match_count = query_result.matches.len() as i32;

    let mut best_dist: Option<u32> = None;
    let mut best_idx: usize = 0;
    for (i, dist) in &query_result.matches {
        if best_dist.map_or(true, |d| *dist < d) {
            best_dist = Some(*dist);
            best_idx = *i;
        }
    }

    let score: f32 = best_dist.map(|d| 1.0 / (1.0 + d as f32)).unwrap_or(0.0_f32);
    let top_idx: i32 = best_idx as i32;

    let out = unsafe { ::core::slice::from_raw_parts_mut(result, result_len) };
    out[0..4].copy_from_slice(&score.to_le_bytes());
    out[4..8].copy_from_slice(&top_idx.to_le_bytes());
    out[8..12].copy_from_slice(&match_count.to_le_bytes());
    out[12..16].copy_from_slice(&(query_result.tier as i32).to_le_bytes());

    match_count
}

/// Return the total number of queries executed so far.
#[no_mangle]
pub extern "C" fn mirror_mesh_stats() -> u64 {
    unsafe {
        match PIPELINE.as_ref() {
            Some(p) => p.query_count(),
            None => 0,
        }
    }
}

/// Encode UTF‑8 text to a 128‑bit binary vector (16 bytes).
/// Writes 16 bytes into `out`; returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn mirror_mesh_encode(data: *const u8, len: usize, out: *mut u8) -> i32 {
    if data.is_null() || out.is_null() {
        return -1;
    }
    if len == 0 {
        return -1;
    }
    let text = match unsafe { ::core::str::from_utf8(::core::slice::from_raw_parts(data, len)) } {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let encoder = Encoder::new(0);
    let bytes = encoder.encode_to_bytes(text);
    unsafe {
        ::core::ptr::copy_nonoverlapping(bytes.as_ptr(), out, 16);
    }
    0
}

// ---------------------------------------------------------------------------
// FFI: Cascading Retrieval
// ---------------------------------------------------------------------------

use ::core::ffi::{c_int, c_float, c_void};

/// Run a 4-stage cascade query on an existing LSH index.
/// Returns count of results written to distances_out/indices_out.
/// Stats are written to stats_out as 6 u64 values:
///   [0] = stage1_candidates, [1] = stage2_candidates, [2] = stage3_candidates,
///   [3] = stage4_candidates, [4] = total_time_us, [5] = reserved
    use std::sync::Mutex;

    static FFI_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn ffi_init_returns_0() {
        let _guard = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(mirror_mesh_init(), 0);
    }

    #[test]
    fn ffi_add_paper_and_query_returns_result() {
        let _guard = FFI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(mirror_mesh_init_with_thresholds(0, 0, 0), 0);

        let paper = [1u8; 16];
        assert_eq!(mirror_mesh_add_paper(paper.as_ptr(), 16), 0);

        let query = [1u8; 16];
        let mut out = [0u8; 16];
        let ret = mirror_mesh_query(query.as_ptr(), 16, out.as_mut_ptr(), 16);
        assert!(ret > 0, "query should return a positive match count, got {}", ret);

        let score = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        let index = i32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        assert!(score > 0.0_f32, "similarity score should be positive, got {score}");
        assert_eq!(index, 0_i32, "best match index should be 0, got {index}");
    }
pub mod cheat_sheet_cascade;
pub mod hybrid_mesh;
pub mod exact_cascade;

pub mod wiring_registry_types;

pub mod wiring_registry_crypto;

pub mod wiring_registry_scanner;

pub mod wiring_registry;
pub mod result_cache;
pub mod learned_router;
pub mod self_learning;
pub mod engine_feeder;
pub mod hash_stage;
pub mod spectral_stage;
pub mod wedge_stage;
pub mod hologram_stage;
pub mod spectral_coords;
pub mod simd_kernels;
pub mod batch_fusion;
pub mod intent_classifier;
pub mod drift_detector;
pub mod sharded_mesh;
pub mod collaborative_engine;

#[cfg(feature = "std")]
pub mod async_ffi;

#[cfg(feature = "std")]
pub mod temporal_orchestrator;

#[cfg(feature = "std")]
pub mod versioned_tables;

#[cfg(feature = "std")]
pub mod shadow_build;
pub mod ffi_unified;
pub mod ffi_autopoiesis;
pub mod ffi_tensor_spectral_512;
pub mod tensor_spectral;
pub mod multi_base_crystal;
#[cfg(feature = "trinity")]
pub mod trinity;
pub mod predictor_tokens;
#[cfg(feature = "living_mesh")]
pub mod congruence_filter;
#[cfg(feature = "living_mesh")]
pub mod energy_overlay;
#[cfg(feature = "living_mesh")]
pub mod snapshot_trait;
#[cfg(feature = "living_mesh")]
pub mod temporal_tracker;
pub mod self_tuner;
pub mod cgt;
pub mod crystal_mesh_384;
pub mod binary_hnsw_384;
pub mod hybrid_mesh_384;
pub mod embedding_store;
pub mod geometric_rerank;
pub mod flat_array;
pub mod flat_query_engine;
pub mod flat_ffi;
pub mod ffi_ism;
pub mod ffi_bipolar;
pub mod ffi_linear_hash;
pub mod ffi_temporal;
pub mod ffi_energy;
pub mod ffi_mesh_snapshot;
pub mod ffi_predictor;
pub mod dynamic_mesh;
pub mod shard;
pub mod cascade_index;
pub mod exact;
pub mod pq;
pub mod ffi_dynamic_mesh;
pub mod ffi_shard;
pub mod ffi_cascade_index;





#[cfg(feature = "intelligent_shard_manager")]
pub mod intelligent_shard_manager;

#[cfg(feature = "watchdog")]
pub mod watchdog;
pub mod ism_error;
pub mod parallel_builder;
pub mod build_advisor;

pub mod build_context;

#[cfg(feature = "intelligent_shard_manager")]
pub mod pams {
    pub use crate::intelligent_shard_manager::*;
}

// ---------------------------------------------------------------------------
// FFI: YP Rust Re-Rank Engine
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
use rerank_engine::RerankEngine;

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn yp_rerank_engine_new(
    npy_path: *const u8,
    path_len: usize,
) -> *mut RerankEngine {
    let path = match std::str::from_utf8(std::slice::from_raw_parts(npy_path, path_len)) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    match RerankEngine::new(path) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn yp_rerank_engine_free(ptr: *mut RerankEngine) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn yp_rerank_engine_search(
    engine: *mut RerankEngine,
    candidates: *const u64,
    n_candidates: usize,
    query_emb: *const f32,
    emb_dim: usize,
    k_final: usize,
    out_ids: *mut u64,
    out_scores: *mut f32,
) -> usize {
    if engine.is_null()
        || candidates.is_null()
        || query_emb.is_null()
        || out_ids.is_null()
        || out_scores.is_null()
    {
        return 0;
    }
    let engine = &*engine;
    let cands = std::slice::from_raw_parts(candidates, n_candidates);
    let query = std::slice::from_raw_parts(query_emb, emb_dim);
    let results = engine.rerank(cands, query, k_final);
    for (i, (id, score)) in results.iter().enumerate() {
        *out_ids.add(i) = *id;
        *out_scores.add(i) = *score;
    }
    results.len()
}
pub mod ffi_core_hologram;
pub mod ffi_soft;

pub mod abstract_store;
pub mod ffi_abstracts;
