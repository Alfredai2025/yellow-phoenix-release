# Yellow Phoenix — Wiring Diagnosis

**Branch:** `phoenix-bench-windows-v2`  
**Audit date:** 2026-07-23  
**Scope:** Rust `mirror_mesh` library + `yp_bridge.py` FFI consumer  
**Build status:** `cargo check --lib` ✅ (12 dead-code warnings), `cargo test --lib` ✅ 204 passed  

---

## 1. Executive Summary

The production query path is narrow and fast:

```text
Python YPEngine.search()
    → RustBridge.encode_text()
    → HYBRID_MESH (O(1) prefix exact match)
    → CollaborativeEngine.query_lazy() / query()
        → HashStage → LearnedRouter / IntentClassifier
        → optional SpectralStage → optional WedgeStage → optional HologramStage
        → ResultCache / SelfLearningTable
    → OptimizedShard (sharded path when enabled)
```

Most of the codebase is **compiled but not on the hot path**. A non-trivial set of source files is **not even declared in `lib.rs`** and therefore never built. Cleaning these up will shorten compile times, reduce binary size, and remove confusing legacy entry points.

---

## 2. Methodology

1. Enumerated every `.rs` file under `src/`.
2. Checked whether each file is declared by its parent `mod.rs` / `lib.rs`.
3. Traced `use crate::...` edges from `CollaborativeEngine` outward.
4. Compared Rust `#[no_mangle] pub extern "C" fn` symbols against Python `yp_*` references.
5. Checked which `RustBridge` / `YPEngine` methods are actually invoked from live scripts.

---

## 3. Module Status

### 3.1 🟢 Green — wired into the live query path

These modules are compiled **and** exercised on a normal `YPEngine.search()` or `enable_sharding()` call.

| Module | Role in query path |
|--------|--------------------|
| `collaborative_engine` | Main adaptive engine; owns all stages and routing |
| `hybrid_mesh` | Dual-resolution index (`CrystalMesh128/512`), prefix map, SIMD distances |
| `hash_stage` | Fast candidate generator from 128-bit / 512-bit PAPs |
| `spectral_stage` | Spectral-coordinate re-rank |
| `wedge_stage` | Wedge/top-K contrastive re-rank |
| `hologram_stage` | Hologram consensus stage |
| `learned_router` | Decides which feature chain to run |
| `intent_classifier` | Maps hash confidence + bucket size to `Intent` |
| `self_learning` | Empirical feature-chain table used by `plan_chain` |
| `result_cache` | Query-result cache |
| `sharded_mesh` | Memory-optimized shard (M3.5) |
| `spectral_coords` | `SpectralCoords` storage used by mesh + shard |
| `simd_kernels` | SIMD PAP distance routines called by `hybrid_mesh` |
| `ffi` | Active `yp_*` symbols consumed by `yp_bridge.py` |

### 3.2 🟡 Yellow — compiled, but off the hot path or optional

These modules are declared in `lib.rs` (or a parent `mod.rs`) and therefore compiled, but they are either not called by `CollaborativeEngine`, only enabled via optional API calls, or belong to legacy/experimental subsystems.

| Module / group | Why it is yellow | Recommended action |
|----------------|------------------|--------------------|
| `batch_fusion` | Wired only if `enable_batch_fusion()` is called | Expose via FFI/Python if throughput mode is still a goal; otherwise feature-gate |
| `drift_detector` | Wired only if `enable_drift_detection()` is called | Keep; add Python toggle |
| `engine_feeder` | Held by engine but **not read** during `query()` | Either wire feeder feedback into scoring or remove from engine struct |
| `memory` | Supports `engine_feeder` + legacy ring buffer | Keep if feeder stays; otherwise remove with feeder |
| `distance` | Used only by non-hot-path/legacy modules | Merge into `hybrid_mesh` or delete after orphans are removed |
| `types` | Used by `core::encoder`, `core::geometric_autopoiesis`, `ffi_kernel` | Keep only if geometric/autopoiesis kernel is retained |
| `algebra` | Used only by `core::geometric_autopoiesis` | Keep only if autopoiesis is retained |
| `crystal` | Old `CrystalMesh`; replaced by `hybrid_mesh::CrystalMesh*` | Remove after confirming tests do not need it |
| `query`, `direct_hash`, `lsh`, `router`, `math`, `solver` | Declared but not referenced by the live engine | Remove or archive |
| `exact_cascade`, `cheat_sheet_cascade` | Old cascade indices; Python wrappers call missing symbols | Remove or archive (superseded by hybrid + sharded) |
| `core::resonance`, `core::encoder` | Old `QueryPipeline` + `Encoder` exposed via legacy `mirror_mesh_*` FFI | Remove legacy FFI or move to `archive/` |
| `core::{hologram,hologram_index,sdm,simd,tiered_query,geometric_autopoiesis}` | Used only by legacy/experimental paths | Keep only if `ffi_kernel` autopoiesis API is needed |
| `ffi_kernel`, `async_ffi` | `ffi_kernel` has no exported symbols; `async_ffi` is not called from Python | Delete or merge into a single async API |
| `temporal_orchestrator`, `versioned_tables`, `shadow_build` | Experimental governance/versioning | Feature-gate or move to a separate crate |
| `wiring_registry*` | Self-audit tooling; references only each other | Move to a build/dev crate or keep under `#[cfg(test)]` |

### 3.3 🔴 Red — source files that are never loaded

These files exist under `src/` but **are not declared in any `mod.rs`/`lib.rs`**. They are not compiled and have no effect on the build. Several form their own disconnected subgraph (e.g., `intelligent` → `cascade_index` → `exact`/`pq`/`query`, etc.).

| File | Notes |
|------|-------|
| `src/cascade_index.rs` | Part of the orphan `intelligent` subgraph |
| `src/cross_reference.rs` | References `crystal`, `dynamic_mesh` (all orphan) |
| `src/cun_disk.rs` | No references |
| `src/disk_paging.rs` | No references |
| `src/disk_query.rs` | No references |
| `src/dual_identity.rs` | No references |
| `src/dynamic_mesh.rs` | Referenced only by other orphans |
| `src/exact.rs` | Referenced only by `cascade_index.rs` |
| `src/intelligent.rs` | Root of an orphan subgraph |
| `src/multi_base_crystal.rs` | Referenced only by orphans |
| `src/multi_base_dynamic.rs` | Referenced only by orphans |
| `src/pq.rs` | Referenced only by `cascade_index.rs` |
| `src/proof_of_life.rs` | No references; separate proof-of-life chain lives in `logs/proof/` |
| `src/slot.rs` | No references |
| `src/temporal_evolution.rs` | No references |
| `src/tensor_spectral.rs` | No references; may contain valuable tensor ideas — review before deleting |

---

## 4. FFI / Python Bridge Wiring

### 4.1 Active Rust FFI symbols (called at runtime)

These `#[no_mangle] pub extern "C" fn` symbols are actually invoked from `yp_bridge.py` / scripts:

| Rust symbol | Python caller | Used for |
|-------------|---------------|----------|
| `yp_encode_text` | `RustBridge.encode_text` | Text → PAP hash |
| `yp_insert_to_mesh` | `RustBridge.insert_to_mesh` | Populate `HYBRID_MESH` |
| `yp_build_edges` | `RustBridge.build_edges` | Edge build (legacy path) |
| `yp_enable_sharding` | `RustBridge.enable_sharding` | Build `OptimizedShard` |
| `yp_query_sharded` | `RustBridge.query_sharded` | Sharded exact lookup |
| `yp_save_mesh` | `RustBridge.save_static_mesh` | Persist `HYBRID_MESH` |
| `yp_load_mesh` | `RustBridge.load_static_mesh` | Load + rebuild prefix map |
| `yp_confidence_score_fast` | `RustBridge.confidence_score_fast` | Fast confidence check in `search()` |

### 4.2 Declared but unused Rust FFI symbols

| Rust symbol | Status |
|-------------|--------|
| `yp_proof_of_life` | Wrapper exists but not called from live code |
| `yp_insert_paper` | Not called from Python |
| `yp_query_mesh` | Not called from Python |
| `yp_mesh_count` | Wrapper exists but not called |
| `yp_temporal_submit`, `yp_temporal_poll`, `yp_temporal_gc` | Async/temporal API, not wired to Python |
| `yp_confidence_score` | Slow variant; Python uses `_fast` |
| `yp_fast_path_eligible` | Not called |
| `mirror_mesh_*` (6 functions in `lib.rs`) | Legacy FFI, completely unused by `yp_bridge.py` |

### 4.3 Python calls to symbols that do **not** exist in Rust

`yp_bridge.py` still references many functions that are **not** compiled into the current library. These paths will fail at runtime if invoked:

```text
yp_autopoiesis_step, yp_benchmark_geometric, yp_cascade_*, yp_cross_check_*,
yp_crystal_query, yp_free_string, yp_free_u32_array, yp_hash256,
yp_hopfield_filter, yp_hybrid_*, yp_insert_to_mesh_bytes, yp_insert_with_col,
yp_mesh_evolve, yp_pq_insert, yp_prime_overlay_*, yp_query_mesh_bytes,
yp_tensor_build, yp_tensor_query, yp_unified_*
```

These mostly live in dead `RustBridge` / `YPEngine` methods (`crystal_query`, `mesh_evolve`, `autopoiesis_step`, `populate_unified_mesh`, etc.).

---

## 5. Broken / Legacy Python Paths

The following `YPEngine` / `RustBridge` methods are **not called by any live script** and invoke missing Rust symbols:

- `YPEngine.spectral_query`, `crystal_query`, `mesh_evolve`, `autopoiesis_step`
- `YPEngine.insert_agreement`, `search_agreement`
- `RustBridge.populate_unified_mesh`, `build_edges_unified`, `unified_mesh_*`, `crystal_*`, `cascade_*`
- `RustBridge.proof_of_life`, `benchmark_geometric`

The `build_edges_unified` call inside `yp_bridge.py` (line ~1801) is particularly risky: it executes only when `_unified_batch` is non-empty, but it calls `yp_unified_mesh_build_edges`, which does not exist.

---

## 6. Priority Connection / Cleanup Plan

### Immediate (this week)

1. **Remove red orphan files** — they are not compiled, so moving them to `archive/rust_dead_modules/` is zero-risk. Back up first.
   - Exception: manually review `tensor_spectral.rs` and `temporal_evolution.rs` for ideas worth porting to the live spectral/sharded path.
2. **Delete legacy `mirror_mesh_*` FFI** from `src/lib.rs` and the static `PIPELINE`/`RING` state unless a consumer still needs it.
3. **Prune dead Python wrappers** — remove all `yp_*` signatures and methods that call non-existent Rust symbols. Keep the active 8 symbols.

### Short term (next milestone)

4. **Decide on optional engine capabilities**:
   - If `batch_fusion` stays, expose `enable_batch_fusion` / `query_batch` via FFI and a Python toggle.
   - If `drift_detector` stays, expose `enable_drift_detection` to Python and wire alerts to logs/metrics.
   - If `engine_feeder` stays, actually apply feed messages to router/learning table; otherwise remove the field.
5. **Resolve `exact_cascade` / `cheat_sheet_cascade`** — they are superseded by `hybrid_mesh` + `sharded_mesh`. Archive them.
6. **Resolve `crystal.rs`** — `hybrid_mesh` already embeds equivalent `CrystalMesh128/512`. Archive `crystal.rs` once tests confirm no dependency.

### Longer term / architectural

7. **Move governance/experimental modules out of the library crate**:
   - `wiring_registry*`, `shadow_build`, `versioned_tables`, `temporal_orchestrator`, `async_ffi`, `ffi_kernel`, `core::geometric_autopoiesis`.
   - Either put them behind feature flags (`--features experimental`) or into a separate `yp_experimental` crate.
8. **Adopt `pre_fault=True` as default for benchmark/reproducible runs**; keep it optional in production to preserve lazy memory behavior.
9. **Add a CI wiring check** that fails if:
   - A new `.rs` file is added but not declared in a parent module, or
   - A Python `yp_*` reference has no matching Rust `#[no_mangle]` symbol.

---

## 7. Quick Reference: Green / Yellow / Red Counts

| Category | Count | Notes |
|----------|-------|-------|
| 🟢 Green (hot path) | ~14 modules | Includes the FFI bridge with active symbols |
| 🟡 Yellow (compiled but optional/legacy) | ~44 modules | Most dead-code warnings originate here |
| 🔴 Red (source files not loaded) | 16 files | Safe to archive after backup |

---

## 8. Appendix: How to Re-run This Audit

```bash
# 1. List declarations vs files
grep -n "^pub mod\|^mod " src/lib.rs
find src -name "*.rs" -not -path "*/bin/*" | sed 's|src/||' | sort

# 2. Trace query-path dependencies
grep -rn "use crate::" src/collaborative_engine.rs

# 3. List Rust FFI symbols
python3 -c "import re; [print(m.group(1)) for p in ['src/lib.rs','src/ffi.rs','src/async_ffi.rs'] for m in re.finditer(r'#\[no_mangle\]\s*pub\s+extern\s+\"C\"\s+fn\s+([a-zA-Z0-9_]+)', open(p).read())]"

# 4. List Python yp_* references
grep -rhoP '\byP_\w+|yp_\w+' yp_bridge.py scripts/ python/ | sort -u

# 5. Build + test
cargo check --lib
cargo test --lib
```
