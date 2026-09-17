// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! collaborative_engine.rs — Milestone 1 Component 5.
//!
//! Adaptive feature-chain engine that runs hash, spectral, wedge, and hologram stages
//! according to a learned router and self-learning table, caches results, accepts
//! runtime feedback from the engine feeder, and returns a consensus-ranked answer.

use crate::drift_detector::{ConsoleAlert, DriftAlert, DriftDetector};
use crate::engine_feeder::EngineFeeder;
use crate::hash_stage::{FeatureResult, HashStage};
use crate::hologram_stage::HologramStage;
use crate::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};
use crate::intent_classifier::{Intent, IntentClassifier};
use crate::learned_router::{FeatureChain, LearnedRouter};
use crate::result_cache::ResultCache;
use crate::self_learning::{SelfLearningTable, Thresholds};
use std::sync::LazyLock;

// M3.6: Load thresholds from self-learning config if available.
static THRESHOLDS: LazyLock<std::sync::RwLock<Thresholds>> =
    LazyLock::new(|| std::sync::RwLock::new(Thresholds::default()));
static THRESHOLD_MTIME: LazyLock<std::sync::RwLock<std::time::SystemTime>> =
    LazyLock::new(|| std::sync::RwLock::new(std::time::SystemTime::UNIX_EPOCH));

const THRESHOLD_PATH: &str = "data/self_learning.json";

fn reload_thresholds_if_stale() {
    let meta = std::fs::metadata(THRESHOLD_PATH).ok();
    let new_mtime = meta.and_then(|m| m.modified().ok());
    let current = *THRESHOLD_MTIME.read().unwrap();
    if let Some(mtime) = new_mtime {
        if mtime > current {
            if let Ok(data) = std::fs::read_to_string(THRESHOLD_PATH) {
                if let Ok(cfg) = serde_json::from_str::<Thresholds>(&data) {
                    *THRESHOLDS.write().unwrap() = cfg;
                    *THRESHOLD_MTIME.write().unwrap() = mtime;
                }
            }
        }
    }
}

pub fn current_thresholds() -> Thresholds {
    reload_thresholds_if_stale();
    THRESHOLDS.read().unwrap().clone()
}
use crate::sharded_mesh::OptimizedShard;
use crate::spectral_stage::SpectralStage;
use crate::wedge_stage::WedgeStage;

use std::sync::Arc;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Packed inputs for a query.
#[derive(Clone, Copy)]
pub struct QueryContext {
    pub pap_128: [u8; PAP_128_BYTES],
    pub pap_512: [u8; PAP_512_BYTES],
    pub features: [f32; 8],
    /// Optional ground-truth paper ID for drift-detection bookkeeping.
    pub expected_id: Option<u64>,
}

/// Final answer from the collaborative engine.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsensusResult {
    pub top1: Option<(u64, f32)>,
    pub top5: Vec<(u64, f32)>,
    pub chain: FeatureChain,
    pub confidence: f32,
    pub latency_us: u64,
    pub from_cache: bool,
}

pub struct CollaborativeEngine {
    hash_stage: HashStage,
    spectral_stage: SpectralStage,
    wedge_stage: WedgeStage,
    hologram_stage: HologramStage,
    router: LearnedRouter,
    learning: SelfLearningTable,
    cache: ResultCache,
    feeder: EngineFeeder,
    metrics: CollaborativeMetrics,
    batch_engine: Option<Arc<crate::batch_fusion::BatchFusionEngine>>,
    intent_classifier: IntentClassifier,
    drift_detector: Option<DriftDetector>,
    drift_alert: Option<Box<dyn DriftAlert + Send + Sync>>,
    pub sharded_mesh: Option<OptimizedShard>,
    sharding_threshold: usize,
}

#[derive(Default, Clone, Debug)]
pub struct CollaborativeMetrics {
    pub queries: u64,
    pub cache_hits: u64,
    pub feedback_injected: u64,
}

impl CollaborativeEngine {
    pub fn new(mesh: HybridMesh) -> Self {
        let mut pap_map: std::collections::HashMap<u64, [u8; crate::hybrid_mesh::PAP_512_BYTES]> =
            std::collections::HashMap::with_capacity(mesh.fine.slots.len());
        let mut coords_map: std::collections::HashMap<u64, crate::spectral_coords::SpectralCoords> =
            std::collections::HashMap::with_capacity(mesh.spectral_coords.len());
        for slot in &mesh.fine.slots {
            pap_map.insert(slot.id, slot.pap);
        }
        coords_map.extend(mesh.spectral_coords.iter().map(|(&k, &v)| (k, v)));
        Self {
            hash_stage: HashStage::new(mesh),
            spectral_stage: SpectralStage::with_paps_and_coords(pap_map, coords_map),
            wedge_stage: WedgeStage::new(),
            hologram_stage: HologramStage::new(),
            router: LearnedRouter::new(),
            learning: SelfLearningTable::with_thresholds(current_thresholds()),
            cache: ResultCache::new(10_000, std::time::Duration::from_millis(30_000)),
            feeder: EngineFeeder::new(),
            metrics: CollaborativeMetrics::default(),
            batch_engine: None,
            intent_classifier: IntentClassifier::new(),
            drift_detector: None,
            drift_alert: None,
            sharded_mesh: None,
            sharding_threshold: current_thresholds().sharding_threshold,
        }
    }

    /// M4.5: build a memory-optimized sharded mesh from the current index.
    ///
    /// `cold_path` is the backing file for the memory-mapped cold tier.
    /// `hot_ratio` controls the fraction of records kept in RAM (default 0.10).
    pub fn enable_sharding(&mut self, cold_path: &Path, hot_ratio: f64, pre_fault: bool) -> std::io::Result<()> {
        let mesh = self.hash_stage.mesh();
        let records = mesh.fine.slots.iter().map(|slot| {
            let coords = mesh
                .spectral_coords
                .get(&slot.id)
                .copied()
                .unwrap_or_default();
            (slot.id, slot.pap, coords)
        });
        let shard = OptimizedShard::build(records, hot_ratio, cold_path)?;
        if pre_fault {
            let touched = shard.pre_fault_cold_pages();
            eprintln!("[sharded] pre-faulted {} cold pages", touched);
        }
        self.sharded_mesh = Some(shard);
        Ok(())
    }

    /// Test-only override for the dataset size at which the sharded path is used.
    pub fn set_sharding_threshold(&mut self, threshold: usize) {
        self.sharding_threshold = threshold;
    }

    fn sharded_query_direct(&self, ctx: &QueryContext) -> Option<ConsensusResult> {
        let shard = self.sharded_mesh.as_ref()?;
        let start = Instant::now();
        shard.query(&ctx.pap_512).map(|(id, _)| {
            let top1 = (id, 1.0f32);
            ConsensusResult {
                top1: Some(top1),
                top5: vec![top1],
                chain: FeatureChain::none(),
                confidence: 1.0,
                latency_us: start.elapsed().as_micros() as u64,
                from_cache: false,
            }
        })
    }

    /// M4.4: enable rolling-window drift detection.
    pub fn enable_drift_detection(&mut self, window_size: usize, threshold: f32, min_samples: usize) {
        self.drift_detector = Some(DriftDetector::new(window_size, threshold, min_samples));
        self.drift_alert = Some(Box::new(ConsoleAlert));
    }

    fn record_drift(&mut self, expected: Option<u64>, top1: Option<(u64, f32)>, features: [f32; 8]) {
        if let Some(ref mut detector) = self.drift_detector {
            let correct = expected.map(|expected_id| top1.map(|(id, _)| id == expected_id).unwrap_or(false));
            detector.record_query(features, correct);
            if correct.is_some() && detector.is_drift() {
                if let Some(ref alert) = self.drift_alert {
                    alert.on_drift(detector.current_r1(), detector.threshold());
                } else {
                    eprintln!("{}", detector.status());
                }
            }
        }
    }

    /// M3.5: enable the batch-fusion worker with the given collection window.
    pub fn enable_batch_fusion(&mut self, window_ms: u64) {
        let mesh = self.hash_stage.mesh().clone();
        self.batch_engine = Some(Arc::new(crate::batch_fusion::BatchFusionEngine::new(
            mesh, window_ms,
        )));
    }

    /// M3.5: submit a query to the batch-fusion queue.
    ///
    /// Requires `enable_batch_fusion()` to have been called; otherwise returns
    /// an empty result.
    pub fn query_batch(&self, ctx: &QueryContext) -> ConsensusResult {
        if let Some(ref engine) = self.batch_engine {
            let batch = engine.submit(*ctx);
            return ConsensusResult {
                top1: batch.top1,
                top5: batch.top5,
                chain: FeatureChain::none(),
                confidence: 0.98,
                latency_us: 0,
                from_cache: false,
            };
        }
        ConsensusResult {
            top1: None,
            top5: Vec::new(),
            chain: FeatureChain::none(),
            confidence: 0.0,
            latency_us: 0,
            from_cache: false,
        }
    }

    /// Submit a small burst of queries asynchronously, then collect all results.
    /// This is intended for throughput benchmarks where many in-flight queries
    /// are needed to amortize the batch-fusion window.
    pub fn query_batch_many(&self, ctxs: &[QueryContext]) -> Vec<ConsensusResult> {
        if let Some(ref engine) = self.batch_engine {
            let receivers: Vec<_> = ctxs.iter().map(|ctx| engine.submit_async(*ctx)).collect();
            return receivers
                .into_iter()
                .map(|rx| {
                    let batch = rx.recv().unwrap_or(crate::batch_fusion::BatchResult {
                        top1: None,
                        top5: Vec::new(),
                    });
                    ConsensusResult {
                        top1: batch.top1,
                        top5: batch.top5,
                        chain: FeatureChain::none(),
                        confidence: 0.98,
                        latency_us: 0,
                        from_cache: false,
                    }
                })
                .collect();
        }
        ctxs.iter()
            .map(|_| ConsensusResult {
                top1: None,
                top5: Vec::new(),
                chain: FeatureChain::none(),
                confidence: 0.0,
                latency_us: 0,
                from_cache: false,
            })
            .collect()
    }

    /// M3.5: fraction of batched queries that shared a bucket group with others.
    pub fn batch_fusion_rate(&self) -> f32 {
        self.batch_engine
            .as_ref()
            .map(|e| e.fusion_rate())
            .unwrap_or(0.0)
    }

    /// Choose which feature chains to run.
    pub fn plan_chain(&self, features: [f32; 8]) -> FeatureChain {
        let pattern = Self::pattern_from_features(features);
        // Prefer empirical table; if unseen (all features), fall back to MLP router.
        let table_chain = self.learning.best_chain(pattern);
        if table_chain.run_spectral && table_chain.run_wedge && table_chain.run_hologram {
            self.router.decide_chain(features)
        } else {
            table_chain
        }
    }

    fn pattern_from_features(features: [f32; 8]) -> u8 {
        let mut p: u8 = 0;
        for (i, &v) in features.iter().enumerate().take(8) {
            if v > 0.0 {
                p |= 1 << i;
            }
        }
        p
    }

    fn cache_key(pap_128: &[u8; PAP_128_BYTES], pap_512: &[u8; PAP_512_BYTES]) -> [u8; 64] {
        let mut key = [0u8; 64];
        key[..PAP_128_BYTES].copy_from_slice(pap_128);
        // Pack the first 48 bytes of the 512-bit PAP into the remaining cache key space.
        key[PAP_128_BYTES..].copy_from_slice(&pap_512[..64 - PAP_128_BYTES]);
        key
    }

    pub fn run_hash(&self, ctx: &QueryContext) -> FeatureResult {
        self.hash_stage.query(&ctx.pap_128, &ctx.pap_512, 20)
    }

    pub fn run_spectral(&self, ctx: &QueryContext, hash: &FeatureResult) -> FeatureResult {
        self.spectral_stage.query(&ctx.pap_128, &ctx.pap_512, hash)
    }

    pub fn run_wedge(&self, _ctx: &QueryContext, spectral: &FeatureResult) -> FeatureResult {
        self.wedge_stage.query(&_ctx.pap_512, spectral)
    }

    pub fn run_hologram(&self, _ctx: &QueryContext, wedge: &FeatureResult) -> FeatureResult {
        self.hologram_stage.query(&_ctx.pap_512, wedge)
    }

    pub fn finalize(
        &self,
        hash: &FeatureResult,
        spectral: &FeatureResult,
        wedge: &FeatureResult,
        hologram: &FeatureResult,
        chain: &FeatureChain,
    ) -> ConsensusResult {
        self.consensus_score(hash, spectral, wedge, hologram, chain)
    }

    /// Run the adaptive chain and return consensus result.
    pub fn query(&mut self, ctx: &QueryContext) -> ConsensusResult {
        let start = Instant::now();
        self.metrics.queries += 1;

        let key = Self::cache_key(&ctx.pap_128, &ctx.pap_512);
        let cached_result = if let Some(cached) = self.cache.lookup(&key) {
            self.metrics.cache_hits += 1;
            let top5: Vec<(u64, f32)> = cached.iter().map(|(id, score, _)| (*id, *score)).collect();
            Some(ConsensusResult {
                top1: top5.first().copied(),
                top5,
                chain: FeatureChain::all(),
                confidence: 0.92,
                latency_us: start.elapsed().as_micros() as u64,
                from_cache: true,
            })
        } else {
            None
        };
        if let Some(result) = cached_result {
            self.record_drift(ctx.expected_id, result.top1, ctx.features);
            return result;
        }

        // M4.5: for very large datasets, prefer the memory-mapped sharded path.
        if self.hash_stage.mesh().fine.slots.len() > self.sharding_threshold {
            if let Some(result) = self.sharded_query_direct(ctx) {
                self.record_drift(ctx.expected_id, result.top1, ctx.features);
                return result;
            }
        }

        // Hash stage always runs as the fast candidate generator.
        // M3.1: use router confidence to decide whether to try the fast bucket path.
        let router_confidence = self.router.confidence(ctx.features);
        let hash = self
            .hash_stage
            .query_with_confidence(&ctx.pap_128, &ctx.pap_512, 20, router_confidence);

        // M4.3: classify intent and route to the right execution depth.
        let bucket_size = self.hash_stage.mesh().bucket_size(&ctx.pap_128);
        let intent = self.intent_classifier.classify(hash.confidence, bucket_size);

        let (spectral, wedge, hologram, chain) = match intent {
            Intent::ExactLookup => {
                // Hash only: skip all later stages.
                let top5: Vec<(u64, f32)> = hash.candidates.iter().take(5).copied().collect();
                let cache_entries: Vec<(u64, f32, String)> = top5
                    .iter()
                    .map(|&(id, score)| (id, score, format!("doc_{}", id)))
                    .collect();
                self.cache.insert(&key, cache_entries);
                let result = ConsensusResult {
                    top1: top5.first().copied(),
                    top5,
                    chain: FeatureChain::none(),
                    confidence: hash.confidence,
                    latency_us: start.elapsed().as_micros() as u64,
                    from_cache: false,
                };
                self.record_drift(ctx.expected_id, result.top1, ctx.features);
                return result;
            }
            Intent::SemanticSearch => {
                let spectral = self.spectral_stage.query(&ctx.pap_128, &ctx.pap_512, &hash);
                let wedge = spectral.clone();
                let hologram = spectral.clone();
                let chain = FeatureChain {
                    run_spectral: true,
                    run_wedge: false,
                    run_hologram: false,
                };
                (spectral, wedge, hologram, chain)
            }
            Intent::Exploration => {
                let chain = self.plan_chain(ctx.features);
                let spectral = if chain.run_spectral {
                    self.spectral_stage.query(&ctx.pap_128, &ctx.pap_512, &hash)
                } else {
                    hash.clone()
                };
                let wedge = if chain.run_wedge {
                    self.wedge_stage.query(&ctx.pap_512, &spectral)
                } else {
                    spectral.clone()
                };
                let hologram = if chain.run_hologram {
                    self.hologram_stage.query(&ctx.pap_512, &wedge)
                } else {
                    wedge.clone()
                };
                (spectral, wedge, hologram, chain)
            }
        };

        let final_result = self.consensus_score(&hash, &spectral, &wedge, &hologram, &chain);

        // Cache top-5 with string label stub.
        let cache_entries: Vec<(u64, f32, String)> = final_result
            .top5
            .iter()
            .map(|&(id, score)| (id, score, format!("doc_{}", id)))
            .collect();
        self.cache.insert(&key, cache_entries);

        let result = ConsensusResult {
            confidence: hologram.confidence,
            latency_us: start.elapsed().as_micros() as u64,
            from_cache: false,
            ..final_result
        };
        self.record_drift(ctx.expected_id, result.top1, ctx.features);
        result
    }

    /// M3.2: Query with lazy feature activation.
    ///
    /// Skips expensive later stages when the hash stage is already confident:
    /// - hash.confidence >= 0.98 + exact match: skip ALL later stages
    /// - hash.confidence >= 0.85: skip wedge + hologram, keep spectral
    /// - otherwise: full pipeline
    pub fn query_lazy(&mut self, ctx: &QueryContext) -> ConsensusResult {
        let start = Instant::now();
        self.metrics.queries += 1;

        let key = Self::cache_key(&ctx.pap_128, &ctx.pap_512);
        if let Some(cached) = self.cache.lookup(&key) {
            self.metrics.cache_hits += 1;
            let top5: Vec<(u64, f32)> = cached.iter().map(|(id, score, _)| (*id, *score)).collect();
            return ConsensusResult {
                top1: top5.first().copied(),
                top5,
                chain: FeatureChain::none(),
                confidence: 0.92,
                latency_us: start.elapsed().as_micros() as u64,
                from_cache: true,
            };
        }

        // M4.5: for very large datasets, prefer the memory-mapped sharded path.
        if self.hash_stage.mesh().fine.slots.len() > self.sharding_threshold {
            if let Some(result) = self.sharded_query_direct(ctx) {
                return result;
            }
        }

        // M3.2: O(1) exact prefix match — world-class lazy path.
        // Bypass router, intent classifier, and geometric stages when the first
        // 8 bytes of pap_128 uniquely identify the paper.
        if let Some(id) = self.hash_stage.mesh().query_prefix_exact(&ctx.pap_128) {
            return ConsensusResult {
                top1: Some((id, 1.0f32)),
                top5: vec![(id, 1.0f32)],
                chain: FeatureChain::none(),
                confidence: 1.0,
                latency_us: start.elapsed().as_micros() as u64,
                from_cache: false,
            };
        }

        // M3.1: use router confidence to decide whether to try the fast bucket path.
        let router_confidence = self.router.confidence(ctx.features);
        let hash = self
            .hash_stage
            .query_with_confidence(&ctx.pap_128, &ctx.pap_512, 20, router_confidence);

        let thresholds = current_thresholds();
        let hash_confidence = hash.confidence;
        let hash_score = hash.score;

        // Fastest path: hash only, no later stages.
        // Skip the result cache entirely here; the fast hash lookup is cheaper than
        // the cache bookkeeping, so caching would only add latency on this path.
        if hash_confidence >= thresholds.skip_all && hash_score >= 0.99 {
            let top5: Vec<(u64, f32)> = hash.candidates.iter().take(5).copied().collect();
            return ConsensusResult {
                top1: top5.first().copied(),
                top5,
                chain: FeatureChain::none(),
                confidence: hash_confidence,
                latency_us: start.elapsed().as_micros() as u64,
                from_cache: false,
            };
        }

        let run_spectral = hash_confidence < thresholds.skip_all;
        let run_wedge = hash_confidence < thresholds.skip_wedge_holo;
        let run_hologram = hash_confidence < thresholds.skip_wedge_holo;

        let spectral = if run_spectral {
            self.spectral_stage.query(&ctx.pap_128, &ctx.pap_512, &hash)
        } else {
            hash.clone()
        };
        let wedge = if run_wedge {
            self.wedge_stage.query(&ctx.pap_512, &spectral)
        } else {
            spectral.clone()
        };
        let hologram = if run_hologram {
            self.hologram_stage.query(&ctx.pap_512, &wedge)
        } else {
            wedge.clone()
        };

        let chain = FeatureChain {
            run_spectral,
            run_wedge,
            run_hologram,
        };
        let final_result = self.consensus_score(&hash, &spectral, &wedge, &hologram, &chain);

        let cache_entries: Vec<(u64, f32, String)> = final_result
            .top5
            .iter()
            .map(|&(id, score)| (id, score, format!("doc_{}", id)))
            .collect();
        self.cache.insert(&key, cache_entries);

        ConsensusResult {
            confidence: if run_hologram { hologram.confidence } else if run_wedge { wedge.confidence } else { hash_confidence },
            latency_us: start.elapsed().as_micros() as u64,
            from_cache: false,
            ..final_result
        }
    }

    /// Weighted consensus across stages that were executed.
    pub fn consensus_score(
        &self,
        hash: &FeatureResult,
        spectral: &FeatureResult,
        wedge: &FeatureResult,
        hologram: &FeatureResult,
        chain: &FeatureChain,
    ) -> ConsensusResult {
        let mut score_map: HashMap<u64, (f32, usize)> = HashMap::new();

        let stages: Vec<(&FeatureResult, f32, bool)> = vec![
            (hash, 0.20, true),         // hash always runs
            (spectral, 0.25, chain.run_spectral),
            (wedge, 0.25, chain.run_wedge),
            (hologram, 0.30, chain.run_hologram),
        ];

        let mut active_weight = 0.0f32;
        for (_, weight, active) in &stages {
            if *active {
                active_weight += weight;
            }
        }
        let norm = if active_weight > 0.0 { active_weight } else { 1.0 };

        for (result, weight, active) in &stages {
            if !active {
                continue;
            }
            for (pos, (id, score)) in result.candidates.iter().enumerate() {
                let positional = 1.0 / (1.0 + pos as f32);
                let entry = score_map.entry(*id).or_insert((0.0, 0usize));
                entry.0 += (score * weight * positional) / norm;
                entry.1 += 1;
            }
        }

        let mut ranked: Vec<(u64, f32)> = score_map
            .into_iter()
            .map(|(id, (score, count))| (id, score * (1.0 + 0.1 * count as f32)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(5);

        let confidence = hologram.confidence;

        ConsensusResult {
            top1: ranked.first().copied(),
            top5: ranked,
            chain: chain.clone(),
            confidence,
            latency_us: 0,
            from_cache: false,
        }
    }

    /// Drain pending feedback from the engine feeder and apply it.
    pub fn process_feedback(&mut self) {
        let feeds = self.feeder.list_feeds();
        for feed in feeds {
            while let Some(msg) = self.feeder.poll(&feed) {
                self.metrics.feedback_injected += 1;
                // Decode 8 feature bits from payload if present.
                if msg.payload.len() >= 1 {
                    let pattern = msg.payload[0];
                    // Heuristic: cascade_miss means the chosen chain failed.
                    let failed = feed.contains("miss") || feed.contains("disagree");
                    let chain = self.learning.best_chain(pattern);
                    self.learning.update(pattern, &chain, !failed);
                    // Reward correct runs: MLP target nudges toward the executed chain.
                    if !failed {
                        let target = chain.to_target();
                        let features = Self::features_from_pattern(pattern);
                        self.router.train_step(features, target, 0.01);
                    }
                }
            }
        }
    }

    fn features_from_pattern(pattern: u8) -> [f32; 8] {
        let mut f = [0.0f32; 8];
        for i in 0..8 {
            if (pattern >> i) & 1 == 1 {
                f[i] = 1.0;
            }
        }
        f
    }

    pub fn metrics(&self) -> &CollaborativeMetrics {
        &self.metrics
    }

    pub fn cache_hit_rate(&self) -> f32 {
        self.cache.hit_rate()
    }

    pub fn result_cache(&mut self) -> &mut ResultCache {
        &mut self.cache
    }

    pub fn router(&mut self) -> &mut LearnedRouter {
        &mut self.router
    }

    pub fn learning_table(&mut self) -> &mut SelfLearningTable {
        &mut self.learning
    }

    pub fn feeder(&mut self) -> &mut EngineFeeder {
        &mut self.feeder
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_mesh::HybridMesh;
    use tempfile::tempdir;

    fn ctx_with(id: u64) -> QueryContext {
        let mut pap_128 = [0u8; PAP_128_BYTES];
        let mut pap_512 = [0u8; PAP_512_BYTES];
        let bytes = id.to_le_bytes();
        for i in 0..PAP_128_BYTES {
            pap_128[i] = bytes[i % 8].wrapping_add(i as u8);
        }
        for i in 0..PAP_512_BYTES {
            pap_512[i] = bytes[i % 8].wrapping_add(i as u8);
        }
        QueryContext {
            pap_128,
            pap_512,
            features: [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            expected_id: None,
        }
    }

    fn build_engine_with(ids: &[u64]) -> CollaborativeEngine {
        let mut mesh = HybridMesh::new(1024, 1024);
        for &id in ids {
            let ctx = ctx_with(id);
            mesh.insert_dual(id, &ctx.pap_128, &ctx.pap_512);
        }
        mesh.coarse.build_edges(8);
        mesh.fine.build_edges(8);
        CollaborativeEngine::new(mesh)
    }

    #[test]
    fn query_returns_top1_and_top5() {
        let mut engine = build_engine_with(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let ctx = ctx_with(3);
        let result = engine.query(&ctx);
        assert!(result.top1.is_some());
        assert!(!result.top5.is_empty());
        assert!(result.confidence > 0.0);
        assert!(!result.from_cache);
    }

    #[test]
    fn cache_returns_second_query_from_cache() {
        let mut engine = build_engine_with(&[10, 20, 30, 40, 50]);
        let ctx = ctx_with(20);
        let r1 = engine.query(&ctx);
        assert!(!r1.from_cache);
        let r2 = engine.query(&ctx);
        assert!(r2.from_cache);
        assert_eq!(engine.metrics().cache_hits, 1);
    }

    #[test]
    fn feedback_updates_learning_and_router() {
        let mut engine = build_engine_with(&[100, 200, 300]);
        let pattern: u8 = 0b0000_0101;
        // A miss injects a negative observation; table still needs ≥4 observations
        // before best_chain deviates from the safe default.
        for _ in 0..4 {
            engine.feeder().inject_raw("cascade_miss", &[pattern]);
        }
        engine.process_feedback();
        assert_eq!(engine.metrics().feedback_injected, 4);
        // The engine is still functional after processing feedback.
        let ctx = ctx_with(200);
        let result = engine.query(&ctx);
        assert!(result.top1.is_some());
    }

    #[test]
    fn plan_chain_respects_learning_table() {
        let mut engine = build_engine_with(&[1, 2, 3]);
        // Teach pattern 0 that hash-only is sufficient (cheapest chain).
        let none = FeatureChain::none();
        for _ in 0..4 {
            engine.learning_table().update(0, &none, true);
        }
        let ctx = QueryContext {
            pap_128: [0u8; PAP_128_BYTES],
            pap_512: [0u8; PAP_512_BYTES],
            features: [0.0; 8],
            expected_id: None,
        };
        let chain = engine.plan_chain(ctx.features);
        assert!(!chain.run_spectral);
        assert!(!chain.run_wedge);
        assert!(!chain.run_hologram);
    }

    #[test]
    fn consensus_score_ranks_candidates() {
        let engine = build_engine_with(&[1, 2, 3]);
        let h = FeatureResult { score: 1.0, confidence: 0.5, candidates: vec![(1, 1.0), (2, 0.8)] };
        let s = FeatureResult { score: 0.9, confidence: 0.6, candidates: vec![(2, 0.9), (1, 0.7)] };
        let w = FeatureResult { score: 0.85, confidence: 0.55, candidates: vec![(1, 0.85), (2, 0.82)] };
        let g = FeatureResult { score: 0.95, confidence: 0.9, candidates: vec![(1, 0.95), (2, 0.6)] };
        let chain = FeatureChain::all();
        let result = engine.consensus_score(&h, &s, &w, &g, &chain);
        assert_eq!(result.top1.unwrap().0, 1);
        assert_eq!(result.top5.len(), 2);
    }

    #[test]
    fn drift_detection_triggers_on_bad_queries() {
        let mut engine = build_engine_with(&[1, 2, 3]);
        engine.enable_drift_detection(20, 0.9, 10);

        // 10 correct self-queries
        for _ in 0..10 {
            let mut ctx = ctx_with(1);
            ctx.expected_id = Some(1);
            engine.query(&ctx);
        }
        assert!(!engine.drift_detector.as_ref().unwrap().is_drift());

        // 10 incorrect queries: retrieve record 2 but claim expected 1
        for _ in 0..10 {
            let mut ctx = ctx_with(2);
            ctx.expected_id = Some(1);
            engine.query(&ctx);
        }
        assert!(engine.drift_detector.as_ref().unwrap().is_drift());
    }

    #[test]
    fn sharded_integration_returns_exact_match() {
        let mut engine = build_engine_with(&[1, 2, 3, 4, 5]);
        let dir = tempdir().unwrap();
        let cold_path = dir.path().join("cold.bin");
        engine.enable_sharding(&cold_path, 0.2, false).unwrap();
        engine.set_sharding_threshold(1); // force sharded path for small test index

        let ctx = ctx_with(3);
        let result = engine.query(&ctx);
        assert_eq!(result.top1.map(|(id, _)| id), Some(3));
        assert!(result.confidence > 0.99);
    }
}
