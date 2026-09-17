// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! temporal_orchestrator.rs — Milestone 2 Component 2.
//!
//! Single-threaded pipeline orchestrator. Each `tick()` advances every
//! in-flight query by one feature stage. This overlaps work across queries:
//! while Query N runs its hash stage, Query N-1 may run spectral, etc.

use crate::async_ffi::AsyncTokenPool;
use crate::collaborative_engine::{CollaborativeEngine, ConsensusResult, QueryContext};
use crate::hash_stage::FeatureResult;
use crate::learned_router::FeatureChain;

const DEFAULT_MAX_IN_FLIGHT: usize = 64;

struct InFlightQuery {
    token: u64,
    ctx: QueryContext,
    chain: FeatureChain,
    stage: u8,
    hash: Option<FeatureResult>,
    spectral: Option<FeatureResult>,
    wedge: Option<FeatureResult>,
    hologram: Option<FeatureResult>,
}

pub struct TemporalOrchestrator {
    engine: CollaborativeEngine,
    token_pool: AsyncTokenPool,
    in_flight: Vec<InFlightQuery>,
    max_in_flight: usize,
}

impl TemporalOrchestrator {
    pub fn new(engine: CollaborativeEngine) -> Self {
        Self {
            engine,
            token_pool: AsyncTokenPool::new(),
            in_flight: Vec::with_capacity(DEFAULT_MAX_IN_FLIGHT),
            max_in_flight: DEFAULT_MAX_IN_FLIGHT,
        }
    }

    pub fn with_capacity(engine: CollaborativeEngine, max_in_flight: usize) -> Self {
        Self {
            engine,
            token_pool: AsyncTokenPool::new(),
            in_flight: Vec::with_capacity(max_in_flight),
            max_in_flight,
        }
    }

    /// Submit a query. Returns a token, or 0 if the pipeline is full.
    pub fn submit(&mut self, ctx: QueryContext) -> u64 {
        if self.in_flight.len() >= self.max_in_flight {
            return 0;
        }
        let chain = self.engine.plan_chain(ctx.features);
        let token = self.token_pool.submit(Vec::new());
        self.in_flight.push(InFlightQuery {
            token,
            ctx,
            chain,
            stage: 0,
            hash: None,
            spectral: None,
            wedge: None,
            hologram: None,
        });
        token
    }

    /// Advance every in-flight query by one stage.
    pub fn tick(&mut self) {
        let mut completed = Vec::new();

        for (idx, q) in self.in_flight.iter_mut().enumerate() {
            match q.stage {
                0 => {
                    q.hash = Some(self.engine.run_hash(&q.ctx));
                    q.stage = 1;
                }
                1 => {
                    q.spectral = Some(if q.chain.run_spectral {
                        self.engine.run_spectral(&q.ctx, q.hash.as_ref().unwrap())
                    } else {
                        q.hash.clone().unwrap()
                    });
                    q.stage = 2;
                }
                2 => {
                    q.wedge = Some(if q.chain.run_wedge {
                        self.engine.run_wedge(&q.ctx, q.spectral.as_ref().unwrap())
                    } else {
                        q.spectral.clone().unwrap()
                    });
                    q.stage = 3;
                }
                3 => {
                    q.hologram = Some(if q.chain.run_hologram {
                        self.engine.run_hologram(&q.ctx, q.wedge.as_ref().unwrap())
                    } else {
                        q.wedge.clone().unwrap()
                    });
                    q.stage = 4;
                }
                4 => {
                    let result = self.engine.finalize(
                        q.hash.as_ref().unwrap(),
                        q.spectral.as_ref().unwrap(),
                        q.wedge.as_ref().unwrap(),
                        q.hologram.as_ref().unwrap(),
                        &q.chain,
                    );
                    let payload = result_to_json(&result);
                    completed.push(idx);
                    self.token_pool.complete(q.token, payload);
                }
                _ => {}
            }
        }

        // Remove completed queries in reverse order to keep indices valid.
        for idx in completed.into_iter().rev() {
            self.in_flight.remove(idx);
        }
    }

    /// Poll for a completed result.
    pub fn poll(&mut self, token: u64) -> Option<Vec<u8>> {
        self.token_pool.poll(token)
    }

    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Run ticks until the token is ready (blocking helper for tests).
    pub fn drain_until(&mut self, token: u64, max_ticks: usize) -> Option<ConsensusResult> {
        for _ in 0..max_ticks {
            if let Some(payload) = self.poll(token) {
                return serde_json::from_slice(&payload).ok();
            }
            self.tick();
        }
        None
    }
}

fn result_to_json(result: &ConsensusResult) -> Vec<u8> {
    serde_json::json!({
        "top1": result.top1,
        "top5": result.top5,
        "chain": {
            "run_spectral": result.chain.run_spectral,
            "run_wedge": result.chain.run_wedge,
            "run_hologram": result.chain.run_hologram,
        },
        "confidence": result.confidence,
        "latency_us": result.latency_us,
        "from_cache": result.from_cache,
    })
    .to_string()
    .into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};

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

    fn build_orchestrator(ids: &[u64]) -> TemporalOrchestrator {
        let mut mesh = HybridMesh::new(1024, 1024);
        for &id in ids {
            let ctx = ctx_with(id);
            mesh.insert_dual(id, &ctx.pap_128, &ctx.pap_512);
        }
        mesh.coarse.build_edges(8);
        mesh.fine.build_edges(8);
        TemporalOrchestrator::new(CollaborativeEngine::new(mesh))
    }

    #[test]
    fn pipeline_advances_one_stage_per_tick() {
        let mut orch = build_orchestrator(&[1, 2, 3, 4, 5]);
        let token = orch.submit(ctx_with(3));
        assert!(token > 0);
        assert_eq!(orch.in_flight_count(), 1);

        orch.tick(); // hash
        assert_eq!(orch.in_flight_count(), 1);
        orch.tick(); // spectral
        orch.tick(); // wedge
        orch.tick(); // hologram
        assert_eq!(orch.in_flight_count(), 1);
        orch.tick(); // finalize
        assert_eq!(orch.in_flight_count(), 0);

        let result = orch.poll(token).expect("result ready");
        let parsed: serde_json::Value = serde_json::from_slice(&result).unwrap();
        assert!(parsed.get("top1").is_some());
    }

    #[test]
    fn two_queries_overlap_stages() {
        let mut orch = build_orchestrator(&[10, 20, 30, 40, 50]);
        let t1 = orch.submit(ctx_with(20));
        let t2 = orch.submit(ctx_with(30));

        // After 1 tick, both have done hash; second hasn't done spectral yet.
        orch.tick();
        assert_eq!(orch.in_flight_count(), 2);

        // Drain both.
        let r1 = orch.drain_until(t1, 10).expect("t1 completes");
        let r2 = orch.drain_until(t2, 10).expect("t2 completes");
        assert!(r1.top1.is_some());
        assert!(r2.top1.is_some());
    }

    #[test]
    fn full_pipeline_returns_correct_top1() {
        let mut orch = build_orchestrator(&[100, 200, 300]);
        let token = orch.submit(ctx_with(200));
        let result = orch.drain_until(token, 10).expect("completes");
        assert_eq!(result.top1.unwrap().0, 200);
    }
}
