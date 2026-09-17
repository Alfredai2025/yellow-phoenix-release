// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! batch_fusion.rs — M3.5: 1ms batched fast-path lookups.
//!
//! Collects incoming queries for a 1ms window, groups them by the first 8 bytes
//! of their 128-bit hash, then executes a single `mesh.query_fast_path()` per
//! group and distributes the results back to the waiters.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use crate::collaborative_engine::QueryContext;
use crate::hybrid_mesh::HybridMesh;

/// Result returned to each query that participated in a fused batch.
#[derive(Clone, Debug)]
pub struct BatchResult {
    pub top1: Option<(u64, f32)>,
    pub top5: Vec<(u64, f32)>,
}

/// Internal request sent from a client thread to the batch worker.
struct BatchRequest {
    ctx: QueryContext,
    resp: mpsc::Sender<BatchResult>,
}

/// Counters used to report how often fusion actually happened.
#[derive(Default)]
struct BatchMetrics {
    total: AtomicU64,
    fused: AtomicU64,
}

/// Shared batching engine. Spawn one worker thread that fuses lookups.
pub struct BatchFusionEngine {
    tx: mpsc::Sender<BatchRequest>,
    metrics: Arc<BatchMetrics>,
    handle: Option<thread::JoinHandle<()>>,
}

impl BatchFusionEngine {
    /// Spawn a worker that collects queries for `window_ms`, groups by the first
    /// 8 bytes of `pap_128`, and executes one fast-path lookup per group.
    pub fn new(mesh: HybridMesh, window_ms: u64) -> Self {
        let (tx, rx) = mpsc::channel::<BatchRequest>();
        let metrics = Arc::new(BatchMetrics::default());
        let worker_metrics = Arc::clone(&metrics);
        let handle = thread::spawn(move || worker_loop(rx, mesh, window_ms, worker_metrics));
        Self {
            tx,
            metrics,
            handle: Some(handle),
        }
    }

    /// Submit a single query and block until the next batch is processed.
    pub fn submit(&self, ctx: QueryContext) -> BatchResult {
        let (tx, rx) = mpsc::channel();
        let req = BatchRequest { ctx, resp: tx };
        // If the worker has exited, return an empty result instead of panicking.
        if self.tx.send(req).is_err() {
            return BatchResult {
                top1: None,
                top5: Vec::new(),
            };
        }
        rx.recv().unwrap_or(BatchResult {
            top1: None,
            top5: Vec::new(),
        })
    }

    /// Submit a single query without blocking.  The returned receiver yields the
    /// result once the batch containing this query has been processed.
    pub fn submit_async(&self, ctx: QueryContext) -> mpsc::Receiver<BatchResult> {
        let (tx, rx) = mpsc::channel();
        let req = BatchRequest { ctx, resp: tx };
        // If the worker has exited the receiver will be disconnected and the
        // caller falls back to an empty result.
        let _ = self.tx.send(req);
        rx
    }

    /// Fraction of submitted queries that were in a bucket group with >1 query.
    pub fn fusion_rate(&self) -> f32 {
        let total = self.metrics.total.load(Ordering::Relaxed);
        let fused = self.metrics.fused.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            fused as f32 / total as f32
        }
    }
}

    // NOTE: Intentionally no `Drop` impl. The worker thread exits on its own
    // when the sending side is dropped, and for benchmark binaries the OS
    // cleans up the detached thread when the process terminates.

fn worker_loop(
    rx: mpsc::Receiver<BatchRequest>,
    mesh: HybridMesh,
    window_ms: u64,
    metrics: Arc<BatchMetrics>,
) {
    let window = Duration::from_millis(window_ms);
    loop {
        // Use a startup timeout so the worker exits if no requests arrive.
        let first = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(req) => req,
            Err(mpsc::RecvTimeoutError::Timeout) => break,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        let mut batch = vec![first];
        let start = Instant::now();
        while start.elapsed() < window {
            let remaining = window - start.elapsed();
            match rx.recv_timeout(remaining) {
                Ok(req) => batch.push(req),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        dispatch_batch(batch, &mesh, &metrics);
    }
}

fn dispatch_batch(batch: Vec<BatchRequest>, mesh: &HybridMesh, metrics: &BatchMetrics) {
    if batch.is_empty() {
        return;
    }

    // Group by the coarse bucket prefix: the first 2 bytes of pap_128 match
    // the 16-bit bucket hash used by CrystalMesh128, so queries landing in the
    // same bucket can share a single fast-path lookup.
    let mut groups: HashMap<[u8; 2], Vec<BatchRequest>> = HashMap::new();
    for req in batch {
        let key = [req.ctx.pap_128[0], req.ctx.pap_128[1]];
        groups.entry(key).or_default().push(req);
    }

    for (_, requests) in groups {
        metrics.total.fetch_add(requests.len() as u64, Ordering::Relaxed);
        if requests.len() > 1 {
            metrics.fused.fetch_add(requests.len() as u64, Ordering::Relaxed);
        }

        // Enumerate the bucket once, then score each query's candidates locally.
        // This fuses the expensive bucket lookup while keeping per-query ranking.
        let rep = &requests[0].ctx;
        let indices = mesh.bucket_slot_indices(&rep.pap_128).unwrap_or_default();

        for req in requests {
            let mut scored: Vec<(f32, u64)> = indices
                .iter()
                .map(|&idx| {
                    let slot = &mesh.coarse.slots[idx];
                    let score = crate::hybrid_mesh::pap_distance_128(&req.ctx.pap_128, &slot.pap);
                    (score, slot.id)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(5);
            let top5: Vec<(u64, f32)> = scored.iter().map(|(score, id)| (*id, *score)).collect();
            let result = BatchResult {
                top1: top5.first().copied(),
                top5,
            };
            let _ = req.resp.send(result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_mesh::{pap_128_from_seed, pap_512_from_seed, HybridMesh};

    #[test]
    fn batch_fusion_returns_results() {
        let mut mesh = HybridMesh::new(1024, 1024);
        for id in 0..16u64 {
            mesh.insert_dual(id, &pap_128_from_seed(id), &pap_512_from_seed(id));
        }
        mesh.coarse.build_edges(8);

        let engine = BatchFusionEngine::new(mesh, 1);
        let ctx = QueryContext {
            pap_128: pap_128_from_seed(7),
            pap_512: pap_512_from_seed(7),
            features: [0.0; 8],
            expected_id: None,
        };
        let result = engine.submit(ctx);
        assert!(result.top1.is_some());
    }
}
