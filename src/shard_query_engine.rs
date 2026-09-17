// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

#[cfg(feature = "intelligent_shard_manager")]
#[cfg(feature = "intelligent_shard_manager")]
use crate::intelligent_shard_manager::{IntelligentShardManager, QueryResult};

/// Thin wrapper that exposes explicit fan-out/merge semantics over an
/// `IntelligentShardManager`.
pub struct ShardQueryEngine {
    manager: IntelligentShardManager,
}

impl ShardQueryEngine {
    /// Wrap an existing manager.
    pub fn new(manager: IntelligentShardManager) -> Self {
        Self { manager }
    }

    /// Sequential fan-out query.
    pub fn query(&self, cell_id: u64) -> Vec<QueryResult> {
        self.manager.query(cell_id)
    }

    /// Parallel fan-out query.
    pub fn query_parallel(&self, cell_id: u64) -> Vec<QueryResult> {
        self.manager.query_parallel(cell_id)
    }

    /// Return global top-k results after deduplication (currently exact-match only).
    pub fn query_top_k(&self, cell_id: u64, _k: usize) -> Vec<QueryResult> {
        // For exact identity lookups, there is at most one result per cell_id.
        // Deduplication is therefore a no-op in v0.3.
        self.manager.query_parallel(cell_id)
    }

    /// Borrow the underlying manager.
    pub fn manager(&self) -> &IntelligentShardManager {
        &self.manager
    }

    /// Mutable borrow to the underlying manager.
    pub fn manager_mut(&mut self) -> &mut IntelligentShardManager {
        &mut self.manager
    }
}
