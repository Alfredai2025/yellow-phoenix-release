// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::ism_error::ISMError;
use crate::parallel_builder::ParallelBuilder;
pub use crate::shard::{IsmPap, Shard, ShardError, ISM_MAX_SHARD_CAPACITY};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Maximum number of shards in ISM v0.3 (four × 50M = 200M).
pub const ISM_MAX_SHARDS: usize = 4;

/// Default shard capacity when no learned params are available.
pub const ISM_DEFAULT_SHARD_CAPACITY: usize = ISM_MAX_SHARD_CAPACITY;

/// Persisted learning state: optimal shard size and observed build cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedParams {
    pub optimal_shard_size: usize,
    pub build_time_per_m: f64,
    pub last_build_timestamp: u64,
}

impl Default for LearnedParams {
    fn default() -> Self {
        Self {
            optimal_shard_size: ISM_DEFAULT_SHARD_CAPACITY,
            build_time_per_m: 0.0,
            last_build_timestamp: 0,
        }
    }
}

/// Result returned by an ISM query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub cell_id: u64,
    pub data: IsmPap,
    pub shard_id: u32,
}

/// Core ISM: splits a corpus into independent shards, builds them in parallel,
/// and fans out queries.
pub struct IntelligentShardManager {
    pub shards: Vec<Shard>,
    pub learned: LearnedParams,
    pub total_count: usize,
    pub last_build_time_s: f64,
}

impl IntelligentShardManager {
    /// Create a fresh manager with default learned parameters.
    pub fn new() -> Self {
        Self {
            shards: Vec::new(),
            learned: LearnedParams::default(),
            total_count: 0,
            last_build_time_s: 0.0,
        }
    }

    /// Create a manager seeded with previously learned parameters.
    pub fn with_learned(learned: LearnedParams) -> Self {
        Self {
            shards: Vec::new(),
            learned,
            total_count: 0,
            last_build_time_s: 0.0,
        }
    }

    /// Load learned parameters from disk, falling back to defaults on error.
    pub fn load_learned(path: &str) -> Self {
        let learned = if Path::new(path).exists() {
            std::fs::read_to_string(path)
                .ok()
                .and_then(|s| serde_json::from_str::<LearnedParams>(&s).ok())
                .unwrap_or_default()
        } else {
            LearnedParams::default()
        };
        Self::with_learned(learned)
    }

    /// Determine the number of shards and capacity per shard for `n` papers.
    /// Respects the hard ceiling of `ISM_MAX_SHARDS` and `ISM_MAX_SHARD_CAPACITY`.
    fn plan(n: usize) -> Result<(usize, usize), ShardError> {
        if n == 0 {
            return Ok((0, 0));
        }
        let max_total = ISM_MAX_SHARDS * ISM_MAX_SHARD_CAPACITY;
        if n > max_total {
            return Err(ShardError::Full {
                capacity: max_total,
            });
        }
        // Use up to 4 shards; size each shard so no shard exceeds 50M.
        let n_shards = ((n + ISM_MAX_SHARD_CAPACITY - 1) / ISM_MAX_SHARD_CAPACITY).max(1);
        let per_shard = ((n + n_shards - 1) / n_shards).max(1);
        Ok((n_shards, per_shard))
    }

    /// Sequential build path (used for small corpora or debugging).
    pub fn build(&mut self, papers: Vec<(u64, IsmPap)>) -> Result<(), ShardError> {
        let start = Instant::now();
        self.shards.clear();
        self.total_count = papers.len();

        let (n_shards, per_shard) = Self::plan(papers.len())?;
        let mut chunks = papers.chunks(per_shard).peekable();
        for shard_id in 0..n_shards {
            let chunk = chunks.next().unwrap_or(&[]);
            let mut shard = Shard::new(shard_id as u32, per_shard);
            for (cell_id, pap) in chunk {
                shard.insert(*cell_id, pap.clone())?;
            }
            self.shards.push(shard);
        }

        self.last_build_time_s = start.elapsed().as_secs_f64();
        self.learn_from_build();
        Ok(())
    }

    /// Parallel build from flat borrowed slices (memory-efficient FFI path).
    /// `ids` and `hashes` must have the same length; each hash is `hash_len` bytes.
    pub fn build_parallel_flat(
        &mut self,
        ids: &[u64],
        hashes: &[u8],
        hash_len: usize,
    ) -> Result<(), ShardError> {
        let start = Instant::now();
        self.shards.clear();
        self.total_count = ids.len();

        if ids.is_empty() {
            self.last_build_time_s = 0.0;
            return Ok(());
        }
        if hashes.len() != ids.len() * hash_len {
            return Err(ShardError::Full { capacity: 0 });
        }

        let (n_shards, per_shard) = Self::plan(ids.len())?;
        let ranges: Vec<(usize, usize)> = (0..n_shards)
            .map(|i| {
                let start = i * per_shard;
                let end = ((i + 1) * per_shard).min(ids.len());
                (start, end)
            })
            .collect();

        self.shards = ParallelBuilder::new(n_shards)
            .build_shards_flat(ids, hashes, hash_len, ranges)?;

        self.last_build_time_s = start.elapsed().as_secs_f64();
        self.learn_from_build();
        Ok(())
    }

    /// Memory-efficient parallel build from a memory-mapped file.
    /// The file must contain `n` fixed-size records: 8-byte LE id + `hash_len` hash bytes.
    pub fn build_parallel_flat_from_file(
        &mut self,
        path: &str,
        n: usize,
        hash_len: usize,
    ) -> Result<(), ShardError> {
        let start = Instant::now();
        self.shards.clear();
        self.total_count = n;

        if n == 0 {
            self.last_build_time_s = 0.0;
            return Ok(());
        }

        let file = std::fs::File::open(path).map_err(|_| ShardError::Full { capacity: 0 })?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|_| ShardError::Full { capacity: 0 })? };
        let record_size = 8 + hash_len;
        if mmap.len() != n * record_size {
            return Err(ShardError::Full { capacity: 0 });
        }

        let (n_shards, per_shard) = Self::plan(n)?;
        let ranges: Vec<(usize, usize)> = (0..n_shards)
            .map(|i| {
                let start = i * per_shard;
                let end = ((i + 1) * per_shard).min(n);
                (start, end)
            })
            .collect();

        self.shards = ParallelBuilder::new(n_shards)
            .build_shards_flat_from_file(&mmap, hash_len, ranges)?;

        self.last_build_time_s = start.elapsed().as_secs_f64();
        self.learn_from_build();
        Ok(())
    }

    /// Sequential shard build from a memory-mapped file.
    /// Avoids concurrent allocation pressure; useful when RAM is tight.
    pub fn build_flat_from_file_sequential(
        &mut self,
        path: &str,
        n: usize,
        hash_len: usize,
    ) -> Result<(), ShardError> {
        let start = Instant::now();
        self.shards.clear();
        self.total_count = n;

        if n == 0 {
            self.last_build_time_s = 0.0;
            return Ok(());
        }

        let file = std::fs::File::open(path).map_err(|_| ShardError::Full { capacity: 0 })?;
        let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|_| ShardError::Full { capacity: 0 })? };
        let record_size = 8 + hash_len;
        if mmap.len() != n * record_size {
            return Err(ShardError::Full { capacity: 0 });
        }

        let (n_shards, per_shard) = Self::plan(n)?;
        for shard_id in 0..n_shards {
            let start_i = shard_id * per_shard;
            let end_i = ((shard_id + 1) * per_shard).min(n);
            let mut shard = Shard::new(shard_id as u32, end_i - start_i);
            for i in start_i..end_i {
                let off = i * record_size;
                let id = u64::from_le_bytes(
                    mmap[off..off + 8].try_into().map_err(|_| ShardError::Full {
                        capacity: 0,
                    })?,
                );
                let hash_off = off + 8;
                let pap: IsmPap = mmap[hash_off..hash_off + hash_len]
                    .try_into()
                    .map_err(|_| ShardError::Full { capacity: 0 })?;
                shard.insert(id, pap)?;
            }
            self.shards.push(shard);
        }

        self.last_build_time_s = start.elapsed().as_secs_f64();
        self.learn_from_build();
        Ok(())
    }

    /// Parallel build with an intelligent watchdog.
    /// Spawns one thread per shard, monitors progress/CPU/memory, and kills
    /// the build if it stalls.
    pub fn build_parallel_with_watchdog(
        &mut self,
        papers: Vec<(u64, IsmPap)>,
    ) -> Result<(), ISMError> {
        use crate::build_context::BuildContext;
        use crate::watchdog::Watchdog;

        let start = Instant::now();
        self.shards.clear();
        self.total_count = papers.len();

        if papers.is_empty() {
            self.last_build_time_s = 0.0;
            return Ok(());
        }

        let (_n_shards, per_shard) = Self::plan(papers.len()).map_err(ISMError::from)?;
        let chunks: Vec<Vec<(u64, IsmPap)>> = papers
            .chunks(per_shard)
            .map(|c| c.to_vec())
            .collect();

        let ctx = Arc::new(BuildContext::new());
        let mut handles = Vec::with_capacity(chunks.len());

        for (shard_id, chunk) in chunks.into_iter().enumerate() {
            let ctx = Arc::clone(&ctx);
            let handle = std::thread::spawn(move || {
                let result: Result<Shard, ISMError> = (|| {
                    let mut shard = Shard::new(shard_id as u32, chunk.len());
                    for (cell_id, pap) in chunk {
                        if ctx.should_stop() {
                            return Err(ISMError::KilledByWatchdog {
                                reason: "watchdog requested stop".to_string(),
                                advice: String::new(),
                            });
                        }
                        shard.insert(cell_id, pap)?;
                        ctx.increment_progress();
                    }
                    Ok(shard)
                })();
                ctx.worker_done();
                result
            });
            handles.push(handle);
        }

        let mut watchdog = Watchdog::new(Arc::clone(&ctx));
        {
            let advisor = crate::build_advisor::BuildAdvisor::new();
            watchdog.set_advice(advisor.advice_for_failure(self.total_count as u64));
        }
        let watchdog_result = watchdog.run();

        let mut shards = Vec::with_capacity(handles.len());
        for handle in handles {
            match handle.join() {
                Ok(Ok(shard)) => shards.push(shard),
                Ok(Err(e)) => {
                    ctx.request_stop();
                    return Err(e);
                }
                Err(_) => {
                    ctx.request_stop();
                    return Err(ISMError::ThreadPanic);
                }
            }
        }

        if let Err(reason) = watchdog_result {
            // The watchdog already composes the learned advice into the reason.
            return Err(ISMError::KilledByWatchdog {
                reason,
                advice: String::new(),
            });
        }

        self.shards = shards;
        self.last_build_time_s = start.elapsed().as_secs_f64();
        self.learn_from_build();
        Ok(())
    }

    /// Parallel build path: spawn one thread per shard.
    pub fn build_parallel(&mut self, papers: Vec<(u64, IsmPap)>) -> Result<(), ShardError> {
        let start = Instant::now();
        self.shards.clear();
        self.total_count = papers.len();

        let (n_shards, per_shard) = Self::plan(papers.len())?;
        if n_shards == 0 {
            self.last_build_time_s = 0.0;
            return Ok(());
        }

        let chunks: Vec<Vec<(u64, IsmPap)>> = papers
            .chunks(per_shard)
            .map(|c| c.to_vec())
            .collect();

        let built = ParallelBuilder::new(n_shards).build_shards(chunks)?;
        self.shards = built;

        self.last_build_time_s = start.elapsed().as_secs_f64();
        self.learn_from_build();
        Ok(())
    }

    /// Update learned parameters using the most recent build metrics.
    pub fn learn_from_build(&mut self) {
        let total_m = self.total_count as f64 / 1_000_000.0;
        if total_m > 0.0 && self.last_build_time_s > 0.0 {
            self.learned.build_time_per_m = self.last_build_time_s / total_m;
        }
        // Keep optimal shard size at the hard ceiling unless build cost explodes.
        // Future versions may shrink this dynamically.
        self.learned.optimal_shard_size = ISM_MAX_SHARD_CAPACITY;
        self.learned.last_build_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
    }

    /// Save learned parameters to disk as JSON.
    pub fn save_learned(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.learned)?;
        std::fs::write(path, json)
    }

    /// Query all shards sequentially and return every matching result.
    pub fn query(&self, cell_id: u64) -> Vec<QueryResult> {
        let mut out = Vec::with_capacity(self.shards.len());
        for shard in &self.shards {
            if let Some(data) = shard.get(cell_id) {
                out.push(QueryResult {
                    cell_id,
                    data: data.clone(),
                    shard_id: shard.id,
                });
            }
        }
        out
    }

    /// Query all shards in parallel and return every matching result.
    pub fn query_parallel(&self, cell_id: u64) -> Vec<QueryResult> {
        if self.shards.len() == 1 {
            return self.query_single(cell_id);
        }
        let n = self.shards.len();
        if n == 0 {
            return Vec::new();
        }

        // Pre-allocate per-thread Option slots to avoid allocations in threads.
        let mut per_shard: Vec<Option<QueryResult>> = Vec::with_capacity(n);
        per_shard.resize_with(n, || None);

        std::thread::scope(|s| {
            let handles: Vec<_> = self
                .shards
                .iter()
                .enumerate()
                .map(|(_idx, shard)| {
                    s.spawn(move || {
                        shard.get(cell_id).map(|data| QueryResult {
                            cell_id,
                            data: data.clone(),
                            shard_id: shard.id,
                        })
                    })
                })
                .collect();

            for (idx, handle) in handles.into_iter().enumerate() {
                per_shard[idx] = handle.join().ok().flatten();
            }
        });

        per_shard.into_iter().flatten().collect()
    }

    /// Single-shard lookup without thread spawn overhead.
    pub fn query_single(&self, cell_id: u64) -> Vec<QueryResult> {
        for shard in &self.shards {
            if let Some(pap) = shard.get(cell_id) {
                return vec![QueryResult {
                    cell_id,
                    shard_id: shard.id,
                    data: *pap,
                }];
            }
        }
        vec![]
    }

    /// Return basic status for FFI/reporting.
    pub fn status_json(&self) -> String {
        serde_json::json!({
            "shard_count": self.shards.len(),
            "total_papers": self.total_count,
            "last_build_time_s": self.last_build_time_s,
            "learned": self.learned,
        })
        .to_string()
    }
}

impl Default for IntelligentShardManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_papers(n: usize) -> Vec<(u64, IsmPap)> {
        (0..n as u64)
            .map(|i| {
                let mut pap = [0u8; 32];
                pap[0] = (i % 256) as u8;
                (i, pap)
            })
            .collect()
    }

    #[test]
    fn plan_fits_200m_in_four_shards() {
        let (shards, per) = IntelligentShardManager::plan(200_000_000).unwrap();
        assert_eq!(shards, 4);
        assert_eq!(per, 50_000_000);
    }

    #[test]
    fn plan_rejects_over_capacity() {
        assert!(IntelligentShardManager::plan(200_000_001).is_err());
    }

    #[test]
    fn sequential_build_and_query() {
        let mut mgr = IntelligentShardManager::new();
        let papers = make_papers(1_000);
        mgr.build(papers.clone()).unwrap();
        assert_eq!(mgr.total_count, 1_000);
        assert!(mgr.shards.len() <= ISM_MAX_SHARDS);
        let results = mgr.query(42);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cell_id, 42);
    }

    #[test]
    fn parallel_build_and_query_matches_sequential() {
        let mut mgr = IntelligentShardManager::new();
        let papers = make_papers(10_000);
        mgr.build_parallel(papers.clone()).unwrap();
        let results = mgr.query_parallel(1234);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].cell_id, 1234);
    }

    #[test]
    fn learned_params_persisted() {
        let mut mgr = IntelligentShardManager::new();
        mgr.build_parallel(make_papers(1_000)).unwrap();
        let path = "/tmp/ism_learned_test.json";
        mgr.save_learned(path).unwrap();
        let loaded = IntelligentShardManager::load_learned(path);
        assert_eq!(loaded.learned.optimal_shard_size, ISM_MAX_SHARD_CAPACITY);
        std::fs::remove_file(path).unwrap();
    }
}
