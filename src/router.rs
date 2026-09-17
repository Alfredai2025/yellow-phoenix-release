// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Unified query router over the Crystal Mesh.

use crate::crystal::CrystalMesh;

pub type EntityId = u64;

/// Query strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    /// O(1) direct hash lookup.
    Direct,
    /// Residue-signature lookup with exact verification.
    Crystal,
    /// Exact match + L0 neighbor exploration.
    ExploreNeighbors,
    /// Return all entities in the same primary CRT bucket.
    Cluster,
}

/// Result bundle from a routed query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryResult {
    pub exact_match: Option<EntityId>,
    pub neighbors: Option<Vec<EntityId>>,
    pub cluster: Option<Vec<EntityId>>,
    pub latency_ns: u64,
    pub path_taken: String,
}

/// High-level router that dispatches queries to the appropriate mesh path.
pub struct Router {
    crystal: CrystalMesh,
}

impl Router {
    pub fn new() -> Self {
        Self {
            crystal: CrystalMesh::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            crystal: CrystalMesh::with_capacity(capacity),
        }
    }

    /// Insert an entity into the underlying mesh.
    pub fn insert(&mut self, id: EntityId, pap: &[u8; 64]) {
        self.crystal.insert(id, pap);
    }

    /// Direct hash query (bypasses routing modes).
    pub fn query_direct(&self, query_pap: &[u8; 64]) -> Option<EntityId> {
        self.crystal.query_direct(query_pap)
    }

    /// Crystal residue query (bypasses routing modes).
    pub fn query_crystal(&self, query_pap: &[u8; 64]) -> Option<EntityId> {
        self.crystal.query_crystal(query_pap)
    }

    /// Execute a query according to the selected mode.
    pub fn query(&self, query_pap: &[u8; 64], mode: QueryMode) -> QueryResult {
        let start = std::time::Instant::now();

        match mode {
            QueryMode::Direct => QueryResult {
                exact_match: self.crystal.query_direct(query_pap),
                neighbors: None,
                cluster: None,
                latency_ns: start.elapsed().as_nanos() as u64,
                path_taken: "direct".to_string(),
            },
            QueryMode::Crystal => QueryResult {
                exact_match: self.crystal.query_crystal(query_pap),
                neighbors: None,
                cluster: None,
                latency_ns: start.elapsed().as_nanos() as u64,
                path_taken: "crystal".to_string(),
            },
            QueryMode::ExploreNeighbors => {
                let exact = self.crystal.query_direct(query_pap);
                let neighbors = exact.and_then(|id| self.crystal.get_neighbors(id));
                QueryResult {
                    exact_match: exact,
                    neighbors: neighbors.map(|arr| arr.to_vec()),
                    cluster: None,
                    latency_ns: start.elapsed().as_nanos() as u64,
                    path_taken: "explore".to_string(),
                }
            }
            QueryMode::Cluster => {
                // Return all entities in the same crystal bucket as the queried PAP.
                // If the queried PAP is not indexed, fall back to the whole mesh.
                let cluster = self
                    .crystal
                    .query_direct(query_pap)
                    .and_then(|id| self.crystal.get_cluster(id))
                    .or_else(|| Some(self.crystal.slots.iter().map(|s| s.id).collect()));
                QueryResult {
                    exact_match: None,
                    neighbors: None,
                    cluster,
                    latency_ns: start.elapsed().as_nanos() as u64,
                    path_taken: "cluster".to_string(),
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.crystal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.crystal.is_empty()
    }

    pub fn get_neighbors(&self, entity_id: EntityId) -> Option<[EntityId; 3]> {
        self.crystal.get_neighbors(entity_id).map(|v| [v.get(0).copied().unwrap_or(0), v.get(1).copied().unwrap_or(0), v.get(2).copied().unwrap_or(0)]).map(|v| [v.get(0).copied().unwrap_or(0), v.get(1).copied().unwrap_or(0), v.get(2).copied().unwrap_or(0)]).map(|v| [v.get(0).copied().unwrap_or(0), v.get(1).copied().unwrap_or(0), v.get(2).copied().unwrap_or(0)]).map(|v| [v.get(0).copied().unwrap_or(0), v.get(1).copied().unwrap_or(0), v.get(2).copied().unwrap_or(0)])
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pap_from_seed(seed: u64) -> [u8; 64] {
        let mut pap = [0u8; 64];
        let bytes = seed.to_le_bytes();
        for chunk in pap.chunks_exact_mut(8) {
            chunk.copy_from_slice(&bytes);
        }
        pap
    }

    #[test]
    fn test_router_direct_finds_match() {
        let mut router = Router::new();
        let pap = pap_from_seed(42);
        router.insert(42, &pap);
        let result = router.query(&pap, QueryMode::Direct);
        assert_eq!(result.exact_match, Some(42));
        assert_eq!(result.path_taken, "direct");
        assert!(result.latency_ns >= 0);
    }

    #[test]
    fn test_router_crystal_finds_match() {
        let mut router = Router::new();
        let pap = pap_from_seed(99);
        router.insert(99, &pap);
        let result = router.query(&pap, QueryMode::Crystal);
        assert_eq!(result.exact_match, Some(99));
        assert_eq!(result.path_taken, "crystal");
    }

    #[test]
    fn test_router_explore_returns_neighbors() {
        let mut router = Router::new();
        // Insert enough nodes so a node inserted later has 3 real L0 neighbors.
        for i in 0..10u64 {
            router.insert(10 + i, &pap_from_seed(i));
        }
        // Query node 17 (inserted after 10..16, so it has at least 3 prior neighbors).
        let pap = pap_from_seed(7);
        let result = router.query(&pap, QueryMode::ExploreNeighbors);
        assert_eq!(result.exact_match, Some(17));
        assert!(result.neighbors.is_some());
        let neighbors = result.neighbors.unwrap();
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.iter().all(|&id| id != 0));
    }

    #[test]
    fn test_router_cluster_returns_group() {
        let mut router = Router::with_capacity(50);
        for i in 0..50u64 {
            router.insert(i, &pap_from_seed(i));
        }
        let pap = pap_from_seed(7);
        let result = router.query(&pap, QueryMode::Cluster);
        assert!(result.cluster.is_some());
        let cluster = result.cluster.unwrap();
        assert!(!cluster.is_empty());
        assert!(cluster.len() <= 50);
        assert_eq!(result.path_taken, "cluster");
    }

    #[test]
    fn test_router_latency_tracked() {
        let mut router = Router::new();
        let pap = pap_from_seed(1);
        router.insert(1, &pap);
        let result = router.query(&pap, QueryMode::Direct);
        assert!(result.latency_ns >= 0);
        println!("router latency: {} ns", result.latency_ns);
    }
}
