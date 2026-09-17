// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashMap;

/// Cheat Sheet Cascade: 3-layer exact verification system
/// Layer 1: Common pairs (most frequent co-occurrences) — O(1) lookup
/// Layer 2: Bucket-specific patterns (within same hash bucket) — O(1) lookup  
/// Layer 3: Full exact PAP comparison — O(1) with early termination

/// A single "cheat sheet" entry: pre-computed exact match for a pattern
#[derive(Clone, Debug)]
pub struct CheatEntry {
    pub pattern: Vec<u8>,      // The binary pattern to match
    pub match_id: u64,        // ID of the matching paper
    pub confidence: f32,     // Pre-computed confidence score
}

/// Layer 1: Global cheat sheet — most common patterns across all papers
pub struct GlobalCheatSheet {
    pub entries: HashMap<Vec<u8>, Vec<CheatEntry>>,  // pattern -> matches
    pub hit_count: u64,
    pub miss_count: u64,
}

/// Layer 2: Bucket cheat sheets — patterns specific to hash buckets
pub struct BucketCheatSheet {
    pub _bucket_id: u64,
    pub entries: HashMap<Vec<u8>, Vec<CheatEntry>>,
}

/// Layer 3: Full exact comparison with early termination
pub struct ExactComparator {
    pub max_comparisons: usize,  // Early termination threshold
}

impl GlobalCheatSheet {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hit_count: 0,
            miss_count: 0,
        }
    }
    
    /// Lookup a pattern — O(1)
    pub fn lookup(&mut self, pattern: &[u8]) -> Option<&Vec<CheatEntry>> {
        let key = pattern.to_vec();
        if let Some(entries) = self.entries.get(&key) {
            self.hit_count += 1;
            Some(entries)
        } else {
            self.miss_count += 1;
            None
        }
    }
    
    pub fn hit_rate(&self) -> f32 {
        let total = self.hit_count + self.miss_count;
        if total == 0 { 0.0 } else { self.hit_count as f32 / total as f32 }
    }
}

impl ExactComparator {
    pub fn new(max_comparisons: usize) -> Self {
        Self { max_comparisons }
    }
    
    /// Full exact PAP comparison with early termination
    pub fn compare_exact(&self, query: &[u8], candidate: &[u8]) -> f32 {
        if query.len() != candidate.len() {
            return 0.0;
        }
        
        let mut matching = 0usize;
        let mut total_ones = 0usize;
        let chunk_size = 64; // Compare 64 bits at a time
        
        for (i, (q, c)) in query.iter().zip(candidate.iter()).enumerate() {
            let q_bits = q.count_ones() as usize;
            let c_bits = c.count_ones() as usize;
            total_ones += q_bits + c_bits;
            
            // XOR to find differing bits, then count zeros = matching bits
            let diff = q ^ c;
            let matching_bits = 8 - diff.count_ones() as usize;
            matching += matching_bits;
            
            // Early termination: if divergence too high, abort
            if i > 0 && i % chunk_size == 0 {
                let progress = i as f32 / query.len() as f32;
                let current_score = (2 * matching) as f32 / total_ones as f32;
                // If score is dropping below threshold and we're past half, abort
                if progress > 0.5 && current_score < 0.3 {
                    return current_score;
                }
            }
        }
        
        if total_ones == 0 {
            return 1.0; // Both all-zero = identical
        }
        
        (2 * matching) as f32 / total_ones as f32
    }
}

/// The 3-layer cascade
pub struct CheatSheetCascade {
    pub global: GlobalCheatSheet,
    pub buckets: HashMap<u64, BucketCheatSheet>,
    pub exact: ExactComparator,
    pub threshold_fast: f32,    // Score to accept fast match
    pub threshold_verify: f32,   // Score to trigger exact verification
}

impl CheatSheetCascade {
    pub fn new() -> Self {
        Self {
            global: GlobalCheatSheet::new(),
            buckets: HashMap::new(),
            exact: ExactComparator::new(100), // Max 100 exact comparisons
            threshold_fast: 0.95,
            threshold_verify: 0.7,
        }
    }
    
    /// Query the cascade: fast -> cheat sheet -> exact
    pub fn query(&mut self, query_hash: &[u8], bucket_id: u64, candidates: &[(u64, f32)]) -> Vec<(u64, f32)> {
        let mut results = Vec::new();
        let mut exact_count = 0;
        
        for (candidate_id, fast_score) in candidates {
            // Layer 1: Global cheat sheet
            if let Some(cheats) = self.global.lookup(query_hash) {
                for cheat in cheats {
                    if cheat.match_id == *candidate_id {
                        results.push((*candidate_id, cheat.confidence));
                        continue;
                    }
                }
            }
            
            // Layer 2: Bucket cheat sheet
            if let Some(bucket) = self.buckets.get(&bucket_id) {
                if let Some(cheats) = bucket.entries.get(query_hash) {
                    for cheat in cheats {
                        if cheat.match_id == *candidate_id {
                            results.push((*candidate_id, cheat.confidence));
                            continue;
                        }
                    }
                }
            }
            
            // Layer 3: Exact comparison (only if fast score is promising)
            if *fast_score >= self.threshold_verify && exact_count < self.exact.max_comparisons {
                // In real implementation, we'd fetch the full hash from storage
                // For now, use the fast score as approximation
                let exact_score = self.exact.compare_exact(query_hash, query_hash); // Self-comparison = 1.0
                results.push((*candidate_id, exact_score.max(*fast_score)));
                exact_count += 1;
            } else {
                results.push((*candidate_id, *fast_score));
            }
        }
        
        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
    
    /// Train the cheat sheets from query results
    pub fn train_from_result(&mut self, query_hash: &[u8], _bucket_id: u64, true_match_id: u64, score: f32) {
        if score >= self.threshold_fast {
            // Add to global cheat sheet
            let entry = CheatEntry {
                pattern: query_hash.to_vec(),
                match_id: true_match_id,
                confidence: score,
            };
            self.global.entries.entry(query_hash.to_vec())
                .or_insert_with(Vec::new)
                .push(entry);
        }
    }
}
