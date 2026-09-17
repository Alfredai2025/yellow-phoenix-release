// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Default, Serialize, Deserialize, Debug, Clone)]
pub struct PredictorMetrics {
    pub hits: u64,
    pub misses: u64,
    pub char3_fallbacks: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TokenPredictor {
    token_index: HashMap<u64, Vec<(u32, u16)>>,
    bigram_index: HashMap<u64, Vec<(u32, u16)>>,
    char3_index: HashMap<u64, Vec<(u32, u16)>>,
    min_confidence: f32,
    min_votes: usize,
    #[serde(default)]
    metrics: PredictorMetrics,
    /// Optional congruence pre-filter gate (v3.6).
    #[cfg(feature = "living_mesh")]
    #[serde(default)]
    congruence_filter_enabled: bool,
}

impl Default for TokenPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenPredictor {
    /// Create a new empty TokenPredictor.
    pub fn new() -> Self {
        Self {
            token_index: HashMap::new(),
            bigram_index: HashMap::new(),
            char3_index: HashMap::new(),
            min_confidence: 0.65,
            min_votes: 2,
            metrics: PredictorMetrics::default(),
            #[cfg(feature = "living_mesh")]
            congruence_filter_enabled: false,
        }
    }

    /// Set the confidence threshold and minimum votes required.
    pub fn set_threshold(&mut self, confidence: f32, votes: usize) {
        self.min_confidence = confidence;
        self.min_votes = votes;
    }

    /// Enable/disable the congruence pre-filter gate.
    #[cfg(feature = "living_mesh")]
    pub fn set_congruence_filter(&mut self, enabled: bool) {
        self.congruence_filter_enabled = enabled;
    }

    /// Apply the congruence pre-filter gate to a predicted bucket.
    /// Returns the bucket only if the arithmetic gate passes.
    #[cfg(feature = "living_mesh")]
    fn apply_congruence_filter(&self, bucket: u32) -> Option<u32> {
        if !self.congruence_filter_enabled {
            return Some(bucket);
        }
        let conf = crate::congruence_filter::bucket_congruence_gate(bucket);
        if conf >= 0.75 {
            Some(bucket)
        } else {
            None
        }
    }

    pub fn metrics(&self) -> &PredictorMetrics {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics = PredictorMetrics::default();
    }

    /// Learn a (query, bucket) pair from search history.
    pub fn learn(&mut self, query: &str, bucket: u32) {
        let lower = query.to_lowercase();
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        for token in &tokens {
            let h = Self::hash_str(token);
            self.token_index.entry(h).or_insert_with(Vec::new).push((bucket, 1));
        }
        for window in tokens.windows(2) {
            let bigram = format!("{} {}", window[0], window[1]);
            let h = Self::hash_str(&bigram);
            self.bigram_index.entry(h).or_insert_with(Vec::new).push((bucket, 1));
        }
    }

    pub fn learn_char3(&mut self, query: &str, bucket: u32) {
        let lower = query.to_lowercase();
        for token in lower.split_whitespace() {
            let chars: Vec<char> = token.chars().collect();
            if chars.len() >= 3 {
                for tri in chars.windows(3) {
                    let tri_str: String = tri.iter().collect();
                    let h = Self::hash_str(&tri_str);
                    self.char3_index.entry(h).or_insert_with(Vec::new).push((bucket, 1));
                }
            }
        }
    }

    /// Predict the best bucket for a query using token voting.
    pub fn predict(&mut self, query: &str) -> Option<(u32, f32)> {
        let lower = query.to_lowercase();
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        let mut votes: HashMap<u32, f32> = HashMap::new();

        // Exact token matches
        for token in &tokens {
            let h = Self::hash_str(token);
            if let Some(candidates) = self.token_index.get(&h) {
                for (bucket, freq) in candidates {
                    *votes.entry(*bucket).or_insert(0.0) += 0.3 * (*freq as f32);
                }
            }
        }

        // Bigram matches
        for window in tokens.windows(2) {
            let bigram = format!("{} {}", window[0], window[1]);
            let h = Self::hash_str(&bigram);
            if let Some(candidates) = self.bigram_index.get(&h) {
                for (bucket, freq) in candidates {
                    *votes.entry(*bucket).or_insert(0.0) += 0.7 * (*freq as f32);
                }
            }
        }

        // Char3 fallback for tokens that had NO exact match
        let mut used_char3 = false;
        for token in &tokens {
            let h = Self::hash_str(token);
            let token_missed = !self.token_index.contains_key(&h);
            if token_missed {
                let chars: Vec<char> = token.chars().collect();
                if chars.len() >= 3 {
                    for tri in chars.windows(3) {
                        let tri_str: String = tri.iter().collect();
                        let h3 = Self::hash_str(&tri_str);
                        if let Some(candidates) = self.char3_index.get(&h3) {
                            used_char3 = true;
                            for (bucket, freq) in candidates {
                                *votes.entry(*bucket).or_insert(0.0) += 0.4 * (*freq as f32);
                            }
                        }
                    }
                }
            }
        }

        if votes.len() < self.min_votes {
            self.metrics.misses += 1;
            return None;
        }

        let total: f32 = votes.values().sum();
        let (&best_bucket, &best_score) = votes.iter().max_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        })?;

        let confidence = best_score / total;
        if confidence >= self.min_confidence {
            #[cfg(feature = "living_mesh")]
            {
                if self.apply_congruence_filter(best_bucket).is_none() {
                    self.metrics.misses += 1;
                    return None;
                }
            }
            self.metrics.hits += 1;
            if used_char3 {
                self.metrics.char3_fallbacks += 1;
            }
            Some((best_bucket, confidence))
        } else {
            self.metrics.misses += 1;
            None
        }
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let predictor = serde_json::from_str(&json)?;
        Ok(predictor)
    }

    fn hash_str(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let mut p = TokenPredictor::new();
        p.set_threshold(0.65, 1);
        p.learn("neural network gradient", 47831);
        let result = p.predict("neural network gradient");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 47831);
    }

    #[test]
    fn test_char3_fallback() {
        let mut p = TokenPredictor::new();
        p.set_threshold(0.65, 1);
        p.learn("transformers architecture", 12345);
        p.learn_char3("transformers architecture", 12345);
        let result = p.predict("transformer architecture");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 12345);
    }

    #[test]
    fn test_partial_char3_for_missed_token() {
        let mut p = TokenPredictor::new();
        p.set_threshold(0.65, 1);
        p.learn("neural network", 40000);
        p.learn_char3("neural network", 40000);
        p.learn("transformers architecture", 50000);
        p.learn_char3("transformers architecture", 50000);
        // "neural" is known, "transformer" is not — char3 should rescue it
        let result = p.predict("neural transformer");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 50000);
    }

    #[test]
    fn test_confidence_rejection() {
        let mut p = TokenPredictor::new();
        p.set_threshold(0.99, 1);
        p.learn("alpha beta", 1);
        p.learn("gamma delta", 2);
        assert!(p.predict("epsilon zeta").is_none());
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut p = TokenPredictor::new();
        p.set_threshold(0.65, 1);
        p.learn("neural network", 40000);
        p.learn_char3("neural network", 40000);
        let tmp = "/tmp/predictor_test_state.json";
        p.save_to_file(tmp).unwrap();

        let mut loaded = TokenPredictor::load_from_file(tmp).unwrap();
        let result = loaded.predict("neural network");
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, 40000);
        std::fs::remove_file(tmp).unwrap();
    }
}
