// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! learned_router.rs — tiny MLP that decides which feature chain to run.
//!
//! Architecture: 8 inputs → 4 hidden → 3 outputs
//! Parameters: 32 (w1) + 4 (b1) + 12 (w2) + 3 (b2) = 51
//! Fallback: if any output is in (0.4, 0.6), run that feature anyway (safe mode).

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

const INPUT_DIM: usize = 8;
const HIDDEN_DIM: usize = 4;
const OUTPUT_DIM: usize = 3;

pub const UNCERTAINTY_LOW: f32 = 0.4;
pub const UNCERTAINTY_HIGH: f32 = 0.6;
pub const LEARNING_RATE: f32 = 0.01;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FeatureChain {
    pub run_spectral: bool,
    pub run_wedge: bool,
    pub run_hologram: bool,
}

impl FeatureChain {
    pub fn all() -> Self {
        Self {
            run_spectral: true,
            run_wedge: true,
            run_hologram: true,
        }
    }

    pub fn none() -> Self {
        Self {
            run_spectral: false,
            run_wedge: false,
            run_hologram: false,
        }
    }

    /// Convert chain to a 3-output training target [spectral, wedge, hologram].
    pub fn to_target(&self) -> [f32; OUTPUT_DIM] {
        [
            if self.run_spectral { 1.0 } else { 0.0 },
            if self.run_wedge { 1.0 } else { 0.0 },
            if self.run_hologram { 1.0 } else { 0.0 },
        ]
    }
}

#[derive(Clone, Debug)]
pub struct LearnedRouter {
    pub w1: [[f32; INPUT_DIM]; HIDDEN_DIM],
    pub b1: [f32; HIDDEN_DIM],
    pub w2: [[f32; HIDDEN_DIM]; OUTPUT_DIM],
    pub b2: [f32; OUTPUT_DIM],
    pub train_count: u64,
}

impl Default for LearnedRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl LearnedRouter {
    /// Create a router with small random weights.
    pub fn new() -> Self {
        let mut router = Self {
            w1: [[0.0; INPUT_DIM]; HIDDEN_DIM],
            b1: [0.0; HIDDEN_DIM],
            w2: [[0.0; HIDDEN_DIM]; OUTPUT_DIM],
            b2: [0.0; OUTPUT_DIM],
            train_count: 0,
        };
        router.reset_weights();
        router
    }

    /// Force the router to emit low confidence and run all feature stages.
    /// Useful for M3 benchmarks that need an "M1-style" baseline.
    pub fn force_baseline(&mut self) {
        self.b2 = [-0.2f32; OUTPUT_DIM];
    }

    /// Force the router to emit high confidence and try the fast hash path.
    pub fn force_fast_path(&mut self) {
        self.b2 = [10.0f32; OUTPUT_DIM];
    }

    /// Reset weights to small random values (±0.1).
    pub fn reset_weights(&mut self) {
        let seed = 0x9e3779b97f4a7c15u64;
        for i in 0..HIDDEN_DIM {
            for j in 0..INPUT_DIM {
                self.w1[i][j] = small_random(seed.wrapping_add((i * INPUT_DIM + j) as u64));
            }
            self.b1[i] = small_random(seed.wrapping_add(1000 + i as u64));
        }
        for i in 0..OUTPUT_DIM {
            for j in 0..HIDDEN_DIM {
                self.w2[i][j] = small_random(seed.wrapping_add(2000 + (i * HIDDEN_DIM + j) as u64));
            }
            self.b2[i] = small_random(seed.wrapping_add(3000 + i as u64));
        }
    }

    /// Forward pass. Returns probabilities for [spectral, wedge, hologram].
    pub fn predict(&self, features: [f32; INPUT_DIM]) -> [f32; OUTPUT_DIM] {
        // Hidden layer with tanh activation.
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut sum = self.b1[i];
            for j in 0..INPUT_DIM {
                sum += self.w1[i][j] * features[j];
            }
            hidden[i] = tanh(sum);
        }

        // Output layer with sigmoid.
        let mut output = [0.0f32; OUTPUT_DIM];
        for i in 0..OUTPUT_DIM {
            let mut sum = self.b2[i];
            for j in 0..HIDDEN_DIM {
                sum += self.w2[i][j] * hidden[j];
            }
            output[i] = sigmoid(sum);
        }
        output
    }

    /// Confidence score: max output probability. Higher = more certain.
    pub fn confidence(&self, features: [f32; INPUT_DIM]) -> f32 {
        let probs = self.predict(features);
        probs.iter().fold(0.0f32, |a, &b| a.max(b))
    }

    /// Decide which features to run.
    /// Safe fallback: if probability is uncertain (0.4–0.6), run the feature anyway.
    pub fn decide_chain(&self, features: [f32; INPUT_DIM]) -> FeatureChain {
        let probs = self.predict(features);
        FeatureChain {
            run_spectral: probs[0] > 0.5 || (probs[0] > UNCERTAINTY_LOW && probs[0] < UNCERTAINTY_HIGH),
            run_wedge: probs[1] > 0.5 || (probs[1] > UNCERTAINTY_LOW && probs[1] < UNCERTAINTY_HIGH),
            run_hologram: probs[2] > 0.5 || (probs[2] > UNCERTAINTY_LOW && probs[2] < UNCERTAINTY_HIGH),
        }
    }

    /// Online SGD update. target is 1.0 if feature was useful, 0.0 otherwise.
    pub fn train(&mut self, features: [f32; INPUT_DIM], targets: [f32; OUTPUT_DIM]) {
        self.train_step(features, targets, LEARNING_RATE);
    }

    pub fn train_step(&mut self, features: [f32; INPUT_DIM], targets: [f32; OUTPUT_DIM], lr: f32) {
        // Forward pass, keeping intermediates.
        let mut hidden = [0.0f32; HIDDEN_DIM];
        let mut hidden_deriv = [0.0f32; HIDDEN_DIM];
        for i in 0..HIDDEN_DIM {
            let mut sum = self.b1[i];
            for j in 0..INPUT_DIM {
                sum += self.w1[i][j] * features[j];
            }
            hidden[i] = tanh(sum);
            hidden_deriv[i] = 1.0 - hidden[i] * hidden[i];
        }

        let probs = self.predict(features);

        // Output layer gradients.
        let mut output_error = [0.0f32; OUTPUT_DIM];
        for i in 0..OUTPUT_DIM {
            output_error[i] = (probs[i] - targets[i]) * probs[i] * (1.0 - probs[i]);
        }

        // Hidden layer gradients.
        let mut hidden_error = [0.0f32; HIDDEN_DIM];
        for j in 0..HIDDEN_DIM {
            let mut sum = 0.0;
            for i in 0..OUTPUT_DIM {
                sum += output_error[i] * self.w2[i][j];
            }
            hidden_error[j] = sum * hidden_deriv[j];
        }

        // Update weights.
        for i in 0..OUTPUT_DIM {
            for j in 0..HIDDEN_DIM {
                self.w2[i][j] -= lr * output_error[i] * hidden[j];
            }
            self.b2[i] -= lr * output_error[i];
        }
        for i in 0..HIDDEN_DIM {
            for j in 0..INPUT_DIM {
                self.w1[i][j] -= lr * hidden_error[i] * features[j];
            }
            self.b1[i] -= lr * hidden_error[i];
        }

        self.train_count += 1;
    }

    /// Save router to binary file (51 f32 values = 204 bytes).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let bytes = self.to_bytes();
        let tmp = path.as_ref().with_extension("tmp");
        let mut file = fs::File::create(&tmp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Load router from binary file.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(51 * 4);
        for i in 0..HIDDEN_DIM {
            for j in 0..INPUT_DIM {
                bytes.extend_from_slice(&self.w1[i][j].to_le_bytes());
            }
        }
        for i in 0..HIDDEN_DIM {
            bytes.extend_from_slice(&self.b1[i].to_le_bytes());
        }
        for i in 0..OUTPUT_DIM {
            for j in 0..HIDDEN_DIM {
                bytes.extend_from_slice(&self.w2[i][j].to_le_bytes());
            }
        }
        for i in 0..OUTPUT_DIM {
            bytes.extend_from_slice(&self.b2[i].to_le_bytes());
        }
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() != 51 * 4 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "router model wrong size"));
        }
        let mut router = Self::new();
        let mut idx = 0;
        for i in 0..HIDDEN_DIM {
            for j in 0..INPUT_DIM {
                router.w1[i][j] = f32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
                idx += 4;
            }
        }
        for i in 0..HIDDEN_DIM {
            router.b1[i] = f32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
            idx += 4;
        }
        for i in 0..OUTPUT_DIM {
            for j in 0..HIDDEN_DIM {
                router.w2[i][j] = f32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
                idx += 4;
            }
        }
        for i in 0..OUTPUT_DIM {
            router.b2[i] = f32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
            idx += 4;
        }
        Ok(router)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

fn small_random(seed: u64) -> f32 {
    // xorshift64* in [-0.1, 0.1]
    let mut x = seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let raw = x.wrapping_mul(0x2545f4914f6cdd1d);
    ((raw as f64) / u64::MAX as f64) as f32 * 0.2 - 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_count_is_51() {
        let router = LearnedRouter::new();
        let bytes = router.to_bytes();
        assert_eq!(bytes.len(), 51 * 4);
    }

    #[test]
    fn predict_in_range() {
        let router = LearnedRouter::new();
        let features = [0.5; INPUT_DIM];
        let probs = router.predict(features);
        for p in probs {
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn decide_chain_safe_fallback() {
        // Manually set weights so all outputs are exactly 0.45 (uncertain region).
        let mut router = LearnedRouter::new();
        router.b2 = [0.0; OUTPUT_DIM]; // sigmoid(0) = 0.5, not uncertain. Need -0.2 for 0.45.
        router.b2 = [-0.2; OUTPUT_DIM];
        let chain = router.decide_chain([0.0; INPUT_DIM]);
        assert!(chain.run_spectral);
        assert!(chain.run_wedge);
        assert!(chain.run_hologram);
    }

    #[test]
    fn training_changes_predictions() {
        let mut router = LearnedRouter::new();
        let features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let before = router.predict(features);
        for _ in 0..100 {
            router.train(features, [1.0, 0.0, 1.0]);
        }
        let after = router.predict(features);
        assert!((before[0] - after[0]).abs() > 1e-3);
        assert!((before[1] - after[1]).abs() > 1e-3);
    }

    #[test]
    fn save_load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("router_model_v1.bin");
        let mut router = LearnedRouter::new();
        router.train([0.5; INPUT_DIM], [1.0, 0.0, 1.0]);
        router.save(&path).unwrap();
        let loaded = LearnedRouter::load(&path).unwrap();
        assert_eq!(router.w1, loaded.w1);
        assert_eq!(router.b1, loaded.b1);
        assert_eq!(router.w2, loaded.w2);
        assert_eq!(router.b2, loaded.b2);
    }
}
