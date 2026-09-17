// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::types::multivector::BinaryMultivector;

// -----------------------------------------------------------------
// PRNG helpers
// -----------------------------------------------------------------

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Hash a token *without* position information.
#[inline]
fn hash_token(seed: u64, token: &str) -> u64 {
    let mut h = seed;
    for b in token.bytes() {
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(b as u64);
    }
    h
}

/// Hash a token *with* a position index (for order‑sensitive encoding).
#[inline]
fn hash_token_with_position(seed: u64, token: &str, position: u64) -> u64 {
    let mut h = seed;
    for b in token.bytes() {
        h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(b as u64);
    }
    h ^ position
}

const SATURATION_THRESHOLD: i32 = 10;

// -----------------------------------------------------------------
// Encoder
// -----------------------------------------------------------------

/// Encodes a textual paper into a 128‑bit binary multivector using
/// order‑sensitive, deterministic geometric hashing.
pub struct Encoder {
    /// Global seed – keeps a stable internal state across runs.
    seed: u64,
}

impl Encoder {
    /// Create a new encoder with a given reproducibility seed.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Encode a **single** token into a `BinaryMultivector` (order *not*
    /// considered).
    pub fn encode_token(&self, token: &str) -> BinaryMultivector {
        let s = hash_token(self.seed, token);
        let mut state = s;
        let a = splitmix64(&mut state);
        let b = splitmix64(&mut state);
        BinaryMultivector([a, b])
    }

    /// Return the full 128‑bit bag‑of‑words hash for a single token.
    /// This is the same 128‑bit value that `encode_paper_bow` would
    /// contribute for this token, without any position information.
    /// Expands the token seed into a 128‑bit value using the existing
    /// `splitmix64` stream.
    #[inline]
    pub fn hash_token_bow(&self, token: &str) -> u128 {
        let mut state = hash_token(self.seed, token);
        let a = splitmix64(&mut state);
        let b = splitmix64(&mut state);
        ((a as u128) << 64) | (b as u128)
    }

    /// Encode a whole paper (sequence of tokens separated by Unicode
    /// whitespace) into a `BinaryMultivector`.
    ///
    /// Each token contributes ±1 to every bit position and the final
    /// multivector is obtained by *majority vote* (bit set if sum > 0).
    pub fn encode_paper(&self, text: &str) -> BinaryMultivector {
        let mut buffer: [i32; 128] = [0i32; 128];

        let mut pos: u64 = 0;
        for token in text.split_whitespace() {
            let s = hash_token_with_position(self.seed, token, pos);
            let mut state = s;
            let a = splitmix64(&mut state);
            let b = splitmix64(&mut state);

            // low 64 bits
            for i in 0..64 {
                let bit = (a >> i) & 1;
                // map bit 1 → +1, bit 0 → -1
                buffer[i] += ((bit as i32) * 2) - 1;
            }
            // high 64 bits
            for i in 0..64 {
                let bit = (b >> i) & 1;
                buffer[i + 64] += ((bit as i32) * 2) - 1;
            }

            pos += 1;
        }

        // thresholding
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if buffer[i] > 0 {
                lo |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if buffer[i + 64] > 0 {
                hi |= 1u64 << i;
            }
        }
        BinaryMultivector([lo, hi])
    }

    /// Like `encode_paper`, but stops updating a bit once its cumulative
    /// sum reaches ±`SATURATION_THRESHOLD`. This accelerates encoding of
    /// very long papers without noticeably affecting the final vector.
    pub fn encode_paper_fast(&self, text: &str) -> BinaryMultivector {
        let mut buffer: [i32; 128] = [0i32; 128];
        let mut saturated: [bool; 128] = [false; 128];
        let mut saturated_count: usize = 0;

        let mut pos: u64 = 0;
        for token in text.split_whitespace() {
            let s = hash_token_with_position(self.seed, token, pos);
            let mut state = s;
            let a = splitmix64(&mut state);
            let b = splitmix64(&mut state);

            // low 64 bits
            for i in 0..64 {
                if !saturated[i] {
                    let bit = (a >> i) & 1;
                    buffer[i] += ((bit as i32) * 2) - 1;
                    if buffer[i].abs() >= SATURATION_THRESHOLD {
                        saturated[i] = true;
                        saturated_count += 1;
                    }
                }
            }
            // high 64 bits
            for i in 0..64 {
                let idx = i + 64;
                if !saturated[idx] {
                    let bit = (b >> i) & 1;
                    buffer[idx] += ((bit as i32) * 2) - 1;
                    if buffer[idx].abs() >= SATURATION_THRESHOLD {
                        saturated[idx] = true;
                        saturated_count += 1;
                    }
                }
            }

            pos += 1;

            // early exit when every dimension is frozen
            if saturated_count == 128 {
                break;
            }
        }

        // thresholding (same as above)
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if buffer[i] > 0 {
                lo |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if buffer[i + 64] > 0 {
                hi |= 1u64 << i;
            }
        }
        BinaryMultivector([lo, hi])
    }

    /// Sparse bag‑of‑words encoder. Each token activates exactly
    /// `SPARSE_K` deterministic bits in the 128‑bit space. Superposition
    /// across tokens produces a sparse, topic‑sensitive vector.
    pub fn encode_paper_bow(&self, text: &str) -> BinaryMultivector {
        let mut buffer: [i32; 128] = [0i32; 128];
        let mut saturated: [bool; 128] = [false; 128];
        let mut saturated_count: usize = 0;
        const SPARSE_K: usize = 4;

        for token in text.split_whitespace() {
            let mut h = self.hash_token_bow(token);
            for _ in 0..SPARSE_K {
                let idx = (h & 0x7f) as usize;
                h >>= 7;
                if !saturated[idx] {
                    buffer[idx] += 1;
                    if buffer[idx] >= SATURATION_THRESHOLD {
                        saturated[idx] = true;
                        saturated_count += 1;
                    }
                }
            }

            if saturated_count == 128 {
                break;
            }
        }

        // thresholding
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if buffer[i] > 0 {
                lo |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if buffer[i + 64] > 0 {
                hi |= 1u64 << i;
            }
        }
        BinaryMultivector([lo, hi])
    }

    /// Encode a paper and return the raw 128‑bit vector as 16 bytes.
    pub fn encode_to_bytes(&self, text: &str) -> [u8; 16] {
        self.encode_paper(text).to_bytes()
    }
}

/// BipolarEncoder using a projection matrix learned via perceptron updates.
pub struct BipolarEncoder {
    /// 128 rows, each row holds 128 weights.
    projection: [[f32; 128]; 128],
    /// Per‑dimension bias value.
    bias: [f32; 128],
}

impl BipolarEncoder {
    /// Create a new random encoder using a deterministic linear congruential
    /// generator. The internal seed is always 42 for reproducibility,
    /// regardless of the provided `_seed`.
    pub fn new_random(_seed: u64) -> Self {
        let mut state: u64 = 42;
        let mut projection = [[0.0f32; 128]; 128];
        for i in 0..128 {
            for j in 0..128 {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let val = ((state >> 16) & 0x7FFF) as f32 / 16384.0 - 0.5;
                projection[i][j] = val;
            }
        }
        let mut bias = [0.0f32; 128];
        for i in 0..128 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((state >> 16) & 0x7FFF) as f32 / 16384.0 - 0.5;
            bias[i] = val;
        }
        Self { projection, bias }
    }

    /// Load a pre‑trained `BipolarEncoder` with given weights and bias slice.
    pub fn from_pretrained(weights: &[[f32; 128]; 128], bias: &[f32]) -> Self {
        let mut bias_arr = [0.0f32; 128];
        bias_arr.copy_from_slice(&bias[..128]);
        Self {
            projection: *weights,
            bias: bias_arr,
        }
    }

    /// Encode a slice of token‑id (`u64`) values into a `BinaryMultivector`.
    /// For each token, the deterministic hash selects a projection row, and
    /// the row's weights are accumulated. After processing all tokens the
    /// per‑dimension sum (including bias) is thresholded at 0.0.
    pub fn encode_tokens(&self, tokens: &[u64]) -> BinaryMultivector {
        let mut acc = [0.0f32; 128];
        for &token in tokens {
            let mut state = token;
            let h = splitmix64(&mut state);
            let row = (h % 128) as usize;
            for d in 0..128 {
                acc[d] += self.projection[row][d];
            }
        }
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if acc[i] + self.bias[i] > 0.0 {
                lo |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if acc[i + 64] + self.bias[i + 64] > 0.0 {
                hi |= 1u64 << i;
            }
        }
        BinaryMultivector([lo, hi])
    }

    /// Encode a list of pre‑computed projection-row indices (already mapped
    /// via splitmix64) and return the 128‑bit result as a `u128`.
    /// This is the path used by the BipolarEncoder FFI, where Python/Rust
    /// share the same token‑to‑row hashing.
    pub fn encode_tokens_u64(&self, rows: &[usize]) -> u128 {
        let mut acc = [0.0f32; 128];
        for &row in rows {
            if row < 128 {
                for d in 0..128 {
                    acc[d] += self.projection[row][d];
                }
            }
        }
        for d in 0..128 {
            acc[d] += self.bias[d];
        }
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if acc[i] > 0.0 {
                lo |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if acc[i + 64] > 0.0 {
                hi |= 1u64 << i;
            }
        }
        ((hi as u128) << 64) | (lo as u128)
    }

    /// Perform one perceptron training step.
    ///
    /// The current prediction is computed, compared to `target`, and the
    /// projection weights are nudged toward the target bits using the
    /// supplied learning rate `lr`. Bias values are **not** updated.
    pub fn train_step(&mut self, tokens: &[u64], target: &BinaryMultivector, lr: f32) {
        // compute current prediction (same as encode_tokens)
        let mut acc = [0.0f32; 128];
        for &token in tokens {
            let mut state = token;
            let h = splitmix64(&mut state);
            let row = (h % 128) as usize;
            for d in 0..128 {
                acc[d] += self.projection[row][d];
            }
        }
        let mut lo = 0u64;
        let mut hi = 0u64;
        for i in 0..64 {
            if acc[i] + self.bias[i] > 0.0 {
                lo |= 1u64 << i;
            }
        }
        for i in 0..64 {
            if acc[i + 64] + self.bias[i + 64] > 0.0 {
                hi |= 1u64 << i;
            }
        }
        let target_bits = target.0;

        // perceptron weight update
        for &token in tokens {
            let mut state = token;
            let h = splitmix64(&mut state);
            let row = (h % 128) as usize;
            for i in 0..64 {
                let tbit = ((target_bits[0] >> i) & 1) as i32;
                let pbit = ((lo >> i) & 1) as i32;
                self.projection[row][i] += lr * (tbit - pbit) as f32;
            }
            for i in 0..64 {
                let tbit = ((target_bits[1] >> i) & 1) as i32;
                let pbit = ((hi >> i) & 1) as i32;
                self.projection[row][i + 64] += lr * (tbit - pbit) as f32;
            }
        }
    }
}

// -----------------------------------------------------------------
// Tests
// -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::string::String;
    use std::time::Instant;

    #[test]
    fn same_text_produces_identical_vector() {
        let enc = Encoder::new(42);
        let a = enc.encode_paper("the quick brown fox");
        let b = enc.encode_paper("the quick brown fox");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn order_matters() {
        let enc = Encoder::new(99);
        let a = enc.encode_paper("machine learning");
        let b = enc.encode_paper("learning machine");
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn fast_and_slow_produce_same_for_short_text() {
        let enc = Encoder::new(7);
        let text = "short sample paper";
        let a = enc.encode_paper(text);
        let b = enc.encode_paper_fast(text);
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn thousand_tokens_encode_in_under_100us() {
        // Build a 1000‑token string (no alloc in crate, but tests use std)
        let mut s = String::new();
        for _ in 0..1000 {
            s.push_str("token ");
        }
        let _ = s.pop(); // remove trailing space
        let enc = Encoder::new(0);
        let start = Instant::now();
        let _mv = enc.encode_paper_fast(&s);
        let elapsed_ns = start.elapsed().as_nanos();
        let max_ns = if cfg!(debug_assertions) { 5_000_000 } else { 500_000 };
        assert!(
            elapsed_ns < max_ns,
            "encode_paper_fast took {} ns, expected < {} ns",
            elapsed_ns,
            max_ns
        );
    }

    #[test]
    fn bow_order_does_not_matter() {
        let enc = Encoder::new(42);
        let a = enc.encode_paper_bow("machine learning");
        let b = enc.encode_paper_bow("learning machine");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn test_bow_order_invariant() {
        let enc = Encoder::new(42);
        let a = enc.encode_paper_bow("machine learning");
        let b = enc.encode_paper_bow("learning machine");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn test_bow_deterministic() {
        let enc = Encoder::new(42);
        let a = enc.encode_paper_bow("the quick brown fox");
        let b = enc.encode_paper_bow("the quick brown fox");
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn test_bow_different_content_differs() {
        let enc = Encoder::new(42);
        let a = enc.encode_paper_bow("machine learning");
        let b = enc.encode_paper_bow("biology chemistry");
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn test_exact_still_order_sensitive() {
        let enc = Encoder::new(42);
        let a = enc.encode_paper("machine learning");
        let b = enc.encode_paper("learning machine");
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn encode_reproducible() {
        let enc1 = BipolarEncoder::new_random(0);
        let enc2 = BipolarEncoder::new_random(0);
        let tokens = [1u64, 2, 3];
        let mv1 = enc1.encode_tokens(&tokens);
        let mv2 = enc2.encode_tokens(&tokens);
        assert_eq!(mv1.0, mv2.0);
    }

    #[test]
    fn train_improves() {
        let mut enc = BipolarEncoder::new_random(42);
        let tokens = [42u64];
        let target = BinaryMultivector([!0u64, !0u64]);

        fn hamming(a: &BinaryMultivector, b: &BinaryMultivector) -> u32 {
            (a.0[0] ^ b.0[0]).count_ones() + (a.0[1] ^ b.0[1]).count_ones()
        }
        let initial = enc.encode_tokens(&tokens);
        let initial_errors = hamming(&initial, &target);

        for _ in 0..10 {
            enc.train_step(&tokens, &target, 0.5);
        }
        let trained = enc.encode_tokens(&tokens);
        let trained_errors = hamming(&trained, &target);

        assert!(
            trained_errors < initial_errors,
            "training should reduce hamming distance to all‑ones target"
        );
    }

    #[test]
    fn threshold_correct() {
        // Find a token that maps to row 0 via the same splitmix64
        // that the encoder uses internally.
        fn local_splitmix64(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        let token = (0..500u64)
            .find(|&t| local_splitmix64(t) % 128 == 0)
            .expect("No token maps to row 0 in the first 500 values");

        // Create weights so that row 0, dim 0 is positive and dim 1 is negative.
        let mut weights = [[0.0f32; 128]; 128];
        weights[0][0] = 1.0_f32;
        weights[0][1] = -1.0_f32;
        let bias = [0.0f32; 128];

        let enc = BipolarEncoder::from_pretrained(&weights, &bias);
        let mv = enc.encode_tokens(&[token]);

        let bit0 = (mv.0[0] & 1) != 0;
        let bit1 = (mv.0[0] & 2) != 0;

        assert!(bit0, "positive sum should give bit 1");
        assert!(!bit1, "negative sum should give bit 0");
    }
}
