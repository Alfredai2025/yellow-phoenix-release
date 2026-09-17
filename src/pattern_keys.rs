// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Mathematical pattern-based holographic keys.
//!
//! Generates bipolar keys from embeddings using:
//!   - Power-of-2 sampling (exponential structure)
//!   - Prime sampling (irregular, breaks periodicity)
//!   - Fibonacci sampling (golden ratio, natural growth)
//!   - Cube sampling (non-linear curvature)
//!
//! Each key is 512-dim bipolar. Similar embeddings → similar keys by construction.

use alloc::vec::Vec;

/// Generate 512-dim bipolar key by sampling embedding at mathematical positions.
/// embedding: 384-dim float32 (normalized)
/// pattern: function that maps index → sample position in embedding
pub fn pattern_key(embedding: &[f32], pattern: &[usize]) -> Vec<f32> {
    let mut key = Vec::with_capacity(512);
    let emb_len = embedding.len();
    for i in 0..512 {
        let pos = pattern[i % pattern.len()] % emb_len;
        // Threshold at 0: positive → +1.0, negative → -1.0
        key.push(if embedding[pos] >= 0.0 { 1.0f32 } else { -1.0f32 });
    }
    key
}

/// Power-of-2 positions: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512...
/// Repeats with wrap for 512 samples.
pub fn power2_pattern() -> Vec<usize> {
    let mut p = Vec::with_capacity(512);
    let mut i = 0;
    while p.len() < 512 {
        let pos = 1usize << (i % 9); // 2^0 to 2^8 = 1 to 256
        p.push(pos);
        i += 1;
    }
    p
}

/// Prime positions: first 512 primes, modulo embedding length.
pub fn prime_pattern() -> Vec<usize> {
    let mut primes = Vec::with_capacity(512);
    let mut n = 2usize;
    while primes.len() < 512 {
        let mut is_prime = true;
        for d in 2..=((n as f64).sqrt() as usize) {
            if n % d == 0 { is_prime = false; break; }
        }
        if is_prime { primes.push(n); }
        n += 1;
    }
    primes
}

/// Fibonacci positions: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377...
pub fn fibonacci_pattern() -> Vec<usize> {
    let mut fib = Vec::with_capacity(512);
    let mut a = 1usize;
    let mut b = 2usize;
    while fib.len() < 512 {
        fib.push(a);
        let next = a + b;
        a = b;
        b = next;
    }
    fib
}

/// Cube positions: 1, 8, 27, 64, 125, 216, 343, 512, 729...
pub fn cube_pattern() -> Vec<usize> {
    let mut cubes = Vec::with_capacity(512);
    let mut i = 1usize;
    while cubes.len() < 512 {
        cubes.push(i * i * i);
        i += 1;
    }
    cubes
}

/// All 4 patterns pre-computed.
pub struct PatternBank {
    pub power2: Vec<usize>,
    pub primes: Vec<usize>,
    pub fibonacci: Vec<usize>,
    pub cubes: Vec<usize>,
}

impl PatternBank {
    pub fn new() -> Self {
        Self {
            power2: power2_pattern(),
            primes: prime_pattern(),
            fibonacci: fibonacci_pattern(),
            cubes: cube_pattern(),
        }
    }

    pub fn generate_keys(&self, embedding: &[f32]) -> [Vec<f32>; 4] {
        [
            pattern_key(embedding, &self.power2),
            pattern_key(embedding, &self.primes),
            pattern_key(embedding, &self.fibonacci),
            pattern_key(embedding, &self.cubes),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power2_pattern() {
        let p = power2_pattern();
        assert_eq!(p[0], 1);
        assert_eq!(p[1], 2);
        assert_eq!(p[2], 4);
        assert_eq!(p[8], 256);
        assert_eq!(p[9], 1); // wraps
        assert_eq!(p.len(), 512);
    }

    #[test]
    fn test_prime_pattern() {
        let p = prime_pattern();
        assert_eq!(p[0], 2);
        assert_eq!(p[1], 3);
        assert_eq!(p[2], 5);
        assert_eq!(p[99], 541); // 100th prime
        assert_eq!(p.len(), 512);
    }

    #[test]
    fn test_fibonacci_pattern() {
        let p = fibonacci_pattern();
        assert_eq!(p[0], 1);
        assert_eq!(p[1], 2);
        assert_eq!(p[2], 3);
        assert_eq!(p[3], 5);
        assert_eq!(p[10], 144);
        assert_eq!(p.len(), 512);
    }

    #[test]
    fn test_similar_embeddings_similar_keys() {
        let bank = PatternBank::new();
        let emb_a = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.9, 0.4, -0.6];
        let emb_b = vec![0.48f32, -0.28, 0.79, -0.12, 0.21, -0.88, 0.41, -0.59]; // very close
        let keys_a = bank.generate_keys(&emb_a);
        let keys_b = bank.generate_keys(&emb_b);
        // At least 3 of 4 channels should have high agreement
        let mut agreements = 0;
        for i in 0..4 {
            let agree: usize = keys_a[i].iter().zip(keys_b[i].iter())
                .map(|(a, b)| if (*a > 0.0) == (*b > 0.0) { 1 } else { 0 })
                .sum();
            if agree > 400 { agreements += 1; } // >78% agreement
        }
        assert!(agreements >= 3, "Expected >=3 channels to agree, got {}", agreements);
    }
}
