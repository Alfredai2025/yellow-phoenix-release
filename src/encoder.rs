// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer


    /// Encode a list of token-row indices and return the 128-bit result as a u128.
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
            if acc[i] > 0.0 { lo |= 1u64 << i; }
        }
        for i in 0..64 {
            if acc[i + 64] > 0.0 { hi |= 1u64 << i; }
        }
        ((hi as u128) << 64) | (lo as u128)
    }
