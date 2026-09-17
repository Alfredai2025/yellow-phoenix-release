// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! SAH Stage 2: Hash Bridge — Eigenvector → 512-bit Beacon Hash
//!
//! Applies ITQ rotation + binarization to convert floating-point
//! eigenvectors into BinaryHNSW-compatible Hash512 values.

use crate::binary_hnsw::Hash512;

/// Convert a single eigenvector to a 512-bit hash.
///
/// `vector`  — length = dim
/// `rotation` — flattened row-major dim×dim ITQ rotation matrix
/// `dim` — must be 512 for current ITQ model
pub fn eigenvector_to_hash512(
    vector: &[f32],
    rotation: &[f32],
    dim: usize,
) -> Hash512 {
    assert_eq!(vector.len(), dim);
    assert_eq!(rotation.len(), dim * dim);

    // rotated = vector @ rotation_matrix
    let mut rotated = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += vector[j] * rotation[j * dim + i];
        }
        rotated[i] = sum;
    }

    // Binarize: positive → 1, negative/zero → 0
    // Bit 0 → MSB of byte 0
    let mut hash = [0u8; 64];
    for i in 0..dim {
        if rotated[i] > 0.0 {
            hash[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    hash
}

/// Batch convert many eigenvectors.
///
/// `vectors` — flattened `count × dim` row-major
/// `out_hashes` — length = count
pub fn batch_eigenvector_to_hash512(
    vectors: &[f32],
    count: usize,
    dim: usize,
    rotation: &[f32],
    out_hashes: &mut [Hash512],
) {
    assert_eq!(vectors.len(), count * dim);
    assert_eq!(rotation.len(), dim * dim);
    assert_eq!(out_hashes.len(), count);

    for b in 0..count {
        let vec = &vectors[b * dim..(b + 1) * dim];
        out_hashes[b] = eigenvector_to_hash512(vec, rotation, dim);
    }
}
