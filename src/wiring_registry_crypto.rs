// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! wiring_registry_crypto.rs — checksums, AES-256-GCM encryption, and HMAC-SHA-256.

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    aead::rand_core::RngCore,
    Aes256Gcm, Nonce,
};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

pub type HmacSha256 = Hmac<Sha256>;

/// Compute SHA-256 of bytes, returned as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Compute SHA-256 of bytes, returned as raw 32 bytes.
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Derive a 32-byte key from a hex string or raw bytes.
/// If `key_material` is exactly 64 hex chars, parse as hex.
/// Otherwise hash the material with SHA-256.
pub fn derive_key(key_material: &str) -> [u8; 32] {
    if key_material.len() == 64 {
        if let Ok(bytes) = hex::decode(key_material) {
            if bytes.len() == 32 {
                let mut out = [0u8; 32];
                out.copy_from_slice(&bytes);
                return out;
            }
        }
    }
    sha256_bytes(key_material.as_bytes())
}

/// Secure backup format:
///   [4 bytes: plaintext JSON length LE]
///   [12 bytes: nonce]
///   [N bytes: AES-256-GCM ciphertext]
///   [32 bytes: HMAC-SHA-256 over nonce || ciphertext]
pub fn encrypt_secure(plaintext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| format!("key init: {e}"))?;

    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| format!("encrypt: {e}"))?;

    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).map_err(|e| format!("hmac init: {e}"))?;
    mac.update(nonce.as_slice());
    mac.update(&ciphertext);
    let tag = mac.finalize().into_bytes();

    let mut out = Vec::new();
    out.extend_from_slice(&(plaintext.len() as u32).to_le_bytes());
    out.extend_from_slice(nonce.as_slice());
    out.extend_from_slice(&ciphertext);
    out.extend_from_slice(&tag);
    Ok(out)
}

pub fn decrypt_secure(ciphertext: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, String> {
    if ciphertext.len() < 4 + 12 + 16 + 32 {
        return Err("secure backup too short".into());
    }
    let plain_len = u32::from_le_bytes([ciphertext[0], ciphertext[1], ciphertext[2], ciphertext[3]]) as usize;
    let nonce = Nonce::from_slice(&ciphertext[4..16]);
    let tag_start = ciphertext.len() - 32;
    let encrypted = &ciphertext[16..tag_start];
    let expected_tag = &ciphertext[tag_start..];

    let mut mac = <HmacSha256 as Mac>::new_from_slice(key).map_err(|e| format!("hmac init: {e}"))?;
    mac.update(nonce.as_slice());
    mac.update(encrypted);
    let actual_tag = mac.finalize().into_bytes();

    if actual_tag.as_slice() != expected_tag {
        return Err("HMAC verification failed".into());
    }

    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| format!("key init: {e}"))?;
    let plaintext = cipher
        .decrypt(nonce, encrypted)
        .map_err(|e| format!("decrypt: {e}"))?;

    if plaintext.len() != plain_len {
        return Err("decrypted length mismatch".into());
    }

    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let key = [1u8; 32];
        let msg = b"registry state";
        let sealed = encrypt_secure(msg, &key).unwrap();
        let opened = decrypt_secure(&sealed, &key).unwrap();
        assert_eq!(opened, msg);
    }

    #[test]
    fn tamper_detected() {
        let key = [1u8; 32];
        let mut sealed = encrypt_secure(b"registry state", &key).unwrap();
        let last = sealed.len() - 1;
        sealed[last] ^= 0xFF;
        assert!(decrypt_secure(&sealed, &key).is_err());
    }

    #[test]
    fn nonce_is_random() {
        let key = [1u8; 32];
        let a = encrypt_secure(b"x", &key).unwrap();
        let b = encrypt_secure(b"x", &key).unwrap();
        // Nonce bytes are at offset 4..16; they should differ with overwhelming probability.
        assert_ne!(&a[4..16], &b[4..16]);
    }
}
