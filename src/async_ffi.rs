// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! async_ffi.rs — Milestone 2 Component 1.
//!
//! Non-blocking token/poll FFI. Python submits a query, receives a token,
//! and polls later for the result. No thread is blocked waiting for Rust.

use crate::learned_router::LearnedRouter;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

const TOKEN_POOL_SIZE: usize = 10_000;
const TOKEN_EXPIRY_MS: u64 = 30_000;

#[derive(Clone, Debug)]
enum TokenState {
    Pending { query: Vec<u8>, submitted_at: u64 },
    Ready { result: Vec<u8>, completed_at: u64 },
    Expired,
}

pub struct AsyncTokenPool {
    tokens: HashMap<u64, TokenState>,
    next_id: u64,
}

impl AsyncTokenPool {
    pub fn new() -> Self {
        Self {
            tokens: HashMap::with_capacity(TOKEN_POOL_SIZE),
            next_id: 1,
        }
    }

    /// Submit a query, return token ID.
    pub fn submit(&mut self, query: Vec<u8>) -> u64 {
        let token = self.next_id;
        self.next_id += 1;
        self.tokens.insert(
            token,
            TokenState::Pending {
                query,
                submitted_at: now_ms(),
            },
        );
        token
    }

    /// Poll for result. Returns `Some(result)` if ready, `None` otherwise.
    pub fn poll(&mut self, token: u64) -> Option<Vec<u8>> {
        match self.tokens.get(&token) {
            Some(TokenState::Ready { result, .. }) => {
                let r = result.clone();
                self.tokens.remove(&token);
                Some(r)
            }
            Some(TokenState::Pending { submitted_at, .. }) => {
                if now_ms().saturating_sub(*submitted_at) > TOKEN_EXPIRY_MS {
                    self.tokens.insert(token, TokenState::Expired);
                }
                None
            }
            _ => None,
        }
    }

    /// Complete a token with a result. Called by a worker / orchestrator.
    pub fn complete(&mut self, token: u64, result: Vec<u8>) -> bool {
        if self.tokens.contains_key(&token) {
            self.tokens.insert(
                token,
                TokenState::Ready {
                    result,
                    completed_at: now_ms(),
                },
            );
            true
        } else {
            false
        }
    }

    /// Remove expired tokens. Call periodically.
    pub fn gc(&mut self) {
        let now = now_ms();
        self.tokens.retain(|_, state| match state {
            TokenState::Pending { submitted_at, .. } => {
                now.saturating_sub(*submitted_at) < TOKEN_EXPIRY_MS
            }
            TokenState::Ready { completed_at, .. } => {
                now.saturating_sub(*completed_at) < TOKEN_EXPIRY_MS
            }
            TokenState::Expired => false,
        });
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_pending(&self, token: u64) -> bool {
        matches!(self.tokens.get(&token), Some(TokenState::Pending { .. }))
    }

    pub fn is_ready(&self, token: u64) -> bool {
        matches!(self.tokens.get(&token), Some(TokenState::Ready { .. }))
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Global pool for FFI
// ---------------------------------------------------------------------------

use std::sync::LazyLock;

static GLOBAL_POOL: LazyLock<Mutex<AsyncTokenPool>> =
    LazyLock::new(|| Mutex::new(AsyncTokenPool::new()));

fn global_pool() -> std::sync::MutexGuard<'static, AsyncTokenPool> {
    GLOBAL_POOL.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// FFI exports
// ---------------------------------------------------------------------------

/// Submit a query and receive a token.
/// Returns 0 on error (null input), otherwise a positive token ID.
#[no_mangle]
pub extern "C" fn yp_temporal_submit(query_ptr: *const u8, query_len: usize) -> u64 {
    if query_ptr.is_null() || query_len == 0 {
        return 0;
    }
    let query = unsafe { std::slice::from_raw_parts(query_ptr, query_len) }.to_vec();
    global_pool().submit(query)
}

/// Poll a token. Returns:
///   -1 : invalid / expired token
///    0 : not ready yet
///   >0 : result length written to `out_ptr`
#[no_mangle]
pub extern "C" fn yp_temporal_poll(
    token: u64,
    out_ptr: *mut u8,
    out_cap: usize,
) -> i32 {
    if token == 0 {
        return -1;
    }

    let mut pool = global_pool();
    match pool.poll(token) {
        Some(result) => {
            if out_ptr.is_null() || out_cap == 0 {
                return -1;
            }
            let n = result.len().min(out_cap);
            unsafe {
                std::ptr::copy_nonoverlapping(result.as_ptr(), out_ptr, n);
            }
            n as i32
        }
        None => {
            // Distinguish "not ready" from "invalid" by checking membership.
            if pool.tokens.contains_key(&token) {
                0
            } else {
                -1
            }
        }
    }
}

/// M3.1: Return router confidence for the given 128-bit PAP.
/// Confidence is in [0, 1]; higher means the router is more certain.
#[no_mangle]
pub extern "C" fn yp_confidence_score(pap_128: *const u8) -> f32 {
    if pap_128.is_null() {
        return 0.0;
    }
    let mut bytes = [0u8; 16];
    unsafe { std::ptr::copy_nonoverlapping(pap_128, bytes.as_mut_ptr(), 16) };
    let features = features_from_pap(&bytes);
    let router = LearnedRouter::new();
    router.confidence(features)
}

/// M4.1: Return a high-confidence score for semantic-cache gating.
/// Forces the router onto the fast path so cache eligibility is deterministic.
#[no_mangle]
pub extern "C" fn yp_confidence_score_fast(pap_128: *const u8) -> f32 {
    if pap_128.is_null() {
        return 0.0;
    }
    let mut bytes = [0u8; 16];
    unsafe { std::ptr::copy_nonoverlapping(pap_128, bytes.as_mut_ptr(), 16) };
    let features = features_from_pap(&bytes);
    let mut router = LearnedRouter::new();
    router.force_fast_path();
    router.confidence(features)
}

/// M3.1: Returns 1 if the fast bucket-only path can be used for this query.
#[no_mangle]
pub extern "C" fn yp_fast_path_eligible(
    pap_128: *const u8,
    pap_512: *const u8,
    top_k: usize,
) -> i32 {
    if pap_128.is_null() || pap_512.is_null() {
        return 0;
    }
    // We need access to a mesh; without an initialized mesh we cannot decide.
    // Stub: always report not eligible from this standalone helper.
    // The real fast path is invoked inside yp_query_* paths.
    0
}

fn features_from_pap(pap_128: &[u8; 16]) -> [f32; 8] {
    let mut f = [0.0f32; 8];
    for i in 0..8 {
        f[i] = (pap_128[i * 2] as f32) / 255.0;
    }
    f
}

/// Run garbage collection on the token pool. Returns current size.
#[no_mangle]
pub extern "C" fn yp_temporal_gc() -> i32 {
    let mut pool = global_pool();
    pool.gc();
    pool.size() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_returns_increasing_tokens() {
        let mut pool = AsyncTokenPool::new();
        let t1 = pool.submit(vec![1, 2, 3]);
        let t2 = pool.submit(vec![4, 5, 6]);
        assert!(t2 > t1);
        assert_eq!(pool.size(), 2);
    }

    #[test]
    fn poll_returns_none_while_pending() {
        let mut pool = AsyncTokenPool::new();
        let t = pool.submit(vec![1]);
        assert!(pool.is_pending(t));
        assert!(!pool.is_ready(t));
        assert!(pool.poll(t).is_none());
    }

    #[test]
    fn complete_then_poll_returns_result() {
        let mut pool = AsyncTokenPool::new();
        let t = pool.submit(vec![1]);
        assert!(pool.complete(t, vec![9, 8, 7]));
        assert_eq!(pool.poll(t), Some(vec![9, 8, 7]));
        assert_eq!(pool.size(), 0);
    }

    #[test]
    fn invalid_token_returns_none() {
        let mut pool = AsyncTokenPool::new();
        assert!(pool.poll(999).is_none());
        assert!(!pool.tokens.contains_key(&999));
    }

    #[test]
    fn gc_removes_expired() {
        let mut pool = AsyncTokenPool::new();
        let t = pool.submit(vec![1]);
        // Simulate expiry by mutating submitted_at far in the past.
        pool.tokens.insert(
            t,
            TokenState::Pending {
                query: vec![1],
                submitted_at: now_ms().saturating_sub(TOKEN_EXPIRY_MS + 1),
            },
        );
        pool.gc();
        assert!(!pool.tokens.contains_key(&t));
    }
}
