// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! engine_feeder.rs — non-blocking ring buffer for YP feeds.
//!
//! Integrates 3 feeds in M1:
//!   - cascade_miss  (/tmp/yp_feed_cascade)
//!   - faiss_disagree (/tmp/yp_feed_faiss)
//!   - domain        (/tmp/yp_feed_domain)
//!
//! Messages are JSON payloads. Missing or corrupt feeds are logged but do not crash queries.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const RING_SIZE: usize = 1024;
const FEED_TIMEOUT_MS: u64 = 5000;

/// A single message from any feed.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeedMessage {
    pub feed: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

/// Health record for a feed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FeedHealth {
    pub last_message: u64,
    pub messages_received: u64,
    pub messages_dropped: u64,
    pub healthy: bool,
}

/// Ring buffer for one feed.
#[derive(Clone, Debug)]
pub struct FeedRing {
    name: String,
    buffer: Vec<FeedMessage>,
    head: usize,
    tail: usize,
    count: usize,
    capacity: usize,
    health: FeedHealth,
}

impl FeedRing {
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            buffer: Vec::with_capacity(capacity),
            head: 0,
            tail: 0,
            count: 0,
            capacity,
            health: FeedHealth::default(),
        }
    }

    pub fn push(&mut self, payload: Vec<u8>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(FeedMessage {
                feed: self.name.clone(),
                payload,
                timestamp: now_ms(),
            });
        } else {
            self.buffer[self.tail] = FeedMessage {
                feed: self.name.clone(),
                payload,
                timestamp: now_ms(),
            };
        }

        self.tail = (self.tail + 1) % self.capacity;
        if self.count == self.capacity {
            self.head = (self.head + 1) % self.capacity;
            self.health.messages_dropped += 1;
        } else {
            self.count += 1;
        }
        self.health.messages_received += 1;
        self.health.last_message = now_ms();
        self.health.healthy = true;
    }

    pub fn pop(&mut self) -> Option<FeedMessage> {
        if self.count == 0 {
            None
        } else {
            let msg = self.buffer[self.head].clone();
            self.head = (self.head + 1) % self.capacity;
            self.count -= 1;
            Some(msg)
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn health(&self) -> &FeedHealth {
        &self.health
    }
}

/// Non-blocking feeder aggregating all configured feeds.
#[derive(Clone, Debug)]
pub struct EngineFeeder {
    rings: HashMap<String, FeedRing>,
}

impl Default for EngineFeeder {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineFeeder {
    pub fn new() -> Self {
        let mut rings = HashMap::new();
        for name in ["cascade_miss", "faiss_disagree", "domain"] {
            rings.insert(name.into(), FeedRing::new(name, RING_SIZE));
        }
        Self { rings }
    }

    /// Inject a raw payload into a named feed.
    pub fn inject(&mut self, feed_name: &str, payload: Vec<u8>) {
        if let Some(ring) = self.rings.get_mut(feed_name) {
            ring.push(payload);
        }
    }

    /// Inject raw bytes into a feed (alias for fuzz/compat).
    pub fn inject_raw(&mut self, feed_name: &str, payload: &[u8]) {
        self.inject(feed_name, payload.to_vec());
    }

    /// Poll the next message from a feed.
    pub fn poll_feed(&mut self, feed_name: &str) -> Option<FeedMessage> {
        self.rings.get_mut(feed_name)?.pop()
    }

    /// Alias for `poll_feed` used by downstream modules.
    pub fn poll(&mut self, feed_name: &str) -> Option<FeedMessage> {
        self.poll_feed(feed_name)
    }

    /// List all registered feed names.
    pub fn list_feeds(&self) -> Vec<String> {
        self.rings.keys().cloned().collect()
    }

    /// Check health of all feeds.
    pub fn health(&mut self) -> HashMap<String, FeedHealth> {
        let now = now_ms();
        let mut result = HashMap::new();
        for (name, ring) in &mut self.rings {
            let timeout = ring.health.last_message + FEED_TIMEOUT_MS;
            ring.health.healthy = ring.health.last_message == 0 || now < timeout;
            result.insert(name.clone(), ring.health.clone());
        }
        result
    }

    /// True if all feeds are alive.
    pub fn is_alive(&self) -> bool {
        !self.rings.is_empty()
    }

    /// Attempt to read from named pipe files. Non-blocking; missing pipes are OK.
    pub fn read_pipes(&mut self, root: impl AsRef<Path>) -> io::Result<()> {
        let root = root.as_ref();
        let pipes = [
            ("cascade_miss", root.join("/tmp/yp_feed_cascade")),
            ("faiss_disagree", root.join("/tmp/yp_feed_faiss")),
            ("domain", root.join("/tmp/yp_feed_domain")),
        ];

        for (name, path) in &pipes {
            if !path.exists() {
                continue;
            }
            let mut file = fs::File::open(path)?;
            let mut buf = Vec::new();
            if file.read_to_end(&mut buf).is_ok() && !buf.is_empty() {
                self.inject(name, buf);
            }
        }
        Ok(())
    }

    /// Default pipe paths exposed for wiring scanners.
    pub fn pipe_paths() -> Vec<(String, PathBuf)> {
        vec![
            ("cascade_miss".into(), PathBuf::from("/tmp/yp_feed_cascade")),
            ("faiss_disagree".into(), PathBuf::from("/tmp/yp_feed_faiss")),
            ("domain".into(), PathBuf::from("/tmp/yp_feed_domain")),
        ]
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_and_poll() {
        let mut feeder = EngineFeeder::new();
        feeder.inject("cascade_miss", b"test".to_vec());
        let msg = feeder.poll_feed("cascade_miss").unwrap();
        assert_eq!(msg.feed, "cascade_miss");
        assert_eq!(msg.payload, b"test");
    }

    #[test]
    fn ring_overwrite_old() {
        let mut ring = FeedRing::new("test", 2);
        ring.push(b"a".to_vec());
        ring.push(b"b".to_vec());
        ring.push(b"c".to_vec());
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop().unwrap().payload, b"b");
        assert_eq!(ring.pop().unwrap().payload, b"c");
    }

    #[test]
    fn feed_corruption_does_not_panic() {
        let mut feeder = EngineFeeder::new();
        for _ in 0..1000 {
            let garbage: Vec<u8> = (0..64).map(|i| (i * 7) as u8).collect();
            feeder.inject_raw("cascade_miss", &garbage);
        }
        assert!(feeder.is_alive());
    }

    #[test]
    fn health_tracks_messages() {
        let mut feeder = EngineFeeder::new();
        feeder.inject("cascade_miss", b"x".to_vec());
        let h = feeder.health();
        assert_eq!(h["cascade_miss"].messages_received, 1);
    }
}
