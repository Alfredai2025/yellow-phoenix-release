// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::fmt;
use crate::shard::ShardError;

#[derive(Debug, Clone)]
pub enum ISMError {
    Shard(String),
    Build(String),
    IO(String),
    KilledByWatchdog { reason: String, advice: String },
    ThreadPanic,
}

impl fmt::Display for ISMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ISMError::Shard(s) => write!(f, "Shard error: {}", s),
            ISMError::Build(s) => write!(f, "Build error: {}", s),
            ISMError::IO(s) => write!(f, "IO error: {}", s),
            ISMError::KilledByWatchdog { reason, advice } => {
                write!(f, "Killed by watchdog: {} | advice: {}", reason, advice)
            }
            ISMError::ThreadPanic => write!(f, "Thread panic"),
        }
    }
}

impl std::error::Error for ISMError {}

impl From<String> for ISMError {
    fn from(s: String) -> Self { ISMError::Build(s) }
}

impl From<std::io::Error> for ISMError {
    fn from(e: std::io::Error) -> Self { ISMError::IO(e.to_string()) }
}

impl From<ShardError> for ISMError {
    fn from(e: ShardError) -> Self { ISMError::Shard(format!("{:?}", e)) }
}
