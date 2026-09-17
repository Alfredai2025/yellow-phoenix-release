// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

pub struct BuildContext;
impl BuildContext {
    pub fn new() -> Self { Self }
    pub fn should_stop(&self) -> bool { false }
    pub fn increment_progress(&self) {}
    pub fn worker_done(&self) {}
    pub fn request_stop(&self) {}
    pub fn is_finished(&self) -> bool { true }
    pub fn total(&self) -> usize { 0 }
    pub fn progress(&self) -> usize { 0 }
}
