// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::build_context::BuildContext;
use std::sync::Arc;
use std::time::{Duration, Instant};
use sysinfo::{Pid, System, RefreshKind, ProcessRefreshKind, ProcessesToUpdate};

const CHECK_INTERVAL_S: u64 = 10;
const WORKING_HARD_GRACE_S: u64 = 30;
const STUCK_BREATHING_ROOM_S: u64 = 60;
const HEALTHY_PAPERS_PER_S: f64 = 1000.0;
const HIGH_CPU: f32 = 90.0;
const LOW_CPU: f32 = 10.0;
const LOW_FREE_RAM_BYTES: u64 = 1_000_000_000; // 1 GB
const HIGH_SWAP_BYTES: u64 = 4_000_000_000; // 4 GB

/// Current health state as decided by the watchdog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WatchdogState {
    Healthy,
    WorkingHard,
    Stuck,
    Dead,
}

impl std::fmt::Display for WatchdogState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatchdogState::Healthy => write!(f, "healthy"),
            WatchdogState::WorkingHard => write!(f, "working hard"),
            WatchdogState::Stuck => write!(f, "stuck"),
            WatchdogState::Dead => write!(f, "dead"),
        }
    }
}

/// A point-in-time snapshot of build progress and system pressure.
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    pub progress: u64,
    pub delta_papers: u64,
    pub elapsed: Duration,
    pub cpu_percent: f32,
    pub free_ram_bytes: u64,
    pub used_swap_bytes: u64,
    pub disk_read_mbps: f32,
    pub disk_write_mbps: f32,
}

impl SystemSnapshot {
    pub fn papers_per_second(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs > 0.0 {
            self.delta_papers as f64 / secs
        } else {
            0.0
        }
    }
}

/// Monitors an in-flight ISM build and kills it if it stalls.
pub struct Watchdog {
    ctx: Arc<BuildContext>,
    system: System,
    pid: Pid,
    state: WatchdogState,
    state_since: Instant,
    working_hard_since: Option<Instant>,
    last_progress: u64,
    last_check: Instant,
    last_disk_read: u64,
    last_disk_written: u64,
    advice: String,
    last_snapshot: Option<SystemSnapshot>,
}

impl Watchdog {
    pub fn new(ctx: Arc<BuildContext>) -> Self {
        let pid = Pid::from_u32(std::process::id());
        let mut system = System::new_with_specifics(
            RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
        );
        // Warm up CPU measurement.
        std::thread::sleep(Duration::from_millis(200));
        system.refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::everything());

        Self {
            ctx,
            system,
            pid,
            state: WatchdogState::Healthy,
            state_since: Instant::now(),
            working_hard_since: None,
            last_progress: 0,
            last_check: Instant::now(),
            last_disk_read: 0,
            last_disk_written: 0,
            advice: String::new(),
            last_snapshot: None,
        }
    }

    /// Run until the build finishes or the watchdog kills it.
    /// Returns `Ok(())` if the build completed naturally, or an error message
    /// describing why it was killed.
    pub fn run(&mut self) -> Result<(), String> {
        let start = Instant::now();
        let mut since_eval = Duration::ZERO;
        loop {
            std::thread::sleep(Duration::from_secs(1));
            since_eval += Duration::from_secs(1);

            if self.ctx.is_finished() {
                return Ok(());
            }

            if since_eval < Duration::from_secs(CHECK_INTERVAL_S) {
                continue;
            }
            since_eval = Duration::ZERO;

            let snapshot = self.take_snapshot();
            self.last_snapshot = Some(snapshot.clone());
            let new_state = self.decide(&snapshot);
            self.transition(new_state);

            println!(
                "🔍 Watchdog [{}]: progress={}/{} (+{}), cpu={:.1}%, free_ram={:.2}GB, swap={:.2}GB",
                self.state,
                snapshot.progress,
                self.ctx.total(),
                snapshot.delta_papers,
                snapshot.cpu_percent,
                snapshot.free_ram_bytes as f64 / 1e9,
                snapshot.used_swap_bytes as f64 / 1e9,
            );

            match self.state {
                WatchdogState::Dead => {
                    let reason = self.compose_reason("Deadlock detected: no progress and CPU is idle.", &snapshot);
                    self.kill(reason.clone());
                    return Err(reason);
                }
                WatchdogState::Stuck => {
                    let stuck_for = self.state_since.elapsed().as_secs();
                    if stuck_for >= STUCK_BREATHING_ROOM_S {
                        let reason = self.compose_reason(
                            &format!(
                                "Build stuck for {}s (no progress, CPU {:.1}%, RAM full, swap thrashing).",
                                stuck_for, snapshot.cpu_percent
                            ),
                            &snapshot,
                        );
                        self.kill(reason.clone());
                        return Err(reason);
                    } else {
                        println!(
                            "⏳ Watchdog: build stuck. Giving {}s breathing room...",
                            STUCK_BREATHING_ROOM_S - stuck_for
                        );
                    }
                }
                WatchdogState::WorkingHard => {
                    if let Some(since) = self.working_hard_since {
                        let grace = since.elapsed().as_secs();
                        if grace >= WORKING_HARD_GRACE_S {
                            println!("⚠️  Watchdog: still working hard after grace period; will treat as stuck if no progress.");
                        } else {
                            println!("⚠️  Watchdog: build working hard, {}s grace remaining...", WORKING_HARD_GRACE_S - grace);
                        }
                    }
                }
                WatchdogState::Healthy => {}
            }

            // Global safety net only; 200M+ Python builds can legitimately run longer.
            if start.elapsed() > Duration::from_secs(3600) {
                let reason = self.compose_reason("Watchdog safety timeout: build exceeded 1 hour.", &snapshot);
                self.kill(reason.clone());
                return Err(reason);
            }
        }
    }

    fn take_snapshot(&mut self) -> SystemSnapshot {
        self.system
            .refresh_processes_specifics(ProcessesToUpdate::All, true, ProcessRefreshKind::everything());
        let now = Instant::now();
        let elapsed = now - self.last_check;
        self.last_check = now;

        let process = self.system.process(self.pid);
        let cpu_percent = process.map(|p| p.cpu_usage()).unwrap_or(0.0);
        let (disk_read_delta, disk_write_delta) = process
            .map(|p| {
                let d = p.disk_usage();
                (
                    d.total_read_bytes.saturating_sub(self.last_disk_read),
                    d.total_written_bytes.saturating_sub(self.last_disk_written),
                )
            })
            .unwrap_or((0, 0));
        if let Some(p) = process {
            self.last_disk_read = p.disk_usage().total_read_bytes;
            self.last_disk_written = p.disk_usage().total_written_bytes;
        }

        let free_ram_bytes = self.system.available_memory();
        let used_swap_bytes = self.system.used_swap();

        let current_progress = self.ctx.progress() as u64;
        let delta_papers = current_progress.saturating_sub(self.last_progress);
        self.last_progress = current_progress;

        let secs = elapsed.as_secs_f64().max(1e-6);
        SystemSnapshot {
            progress: current_progress,
            delta_papers,
            elapsed,
            cpu_percent,
            free_ram_bytes,
            used_swap_bytes,
            disk_read_mbps: ((disk_read_delta as f64 / secs) / (1024.0 * 1024.0)) as f32,
            disk_write_mbps: ((disk_write_delta as f64 / secs) / (1024.0 * 1024.0)) as f32,
        }
    }

    fn decide(&self, snap: &SystemSnapshot) -> WatchdogState {
        let pps = snap.papers_per_second();
        let memory_full = snap.free_ram_bytes < LOW_FREE_RAM_BYTES
            || snap.used_swap_bytes > HIGH_SWAP_BYTES;

        if pps == 0.0 && snap.cpu_percent < LOW_CPU {
            return WatchdogState::Dead;
        }
        if pps == 0.0 && snap.cpu_percent >= HIGH_CPU && memory_full {
            return WatchdogState::Stuck;
        }
        if pps > 0.0 && snap.cpu_percent >= HIGH_CPU {
            return WatchdogState::WorkingHard;
        }
        if pps >= HEALTHY_PAPERS_PER_S && snap.cpu_percent < HIGH_CPU {
            return WatchdogState::Healthy;
        }
        // Default: if there is progress, call it healthy.
        if pps > 0.0 {
            WatchdogState::Healthy
        } else {
            // No progress but not dead/stuck by the strict rules; wait.
            WatchdogState::Stuck
        }
    }

    fn transition(&mut self, new_state: WatchdogState) {
        if new_state != self.state {
            self.state = new_state;
            self.state_since = Instant::now();
            match new_state {
                WatchdogState::WorkingHard if self.working_hard_since.is_none() => {
                    self.working_hard_since = Some(Instant::now());
                }
                WatchdogState::Healthy => {
                    self.working_hard_since = None;
                }
                _ => {}
            }
        }
    }

    fn compose_reason(&self, base: &str, snapshot: &SystemSnapshot) -> String {
        let safe_limit = if !self.advice.is_empty() {
            format!("\nLearned guidance: {}", self.advice)
        } else {
            String::new()
        };
        format!(
            "{}\nSnapshot at kill — progress={}/{} (+{} in {:.1}s), cpu={:.1}%, free_ram={:.2}GB, swap={:.2}GB, read={:.1}MB/s, write={:.1}MB/s{}",
            base,
            snapshot.progress,
            self.ctx.total(),
            snapshot.delta_papers,
            snapshot.elapsed.as_secs_f64(),
            snapshot.cpu_percent,
            snapshot.free_ram_bytes as f64 / 1e9,
            snapshot.used_swap_bytes as f64 / 1e9,
            snapshot.disk_read_mbps,
            snapshot.disk_write_mbps,
            safe_limit
        )
    }

    fn kill(&mut self, reason: String) {
        println!("\n❌ WATCHDOG TRIGGERED\n{}\n", reason);
        self.ctx.request_stop();
        // Give workers a moment to observe the stop signal.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !self.ctx.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    pub fn set_advice(&mut self, advice: String) {
        self.advice = advice;
    }

    pub fn advice(&self) -> &str {
        &self.advice
    }
}
