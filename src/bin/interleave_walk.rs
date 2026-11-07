//! interleave_walk — synthetic K-way query-interleaving benchmark (PUBLIC-SAFE).
//!
//! Question it answers, on any silicon: **does K-way interleaving of independent
//! graph-walk queries speed up a memory-latency-bound walk?**
//!
//! This benchmark is FULLY SYNTHETIC: no data files, no downloads, no external
//! inputs. It measures MEMORY-SYSTEM PHYSICS (cache/DRAM latency hiding under
//! interleaved dependent-pointer-chasing workloads) — it does NOT measure ANN
//! recall, graph quality, or index construction quality, and its "distance" is
//! a synthetic popcount over random bytes, not an embedding metric.
//!
//! Model (deliberately generic, written fresh for this file):
//!   - Graph: N vertices, one contiguous Vec of fixed-stride records (320 B).
//!     Each record = 32 edge slots (u32 vertex ids, unused = u32::MAX)
//!     + one 32-byte 2-bit code (random bytes) + a rank constant + padding.
//!   - Distance: XOR + popcount over the 32-byte code (4 x u64 chunks),
//!     scalar POPCNT-class, portable. SIMD is not the point of this bench.
//!   - Walk: greedy best-first from a fixed entry vertex, exactly
//!     POPS_PER_QUERY pops per query. Each pop visits the best UNVISITED
//!     neighbor of the current vertex by code distance (ties by id). The
//!     sequence is fully deterministic; interleaving K queries may change
//!     TIMING only, never any query's pop sequence.
//!   - Visited set: epoch-tagged marker arrays, one per in-flight query slot
//!     (K arrays of N words; the windowed scheduler bounds the number of
//!     in-flight queries to K, so slot j is reused across windows with a
//!     fresh epoch each window — O(1) reset, no clearing). A single shared
//!     array is NOT sufficient: one word can hold only one epoch, so two
//!     concurrent queries visiting the same vertex would clobber each
//!     other's marks and change the pop sequence — exactly what the GUARD
//!     forbids. K=1 uses a single array, i.e. the classic scheme.
//!   - Modes: K = 1 (serial reference), 2, 4, 8 ... strict round-robin over K
//!     independent query states, one pop per query per turn.
//!
//! GUARD: after every K run, each query's visited-vertex sequence hash is
//! compared against the K=1 reference. Any mismatch -> exit code 1. On match
//! the final line prints GUARD: PASS. Scheduling must never change results.
//!
//! Env overrides:
//!   NQ      total queries per run            (default 2000)
//!   N       graph vertices                   (default 1_000_000)
//!   REPS    repetitions per K (median shown) (default 3)
//!   K_LIST  comma-separated K values         (default "1,2,4,8")
//!
//! Note on silicon: cross-machine QPS comparisons are only meaningful with
//! the silicon basis stated; this bench prints none, so treat absolute QPS
//! as machine-specific and compare K-scaling RATIOS within one machine/run.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EDGE_SLOTS: usize = 32;
const CODE_BYTES: usize = 32;
const RECORD_STRIDE: usize = 320; // bytes; see Record layout check in main
const POPS_PER_QUERY: usize = 134;
const ENTRY_VERTEX: u32 = 0;
const SEED: u64 = 0x243F_6A88_85A3_08D3; // pi digits, fixed -> fully reproducible

/// Fused per-vertex record, 320 B, one contiguous Vec. The `_pad` field is
/// deliberate: real-world fused records are not 160 B; keeping the stride at
/// 320 B keeps the walk genuinely memory-bound (one record per cache-line pair)
/// rather than accidentally cache-resident.
#[repr(C)]
struct Record {
    edges: [u32; EDGE_SLOTS], // 128 B
    code: [u8; CODE_BYTES],   //  32 B
    rank: u32,                //   4 B  per-vertex constant (code popcount)
    _pad: [u8; 156],          // 156 B
}
const _: () = assert!(std::mem::size_of::<Record>() == RECORD_STRIDE);

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*), fixed seed
// ---------------------------------------------------------------------------

struct XorShift(u64);

impl XorShift {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

// ---------------------------------------------------------------------------
// Distance: XOR + popcount over 32 bytes (scalar POPCNT-class, portable)
// ---------------------------------------------------------------------------

#[inline]
fn code_dist(a: &[u8; CODE_BYTES], b: &[u8; CODE_BYTES]) -> u32 {
    let mut d = 0u32;
    for i in 0..4 {
        let x = u64::from_le_bytes(a[i * 8..i * 8 + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
        d += x.count_ones();
    }
    d
}

// ---------------------------------------------------------------------------
// Graph construction (fully synthetic, deterministic)
// ---------------------------------------------------------------------------

fn build_graph(n: usize, rng: &mut XorShift) -> Vec<Record> {
    let mut g = Vec::with_capacity(n);
    for id in 0..n {
        let mut rec = Record {
            edges: [u32::MAX; EDGE_SLOTS],
            code: [0u8; CODE_BYTES],
            rank: 0,
            _pad: [0u8; 156],
        };
        for e in rec.edges.iter_mut() {
            // Random neighbor; avoid trivial self-loops (deterministic).
            let mut v = rng.below(n as u64) as u32;
            if v as usize == id {
                v = ((v as usize + 1) % n) as u32;
            }
            *e = v;
        }
        for b in rec.code.iter_mut() {
            *b = rng.next_u64() as u8;
        }
        // Rank constant: popcount of the code. Stored to mimic the rank
        // metadata present in real fused records; not used by the walk rule
        // (tie-break is by id, per spec) but part of the realistic stride.
        rec.rank = rec
            .code
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()).count_ones())
            .sum();
        g.push(rec);
    }
    g
}

fn gen_query_codes(nq: usize, rng: &mut XorShift) -> Vec<[u8; CODE_BYTES]> {
    (0..nq)
        .map(|_| {
            let mut c = [0u8; CODE_BYTES];
            for b in c.iter_mut() {
                *b = rng.next_u64() as u8;
            }
            c
        })
        .collect()
}

// ---------------------------------------------------------------------------
// One interleaved run with K-way round-robin
// ---------------------------------------------------------------------------

struct QueryState {
    code: [u8; CODE_BYTES],
    current: u32,
    epoch: u32, // set per window: fresh epochs make slot reuse O(1) reset
    pops: usize,
    hash: u64,
    started: Option<Instant>,
    ended: Option<Instant>,
}

#[inline]
fn seq_hash(h: u64, v: u32) -> u64 {
    // FNV-1a step over the visited vertex id.
    let mut h = h ^ v as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01B3);
    h
}

/// Deterministic escape when the current vertex has no unvisited neighbor:
/// a fixed pseudo-random jump. Determinism is what matters — the sequence
/// must be identical at every K.
#[inline]
fn escape(current: u32, pop_index: usize, n: usize) -> u32 {
    let x = (current as u64 ^ (pop_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_mul(0xD6E8_FEB8_6659_FD93);
    (x % n as u64) as u32
}

#[inline]
fn one_pop(graph: &[Record], visited: &mut [u32], q: &mut QueryState, n: usize) {
    if q.started.is_none() {
        q.started = Some(Instant::now());
    }
    let rec = &graph[q.current as usize];
    let mut best: Option<(u32, u32)> = None; // (dist, neighbor id)
    for &e in rec.edges.iter() {
        if e == u32::MAX {
            continue;
        }
        if visited[e as usize] == q.epoch {
            continue; // already visited by THIS query
        }
        let d = code_dist(&graph[e as usize].code, &q.code);
        match best {
            Some((bd, bid)) if !(d < bd || (d == bd && e < bid)) => {}
            _ => best = Some((d, e)),
        }
    }
    let next = match best {
        Some((_, id)) => id,
        None => escape(q.current, q.pops, n),
    };
    visited[next as usize] = q.epoch;
    q.hash = seq_hash(q.hash, next);
    q.current = next;
    q.pops += 1;
    if q.pops >= POPS_PER_QUERY {
        q.ended = Some(Instant::now());
    }
}

struct RunResult {
    hashes: Vec<u64>,       // per-query visited-sequence hash, query order
    latencies_us: Vec<f64>, // per-query wall latency under K-concurrency
    qps: f64,
    total_pops: u64,
}

fn run_k(graph: &[Record], codes: &[[u8; CODE_BYTES]], k: usize, n: usize) -> RunResult {
    let nq = codes.len();
    // One epoch-tagged marker array per in-flight slot (the windowed scheduler
    // bounds the number of in-flight queries to K, so slot j is reused across
    // windows with a fresh epoch each window — O(1) reset, no clearing). A
    // single shared array is NOT sufficient: one word holds one epoch, so two
    // concurrent queries visiting the same vertex would clobber each other's
    // marks and change the pop sequence — exactly what the GUARD forbids.
    // Memory cost: K * N * 4 bytes (32 MB at K=8, N=1M).
    let mut visited: Vec<Vec<u32>> = (0..k).map(|_| vec![0u32; n]).collect();
    let mut epoch_base = 0u32;
    let mut states: Vec<QueryState> = codes
        .iter()
        .map(|c| QueryState {
            code: *c,
            current: ENTRY_VERTEX,
            epoch: 0, // assigned per window below
            pops: 0,
            hash: 0xcbf2_9ce4_8422_2325, // FNV offset basis
            started: None,
            ended: None,
        })
        .collect();

    let t0 = Instant::now();
    // Strict round-robin: the nq queries are processed in windows of K; within
    // each window every active query gets exactly one pop per turn, so each
    // query runs concurrently with K-1 others for its whole lifetime (which is
    // what latency-hiding measurement needs). For K=1 this is a plain serial
    // walk: window of one, one pop per turn.
    let mut window_start = 0usize;
    while window_start < nq {
        let wend = (window_start + k).min(nq);
        // Fresh epochs for this window: slots are reused across windows and
        // never cleared — a query only recognizes marks tagged with its own
        // epoch, so bumping the base is an O(1) reset.
        epoch_base = epoch_base.wrapping_add(k as u32);
        for i in window_start..wend {
            states[i].epoch = epoch_base + (i - window_start) as u32;
        }
        loop {
            let mut all_done = true;
            for i in window_start..wend {
                if states[i].pops < POPS_PER_QUERY {
                    let slot = i - window_start;
                    one_pop(graph, &mut visited[slot], &mut states[i], n);
                    all_done = false;
                }
            }
            if all_done {
                break;
            }
        }
        window_start = wend;
    }
    let wall = t0.elapsed();
    let hashes = states.iter().map(|q| q.hash).collect();
    let latencies_us: Vec<f64> = states
        .iter()
        .map(|q| match (q.started, q.ended) {
            (Some(a), Some(b)) => b.duration_since(a).as_secs_f64() * 1e6,
            _ => f64::NAN,
        })
        .collect();
    RunResult {
        hashes,
        latencies_us,
        qps: nq as f64 / wall.as_secs_f64(),
        total_pops: nq as u64 * POPS_PER_QUERY as u64,
    }
}

// ---------------------------------------------------------------------------
// Small stats helpers
// ---------------------------------------------------------------------------

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len();
    if m == 0 {
        return f64::NAN;
    }
    if m % 2 == 1 {
        v[m / 2]
    } else {
        (v[m / 2 - 1] + v[m / 2]) / 2.0
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let nq_default = 2000;
    let n = env_usize("N", 1_000_000);
    let nq_req = env_usize("NQ", nq_default);
    let reps = env_usize("REPS", 3);
    let k_list: Vec<usize> = env_string("K_LIST", "1,2,4,8")
        .split(',')
        .map(|s| s.trim().parse().expect("K_LIST must be comma-separated integers"))
        .collect();

    println!("=== interleave_walk: synthetic K-way graph-walk interleave scout ===");
    println!("memory-system physics only; NOT an ANN recall/quality benchmark");
    println!(
        "N={}  NQ(requested)={}  REPS={}  K_LIST={:?}  pops/query={}  record stride={}B",
        n, nq_req, reps, k_list, POPS_PER_QUERY, RECORD_STRIDE
    );
    println!();

    // Deterministic graph + query codes: generated ONCE from the fixed seed,
    // identical for every K, so sequence-hash comparison across K is exact.
    let mut rng = XorShift(SEED);
    let graph = build_graph(n, &mut rng);
    // Round NQ up to a multiple of max K so every window is full.
    let k_max = *k_list.iter().max().unwrap();
    let nq = nq_req.div_ceil(k_max) * k_max;
    if nq != nq_req {
        println!("note: NQ rounded up to {} (multiple of max K={})", nq, k_max);
    }
    let codes = gen_query_codes(nq, &mut rng);

    println!("{:<6} {:>14} {:>14} {:>16}", "K", "p50 lat (us)", "QPS", "pops total");
    println!("{:-<6} {:-<14} {:-<14} {:-<16}", "", "", "", "");

    let mut reference: Option<Vec<u64>> = None;
    let mut guard_ok = true;

    for &k in &k_list {
        // Per-rep results; report the median-rep row.
        let mut rep_qps: Vec<f64> = Vec::new();
        let mut rep_p50: Vec<f64> = Vec::new();
        let mut last_hashes: Vec<u64> = Vec::new();
        let mut last_pops: u64 = 0;

        for _ in 0..reps {
            let r = run_k(&graph, &codes, k, n);
            last_pops = r.total_pops;
            last_hashes = r.hashes.clone();
            rep_qps.push(r.qps);
            // p50 over per-query latencies within this rep (drop NaN defensively)
            let lat: Vec<f64> = r.latencies_us.into_iter().filter(|x| x.is_finite()).collect();
            rep_p50.push(median(lat));
        }

        // GUARD: per-query sequence hashes must match the K=1 reference.
        match &reference {
            None => reference = Some(last_hashes.clone()),
            Some(rf) => {
                let bad = last_hashes
                    .iter()
                    .zip(rf.iter())
                    .position(|(a, b)| a != b);
                if let Some(i) = bad {
                    eprintln!(
                        "GUARD FAIL at K={}: query {} sequence hash {:016x} != K=1 reference {:016x}",
                        k, i, last_hashes[i], rf[i]
                    );
                    guard_ok = false;
                }
            }
        }

        println!(
            "{:<6} {:>14.1} {:>14.0} {:>16}",
            k,
            median(rep_p50.clone()),
            median(rep_qps.clone()),
            last_pops
        );
    }

    println!();
    if guard_ok {
        println!(
            "GUARD: PASS ({} queries x K in {:?}: visited-vertex sequences identical to K=1)",
            nq, k_list
        );
    } else {
        println!("GUARD: FAIL — interleaving changed at least one pop sequence");
        std::process::exit(1);
    }
}
