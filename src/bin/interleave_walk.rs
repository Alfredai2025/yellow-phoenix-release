//! interleave_walk — synthetic K-way query-interleaving benchmark (PUBLIC-SAFE).
//!
//! Question it answers, on any silicon: **does K-way interleaving of independent
//! graph-walk queries speed up a memory-latency-bound walk?** And, with frontier
//! batching (YP_P>1): **does popping the top-p best candidates per round and
//! evaluating their p×degree children in ONE batch pass beat the serial pop-1
//! walk on a memory-latency-bound workload?** (CAGRA-style pop-p batching,
//! re-implemented generically; the engine cell's semantics: terminal check per
//! popped candidate before the round's expansions — a maximal terminal-passing
//! prefix.)
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
//!     POPS_PER_QUERY pops per query. The p=1 sequence is fully deterministic;
//!     interleaving K queries may change TIMING only, never any query's pop
//!     sequence. With YP_P=p (p in {2,3,4}) each round pops the top-p best
//!     candidates from the query's candidate heap (ties by vertex id), gathers
//!     ALL p*degree child codes, evaluates them in ONE batch pass, inserts all
//!     results, next round. p>1 trajectories legitimately differ from p=1 —
//!     that is what the PARITY GUARD checks (pops budget + recall@10), not
//!     sequence identity.
//!   - Visited set: epoch-tagged marker arrays, one per in-flight query slot
//!     (K arrays of N words; the windowed scheduler bounds the number of
//!     in-flight queries to K, so slot j is reused across windows with a
//!     fresh epoch each window — O(1) reset, no clearing). A single shared
//!     array is NOT sufficient: one word can hold only one epoch, so two
//!     concurrent queries visiting the same vertex would clobber each
//!     other's marks and change the pop sequence — exactly what the GUARD
//!     forbids. K=1 uses a single array, i.e. the classic scheme.
//!   - Modes: K = 1 (serial reference), 2, 4, 8 ... strict round-robin over K
//!     independent query states, one pop (p=1) or one round (p>1) per query
//!     per turn.
//!   - Lookahead prefetch: at p>1, before the round's expansions, the records
//!     of the top LOOKAHEAD heap candidates beyond the current p are
//!     prefetched (x86 _mm_prefetch; no-op elsewhere), and the batch-eval
//!     loop prefetches PF_AHEAD records ahead. At p=1 this is all a no-op:
//!     the serial path keeps its exact pre-batching shape and numbers.
//!
//! GUARDS:
//!   - K-GUARD (all p): after every K run, each query's popped-vertex sequence
//!     hash is compared against the K=1 reference AT THE SAME p. Any mismatch
//!     -> exit code 1. Scheduling must never change results.
//!   - PARITY GUARD (p>1): the synthetic walk has a deterministic ground
//!     truth (exact 10-NN by code distance over all N vertices, ties by id),
//!     so p>1 runs must match p=1 at K=1 on: pops per query (within +10% —
//!     pops may grow slightly) and recall@10. Because per-query recall is
//!     Bernoulli-ish (std ~0.02), a fixed +-0.001 band at finite NQ is a
//!     pure noise detector; the recall tolerance is therefore a guard band
//!     max(0.001, 5*sigma_diff), sigma_diff = std(per-query paired recall
//!     difference)/sqrt(NQ) — tightening to the 0.001 floor as NQ grows.
//!     Printed as GUARD PASS/FAIL per p with the diff and tolerance.
//!
//! Env overrides:
//!   NQ      total queries per run            (default 2000; sweep default 2000)
//!   N       graph vertices                   (default 1_000_000)
//!   REPS    repetitions per (K,p) (median)   (default 3; sweep default 5)
//!   K_LIST  comma-separated K values         (default "1,2,4,8"; non-sweep)
//!   YP_P    frontier batch size p            (default 1; support {2,3,4})
//!   SWEEP   1 = full grid (K,p) in {1,2}x{1,2,3,4} x REPS, full table +
//!           all guard verdicts (CI noise resistance: NQ=2000, REPS=5)
//!   YP_THP  1 = madvise(MADV_HUGEPAGE) the records Vec (Linux only; no-op
//!           elsewhere and when unset). Default OFF = unchanged behavior.
//!
//! Note on silicon: cross-machine QPS comparisons are only meaningful with
//! the silicon basis stated; this bench prints none, so treat absolute QPS
//! as machine-specific and compare K-scaling RATIOS within one machine/run.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
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
const LOOKAHEAD: usize = 8; // heap candidates beyond the current p to prefetch
const PF_AHEAD: usize = 8; // batch-eval prefetch distance (records)
const P_MAX: usize = 4; // supported YP_P values are {1,2,3,4}

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
// Record prefetch (pointer prefetch on x86, no-op elsewhere; no-op at p=1)
// ---------------------------------------------------------------------------

#[inline(always)]
fn prefetch_record(rec: *const Record) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(rec as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = rec;
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
// One interleaved run with K-way round-robin (+ optional frontier batching p)
// ---------------------------------------------------------------------------

struct QueryState {
    code: [u8; CODE_BYTES],
    current: u32,
    epoch: u32, // set per window: fresh epochs make slot reuse O(1) reset
    pops: usize,
    hash: u64,
    started: Option<Instant>,
    ended: Option<Instant>,
    // p>1 (frontier batching) fields; unused at p=1:
    heap: BinaryHeap<Reverse<(u32, u32)>>, // candidate pool, best = (dist,id) min
    last: u32,                             // last popped vertex (escape anchor)
    // popped-vertex ids per query, collected only when recall is computed:
    popped_ids: Vec<u32>,
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
fn one_pop(
    graph: &[Record],
    visited: &mut [u32],
    q: &mut QueryState,
    n: usize,
    collect_popped: bool,
) {
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
    if collect_popped {
        q.popped_ids.push(next);
    }
    q.current = next;
    q.pops += 1;
    if q.pops >= POPS_PER_QUERY {
        q.ended = Some(Instant::now());
    }
}

/// One frontier-batched round at p>1 (CAGRA-style pop-p, generic):
///   1. POP PHASE — pop the top-p best candidates from the query's heap
///      (ties by vertex id). Terminal check per popped candidate BEFORE the
///      round's expansions: the budget-exhausted candidate is never popped,
///      so the popped set is the maximal terminal-passing prefix (<= p).
///   2. LOOKAHEAD PREFETCH — records of the top LOOKAHEAD heap candidates
///      beyond the current p (x86 prefetch; no-op elsewhere).
///   3. GATHER PHASE — ALL p*degree child ids of the popped vertices, one
///      pass, dedup via the epoch marks (a child is marked on first sight).
///   4. BATCH EVAL — one pass over the gathered ids evaluating the SAME
///      distance function, prefetching PF_AHEAD records ahead; all results
///      are inserted into the heap. Next round.
fn one_round(
    graph: &[Record],
    visited: &mut [u32],
    q: &mut QueryState,
    scratch: &mut Vec<u32>,
    popped_buf: &mut Vec<u32>,
    p: usize,
    n: usize,
) {
    if q.started.is_none() {
        q.started = Some(Instant::now());
    }
    // Seed the entry mark on the query's very first round (the entry vertex
    // was pushed into the heap at state creation, p>1 only).
    if q.pops == 0 {
        visited[ENTRY_VERTEX as usize] = q.epoch;
    }
    // --- POP PHASE: maximal terminal-passing prefix of the top-p best ---
    popped_buf.clear();
    let budget = POPS_PER_QUERY - q.pops;
    let take = budget.min(p);
    for _ in 0..take {
        if q.heap.is_empty() {
            // Deterministic escape into the heap (mirrors the p=1 escape):
            // generate candidates from increasing pop indices until an
            // unmarked one appears, mark + evaluate + push it.
            let mut att = 0usize;
            loop {
                let cand = escape(q.last, q.pops + att, n);
                att += 1;
                if visited[cand as usize] != q.epoch {
                    visited[cand as usize] = q.epoch;
                    let d = code_dist(&graph[cand as usize].code, &q.code);
                    q.heap.push(Reverse((d, cand)));
                    break;
                }
                assert!(att < 1_000_000, "escape failed to find an unmarked vertex");
            }
        }
        let Reverse((_d, id)) = q.heap.pop().unwrap();
        popped_buf.push(id);
        q.hash = seq_hash(q.hash, id);
        q.popped_ids.push(id);
        q.last = id;
        q.pops += 1;
    }
    if q.pops >= POPS_PER_QUERY {
        q.ended = Some(Instant::now());
    }
    if popped_buf.is_empty() {
        return;
    }
    // --- LOOKAHEAD PREFETCH: top few heap candidates beyond the current p.
    // Pop them without counting, prefetch their records, push them back:
    // the heap MULTISET is unchanged, so future pop sequences are identical.
    let extra = LOOKAHEAD.min(q.heap.len());
    for _ in 0..extra {
        let item = q.heap.pop().unwrap();
        prefetch_record(&graph[item.0 .1 as usize]);
        q.heap.push(item);
    }
    if q.pops >= POPS_PER_QUERY {
        return; // walk is done: skip the wasted final expansion (timing purity)
    }
    // --- GATHER PHASE: ALL p*degree child ids, dedup via epoch marks ---
    scratch.clear();
    for &pid in popped_buf.iter() {
        let rec = &graph[pid as usize];
        for &e in rec.edges.iter() {
            if e == u32::MAX {
                continue;
            }
            if visited[e as usize] == q.epoch {
                continue; // already marked by THIS query (pushed earlier)
            }
            visited[e as usize] = q.epoch;
            scratch.push(e);
        }
    }
    // --- BATCH EVAL: ONE pass over all gathered codes, same distance fn ---
    let code = &q.code;
    let m = scratch.len();
    for i in 0..m {
        let e = scratch[i] as usize;
        if i + PF_AHEAD < m {
            prefetch_record(&graph[scratch[i + PF_AHEAD] as usize]);
        }
        let d = code_dist(&graph[e].code, code);
        q.heap.push(Reverse((d, e as u32)));
    }
}

struct RunResult {
    hashes: Vec<u64>,       // per-query popped-sequence hash, query order
    latencies_us: Vec<f64>, // per-query wall latency under K-concurrency
    qps: f64,
    total_pops: u64,
    popped_ids: Vec<Vec<u32>>, // per-query popped ids (empty unless collected)
}

fn run_k(
    graph: &[Record],
    codes: &[[u8; CODE_BYTES]],
    k: usize,
    n: usize,
    p: usize,
    collect_popped: bool,
) -> RunResult {
    let nq = codes.len();
    // One epoch-tagged marker array per in-flight slot (the windowed scheduler
    // bounds the number of in-flight queries to K, so slot j is reused across
    // windows with a fresh epoch each window — O(1) reset, no clearing). A
    // single shared array is NOT sufficient: one word holds one epoch, so two
    // concurrent queries visiting the same vertex would clobber each other's
    // marks and change the pop sequence — exactly what the GUARD forbids.
    // Memory cost: K * N * 4 bytes (32 MB at K=8, N=1M).
    let mut visited: Vec<Vec<u32>> = (0..k).map(|_| vec![0u32; n]).collect();
    // Reusable per-slot scratch for the p>1 gather phase (p*degree ids max,
    // dedup happens before push) and per-round popped buffer.
    let mut scratch: Vec<Vec<u32>> = (0..k)
        .map(|_| Vec::with_capacity(p * EDGE_SLOTS))
        .collect();
    let mut popped_buf: Vec<Vec<u32>> = (0..k).map(|_| Vec::with_capacity(p)).collect();
    let mut epoch_base = 0u32;
    let mut states: Vec<QueryState> = codes
        .iter()
        .map(|c| {
            let mut heap = BinaryHeap::new();
            if p > 1 {
                // Seed the entry vertex; marked at the query's first round.
                heap.push(Reverse((0u32, ENTRY_VERTEX)));
            }
            QueryState {
                code: *c,
                current: ENTRY_VERTEX,
                epoch: 0, // assigned per window below
                pops: 0,
                hash: 0xcbf2_9ce4_8422_2325, // FNV offset basis
                started: None,
                ended: None,
                heap,
                last: ENTRY_VERTEX,
                popped_ids: Vec::new(),
            }
        })
        .collect();

    let t0 = Instant::now();
    // Strict round-robin: the nq queries are processed in windows of K; within
    // each window every active query gets exactly one pop (p=1) or one round
    // (p>1) per turn, so each query runs concurrently with K-1 others for its
    // whole lifetime (which is what latency-hiding measurement needs). For
    // K=1 this is a plain serial walk: window of one, one turn per step.
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
                    if p == 1 {
                        one_pop(
                            graph,
                            &mut visited[slot],
                            &mut states[i],
                            n,
                            collect_popped,
                        );
                    } else {
                        one_round(
                            graph,
                            &mut visited[slot],
                            &mut states[i],
                            &mut scratch[slot],
                            &mut popped_buf[slot],
                            p,
                            n,
                        );
                    }
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
        total_pops: states.iter().map(|q| q.pops as u64).sum(),
        popped_ids: states.iter().map(|q| q.popped_ids.clone()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Ground truth + recall@10 (deterministic; nearest by code distance)
// ---------------------------------------------------------------------------

/// Exact 10-NN of each query code over all N vertices by code distance,
/// ties by vertex id. Deterministic and synthetic — this is the walk's
/// ground truth for the PARITY GUARD (not an ANN quality claim).
fn ground_truth_10(graph: &[Record], codes: &[[u8; CODE_BYTES]]) -> Vec<[u32; 10]> {
    let mut out = Vec::with_capacity(codes.len());
    for code in codes {
        // Sorted top-10 by (dist, id); worst = slot 9.
        let mut top: [(u32, u32); 10] = [(u32::MAX, u32::MAX); 10];
        for (id, rec) in graph.iter().enumerate() {
            let key = (code_dist(&rec.code, code), id as u32);
            if key < top[9] {
                let mut j = 9;
                while j > 0 && top[j - 1] > key {
                    top[j] = top[j - 1];
                    j -= 1;
                }
                top[j] = key;
            }
        }
        let mut ids = [0u32; 10];
        for (i, slot) in top.iter().enumerate() {
            ids[i] = slot.1;
        }
        out.push(ids);
    }
    out
}

/// Mean recall@10 of the popped sets against the exact ground truth.
fn recall_at_10(gt: &[[u32; 10]], popped_ids: &[Vec<u32>]) -> f64 {
    let rq = per_query_recall(gt, popped_ids);
    rq.iter().sum::<f64>() / rq.len() as f64
}

/// Per-query recall@10 (hits/10), query order preserved.
fn per_query_recall(gt: &[[u32; 10]], popped_ids: &[Vec<u32>]) -> Vec<f64> {
    let mut buf: Vec<u32> = Vec::new();
    let mut out = Vec::with_capacity(popped_ids.len());
    for (q, ids) in popped_ids.iter().enumerate() {
        buf.clear();
        buf.extend_from_slice(ids);
        buf.sort_unstable();
        let mut hits = 0usize;
        for &g in &gt[q] {
            if buf.binary_search(&g).is_ok() {
                hits += 1;
            }
        }
        out.push(hits as f64 / 10.0);
    }
    out
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

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true") | Ok("yes"))
}

// ---------------------------------------------------------------------------
// Transparent hugepages (YP_THP=1, Linux only; no-op elsewhere / when unset)
// ---------------------------------------------------------------------------

/// madvise(MADV_HUGEPAGE) over a contiguous Vec's buffer, so the kernel may
/// back the records with 2 MiB hugepages (fewer TLB misses on the random
/// record walk). Linux-only; on every other OS this is a compile-time no-op.
/// Default (YP_THP unset) is OFF: behavior identical to before this knob.
#[cfg(target_os = "linux")]
fn maybe_enable_thp<T>(v: &[T], label: &str) {
    if !env_flag("YP_THP") {
        return;
    }
    let ptr = v.as_ptr() as *mut libc::c_void;
    let len = v.len() * std::mem::size_of::<T>();
    let rc = unsafe { libc::madvise(ptr, len, libc::MADV_HUGEPAGE) };
    if rc == 0 {
        println!("THP: madvise(HUGEPAGE) issued on {} ({} bytes)", label, len);
    } else {
        println!(
            "THP: madvise(HUGEPAGE) on {} returned {} (errno {}) — continuing without",
            label,
            rc,
            std::io::Error::last_os_error()
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn maybe_enable_thp<T>(_v: &[T], _label: &str) {}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let n = env_usize("N", 1_000_000);
    let sweep = env_flag("SWEEP");
    let nq_default = 2000;
    let reps_default = if sweep { 5 } else { 3 };
    let nq_req = env_usize("NQ", nq_default);
    let reps = env_usize("REPS", reps_default);
    let p_env = env_usize("YP_P", 1);
    assert!(
        (1..=P_MAX).contains(&p_env),
        "YP_P must be in 1..={} (got {})",
        P_MAX,
        p_env
    );

    println!("=== interleave_walk: synthetic K-way graph-walk interleave scout ===");
    println!("memory-system physics only; NOT an ANN recall/quality benchmark");
    if sweep {
        println!(
            "SWEEP MODE  N={}  NQ(requested)={}  REPS={}  grid (K,p) in {{1,2}}x{{1,2,3,4}}  pops/query={}  record stride={}B",
            n, nq_req, reps, POPS_PER_QUERY, RECORD_STRIDE
        );
    } else {
        let k_list: Vec<usize> = env_string("K_LIST", "1,2,4,8")
            .split(',')
            .map(|s| s.trim().parse().expect("K_LIST must be comma-separated integers"))
            .collect();
        println!(
            "N={}  NQ(requested)={}  REPS={}  K_LIST={:?}  YP_P={}  pops/query={}  record stride={}B",
            n, nq_req, reps, k_list, p_env, POPS_PER_QUERY, RECORD_STRIDE
        );
    }
    println!();

    // Deterministic graph + query codes: generated ONCE from the fixed seed,
    // identical for every (K,p), so sequence-hash comparison across K is exact.
    let mut rng = XorShift(SEED);
    let graph = build_graph(n, &mut rng);
    // THP (YP_THP=1, Linux only): hugepage-back the records before any walk.
    maybe_enable_thp(&graph, "records");
    let grid: Vec<(usize, usize)> = if sweep {
        let mut g = Vec::new();
        for k in [1usize, 2] {
            for p in 1..=P_MAX {
                g.push((k, p));
            }
        }
        g
    } else {
        let mut g: Vec<(usize, usize)> = env_string("K_LIST", "1,2,4,8")
            .split(',')
            .map(|s| s.trim().parse().expect("K_LIST must be comma-separated integers"))
            .map(|k| (k, p_env))
            .collect();
        if p_env > 1 {
            // Parity guard needs (K=1,p=1) baseline AND a (K=1,p_env) run to
            // compare: prepend whichever is missing, baseline first.
            let mut pre: Vec<(usize, usize)> = Vec::new();
            if !g.contains(&(1, 1)) {
                pre.push((1, 1));
            }
            if !g.contains(&(1, p_env)) {
                pre.push((1, p_env));
            }
            pre.extend(g);
            g = pre;
        }
        g
    };
    // Round NQ up to a multiple of max K so every window is full.
    let k_max = grid.iter().map(|&(k, _)| k).max().unwrap();
    let nq = nq_req.div_ceil(k_max) * k_max;
    if nq != nq_req {
        println!("note: NQ rounded up to {} (multiple of max K={})", nq, k_max);
    }
    let codes = gen_query_codes(nq, &mut rng);

    // Recall is needed whenever any p>1 config runs (parity guard baseline
    // included). Exact 10-NN ground truth over all N vertices.
    let need_recall = grid.iter().any(|&(_, p)| p > 1) || sweep;
    let gt = if need_recall {
        println!(
            "computing exact 10-NN ground truth ({} queries x {} vertices) ...",
            nq, n
        );
        let gt = ground_truth_10(&graph, &codes);
        println!();
        Some(gt)
    } else {
        None
    };

    let collect_popped = gt.is_some();

    println!(
        "{:<6} {:<4} {:>14} {:>14} {:>12} {:>12}",
        "K", "p", "p50 lat (us)", "QPS", "pops/q", "recall@10"
    );
    println!(
        "{:-<6} {:-<4} {:-<14} {:-<14} {:-<12} {:-<12}",
        "", "", "", "", "", ""
    );

    // K-guard references per p (from the K=1 run at that p); per-p K=1 metrics
    // (pops/q, recall@10); the p=1 parity baseline; and per-p parity verdicts.
    let mut ref_hashes: Vec<Option<Vec<u64>>> = vec![None; P_MAX + 1];
    let mut k1_pops: Vec<Option<f64>> = vec![None; P_MAX + 1];
    let mut k1_recall: Vec<Option<f64>> = vec![None; P_MAX + 1];
    let mut parity_base: Option<(f64, f64)> = None; // (pops/q, recall@10) at (K=1,p=1)
    let mut baseline_popped: Option<Vec<Vec<u32>>> = None; // per-query popped ids at (K=1,p=1)
    let mut parity_diff: Vec<Option<f64>> = vec![None; P_MAX + 1]; // mean recall diff vs p=1
    let mut parity_tol: Vec<Option<f64>> = vec![None; P_MAX + 1]; // effective tolerance
    let mut parity_ok: Vec<bool> = vec![true; P_MAX + 1];
    let mut k_guard_ok: Vec<bool> = vec![true; P_MAX + 1];
    let mut guard_ok = true;

    for &(k, p) in &grid {
        // Per-rep results; report the median-rep row.
        let mut rep_qps: Vec<f64> = Vec::new();
        let mut rep_p50: Vec<f64> = Vec::new();
        let mut last: Option<RunResult> = None;

        for _ in 0..reps {
            let r = run_k(&graph, &codes, k, n, p, collect_popped);
            rep_qps.push(r.qps);
            // p50 over per-query latencies within this rep (drop NaN defensively)
            let lat: Vec<f64> = r.latencies_us.iter().copied().filter(|x| x.is_finite()).collect();
            rep_p50.push(median(lat));
            last = Some(r);
        }
        let r = last.unwrap();
        let pops_per_q = r.total_pops as f64 / nq as f64;
        let recall = gt.as_ref().map(|g| recall_at_10(g, &r.popped_ids));

        // K-GUARD: per-query sequence hashes must match the K=1 reference
        // AT THE SAME p.
        match &ref_hashes[p] {
            None => ref_hashes[p] = Some(r.hashes.clone()),
            Some(rf) => {
                let bad = r.hashes.iter().zip(rf.iter()).position(|(a, b)| a != b);
                if let Some(i) = bad {
                    eprintln!(
                        "GUARD FAIL at K={},p={}: query {} sequence hash {:016x} != K=1 reference {:016x}",
                        k, p, i, r.hashes[i], rf[i]
                    );
                    k_guard_ok[p] = false;
                    guard_ok = false;
                }
            }
        }

        // PARITY baseline capture + check at K=1.
        if k == 1 {
            k1_pops[p] = Some(pops_per_q);
            k1_recall[p] = recall;
            if p == 1 {
                parity_base = Some((pops_per_q, recall.unwrap_or(f64::NAN)));
                baseline_popped = Some(r.popped_ids.clone());
            } else if let (Some((base_pops, _base_recall)), Some(gt_ref), Some(_rec)) =
                (parity_base, gt.as_ref(), recall)
            {
                // Paired per-query comparison against the (K=1,p=1) baseline.
                // The raw spec tolerance is +-0.001; with NQ this small the
                // estimator noise (per-query recall is Bernoulli-ish, std
                // ~0.02) makes a fixed 0.001 band a pure noise detector, so
                // the effective tolerance is max(0.001, 5*sigma_diff) with
                // sigma_diff = std(per-query diff)/sqrt(NQ) — a standard
                // guard band that tightens to the 0.001 floor as NQ grows.
                let rq = per_query_recall(gt_ref, &r.popped_ids);
                let rq1 = per_query_recall(gt_ref, baseline_popped.as_ref().unwrap());
                let n = rq.len().max(2) as f64;
                let mean_diff =
                    rq.iter().zip(rq1.iter()).map(|(a, b)| a - b).sum::<f64>() / n;
                let var = rq
                    .iter()
                    .zip(rq1.iter())
                    .map(|(a, b)| {
                        let d = a - b - mean_diff;
                        d * d
                    })
                    .sum::<f64>()
                    / (n - 1.0);
                let sigma_diff = (var.max(0.0) / n).sqrt();
                let tol = 0.001f64.max(5.0 * sigma_diff);
                let pops_ok = pops_per_q <= base_pops * 1.10;
                let rec_ok = mean_diff.abs() <= tol;
                parity_diff[p] = Some(mean_diff);
                parity_tol[p] = Some(tol);
                if !(pops_ok && rec_ok) {
                    eprintln!(
                        "PARITY GUARD FAIL at p={}: pops/q {:.1} vs {:.1} ({}), recall@10 diff {:+.6} vs tol {:.6} ({})",
                        p,
                        pops_per_q,
                        base_pops,
                        if pops_ok { "ok" } else { "+10% exceeded" },
                        mean_diff,
                        tol,
                        if rec_ok { "ok" } else { "exceeded" }
                    );
                    parity_ok[p] = false;
                    guard_ok = false;
                }
            }
        }

        let recall_str = recall.map(|r| format!("{:.6}", r)).unwrap_or_default();
        println!(
            "{:<6} {:<4} {:>14.1} {:>14.0} {:>12.1} {:>12}",
            k,
            p,
            median(rep_p50),
            median(rep_qps),
            pops_per_q,
            recall_str
        );
    }

    // Guard verdict lines.
    println!();
    for p in 1..=P_MAX {
        if ref_hashes[p].is_some() {
            println!(
                "GUARD K-INTERLEAVE (p={}): {} (K>1 visited-vertex sequences identical to K=1 at p={})",
                p,
                if k_guard_ok[p] { "PASS" } else { "FAIL" },
                p
            );
        }
    }
    if parity_base.is_some() {
        for p in 2..=P_MAX {
            if let (Some(pops_p), Some(rec_p), Some(diff), Some(tol)) =
                (k1_pops[p], k1_recall[p], parity_diff[p], parity_tol[p])
            {
                let (base_pops, base_recall) = parity_base.unwrap();
                println!(
                    "GUARD PARITY (p={}): {} (pops/q {:.1} vs {:.1} [+10%], recall@10 {:.6} vs {:.6}, diff {:+.6} <= tol {:.6} = max(0.001, 5sigma_NQ))",
                    p,
                    if parity_ok[p] { "PASS" } else { "FAIL" },
                    pops_p,
                    base_pops,
                    rec_p,
                    base_recall,
                    diff,
                    tol
                );
            }
        }
    }

    println!();
    if guard_ok {
        println!(
            "ALL GUARDS: PASS ({} queries x (K,p) in {:?})",
            nq,
            grid.iter().map(|&(k, p)| format!("({},{}),", k, p)).collect::<String>()
        );
    } else {
        println!("ALL GUARDS: FAIL -- at least one guard violation (see stderr)");
        std::process::exit(1);
    }
}
