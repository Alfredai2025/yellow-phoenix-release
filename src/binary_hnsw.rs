// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! Binary HNSW index for 512-bit ITQ hashes.
//!
//! Uses pure Hamming distance (popcount of XOR) so it works on the compact
//! 64-byte hashes already produced by the ITQ encoder.  All popcount is
//! portable (`u64::count_ones`); an AVX-512 fast path can be added later as
//! an optional feature.
//!
//! This is a standard navigable small-world graph (HNSW) adapted to binary
//! vectors.  It mirrors the parameters of a typical FAISS HNSW index but
//! operates on 64-byte hashes, giving a large memory advantage over a 384-d
//! float index and much cheaper distance computation.
//!
//! Phase 2d: inline-node layout.  Neighbor lists live in a single flat arena
//! (`neighbor_arena: Vec<u32>`) instead of `Vec<Vec<u32>>` per node.  Each
//! node stores an offset and per-layer counts, eliminating per-node heap
//! allocations and reducing pointer chasing during traversal.

use alloc::vec::Vec;
use core::cmp::Reverse;
use hashbrown::HashSet;
use rand::Rng;

/// A 512-bit binary hash, packed into 64 bytes.
pub type Hash512 = [u8; 64];

/// Number of bytes in a 512-bit hash.
pub const HASH512_BYTES: usize = 64;

/// Software prefetch hint for the query hot path (M1 memory discipline).
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "aarch64")]
    // `_prefetch` intrinsic is still unstable (rust#117217); `prfm pldl1keep`
    // is the identical instruction via stable inline asm.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, preserves_flags)
        )
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(
            ptr.cast::<core::ffi::c_char>(),
            core::arch::x86_64::_MM_HINT_T0,
        )
    }
}

thread_local! {
    /// Query-side hop counter (nodes visited per search_layer call); read by benches.
    pub static HOPS: std::cell::Cell<u64> = std::cell::Cell::new(0);
    /// M1 memory discipline: reused per-query search buffers (visited set +
    /// candidate/found heaps). Avoids three heap allocations per layer visit;
    /// `clear()` retains capacity across queries.
    static LAYER_SCRATCH: std::cell::RefCell<(
        hashbrown::HashSet<u32>,
        alloc::collections::BinaryHeap<Reverse<(u32, u32)>>,
        alloc::collections::BinaryHeap<(u32, u32)>,
    )> = std::cell::RefCell::new((
        hashbrown::HashSet::new(),
        alloc::collections::BinaryHeap::new(),
        alloc::collections::BinaryHeap::new(),
    ));
    /// In-graph fusion state, integer fast path. FSKETCH = raw ptr to leaked
    /// [i32; 2*n] fixed-point sketch coords; FDEN = 0 disables fusion.
    static FSKETCH: std::cell::Cell<*const i32> = const { std::cell::Cell::new(std::ptr::null()) };
    static FQX: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    static FQY: std::cell::Cell<i32> = const { std::cell::Cell::new(0) };
    static FDEN: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
    static FWNUM: std::cell::Cell<i64> = const { std::cell::Cell::new(0) };
}

/// Enable integer in-graph fusion. `sketch` = 2 fixed-point i32 coords per node
/// (raw coords * 1024). `den` = (rms_radius * 1024)^2; `wnum` = 64 * weight.
/// The sketch vec is leaked (bench lifetime).
pub fn set_fusion_int(sketch: Vec<i32>, den: i64, wnum: i64) {
    let p = sketch.leak().as_ptr();
    FSKETCH.with(|c| c.set(p));
    FDEN.with(|c| c.set(den));
    FWNUM.with(|c| c.set(wnum));
}
pub fn set_fusion_q(q: [i32; 2]) {
    FQX.with(|c| c.set(q[0]));
    FQY.with(|c| c.set(q[1]));
}
pub fn clear_fusion() {
    FSKETCH.with(|c| c.set(std::ptr::null()));
    FDEN.with(|c| c.set(0));
}
/// Fused score: hamming*64 + min(64*w*d_sketch^2/rms^2, 4096). All integer.
#[inline]
pub fn fused_score(idx: u32, dh: u32) -> u32 {
    let den = FDEN.with(|c| c.get());
    if den == 0 {
        return dh;
    }
    let p = FSKETCH.with(|c| c.get());
    let i = idx as usize * 2;
    let (sx, sy, qx, qy) = unsafe { (*p.add(i), *p.add(i + 1), FQX.with(|c| c.get()), FQY.with(|c| c.get())) };
    let dx = (sx - qx) as i64;
    let dy = (sy - qy) as i64;
    let d2 = dx * dx + dy * dy;
    let term = ((d2 * FWNUM.with(|c| c.get())) / den).min(4096) as u32; // clamp in i64 BEFORE cast
    dh.saturating_mul(64).saturating_add(term)
}

/// Number of bits in a 512-bit hash.
pub const HASH512_BITS: u32 = 512;

/// Default maximum number of bi-directional links per layer (`M` in the
/// Malkov-Yashunin HNSW paper).
pub const DEFAULT_M: usize = 16;

/// Default `efConstruction` — size of the dynamic candidate list used during
/// graph construction.
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Default `efSearch` — size of the dynamic candidate list used during query.
pub const DEFAULT_EF_SEARCH: usize = 128;

/// Hard cap on the number of layers.  With `M = 32` the probability of
/// exceeding 16 layers is vanishingly small.
const MAX_LAYERS: usize = 16;

pub mod simd;

/// Compute the Hamming distance between two 512-bit hashes.
#[inline]
pub fn hamming_distance(a: &Hash512, b: &Hash512) -> u32 {
    simd::hamming_distance_512(a, b)
}

/// A node in the binary HNSW graph.
///
/// Neighbor IDs are stored in a flat arena owned by `BinaryHNSW`.
/// `neighbor_start` is the byte offset (in `u32` elements) into that arena,
/// and `layer_counts[layer]` is the number of neighbors at that layer.
#[derive(Clone, Debug)]
pub struct BinaryNode {
    /// User-facing paper / record identifier.
    pub id: u64,
    /// 512-bit binary hash.
    pub hash: Hash512,
    /// Beacon tag: 0x00=document, 0x01=concept, 0x02=reasoning, 0x03=topic.
    pub tag: u8,
    /// Number of layers this node participates in.
    pub num_layers: u8,
    /// Offset into `BinaryHNSW::neighbor_arena` where this node's primary neighbors begin.
    pub neighbor_start: u32,
    /// Number of primary neighbors stored at each layer.
    pub layer_counts: [u8; MAX_LAYERS],
    /// Offset into `BinaryHNSW::neighbor_arena` where this node's alternative neighbors begin.
    pub alt_neighbor_start: u32,
    /// Number of alternative neighbors stored at each layer.
    pub alt_layer_counts: [u8; MAX_LAYERS],
    /// Element offset into the mapped edges section (v5 mmap arenas only).
    /// Unused (0) for owned arenas, where `neighbor_start`/`layer_offset`
    /// locate neighbors instead.
    pub edge_off: u32,
}

impl BinaryNode {
    /// Create a new node with no neighbors and default tag 0.
    pub fn new(id: u64, hash: Hash512) -> Self {
        Self::with_tag(id, hash, 0)
    }

    /// Create a new node with an explicit beacon tag.
    pub fn with_tag(id: u64, hash: Hash512, tag: u8) -> Self {
        Self {
            id,
            hash,
            tag,
            num_layers: 0,
            neighbor_start: 0,
            layer_counts: [0; MAX_LAYERS],
            alt_neighbor_start: 0,
            alt_layer_counts: [0; MAX_LAYERS],
            edge_off: 0,
        }
    }

    /// Highest layer this node participates in.
    pub fn top_layer(&self) -> usize {
        self.num_layers.saturating_sub(1) as usize
    }
}

/// Neighbor storage backend.
///
/// `Owned` is the classic padded in-memory arena used for building and for
/// v4 files. `Mapped` memory-maps a v5 file's edge section so multi-GB
/// indexes load in milliseconds and pages fault in on demand (RSS grows
/// with pages actually touched by queries, not with file size).
enum Arena {
    Owned(Vec<u32>),
    Mapped(MappedArena),
}

struct MappedArena {
    base: *mut libc::c_void,
    map_len: usize,
    /// Start of the edges section, as a u32 pointer (4-byte aligned by layout).
    elem: *const u32,
    elem_len: usize,
    _file: std::fs::File,
    path: std::path::PathBuf,
}

// The mapping is read-only (PROT_READ / MAP_PRIVATE); sharing across threads
// is safe and required for the static RwLock<BinaryHNSW> storage.
unsafe impl Send for MappedArena {}
unsafe impl Sync for MappedArena {}

impl MappedArena {
    /// Open and memory-map a v5 file (read-only, private).
    fn map(path: &std::path::Path) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;
        let file = std::fs::File::open(path)?;
        let map_len = file.metadata()?.len() as usize;
        if map_len == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "empty v5 file"));
        }
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                map_len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if base as isize == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            base,
            map_len,
            elem: std::ptr::null(),
            elem_len: 0,
            _file: file,
            path: path.to_path_buf(),
        })
    }
}

impl Clone for MappedArena {
    fn clone(&self) -> Self {
        // Re-open and re-map the same file.
        Self::map(&self.path).expect("re-mmap cloned MappedArena")
    }
}

impl core::fmt::Debug for MappedArena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MappedArena")
            .field("elem_len", &self.elem_len)
            .field("path", &self.path)
            .finish()
    }
}

impl Clone for Arena {
    fn clone(&self) -> Self {
        match self {
            Arena::Owned(v) => Arena::Owned(v.clone()),
            Arena::Mapped(m) => Arena::Mapped(m.clone()),
        }
    }
}

impl core::fmt::Debug for Arena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Arena::Owned(v) => f.debug_tuple("Owned").field(&v.len()).finish(),
            Arena::Mapped(m) => f.debug_tuple("Mapped").field(m).finish(),
        }
    }
}

impl Drop for MappedArena {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base, self.map_len); }
    }
}

impl Arena {
    fn len_elems(&self) -> usize {
        match self {
            Arena::Owned(v) => v.len(),
            Arena::Mapped(m) => m.elem_len,
        }
    }

    /// Byte length of the arena (for madvise).
    fn len_bytes(&self) -> usize {
        self.len_elems() * core::mem::size_of::<u32>()
    }

    fn base_ptr(&self) -> *mut libc::c_void {
        match self {
            Arena::Owned(v) => v.as_ptr() as *mut libc::c_void,
            Arena::Mapped(m) => m.elem as *mut libc::c_void,
        }
    }

    /// Mutable access — insert/build path (Owned only).
    fn owned_mut(&mut self) -> &mut Vec<u32> {
        match self {
            Arena::Owned(v) => v,
            Arena::Mapped(_) => panic!("mutating a memory-mapped (v5) HNSW is not supported"),
        }
    }

    /// Immutable slice over an Owned arena. Mapped arenas have a different
    /// (compact, unpadded) layout; use `neighbor_ids` instead.
    fn owned_slice(&self) -> &[u32] {
        match self {
            Arena::Owned(v) => v,
            Arena::Mapped(_) => panic!("padded arena access on a memory-mapped (v5) HNSW"),
        }
    }
}

/// Thread-safe `usize` flag (e.g. `ef_search`): readable/writable under a
/// SHARED lock, so query-parameter updates never need exclusive access to
/// the index (which would block the UI behind warm-up on iOS).
#[derive(Debug)]
struct SharedUsize(std::sync::atomic::AtomicUsize);

impl SharedUsize {
    fn new(v: usize) -> Self {
        Self(std::sync::atomic::AtomicUsize::new(v))
    }
    fn get(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }
    fn set(&self, v: usize) {
        self.0.store(v, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Clone for SharedUsize {
    fn clone(&self) -> Self {
        Self::new(self.get())
    }
}

/// Binary HNSW index with inline neighbor storage.
#[derive(Clone, Debug)]
pub struct BinaryHNSW {
    /// All nodes, indexed by their internal `u32` index.
    nodes: Vec<BinaryNode>,
    /// Flat arena holding all neighbor IDs for all nodes.
    arena: Arena,
    /// Highest layer currently present in the graph.
    max_layers: usize,
    /// `M` — max neighbors per layer.
    m: usize,
    /// `m_L` — layer probability factor (`1 / ln(M)`).
    m_l: f64,
    /// `efConstruction`.
    ef_construction: usize,
    /// `efSearch`.
    ef_search: SharedUsize,
    /// Entry point node index for search.
    enter_point: Option<u32>,
}

/// Warm-up readahead budget for `madvise(MADV_WILLNEED)`. On iOS this pages
/// the region into resident memory, so budget it against what the kernel
/// would actually give this process before jetsam: a quarter of currently
/// available RAM, clamped. Non-iOS keeps the historical full-length hint.
#[cfg(target_os = "ios")]
fn readahead_budget() -> usize {
    const FLOOR: usize = 32 * 1024 * 1024;
    const HARD_CAP: usize = 512 * 1024 * 1024;
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()),
    );
    sys.refresh_memory();
    let avail = sys.available_memory() as usize;
    (avail / 4).clamp(FLOOR, HARD_CAP)
}

/// Out-of-memory error for fallible index allocations.
fn io_error_oom(what: &str) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::OutOfMemory,
        format!("out of memory reserving {what}"),
    )
}

#[cfg(not(target_os = "ios"))]
fn readahead_budget() -> usize {
    usize::MAX
}

impl BinaryHNSW {
    /// Create a new index with default HNSW parameters.
    pub fn new() -> Self {
        Self::with_params(DEFAULT_M, DEFAULT_EF_CONSTRUCTION, DEFAULT_EF_SEARCH)
    }

    /// Create an index with explicit parameters.
    pub fn with_params(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let m = m.max(2);
        let m_l = 1.0 / (m as f64).ln();
        Self {
            nodes: Vec::new(),
            arena: Arena::Owned(Vec::new()),
            max_layers: 0,
            m,
            m_l,
            ef_construction,
            ef_search: SharedUsize::new(ef_search),
            enter_point: None,
        }
    }

    /// Set query-time beam width (`efSearch`) without rebuilding the graph.
    /// Atomic — safe under a shared read lock.
    pub fn set_ef_search(&self, ef_search: usize) {
        self.ef_search.set(ef_search.max(1));
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// User-facing label (id) of the node at an internal index, as returned
    /// by `search`/`search_with_ef` result tuples. Used by eval harnesses
    /// that need to map internal indices back to corpus ids.
    pub fn node_label(&self, idx: u32) -> u64 {
        self.nodes[idx as usize].id
    }

    /// Read-only access to a node's 512-bit code by internal index.
    pub fn node_hash(&self, idx: u32) -> &Hash512 {
        &self.nodes[idx as usize].hash
    }

    /// Copy of a node's primary neighbor IDs at a layer.
    pub fn neighbors(&self, idx: u32, layer: usize) -> Vec<u32> {
        self.neighbor_ids(idx, layer, true)
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// True if the index contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Approximate resident memory in bytes (virtual size for mmap arenas).
    pub fn memory_bytes(&self) -> u64 {
        let nodes = (self.nodes.len() * core::mem::size_of::<BinaryNode>()) as u64;
        nodes + self.arena.len_bytes() as u64
    }

    /// Hint the kernel to drop the neighbor arena pages immediately.
    /// Call before dropping the index to reduce peak resident memory.
    /// For mmap-backed (v5) arenas this purges clean file pages; they are
    /// re-faulted from the file on next access.
    pub fn advise_dontneed(&self) {
        let len = self.arena.len_bytes();
        if len > 0 {
            let ptr = self.arena.base_ptr();
            unsafe { libc::madvise(ptr, len, libc::MADV_DONTNEED); }
        }
    }

    /// Cold-start warm-up: hint the kernel to prefetch the mmap'd neighbor
    /// arena and run sampled self-hash queries to force-fault the hot graph
    /// pages. Call on a background thread right after loading a v5 index so
    /// the first real query pays no page faults. `n_queries` self-hash probes
    /// are spread evenly across the id space and searched at beam width `ef`.
    /// Restores the previous `ef_search` afterwards; single-threaded CPU cost
    /// is a few seconds for a 10M-node index.
    pub fn warm_up(&self, n_queries: usize, ef: usize) {
        // 1. Async kernel readahead of the neighbor arena — CAPPED. An
        // unbounded MADV_WILLNEED on a multi-GB mmap makes the kernel page
        // the ENTIRE file into resident memory at once; the 5.1M graph
        // (2.5GB edges + 571MB node records) hit 3.54GB resident and the app
        // was jetsam-killed at the iOS per-process limit. Demand paging with
        // a bounded, free-memory-aware readahead keeps startup warm without
        // the memory bomb.
        let len = self.arena.len_bytes().min(readahead_budget());
        if len > 0 {
            let ptr = self.arena.base_ptr();
            unsafe { libc::madvise(ptr, len, libc::MADV_WILLNEED); }
        }

        // 2. Sampled self-hash queries at a wide beam to touch hot pages.
        // Uses search_with_ef — no mutation, so the FFI layer can hold a
        // READ lock and user queries are never blocked by warm-up.
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        let n_queries = n_queries.clamp(1, n);
        let ef = ef.max(1);
        let step = (n / n_queries).max(1);
        for i in (0..n).step_by(step).take(n_queries) {
            let hash = self.nodes[i].hash;
            let _ = self.search_with_ef(&hash, 1, ef);
        }
    }

    /// Current maximum layer index (top layer).
    pub fn top_layer(&self) -> usize {
        self.max_layers
    }

    /// Number of layers in the graph (top_layer + 1).
    pub fn layer_count(&self) -> usize {
        self.max_layers + 1
    }

    /// Reference to a node by internal index.
    pub fn node(&self, idx: u32) -> Option<&BinaryNode> {
        self.nodes.get(idx as usize)
    }

    /// Retrieve the 512-bit hash of the node with the given user id.
    pub fn hash_by_id(&self, id: u64) -> Option<&Hash512> {
        self.nodes.iter().find(|n| n.id == id).map(|n| &n.hash)
    }

    /// Current entry point (top-layer node index).
    pub fn enter_point(&self) -> Option<u32> {
        self.enter_point
    }

    /// Maximum number of neighbors allowed at a given layer.
    #[inline(always)]
    fn max_neighbors_for_layer(&self, layer: usize) -> usize {
        if layer == 0 { self.m * 2 } else { self.m }
    }

    /// Total neighbor capacity needed for a node that participates in layers
    /// 0..=top_layer.  With multi-edge storage we keep primary + alternative
    /// arenas, so the total capacity is doubled.
    #[inline(always)]
    fn neighbor_capacity(&self, top_layer: usize) -> usize {
        (self.max_neighbors_for_layer(0) + top_layer * self.m) * 2
    }

    /// Total primary-neighbor capacity for a node.
    #[inline(always)]
    fn primary_capacity(&self, top_layer: usize) -> usize {
        self.max_neighbors_for_layer(0) + top_layer * self.m
    }

    /// Reserve contiguous blocks in `neighbor_arena` for primary and
    /// alternative edges and return both offsets.
    fn alloc_neighbors(&mut self, top_layer: usize) -> (u32, u32) {
        let primary_cap = self.primary_capacity(top_layer);
        let total_cap = self.neighbor_capacity(top_layer);
        let arena = self.arena.owned_mut();
        let primary_offset = arena.len();
        let alt_offset = primary_offset + primary_cap;
        arena.resize(primary_offset + total_cap, 0);
        (primary_offset as u32, alt_offset as u32)
    }

    /// Element offset into the arena for a node's primary or alternative
    /// neighbor slot at `layer`.  Uses fixed per-layer capacities so
    /// lower-layer growth never shifts higher-layer storage.
    fn layer_offset(&self, idx: u32, layer: usize, primary: bool) -> usize {
        let node = &self.nodes[idx as usize];
        let start = if primary {
            node.neighbor_start as usize
        } else {
            node.alt_neighbor_start as usize
        };
        start + (0..layer).map(|l| self.max_neighbors_for_layer(l)).sum::<usize>()
    }

    /// Immutable view of a node's primary or alternative neighbors at a layer.
    /// Owned (padded) arenas only — the v5 mmap layout is compact per-layer.
    fn layer_neighbors(&self, idx: u32, layer: usize, primary: bool) -> &[u32] {
        let node = &self.nodes[idx as usize];
        if layer >= node.num_layers as usize {
            return &[];
        }
        let offset = self.layer_offset(idx, layer, primary);
        let count = if primary {
            node.layer_counts[layer]
        } else {
            node.alt_layer_counts[layer]
        } as usize;
        &self.arena.owned_slice()[offset..offset + count]
    }

    /// Mutable view of a node's primary or alternative neighbors at a layer.
    fn layer_neighbors_mut(&mut self, idx: u32, layer: usize, primary: bool) -> &mut [u32] {
        let node = &self.nodes[idx as usize];
        if layer >= node.num_layers as usize {
            return &mut [];
        }
        let offset = self.layer_offset(idx, layer, primary);
        let count = if primary {
            node.layer_counts[layer]
        } else {
            node.alt_layer_counts[layer]
        } as usize;
        &mut self.arena.owned_mut()[offset..offset + count]
    }

    /// Copy a node's primary or alternative neighbor IDs at a layer into a Vec.
    /// This avoids borrow issues when traversing the graph. Works with both
    /// Owned (v4) and Mapped (v5) arenas.
    fn neighbor_ids(&self, idx: u32, layer: usize, primary: bool) -> Vec<u32> {
        match &self.arena {
            Arena::Owned(_) => self.layer_neighbors(idx, layer, primary).to_vec(),
            Arena::Mapped(_) => self.mapped_neighbor_ids(idx, layer, primary),
        }
    }

    /// Read neighbor IDs from a v5 mmap arena. The edges section is a
    /// per-node sequence of (primary_count, primary_edges, alt_count,
    /// alt_edges) for each layer the node participates in — compact, no
    /// padding. `edge_off` is an element offset into the mapped region.
    fn mapped_neighbor_ids(&self, idx: u32, layer: usize, primary: bool) -> Vec<u32> {
        let (elem, elem_len) = match &self.arena {
            Arena::Mapped(m) => (m.elem, m.elem_len),
            Arena::Owned(_) => unreachable!(),
        };
        let node = &self.nodes[idx as usize];
        if layer >= node.num_layers as usize {
            return Vec::new();
        }
        let mut off = node.edge_off as usize;
        if off >= elem_len {
            return Vec::new();
        }
        unsafe {
            for l in 0..node.num_layers as usize {
                let pc = u32::from_le(elem.add(off).read_unaligned()) as usize;
                off += 1;
                if l == layer && primary {
                    let end = off.saturating_add(pc).min(elem_len);
                    let mut out = Vec::with_capacity(end - off);
                    for i in off..end {
                        out.push(u32::from_le(elem.add(i).read_unaligned()));
                    }
                    return out;
                }
                off = off.saturating_add(pc);
                if off >= elem_len {
                    return Vec::new();
                }
                let ac = u32::from_le(elem.add(off).read_unaligned()) as usize;
                off += 1;
                if l == layer {
                    let end = off.saturating_add(ac).min(elem_len);
                    let mut out = Vec::with_capacity(end - off);
                    for i in off..end {
                        out.push(u32::from_le(elem.add(i).read_unaligned()));
                    }
                    return out;
                }
                off = off.saturating_add(ac);
                if off >= elem_len {
                    return Vec::new();
                }
            }
        }
        Vec::new()
    }

    /// Push a forward neighbor (during insertion). Caller must ensure capacity.
    /// `primary` selects the primary or alternative edge arena.
    fn push_forward_neighbor(&mut self, idx: u32, layer: usize, nbr: u32, primary: bool) {
        assert!((idx as usize) < self.nodes.len(), "push_forward idx {} out of {}", idx, self.nodes.len());
        assert!((nbr as usize) < self.nodes.len(), "push_forward nbr {} out of {}", nbr, self.nodes.len());
        let count = if primary {
            self.nodes[idx as usize].layer_counts[layer]
        } else {
            self.nodes[idx as usize].alt_layer_counts[layer]
        } as usize;
        debug_assert!(count < self.max_neighbors_for_layer(layer));
        let offset = self.layer_offset(idx, layer, primary);
        assert!(offset + count < self.arena.len_elems(), "push_forward arena overflow");
        self.arena.owned_mut()[offset + count] = nbr;
        if primary {
            self.nodes[idx as usize].layer_counts[layer] = (count + 1) as u8;
        } else {
            self.nodes[idx as usize].alt_layer_counts[layer] = (count + 1) as u8;
        }
    }

    /// Add a backward neighbor and prune to `cap` closest by Hamming distance.
    /// Runs in O(cap log cap) with at most one small temporary allocation.
    /// `primary` selects the primary or alternative edge arena.
    fn add_backward_neighbor(&mut self, idx: u32, layer: usize, nbr: u32, cap: usize, primary: bool) {
        assert!((idx as usize) < self.nodes.len(), "add_backward idx {} out of {}", idx, self.nodes.len());
        assert!((nbr as usize) < self.nodes.len(), "add_backward nbr {} out of {}", nbr, self.nodes.len());
        let node_hash = self.nodes[idx as usize].hash;
        let current: Vec<u32> = self.layer_neighbors(idx, layer, primary).to_vec();

        let mut scored: Vec<(u32, u32)> = Vec::with_capacity(current.len() + 1);
        for nb in current {
            scored.push((hamming_distance(&node_hash, &self.nodes[nb as usize].hash), nb));
        }
        scored.push((hamming_distance(&node_hash, &self.nodes[nbr as usize].hash), nbr));

        scored.sort_by_key(|x| x.0);
        // Deduplicate (shouldn't happen, but cheap insurance).
        let mut seen = HashSet::with_capacity(scored.len());
        scored.retain(|x| seen.insert(x.1));
        scored.truncate(cap);

        let offset = self.layer_offset(idx, layer, primary);
        for (i, (_, nb)) in scored.iter().enumerate() {
            self.arena.owned_mut()[offset + i] = *nb;
        }
        if primary {
            self.nodes[idx as usize].layer_counts[layer] = scored.len() as u8;
        } else {
            self.nodes[idx as usize].alt_layer_counts[layer] = scored.len() as u8;
        }
    }

    /// Insert a new point with the default tag 0.
    pub fn insert(&mut self, id: u64, hash: Hash512) {
        self.insert_with_tag(id, hash, 0);
    }

    /// Insert a new point with an explicit beacon tag.
    pub fn insert_with_tag(&mut self, id: u64, hash: Hash512, tag: u8) {
        if self.nodes.is_empty() {
            let (start, alt_start) = self.alloc_neighbors(0);
            let mut layer_counts = [0u8; MAX_LAYERS];
            let mut alt_layer_counts = [0u8; MAX_LAYERS];
            layer_counts[0] = 0;
            alt_layer_counts[0] = 0;
            self.nodes.push(BinaryNode {
                id,
                hash,
                tag,
                num_layers: 1,
                neighbor_start: start,
                layer_counts,
                alt_neighbor_start: alt_start,
                alt_layer_counts,
                edge_off: 0,
            });
            self.enter_point = Some(0);
            self.max_layers = 0;
            return;
        }

        let new_level = self.random_level();
        let new_idx = self.nodes.len() as u32;
        let (start, alt_start) = self.alloc_neighbors(new_level);
        self.nodes.push(BinaryNode {
            id,
            hash,
            tag,
            num_layers: (new_level + 1) as u8,
            neighbor_start: start,
            layer_counts: [0u8; MAX_LAYERS],
            alt_neighbor_start: alt_start,
            alt_layer_counts: [0u8; MAX_LAYERS],
            edge_off: 0,
        });

        let top_layer = self.max_layers;
        let mut ep = self.enter_point.unwrap();

        // Descend from the top layer down to one above the new node's layer.
        for layer in (new_level + 1..=top_layer).rev() {
            let nearest = self.search_layer(&hash, ep, 1, layer);
            ep = nearest[0].1;
        }

        // Connect the new node in every layer it participates in.
        let start_layer = new_level.min(top_layer);
        for layer in (0..=start_layer).rev() {
            let candidates = self.search_layer(&hash, ep, self.ef_construction, layer);
            let selected = self.select_neighbors(&candidates, self.m * 2);

            // Primary edges: first M candidates.
            for &(_, nb_idx) in selected.iter().take(self.m) {
                self.push_forward_neighbor(new_idx, layer, nb_idx, /*primary=*/true);
                let cap = self.max_neighbors_for_layer(layer);
                self.add_backward_neighbor(nb_idx, layer, new_idx, cap, /*primary=*/true);
            }

            // Alternative edges: second M candidates (different hash "angle").
            for &(_, nb_idx) in selected.iter().skip(self.m).take(self.m) {
                self.push_forward_neighbor(new_idx, layer, nb_idx, /*primary=*/false);
                let cap = self.max_neighbors_for_layer(layer);
                self.add_backward_neighbor(nb_idx, layer, new_idx, cap, /*primary=*/false);
            }

            if let Some((_, best_idx)) = selected.first() {
                ep = *best_idx;
            }
        }

        if new_level > top_layer {
            self.max_layers = new_level;
            self.enter_point = Some(new_idx);
        }
    }

    /// Search for the `k` nearest neighbors of `query`.
    pub fn search(&self, query: &Hash512, k: usize) -> Vec<(u32, u32, u8)> {
        self.search_with_ef(query, k, self.ef_search.get())
    }

    /// Search with an explicit beam width, ignoring the stored `ef_search`.
    /// Does not mutate the index, so it can run under a shared read lock
    /// (used by warm-up — see `yp_hnsw_warm_up`).
    pub fn search_with_ef(&self, query: &Hash512, k: usize, ef: usize) -> Vec<(u32, u32, u8)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(self.nodes.len());
        let mut ep = self.enter_point.unwrap();

        // Descend from the top layer down to layer 1.
        for layer in (1..=self.max_layers).rev() {
            let nearest = self.search_layer(query, ep, 1, layer);
            ep = nearest[0].1;
        }

        // Final search at layer 0 with the full beam width.
        let mut results = self.search_layer(query, ep, ef.max(k), 0);
        results.truncate(k);
        results
            .into_iter()
            .map(|(dist, idx)| (dist, idx, self.nodes[idx as usize].tag))
            .collect()
    }

    /// Search starting from a given node with explicit ef (phyllotactic entry test).
    pub fn search_with_ef_from(&self, query: &Hash512, k: usize, ef: usize, entry_point: u32) -> Vec<(u32, u32, u8)> {
        if self.is_empty() || k == 0 || (entry_point as usize) >= self.nodes.len() {
            return self.search_with_ef(query, k, ef);
        }
        let k = k.min(self.nodes.len());
        let mut ep = entry_point;
        for layer in (1..=self.max_layers).rev() {
            let nearest = self.search_layer(query, ep, 1, layer);
            ep = nearest[0].1;
        }
        let mut results = self.search_layer(query, ep, ef.max(k), 0);
        results.truncate(k);
        results.into_iter().map(|(dist, idx)| (dist, idx, self.nodes[idx as usize].tag)).collect()
    }

    /// Geometry-aware pruning: cut nb only if it is redundant in BOTH hamming
    /// space AND the PCA-2 sketch plane. Hamming-redundant but geometrically
    /// diverse edges survive -> deeper cuts at equal recall tax.
    /// `geo_r` = sketch-distance gate (e.g., RMS radius of the coords).
    pub fn prune_diverse_geo(&mut self, alpha: f32, layer: usize, sketch: &[f32], geo_r: f32) -> usize {
        let sd2 = |a: u32, b: u32| -> f32 {
            let i = a as usize * 2;
            let j = b as usize * 2;
            let dx = sketch[i] - sketch[j];
            let dy = sketch[i + 1] - sketch[j + 1];
            (dx * dx + dy * dy).sqrt()
        };
        let mut removed = 0usize;
        for idx in 0..self.nodes.len() as u32 {
            if layer >= self.nodes[idx as usize].num_layers as usize {
                continue;
            }
            let (kept, old_len) = {
                let node_hash = &self.nodes[idx as usize].hash;
                let nbs = self.neighbor_ids(idx, layer, true);
                if nbs.len() <= 2 {
                    continue;
                }
                let mut by_dist: Vec<(u32, u32)> = nbs
                    .iter()
                    .map(|&nb| (hamming_distance(node_hash, &self.nodes[nb as usize].hash), nb))
                    .collect();
                by_dist.sort_unstable();
                let mut kept: Vec<u32> = Vec::with_capacity(by_dist.len());
                for (d, nb) in by_dist {
                    let nb_hash = &self.nodes[nb as usize].hash;
                    let redundant = kept.iter().any(|&p| {
                        (hamming_distance(nb_hash, &self.nodes[p as usize].hash) as f32) < alpha * (d as f32)
                            && sd2(nb, p) < geo_r
                    });
                    if !redundant {
                        kept.push(nb);
                    }
                }
                (kept, nbs.len())
            };
            if kept.len() < old_len {
                let slot = self.layer_neighbors_mut(idx, layer, true);
                let n = kept.len().min(slot.len());
                for (i, &nb) in kept.iter().take(n).enumerate() {
                    slot[i] = nb;
                }
                self.nodes[idx as usize].layer_counts[layer] = n as u8;
                removed += old_len - n;
            }
        }
        removed
    }

    /// Append one-way "bridge" edges into freed layer-0 primary slots (nodes
    /// that were pruned below capacity). `bridge_of(node_idx)` returns the
    /// bridge target (e.g., its coarse-cell anchor). Returns edges added.
    pub fn add_bridge_edges<F: Fn(usize) -> u32>(&mut self, n: usize, bridge_of: F) -> usize {
        let cap = self.max_neighbors_for_layer(0);
        let mut added = 0usize;
        for idx in 0..n as u32 {
            let node = &self.nodes[idx as usize];
            if node.num_layers == 0 || node.layer_counts[0] as usize >= cap {
                continue;
            }
            let b = bridge_of(idx as usize);
            if b == u32::MAX || b == idx {
                continue;
            }
            if self.neighbor_ids(idx, 0, true).contains(&b) {
                continue;
            }
            let off = self.layer_offset(idx, 0, true);
            let cnt = self.nodes[idx as usize].layer_counts[0] as usize;
            self.arena.owned_mut()[off + cnt] = b;
            self.nodes[idx as usize].layer_counts[0] += 1;
            added += 1;
        }
        added
    }

    /// Vamana-style diverse-neighbor pruning (post-build, owned/v4 arenas).
    /// For each node's primary neighbor list at `layer`: sort neighbors by
    /// hamming distance to the node, then greedily drop any neighbor whose
    /// hamming distance to an already-kept neighbor is < alpha * dist(node, nb).
    /// Fewer redundant edges -> tighter ef-beam -> fewer hops per query.
    /// Returns the number of edges removed.
    pub fn prune_diverse(&mut self, alpha: f32, layer: usize, reverse: bool) -> usize {
        let mut removed = 0usize;
        for idx in 0..self.nodes.len() as u32 {
            if layer >= self.nodes[idx as usize].num_layers as usize {
                continue;
            }
            let (kept, old_len) = {
                let node_hash = &self.nodes[idx as usize].hash;
                let nbs = self.neighbor_ids(idx, layer, true);
                if nbs.len() <= 2 {
                    continue;
                }
                let mut by_dist: Vec<(u32, u32)> = nbs
                    .iter()
                    .map(|&nb| (hamming_distance(node_hash, &self.nodes[nb as usize].hash), nb))
                    .collect();
                if reverse {
                    // reverse-distance order: longest edges kept first (control for order bias)
                    by_dist.sort_unstable_by(|a, b| b.0.cmp(&a.0));
                } else {
                    by_dist.sort_unstable();
                }
                let mut kept: Vec<u32> = Vec::with_capacity(by_dist.len());
                for (d, nb) in by_dist {
                    let nb_hash = &self.nodes[nb as usize].hash;
                    // d==0 (identical ITQ codes): thresh 1 dedupes exact duplicates too
                    let thresh = if d == 0 { 1u32 } else { (alpha * (d as f32)) as u32 };
                    let redundant = kept.iter().any(|&p| {
                        hamming_distance(nb_hash, &self.nodes[p as usize].hash) < thresh
                    });
                    if !redundant {
                        kept.push(nb);
                    }
                }
                (kept, nbs.len())
            };
            if kept.len() < old_len {
                let slot = self.layer_neighbors_mut(idx, layer, true);
                let n = kept.len().min(slot.len());
                for (i, &nb) in kept.iter().take(n).enumerate() {
                    slot[i] = nb;
                }
                self.nodes[idx as usize].layer_counts[layer] = n as u8;
                removed += old_len - n;
            }
        }
        removed
    }

    /// Diameter-aware adaptive pruning (GLM council design). The naive rule
    /// cuts long edges preferentially (RHS grows with edge length), destroying
    /// shortcuts. Here alpha_eff = base_alpha / (dist_ratio + 0.1): short
    /// (redundant local) edges get a large alpha_eff -> cut aggressively;
    /// long (shortcut) edges get a small alpha_eff -> protected. A min-degree
    /// floor guarantees connectivity. Returns edges removed.
    pub fn prune_diverse_adaptive(&mut self, base_alpha: f32, layer: usize, min_degree: usize) -> usize {
        let mut removed = 0usize;
        for idx in 0..self.nodes.len() as u32 {
            if layer >= self.nodes[idx as usize].num_layers as usize {
                continue;
            }
            let (kept, old_len) = {
                let node_hash = &self.nodes[idx as usize].hash;
                let nbs = self.neighbor_ids(idx, layer, true);
                if nbs.len() <= min_degree.max(2) {
                    continue;
                }
                let mut by_dist: Vec<(u32, u32)> = nbs
                    .iter()
                    .map(|&nb| (hamming_distance(node_hash, &self.nodes[nb as usize].hash), nb))
                    .collect();
                by_dist.sort_unstable();
                let median = by_dist[by_dist.len() / 2].0.max(1) as f32;
                let mut kept: Vec<u32> = Vec::with_capacity(by_dist.len());
                for (d, nb) in by_dist {
                    let dist_ratio = (d as f32) / median;
                    let alpha_eff = base_alpha / (dist_ratio + 0.1);
                    let nb_hash = &self.nodes[nb as usize].hash;
                    let redundant = kept.iter().any(|&p| {
                        (hamming_distance(nb_hash, &self.nodes[p as usize].hash) as f32)
                            < alpha_eff * (d as f32)
                    });
                    if !redundant || kept.len() < min_degree {
                        kept.push(nb);
                    }
                }
                (kept, nbs.len())
            };
            if kept.len() < old_len {
                let slot = self.layer_neighbors_mut(idx, layer, true);
                let n = kept.len().min(slot.len());
                for (i, &nb) in kept.iter().take(n).enumerate() {
                    slot[i] = nb;
                }
                self.nodes[idx as usize].layer_counts[layer] = n as u8;
                removed += old_len - n;
            }
        }
        removed
    }

    /// Flat search: skip the upper-layer descent entirely and start the
    /// layer-0 beam directly from `entry_point` (compass-entry experiment).
    pub fn search_flat_from(&self, query: &Hash512, k: usize, ef: usize, entry_point: u32) -> Vec<(u32, u32, u8)> {
        if self.is_empty() || k == 0 || (entry_point as usize) >= self.nodes.len() {
            return self.search_with_ef(query, k, ef);
        }
        let k = k.min(self.nodes.len());
        let mut results = self.search_layer(query, entry_point, ef.max(k), 0);
        results.truncate(k);
        results.into_iter().map(|(dist, idx)| (dist, idx, self.nodes[idx as usize].tag)).collect()
    }

    /// Search starting from `entry_point` instead of the default random entry point.
    /// Mirrors the standard HNSW greedy descent but with a warm-start node.
    pub fn search_from(
        &self,
        query: &Hash512,
        k: usize,
        entry_point: u32,
    ) -> Vec<(u32, u32, u8)> {
        if self.is_empty() || k == 0 || (entry_point as usize) >= self.nodes.len() {
            return self.search(query, k);
        }

        let k = k.min(self.nodes.len());
        let mut ep = entry_point;

        // Descend from the top layer down to layer 1.
        for layer in (1..=self.max_layers).rev() {
            let nearest = self.search_layer(query, ep, 1, layer);
            ep = nearest[0].1;
        }

        // Final search at layer 0 with the full beam width.
        let mut results = self.search_layer(query, ep, self.ef_search.get().max(k), 0);
        results.truncate(k);
        results
            .into_iter()
            .map(|(dist, idx)| (dist, idx, self.nodes[idx as usize].tag))
            .collect()
    }

    /// Greedy / beam search within a single layer.
    fn search_layer(
        &self,
        query: &Hash512,
        ep: u32,
        ef: usize,
        layer: usize,
    ) -> Vec<(u32, u32)> {
        let ef = ef.max(1);
        HOPS.with(|c| c.set(c.get() + 1));
        LAYER_SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let (visited, candidates, found) = &mut *scratch;
            visited.clear();
            candidates.clear();
            found.clear();

            let ep_dist = fused_score(ep, hamming_distance(query, &self.nodes[ep as usize].hash));
            visited.insert(ep);
            candidates.push(Reverse((ep_dist, ep)));
            found.push((ep_dist, ep));

            // Shared per-neighbor scan. `count_hops` preserves the original
            // HOPS semantics (primary edges only). First pass issues software
            // prefetches for the node lines we are about to touch (M1).
            let scan =
                |nbs: &[u32],
                 count_hops: bool,
                 visited: &mut hashbrown::HashSet<u32>,
                 candidates: &mut alloc::collections::BinaryHeap<Reverse<(u32, u32)>>,
                 found: &mut alloc::collections::BinaryHeap<(u32, u32)>| {
                    for &nb in nbs {
                        prefetch_read(&self.nodes[nb as usize] as *const BinaryNode);
                    }
                    for &nb in nbs {
                        if (nb as usize) >= self.nodes.len() {
                            panic!("search_layer found invalid neighbor {} (nodes={})", nb, self.nodes.len());
                        }
                        if visited.insert(nb) {
                            if count_hops {
                                HOPS.with(|c| c.set(c.get() + 1));
                            }
                            let nd = fused_score(
                                nb,
                                hamming_distance(query, &self.nodes[nb as usize].hash),
                            );
                            let should_add = found.len() < ef || nd < found.peek().unwrap().0;
                            if should_add {
                                candidates.push(Reverse((nd, nb)));
                                found.push((nd, nb));
                                if found.len() > ef {
                                    found.pop();
                                }
                            }
                        }
                    }
                };

            while let Some(Reverse((cd, ci))) = candidates.pop() {
                if found.len() >= ef {
                    let worst = found.peek().unwrap().0;
                    if cd > worst {
                        break;
                    }
                }

                match &self.arena {
                    Arena::Owned(_) => {
                        let nbs = self.layer_neighbors(ci, layer, /*primary=*/true);
                        scan(nbs, true, visited, candidates, found);
                        // Also probe alternative edges to increase effective branching.
                        let nbs = self.layer_neighbors(ci, layer, /*primary=*/false);
                        scan(nbs, false, visited, candidates, found);
                    }
                    Arena::Mapped(_) => {
                        let nbs = self.mapped_neighbor_ids(ci, layer, /*primary=*/true);
                        scan(&nbs, true, visited, candidates, found);
                        let nbs = self.mapped_neighbor_ids(ci, layer, /*primary=*/false);
                        scan(&nbs, false, visited, candidates, found);
                    }
                }
            }

            let mut results: Vec<(u32, u32)> = found.drain().collect();
            results.sort_by_key(|x| x.0);
            results
        })
    }

    /// Select the `m` closest candidates from a sorted candidate list.
    fn select_neighbors(&self, candidates: &[(u32, u32)], m: usize) -> Vec<(u32, u32)> {
        candidates.iter().take(m).copied().collect()
    }

    /// Sample a layer for a new node using the HNSW layer distribution.
    fn random_level(&self) -> usize {
        let mut level = 0;
        let mut rng = rand::rng();
        while level < MAX_LAYERS - 1 && rng.random_bool(self.m_l) {
            level += 1;
        }
        level
    }

    /// Save index to disk in compact binary format (version 4, multi-edge).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        const MAGIC: &[u8] = b"YPH5";
        const VERSION: u8 = 4;
        file.write_all(MAGIC)?;
        file.write_all(&[VERSION])?;
        file.write_all(&(self.m as u64).to_le_bytes())?;
        file.write_all(&(self.ef_construction as u64).to_le_bytes())?;
        file.write_all(&(self.ef_search.get() as u64).to_le_bytes())?;
        file.write_all(&(self.max_layers as u64).to_le_bytes())?;
        let ep = self.enter_point.unwrap_or(u32::MAX);
        file.write_all(&ep.to_le_bytes())?;
        file.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        for (idx, node) in self.nodes.iter().enumerate() {
            file.write_all(&node.id.to_le_bytes())?;
            file.write_all(&node.hash)?;
            file.write_all(&[node.tag])?;
            file.write_all(&[node.num_layers])?;
            file.write_all(&node.layer_counts)?;
            file.write_all(&node.alt_layer_counts)?;
            for layer in 0..node.num_layers as usize {
                // Primary edges
                let count = node.layer_counts[layer] as u32;
                file.write_all(&count.to_le_bytes())?;
                let offset = self.layer_offset(idx as u32, layer, true);
                for i in 0..count as usize {
                    file.write_all(&self.arena.owned_slice()[offset + i].to_le_bytes())?;
                }
                // Alternative edges
                let alt_count = node.alt_layer_counts[layer] as u32;
                file.write_all(&alt_count.to_le_bytes())?;
                let alt_offset = self.layer_offset(idx as u32, layer, false);
                for i in 0..alt_count as usize {
                    file.write_all(&self.arena.owned_slice()[alt_offset + i].to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Load index from disk. Sniffs the format version: v4 files are parsed
    /// into an in-memory (owned) arena; v5 files are memory-mapped so the
    /// edge pages fault in on demand and multi-GB indexes do not exhaust
    /// device memory at load time.
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"YPH5" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut version = [0u8; 1];
        file.read_exact(&mut version)?;
        match version[0] {
            4 => {
                use std::io::BufReader;
                file.seek(SeekFrom::Start(0))?;
                let mut reader = BufReader::with_capacity(1024 * 1024, file);
                Self::load_from_reader(&mut reader)
            }
            5 => Self::load_v5(path),
            x => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported version {}", x),
            )),
        }
    }

    /// Load a v5 (mmap-backed) index: node records are read into RAM; the
    /// edges section stays mapped in the address space and is paged in by
    /// the kernel on first touch.
    fn load_v5(path: &str) -> std::io::Result<Self> {
        const HEADER_LEN: usize = 56;
        const REC_LEN: usize = 112;
        let mut mapped = MappedArena::map(std::path::Path::new(path))?;
        let base = mapped.base;
        if mapped.map_len < HEADER_LEN {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "v5 file too small"));
        }
        unsafe fn r_u64(p: *const libc::c_void, off: usize) -> u64 {
            let mut b = [0u8; 8];
            core::ptr::copy_nonoverlapping((p as *const u8).add(off), b.as_mut_ptr(), 8);
            u64::from_le_bytes(b)
        }
        unsafe fn r_u32(p: *const libc::c_void, off: usize) -> u32 {
            let mut b = [0u8; 4];
            core::ptr::copy_nonoverlapping((p as *const u8).add(off), b.as_mut_ptr(), 4);
            u32::from_le_bytes(b)
        }

        let m = unsafe { r_u64(base, 8) } as usize;
        let ef_construction = unsafe { r_u64(base, 16) } as usize;
        let ef_search = unsafe { r_u64(base, 24) } as usize;
        let max_layers = unsafe { r_u64(base, 32) } as usize;
        let ep_raw = unsafe { r_u32(base, 40) };
        let enter_point = if ep_raw == u32::MAX { None } else { Some(ep_raw) };
        let node_count = unsafe { r_u64(base, 44) } as usize;

        let recs_off = HEADER_LEN;
        let edges_off = REC_LEN.checked_mul(node_count)
            .and_then(|v| v.checked_add(HEADER_LEN))
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "v5 node count overflows"))?;
        if mapped.map_len < edges_off {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "v5 file truncated (nodes)"));
        }

        let mut nodes: Vec<BinaryNode> = Vec::new();
        // Fallible allocation: `with_capacity` aborts the process when the
        // allocator fails (iOS Rust default) — that was the 2026-09-08 10:31
        // "benchmark run all datasets" freeze/crash (SIGABRT in load_v5).
        // try_reserve returns an error that yp_load_index_chunk already
        // surfaces as rc=0 + rustError() to the UI.
        nodes.try_reserve_exact(node_count).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::OutOfMemory,
                format!("not enough memory for {node_count} HNSW nodes ({})",
                    std::any::type_name::<BinaryNode>()))
        })?;
        for i in 0..node_count {
            let rec = unsafe { base.add(recs_off + i * REC_LEN) };
            let id = unsafe { r_u64(rec, 0) };
            let mut hash = [0u8; HASH512_BYTES];
            unsafe { core::ptr::copy_nonoverlapping((rec as *const u8).add(8), hash.as_mut_ptr(), HASH512_BYTES); }
            let tag = unsafe { *(rec as *const u8).add(72) };
            let num_layers = unsafe { *(rec as *const u8).add(73) };
            let mut layer_counts = [0u8; MAX_LAYERS];
            unsafe { core::ptr::copy_nonoverlapping((rec as *const u8).add(74), layer_counts.as_mut_ptr(), MAX_LAYERS); }
            let mut alt_layer_counts = [0u8; MAX_LAYERS];
            unsafe { core::ptr::copy_nonoverlapping((rec as *const u8).add(90), alt_layer_counts.as_mut_ptr(), MAX_LAYERS); }
            let edge_off = unsafe { r_u32(rec, 106) };
            nodes.push(BinaryNode {
                id,
                hash,
                tag,
                num_layers,
                neighbor_start: 0,
                layer_counts,
                alt_neighbor_start: 0,
                alt_layer_counts,
                edge_off,
            });
        }

        let elem = unsafe { base.add(edges_off) as *const u32 };
        let elem_len = (mapped.map_len - edges_off) / 4;
        mapped.elem = elem;
        mapped.elem_len = elem_len;
        let arena = Arena::Mapped(mapped);
        let m_l = 1.0 / (m.max(2) as f64).ln();
        Ok(Self {
            nodes,
            arena,
            max_layers,
            m,
            m_l,
            ef_construction,
            ef_search: SharedUsize::new(ef_search),
            enter_point,
        })
    }

    /// Save index in v5 mmap-friendly format: fixed-size node records with
    /// absolute edge offsets, followed by a compact (unpadded) edge blob
    /// section. Loading v5 memory-maps the file and faults edges on demand.
    /// Requires an Owned (in-memory) arena.
    pub fn save_v5(&self, path: &str) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        const HEADER_LEN: usize = 56;
        const REC_LEN: usize = 112;
        let n = self.nodes.len();

        // Per-node edge offsets (element offset into the edges section).
        let mut offs = Vec::with_capacity(n);
        let mut run: usize = 0;
        for node in &self.nodes {
            offs.push(run as u32);
            for l in 0..node.num_layers as usize {
                run += 2 + node.layer_counts[l] as usize + node.alt_layer_counts[l] as usize;
            }
        }

        let mut w = BufWriter::with_capacity(8 * 1024 * 1024, std::fs::File::create(path)?);
        w.write_all(b"YPH5")?;
        w.write_all(&[5u8])?;
        w.write_all(&[0u8; 3])?;
        w.write_all(&(self.m as u64).to_le_bytes())?;
        w.write_all(&(self.ef_construction as u64).to_le_bytes())?;
        w.write_all(&(self.ef_search.get() as u64).to_le_bytes())?;
        w.write_all(&(self.max_layers as u64).to_le_bytes())?;
        w.write_all(&self.enter_point.unwrap_or(u32::MAX).to_le_bytes())?;
        w.write_all(&(n as u64).to_le_bytes())?;
        w.write_all(&[0u8; 4])?;

        let mut rec = [0u8; REC_LEN];
        for (i, node) in self.nodes.iter().enumerate() {
            rec[0..8].copy_from_slice(&node.id.to_le_bytes());
            rec[8..72].copy_from_slice(&node.hash);
            rec[72] = node.tag;
            rec[73] = node.num_layers;
            rec[74..90].copy_from_slice(&node.layer_counts);
            rec[90..106].copy_from_slice(&node.alt_layer_counts);
            rec[106..110].copy_from_slice(&offs[i].to_le_bytes());
            rec[110] = 0;
            rec[111] = 0;
            w.write_all(&rec)?;
        }

        for (idx, node) in self.nodes.iter().enumerate() {
            for l in 0..node.num_layers as usize {
                let p = self.layer_neighbors(idx as u32, l, true);
                w.write_all(&(p.len() as u32).to_le_bytes())?;
                for e in p {
                    w.write_all(&e.to_le_bytes())?;
                }
                let a = self.layer_neighbors(idx as u32, l, false);
                w.write_all(&(a.len() as u32).to_le_bytes())?;
                for e in a {
                    w.write_all(&e.to_le_bytes())?;
                }
            }
        }
        w.flush()?;
        let _ = HEADER_LEN;
        Ok(())
    }

    /// Parse a v4-serialized index from a contiguous byte buffer.
    /// (v5 indexes are mmap-backed and must be loaded via `load`.)
    /// Validates only the header magic/version and basic size/trailing-byte checks.
    pub fn load_from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        use std::io::Cursor;
        Self::load_from_reader(&mut Cursor::new(bytes))
    }

    /// Parse a serialized index from any `Read` source.
    fn load_from_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        fn read_u64<R: std::io::Read>(reader: &mut R) -> std::io::Result<u64> {
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(u64::from_le_bytes(buf))
        }
        fn read_u32<R: std::io::Read>(reader: &mut R) -> std::io::Result<u32> {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        }
        fn read_byte<R: std::io::Read>(reader: &mut R) -> std::io::Result<u8> {
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(buf[0])
        }

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"YPH5" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
        }

        let version = read_byte(reader)?;
        if version != 4 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported version"));
        }

        let m = read_u64(reader)? as usize;
        let ef_construction = read_u64(reader)? as usize;
        let ef_search = read_u64(reader)? as usize;
        let max_layers = read_u64(reader)? as usize;
        let ep_raw = read_u32(reader)?;
        let enter_point = if ep_raw == u32::MAX { None } else { Some(ep_raw) };
        let node_count = read_u64(reader)? as usize;

        // Fallible allocation: on a memory-tight device (large corpus switch
        // with the previous engine's pages not yet reclaimed) a plain
        // with_capacity/resize aborts the whole process via rust_oom. Return
        // an io error instead so the caller surfaces a clean load failure.
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(node_count)
            .map_err(|_| io_error_oom("HNSW nodes"))?;
        let mut neighbor_arena = Vec::new();

        for _ in 0..node_count {
            let id = read_u64(reader)?;
            let mut hash = [0u8; HASH512_BYTES];
            reader.read_exact(&mut hash)?;
            let tag = read_byte(reader)?;
            let num_layers = read_byte(reader)?;
            let mut layer_counts = [0u8; MAX_LAYERS];
            reader.read_exact(&mut layer_counts)?;
            let mut alt_layer_counts = [0u8; MAX_LAYERS];
            reader.read_exact(&mut alt_layer_counts)?;

            let top_layer = num_layers.saturating_sub(1) as usize;
            let primary_cap = if num_layers == 0 { 0 } else { m * 2 + top_layer * m };
            let total_cap = primary_cap * 2;
            let start = neighbor_arena.len();
            let alt_start = start + primary_cap;
            let additional = (start + total_cap) - neighbor_arena.len();
            if neighbor_arena.try_reserve(additional).is_err() {
                return Err(io_error_oom("HNSW neighbor arena"));
            }
            neighbor_arena.resize(start + total_cap, 0);

            let mut write_offset = 0usize;
            let mut alt_write_offset = 0usize;
            for layer in 0..num_layers as usize {
                // Primary edges
                let count = read_u32(reader)? as usize;
                let slot_capacity = Self::max_neighbors_for_layer_static(layer, m);
                if count > slot_capacity {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "layer neighbor count exceeds capacity"));
                }
                for i in 0..count {
                    neighbor_arena[start + write_offset + i] = read_u32(reader)?;
                }
                write_offset += slot_capacity;

                // Alternative edges
                let alt_count = read_u32(reader)? as usize;
                if alt_count > slot_capacity {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "alt layer neighbor count exceeds capacity"));
                }
                for i in 0..alt_count {
                    neighbor_arena[alt_start + alt_write_offset + i] = read_u32(reader)?;
                }
                alt_write_offset += slot_capacity;
            }

            nodes.push(BinaryNode {
                id,
                hash,
                tag,
                num_layers,
                neighbor_start: start as u32,
                layer_counts,
                alt_neighbor_start: alt_start as u32,
                alt_layer_counts,
                edge_off: 0,
            });
        }

        let m_l = 1.0 / (m as f64).ln();
        Ok(Self {
            nodes,
            arena: Arena::Owned(neighbor_arena),
            max_layers,
            m,
            m_l,
            ef_construction,
            ef_search: SharedUsize::new(ef_search),
            enter_point,
        })
    }

    #[inline(always)]
    fn max_neighbors_for_layer_static(layer: usize, m: usize) -> usize {
        if layer == 0 { m * 2 } else { m }
    }

    /// Warm-start: load from disk if valid, else build fresh.
    pub fn warm_start(path: &str, m: usize, ef_construction: usize, ef_search: usize) -> Self {
        match Self::load(path) {
            Ok(idx) => {
                eprintln!("[BinaryHNSW] Warm-start loaded {} nodes from {}", idx.len(), path);
                idx
            }
            Err(e) => {
                eprintln!("[BinaryHNSW] Warm-start failed ({}), building fresh", e);
                Self::with_params(m, ef_construction, ef_search)
            }
        }
    }
}

impl Default for BinaryHNSW {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_hash() -> Hash512 {
        let mut h = [0u8; HASH512_BYTES];
        rand::rng().fill(&mut h);
        h
    }

    #[test]
    fn test_insert_and_search() {
        let mut idx = BinaryHNSW::new();
        let target = random_hash();
        idx.insert(0, target);
        let res = idx.search(&target, 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].1, 0);
        assert_eq!(res[0].2, 0);
    }

    #[test]
    fn test_batch_insert() {
        let mut idx = BinaryHNSW::new();
        for i in 0..100 {
            idx.insert(i as u64, random_hash());
        }
        assert_eq!(idx.len(), 100);
    }

    #[test]
    fn test_self_query() {
        let mut idx = BinaryHNSW::with_params(16, 200, 100);
        let mut hashes = Vec::new();
        for i in 0..10_000 {
            let h = random_hash();
            idx.insert(i as u64, h);
            hashes.push(h);
        }
        assert_eq!(idx.len(), 10_000);
        let mut hits = 0;
        for (i, h) in hashes.iter().enumerate().take(100) {
            let res = idx.search(h, 1);
            if res.first().map(|r| r.1).unwrap_or(u32::MAX) == i as u32 {
                hits += 1;
            }
        }
        assert!(hits >= 95, "self-query hits: {hits}");
    }

    #[test]
    fn search_returns_nearest_neighbors() {
        let mut idx = BinaryHNSW::with_params(16, 200, 100);
        let query = random_hash();
        let mut nearest_idx = 0;
        let mut nearest_dist = u32::MAX;
        for i in 0..1_000 {
            let h = random_hash();
            let d = hamming_distance(&query, &h);
            if d < nearest_dist {
                nearest_dist = d;
                nearest_idx = i;
            }
            idx.insert(i as u64, h);
        }
        let res = idx.search(&query, 10);
        assert!(!res.is_empty());
        let found = res.iter().any(|&(_, idx, _)| idx == nearest_idx);
        assert!(found, "brute-force nearest not in top-10 HNSW result");
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut idx = BinaryHNSW::with_params(16, 64, 32);
        let mut hashes = Vec::new();
        for i in 0..1_000 {
            let h = random_hash();
            idx.insert(i as u64, h);
            hashes.push(h);
        }
        idx.save("/tmp/binary_hnsw_test.bin").unwrap();
        let idx2 = BinaryHNSW::load("/tmp/binary_hnsw_test.bin").unwrap();
        assert_eq!(idx2.len(), 1_000);
        assert_eq!(idx2.m, 16);
        assert_eq!(idx2.ef_construction, 64);
        assert_eq!(idx2.ef_search.get(), 32);
        for (i, h) in hashes.iter().enumerate().take(10) {
            let r1 = idx.search(h, 1);
            let r2 = idx2.search(h, 1);
            assert_eq!(r1, r2, "query mismatch for node {}", i);
        }
    }

    #[test]
    fn test_warm_start_valid() {
        let mut idx = BinaryHNSW::with_params(16, 64, 32);
        for i in 0..100 {
            idx.insert(i as u64, random_hash());
        }
        idx.save("/tmp/binary_hnsw_warm.bin").unwrap();
        let idx2 = BinaryHNSW::warm_start("/tmp/binary_hnsw_warm.bin", 16, 64, 32);
        assert_eq!(idx2.len(), 100);
    }

    #[test]
    fn test_warm_start_missing_file() {
        let idx = BinaryHNSW::warm_start("/tmp/nonexistent_yph5.bin", 16, 64, 32);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_corrupt_file_rejected() {
        use std::io::Write;
        let mut file = std::fs::File::create("/tmp/binary_hnsw_corrupt.bin").unwrap();
        file.write_all(b"BAD!").unwrap();
        drop(file);
        let result = BinaryHNSW::load("/tmp/binary_hnsw_corrupt.bin");
        assert!(result.is_err(), "corrupt file should be rejected");
    }
}
