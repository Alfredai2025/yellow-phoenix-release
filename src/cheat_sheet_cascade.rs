// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use std::ffi::CStr;
use std::fs::File;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::time::{Instant, Duration};
use std::ffi::{c_char, CString};
use std::slice;

pub type BinaryVector = [u8; 64];
pub type VectorId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Signature {
    pub hash: u64,
    pub popcount: u16,
    pub residue_mod3: u8,
    pub residue_mod4: u8,
    pub residue_mod6: u8,
    pub residue_mod8: u8,
}

impl Signature {
    pub fn from_vector(v: &BinaryVector) -> Self {
        let hash = fnv1a_hash(v);
        let popcount = popcount_512(v);
        let sum: u64 = v.iter().map(|&b| b as u64).sum();
        Self {
            hash,
            popcount,
            residue_mod3: (sum % 3) as u8,
            residue_mod4: (sum % 4) as u8,
            residue_mod6: (sum % 6) as u8,
            residue_mod8: (sum % 8) as u8,
        }
    }
    pub fn residues(&self) -> [u8; 4] {
        [self.residue_mod3, self.residue_mod4, self.residue_mod6, self.residue_mod8]
    }
}

fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn popcount_512(v: &[u8; 64]) -> u16 {
    v.iter().map(|&b| b.count_ones() as u16).sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel { Residue, Prime, PapBand }

#[derive(Debug, Clone, Default)]
pub struct CascadeStats {
    pub total_queries: u64,
    pub fast_hits: u64,
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l3_hits: u64,
    pub exact_fallbacks: u64,
    pub total_time_us: f64,
}

impl CascadeStats {
    pub fn avg_time_us(&self) -> f64 {
        if self.total_queries == 0 { 0.0 } else { self.total_time_us / self.total_queries as f64 }
    }
    pub fn hit_rate(&self) -> f64 {
        if self.total_queries == 0 { 0.0 } else {
            (self.fast_hits + self.l1_hits + self.l2_hits + self.l3_hits) as f64 / self.total_queries as f64
        }
    }
    pub fn exact_rate(&self) -> f64 {
        if self.total_queries == 0 { 0.0 } else { self.exact_fallbacks as f64 / self.total_queries as f64 }
    }
}

#[derive(Debug, Clone)]
pub struct CheatSheetLayer {
    mappings: HashMap<u64, (Vec<u64>, u64, u64)>,
    channel: Channel,
    signature_fn: fn(&Signature) -> u64,
}

impl CheatSheetLayer {
    pub fn new(channel: Channel, sig_fn: fn(&Signature) -> u64) -> Self {
        Self { mappings: HashMap::new(), channel, signature_fn: sig_fn }
    }

    pub fn learn(&mut self, sig: &Signature, cell: u64) {
        let key = (self.signature_fn)(sig);
        let entry = self.mappings.entry(key).or_insert((Vec::new(), 0, 0));
        entry.0.push(cell);
        entry.1 += 1;
        entry.2 += 1;
        if self.mappings.len() > 10_000_000 {
            self.prune();
        }
    }

    pub fn predict(&self, sig: &Signature) -> Option<(Vec<u64>, f64)> {
        let key = (self.signature_fn)(sig);
        self.mappings.get(&key).map(|(cells, hits, total)| {
            let confidence = *hits as f64 / *total as f64;
            let unique: std::collections::HashSet<u64> = cells.iter().cloned().collect();
            (unique.into_iter().collect(), confidence)
        })
    }

    fn prune(&mut self) {
        let mut ratios: Vec<(u64, f64)> = self.mappings.iter().map(|(k, v)| {
            let ratio = v.1 as f64 / v.2 as f64;
            (*k, ratio)
        }).collect();
        ratios.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let to_remove = ratios.len() / 10;
        for i in 0..to_remove {
            self.mappings.remove(&ratios[i].0);
        }
    }

    pub fn len(&self) -> usize { self.mappings.len() }
}

#[derive(Debug, Clone)]
pub struct CheatSheetCascade {
    pub layer1: CheatSheetLayer,
    pub layer2: CheatSheetLayer,
    pub layer3: CheatSheetLayer,
    pub stats: CascadeStats,
}

impl CheatSheetCascade {
    pub fn new() -> Self {
        Self {
            layer1: CheatSheetLayer::new(Channel::Residue, |sig: &Signature| {
                let mut hasher = DefaultHasher::new();
                sig.residues().hash(&mut hasher);
                hasher.finish()
            }),
            layer2: CheatSheetLayer::new(Channel::Prime, |sig: &Signature| {
                let mut hasher = DefaultHasher::new();
                let pc = sig.popcount as u64;
                (pc % 5, pc % 7, sig.hash % 11, sig.hash % 13).hash(&mut hasher);
                hasher.finish()
            }),
            layer3: CheatSheetLayer::new(Channel::PapBand, |sig: &Signature| {
                let mut hasher = DefaultHasher::new();
                let band = (sig.popcount as usize / 32) as u64;
                let h_frag = sig.hash >> 32;
                (band, h_frag).hash(&mut hasher);
                hasher.finish()
            }),
            stats: CascadeStats::default(),
        }
    }

    pub fn query(&self, sig: &Signature) -> (Vec<u64>, Option<usize>, f64) {
        let start = Instant::now();

        if let Some((cells, conf)) = self.layer1.predict(sig) {
            if conf >= 0.85 {
                let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
                return (cells, Some(1), elapsed);
            }
        }

        if let Some((cells, conf)) = self.layer2.predict(sig) {
            if conf >= 0.85 {
                let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
                return (cells, Some(2), elapsed);
            }
        }

        if let Some((cells, conf)) = self.layer3.predict(sig) {
            if conf >= 0.85 {
                let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
                return (cells, Some(3), elapsed);
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1_000_000.0;
        (Vec::new(), None, elapsed)
    }

    pub fn learn(&mut self, sig: &Signature, fast_cells: &[u64], exact_cell: u64) {
        if !fast_cells.contains(&exact_cell) {
            self.layer1.learn(sig, exact_cell);
            self.layer2.learn(sig, exact_cell);
            self.layer3.learn(sig, exact_cell);
        }
    }

    pub fn memory_usage(&self) -> usize {
        let l1 = self.layer1.len() * (16 + 32);
        let l2 = self.layer2.len() * (16 + 32);
        let l3 = self.layer3.len() * (16 + 32);
        l1 + l2 + l3
    }
}

pub struct HybridOrchestrator {
    pub cascade: CheatSheetCascade,
    pub stats: CascadeStats,
}

impl HybridOrchestrator {
    pub fn new() -> Self {
        Self { cascade: CheatSheetCascade::new(), stats: CascadeStats::default() }
    }

    pub fn query(&mut self, vector: &BinaryVector) -> (Vec<u64>, Option<usize>, f64) {
        let start = Instant::now();
        let sig = Signature::from_vector(vector);
        let (cascade_cells, cascade_layer, _cascade_time) = self.cascade.query(&sig);

        if !cascade_cells.is_empty() {
            let layer = cascade_layer.unwrap_or(1);
            self.record_hit(layer, start.elapsed());
            let elapsed = Self::to_micros(start.elapsed());
            return (cascade_cells, Some(layer), elapsed);
        }

        let exact_results = vec![0u64];
        self.record_exact_fallback(start.elapsed());
        let elapsed = Self::to_micros(start.elapsed());
        (exact_results, None, elapsed)
    }

    fn record_hit(&mut self, layer: usize, elapsed: Duration) {
        self.stats.total_queries += 1;
        match layer {
            1 => self.stats.l1_hits += 1,
            2 => self.stats.l2_hits += 1,
            3 => self.stats.l3_hits += 1,
            _ => {}
        }
        self.stats.total_time_us += Self::to_micros(elapsed);
    }

    fn record_exact_fallback(&mut self, elapsed: Duration) {
        self.stats.total_queries += 1;
        self.stats.exact_fallbacks += 1;
        self.stats.total_time_us += Self::to_micros(elapsed);
    }

    fn to_micros(d: Duration) -> f64 { d.as_secs_f64() * 1_000_000.0 }

    pub fn get_stats(&self) -> &CascadeStats { &self.stats }

    pub fn print_stats(&self) {
        let s = &self.stats;
        let total = s.total_queries as f64;
        if total == 0.0 {
            println!("No queries yet.");
            return;
        }
        println!("\n=== YP v3 Cheat Sheet Stats ===");
        println!("Total queries: {}", s.total_queries);
        println!("L1 hits: {} ({:.1}%)", s.l1_hits, 100.0 * s.l1_hits as f64 / total);
        println!("L2 hits: {} ({:.1}%)", s.l2_hits, 100.0 * s.l2_hits as f64 / total);
        println!("L3 hits: {} ({:.1}%)", s.l3_hits, 100.0 * s.l3_hits as f64 / total);
        println!("Exact fallbacks: {} ({:.1}%)", s.exact_fallbacks, 100.0 * s.exact_fallbacks as f64 / total);
        println!("Hit rate: {:.1}%", s.hit_rate() * 100.0);
        println!("Avg time: {:.2} us", s.avg_time_us());
        println!("Memory: ~{} MB", self.cascade.memory_usage() / 1_048_576);
    }
}

// ============================================================================
// OPTIMIZED FFI — Raw integers, no JSON, batch support
// ============================================================================
static mut ORCHESTRATOR: Option<HybridOrchestrator> = None;

#[no_mangle]
pub extern "C" fn yp_cascade_init() {
    unsafe { unsafe { ORCHESTRATOR = Some(HybridOrchestrator::new()); } }
}

// Single query: returns cell directly, layer in out param
#[no_mangle]
pub extern "C" fn yp_cascade_query_fast(
    vector: *const u8,
    len: usize,
    out_cell: *mut u64,
    out_layer: *mut i32,
) -> f64 {
    if vector.is_null() || len != 64 || out_cell.is_null() || out_layer.is_null() {
        return -1.0;
    }
    let bytes = unsafe { slice::from_raw_parts(vector, len) };
    let mut arr = [0u8; 64];
    arr.copy_from_slice(bytes);
    
    unsafe {
        if ORCHESTRATOR.is_none() { return -1.0; }
        let orch = ORCHESTRATOR.as_mut().unwrap();
        let (cells, layer, time_us) = orch.query(&arr);
        *out_cell = cells.first().copied().unwrap_or(0);
        *out_layer = match layer {
            Some(1) => 1,
            Some(2) => 2,
            Some(3) => 3,
            _ => 0,
        };
        time_us
    }
}

#[no_mangle]
pub extern "C" fn yp_cascade_learn(vector: *const u8, len: usize, exact_cell: u64) {
    if vector.is_null() || len != 64 { return; }
    let bytes = unsafe { slice::from_raw_parts(vector, len) };
    let mut arr = [0u8; 64];
    arr.copy_from_slice(bytes);
    let sig = Signature::from_vector(&arr);
    unsafe {
        if let Some(ref mut orch) = ORCHESTRATOR {
            let (fast_cells, _, _) = orch.cascade.query(&sig);
            orch.cascade.learn(&sig, &fast_cells, exact_cell);
        }
    }
}

// Helper for batch query
fn duration_to_micros(d: Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

// Batch query: n vectors, returns n cells + n layers + total time
#[no_mangle]
pub extern "C" fn yp_cascade_query_batch(
    vectors: *const u8,      // flat: n * 64 bytes
    n: usize,
    out_cells: *mut u64,      // n uint64s
    out_layers: *mut i32,    // n int32s
) -> f64 {
    if vectors.is_null() || out_cells.is_null() || out_layers.is_null() || n == 0 {
        return -1.0;
    }
    let total_start = Instant::now();
    let all_bytes = unsafe { slice::from_raw_parts(vectors, n * 64) };
    
    unsafe {
        if ORCHESTRATOR.is_none() { return -1.0; }
        let orch = ORCHESTRATOR.as_mut().unwrap();
        
        for i in 0..n {
            let offset = i * 64;
            let arr = &all_bytes[offset..offset + 64];
            let mut v = [0u8; 64];
            v.copy_from_slice(arr);
            let (cells, layer, _) = orch.query(&v);
            *out_cells.add(i) = cells.first().copied().unwrap_or(0);
            *out_layers.add(i) = match layer {
                Some(1) => 1,
                Some(2) => 2,
                Some(3) => 3,
                _ => 0,
            };
        }
    }
    duration_to_micros(total_start.elapsed())
}

#[no_mangle]
pub extern "C" fn yp_cascade_stats_json() -> *mut c_char {
    unsafe {
        if ORCHESTRATOR.is_none() {
            return CString::new("{\"error\":\"not initialized\"}").unwrap().into_raw();
        }
        let orch = ORCHESTRATOR.as_ref().unwrap();
        let s = &orch.stats;
        let json = format!(
            "{{\"total\":{},\"l1\":{},\"l2\":{},\"l3\":{},\"exact\":{},\"hit_rate\":{:.4},\"avg_us\":{:.2}}}",
            s.total_queries, s.l1_hits, s.l2_hits, s.l3_hits, s.exact_fallbacks,
            s.hit_rate(), s.avg_time_us()
        );
        CString::new(json).unwrap().into_raw()
    }
}

#[no_mangle]
pub extern "C" fn yp_free_string(s: *mut c_char) {
    if s.is_null() { return; }
    unsafe { let _ = CString::from_raw(s); }
}

#[no_mangle]
pub extern "C" fn yp_cascade_print_stats() {
    unsafe {
        if let Some(ref orch) = ORCHESTRATOR {
            orch.print_stats();
        }
    }
}

// ============================================================================
// SERVICE API — for yp_cascade_service binary
// ============================================================================
use std::sync::{Mutex, Once};
use std::io::Write;
use std::io::Read;

static SERVICE_INIT: Once = Once::new();
static mut SERVICE_ORCH: Option<Mutex<HybridOrchestrator>> = None;

pub fn service_init() {
    SERVICE_INIT.call_once(|| {
        unsafe {
            SERVICE_ORCH = Some(Mutex::new(HybridOrchestrator::new()));
        }
    });
}

pub fn service_query(vector: &BinaryVector) -> (u64, i32, f64) {
    let mut orch = unsafe { SERVICE_ORCH.as_ref().unwrap().lock().unwrap() };
    let (cells, layer, time_us) = orch.query(vector);
    let cell = cells.first().copied().unwrap_or(0);
    let layer_i = match layer {
        Some(1) => 1,
        Some(2) => 2,
        Some(3) => 3,
        _ => 0,
    };
    (cell, layer_i, time_us)
}

pub fn service_learn(vector: &BinaryVector, exact_cell: u64) {
    let mut orch = unsafe { SERVICE_ORCH.as_ref().unwrap().lock().unwrap() };
    let sig = Signature::from_vector(vector);
    let (fast_cells, _, _) = orch.cascade.query(&sig);
    orch.cascade.learn(&sig, &fast_cells, exact_cell);
}

pub fn service_print_stats() {
    let orch = unsafe { SERVICE_ORCH.as_ref().unwrap().lock().unwrap() };
    orch.print_stats();
}


#[no_mangle]
pub extern "C" fn yp_cascade_save(path: *const c_char) -> i32 {
    let path = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let mut file = match File::create(&*path) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    
    // Header: YPCS + version 2
    if file.write_all(b"YPCS").is_err() { return -2; }
    if file.write_all(&2u32.to_le_bytes()).is_err() { return -3; }
    
    let mut orch = match unsafe { ORCHESTRATOR.as_mut() } {
        Some(o) => o,
        None => return -5,
    };
    
    // Write 3 layers
    let layers = [&orch.cascade.layer1, &orch.cascade.layer2, &orch.cascade.layer3];
    if file.write_all(&(layers.len() as u32).to_le_bytes()).is_err() { return -6; }
    
    for layer in &layers {
        let n = layer.mappings.len() as u64;
        if file.write_all(&n.to_le_bytes()).is_err() { return -7; }
        for (key, (cells, hits, total)) in layer.mappings.iter() {
            if file.write_all(&key.to_le_bytes()).is_err() { return -8; }
            if file.write_all(&hits.to_le_bytes()).is_err() { return -9; }
            if file.write_all(&total.to_le_bytes()).is_err() { return -10; }
            let cell_n = cells.len() as u64;
            if file.write_all(&cell_n.to_le_bytes()).is_err() { return -11; }
            for cell in cells {
                if file.write_all(&cell.to_le_bytes()).is_err() { return -12; }
            }
        }
    }
    
    0
}

#[no_mangle]
pub extern "C" fn yp_cascade_load(path: *const c_char) -> i32 {
    let path = unsafe { CStr::from_ptr(path).to_string_lossy() };
    let mut file = match File::open(&*path) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    
    let mut buf = [0u8; 4];
    if file.read_exact(&mut buf).is_err() { return -2; }
    if &buf != b"YPCS" { return -3; }
    
    let mut ver = [0u8; 4];
    if file.read_exact(&mut ver).is_err() { return -4; }
    let version = u32::from_le_bytes(ver);
    if version != 2 { return -5; }
    
    // Ensure orchestrator is initialized
    if unsafe { ORCHESTRATOR.is_none() } {
        unsafe { ORCHESTRATOR = Some(HybridOrchestrator::new()); }
    }
    
    let mut orch = match unsafe { ORCHESTRATOR.as_mut() } {
        Some(o) => o,
        None => return -7,
    };
    
    let mut layer_count_buf = [0u8; 4];
    if file.read_exact(&mut layer_count_buf).is_err() { return -8; }
    let layer_count = u32::from_le_bytes(layer_count_buf) as usize;
    if layer_count != 3 { return -9; }
    
    // Helper: populate a CheatSheetLayer from file
    fn read_layer(file: &mut std::fs::File, mut layer: CheatSheetLayer) -> Result<CheatSheetLayer, i32> {
        let mut n_buf = [0u8; 8];
        if file.read_exact(&mut n_buf).is_err() { return Err(-10); }
        let n = u64::from_le_bytes(n_buf) as usize;
        
        for _ in 0..n {
            let mut key_buf = [0u8; 8];
            if file.read_exact(&mut key_buf).is_err() { return Err(-11); }
            let key = u64::from_le_bytes(key_buf);
            
            let mut hits_buf = [0u8; 8];
            if file.read_exact(&mut hits_buf).is_err() { return Err(-12); }
            let hits = u64::from_le_bytes(hits_buf);
            
            let mut total_buf = [0u8; 8];
            if file.read_exact(&mut total_buf).is_err() { return Err(-13); }
            let total = u64::from_le_bytes(total_buf);
            
            let mut cell_n_buf = [0u8; 8];
            if file.read_exact(&mut cell_n_buf).is_err() { return Err(-14); }
            let cell_n = u64::from_le_bytes(cell_n_buf) as usize;
            
            let mut cells = Vec::with_capacity(cell_n);
            for _ in 0..cell_n {
                let mut cell_buf = [0u8; 8];
                if file.read_exact(&mut cell_buf).is_err() { return Err(-15); }
                cells.push(u64::from_le_bytes(cell_buf));
            }
            layer.mappings.insert(key, (cells, hits, total));
        }
        Ok(layer)
    }
    
    // Read each layer with correct channel + signature function
    let layer1 = CheatSheetLayer::new(Channel::Residue, |sig: &Signature| {
        let mut hasher = DefaultHasher::new();
        sig.residues().hash(&mut hasher);
        hasher.finish()
    });
    let layer1 = match read_layer(&mut file, layer1) { Ok(l) => l, Err(e) => return e };
    
    let layer2 = CheatSheetLayer::new(Channel::Prime, |sig: &Signature| {
        let mut hasher = DefaultHasher::new();
        let pc = sig.popcount as u64;
        (pc % 5, pc % 7, sig.hash % 11, sig.hash % 13).hash(&mut hasher);
        hasher.finish()
    });
    let layer2 = match read_layer(&mut file, layer2) { Ok(l) => l, Err(e) => return e };
    
    let layer3 = CheatSheetLayer::new(Channel::PapBand, |sig: &Signature| {
        let mut hasher = DefaultHasher::new();
        let band = (sig.popcount as usize / 32) as u64;
        let h_frag = sig.hash >> 32;
        (band, h_frag).hash(&mut hasher);
        hasher.finish()
    });
    let layer3 = match read_layer(&mut file, layer3) { Ok(l) => l, Err(e) => return e };
    
    orch.cascade.layer1 = layer1;
    orch.cascade.layer2 = layer2;
    orch.cascade.layer3 = layer3;
    
    0
}


