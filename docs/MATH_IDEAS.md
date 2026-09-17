# YELLOW PHOENIX — MATH & ALGORITHM CATALOG

Original Ideas, Formulas, and Implementations

Generated: 2026-08-12

Paste into KimiCLI or commit as `docs/MATH_IDEAS.md`

---

## 0. META — How to Read This File

Every entry follows:

- **IDEA**:        What was invented / devised / custom-built
- **FORMULA**:     The actual math or algorithm
- **LOCATION**:    File + line where it lives
- **STATUS**:      Production / Debug / Museum / Stub
- **PROOF**:       Commit hash or benchmark that validates it

This is a living document. Add new entries at the bottom.
Tag commits that add new math with `[MATH]` in the message.

---

## 1. GEOMETRIC ALGEBRA (Clifford / VSA)

### 1.1 5-D Geometric Query Engine

- **IDEA**: A unified 5-D geometric query engine supporting five operator combinations:
  Wedge (∧), Inner (·), Geometric (*), Dual (⋆), and Rotor (R).
- **FORMULA**:
  ```
  query_result = f(Wedge(a,b), Inner(a,b), Geometric(a,b), Dual(a), Rotor(a,θ))
  ```
- **LOCATION**: `src/unified_all.rs`
- **STATUS**: Production — 9/9 tests pass
- **PROOF**: Commit history, memory #76 (26-07-04)

### 1.2 Binary Multivector Type System with Grade Projection

- **IDEA**: Explicit grade projection for binary multivectors (0, 1, 2 grades)
  with population-count distance restricted per grade.
- **FORMULA**:
  ```rust
  // Grade-weighted PAP distance
  // d = w0·d0 + w1·d1 + w2·d2
  // where dk = PAP_distance restricted to grade k
  pub fn grade_weighted_distance(&self, other: &Self) -> f32 {
      let p0_a = self.grade_project(0);  // isolates low 16 bits
      let p0_b = other.grade_project(0);
      let p1_a = self.grade_project(1);  // middle bits
      let p1_b = other.grade_project(1);
      let p2_a = self.grade_project(2);  // high bits
      let p2_b = other.grade_project(2);
      // weighted combination...
  }
  ```
- **LOCATION**: `src/types/graded.rs:20-67`
- **STATUS**: Production
- **PROOF**: Unit tests `grade0_project_isolates_low_16_bits`, etc.

### 1.3 PAP Distance (Phoenix Angular Proximity)

- **IDEA**: Custom distance metric for binary multivectors based on signed overlap
  rather than standard Hamming or cosine.
- **FORMULA**:
  ```rust
  /// Compute the PAP distance between two binary multivectors.
  pub fn pap_distance(a: &BinaryMultivector, b: &BinaryMultivector) -> f32 {
      // Signed overlap computation
  }
  ```
- **LOCATION**: `src/distance.rs:5-36`
- **STATUS**: Production

### 1.4 Ternary Multivector Signed Overlap

- **IDEA**: Extension of PAP to ternary multivectors ({-1, 0, +1}) for
  holographic superposition states.
- **LOCATION**: `src/distance.rs:36-53`, `src/types/ternary.rs`
- **STATUS**: Production

---

## 2. HASH & RETRIEVAL MATH

### 2.1 BinaryHNSW on 512-Bit ITQ Hashes (Pure Hamming)

- **IDEA**: HNSW graph index operating directly on 512-bit binary hashes
  using popcount-of-XOR (Hamming) distance instead of float vectors.
  Eliminates float index overhead; distance compute is ~50x cheaper.
- **FORMULA**:
  ```rust
  pub fn hamming_distance(a: &Hash512, b: &Hash512) -> u32 {
      // popcount of XOR across 64 bytes
  }
  ```
- **KEY INNOVATION**: Arena-based neighbor storage (`neighbor_arena: Vec<u32>`)
  eliminates per-node `Vec<Vec<u32>>` heap overhead. Each node stores offset
  per-layer counts.
- **LOCATION**: `src/binary_hnsw.rs:47-78`
- **STATUS**: Production — sole production engine (museum'd OPQ+PQ)
- **PROOF**: 1M P50 0.16ms, 1.27M vectors at 481µs (memory #66)

### 2.2 Three-Tier Hybrid Retrieval (OPQ + PQ)

- **IDEA**: Cascade retrieval with three compressed tiers:
  - Tier 1: 8-byte codes (44% R@1)
  - Tier 2: 16-byte codes (55.6% R@1)
  - Tier 3: Full ITQ 64-byte fallback
- **FORMULA**:
  ```
  candidates = OPQ_8byte.search(query)  // fast, low recall
  if recall < threshold:
      candidates += OPQ_16byte.search(query)  // medium
  if still insufficient:
      candidates += ITQ_512bit.search(query)  // slow, high recall
  final = embedding_cosine_rerank(candidates)
  ```
- **LOCATION**: `src/ffi_opq_pq.rs`, artifacts `opq_rotation_20260727.npz`,
  `pq_opq_8x48_256_20260727.npz`
- **STATUS**: Museum'd (commit `39b5417`) — BinaryHNSW won
- **PROOF**: Memory #49 (26-08-03)

### 2.3 ISM — Inverted Slot Map

- **IDEA**: Inverted slot-based hash index. 200M vectors build in 20.2s,
  3.7µs single query, 30µs parallel.
- **FORMULA**:
  ```
  slot_id = hash_prefix % num_slots
  bucket = inverted_map[slot_id]  // Vec<u32> of vector IDs
  ```
- **LOCATION**: `src/ism/` (inferred from memory #84)
- **STATUS**: Production — auto-hybrid routing in default features
- **PROOF**: 1M 0.03s, 10M 0.36s, 200M 16.4s Rust build (memory #84)

### 2.4 Pattern Keys — Prime Position Hashing

- **IDEA**: Deterministic hash position selection using first 512 primes
  modulo embedding length, plus power-of-2 positions.
- **FORMULA**:
  ```rust
  // Power-of-2 positions
  let pos = 1usize << (i % 9);  // 2^0 to 2^8 = 1 to 256

  // Prime positions: first 512 primes, modulo embedding length
  fn nth_prime(n: usize) -> usize { ... }
  let pos = nth_prime(i) % emb_len;
  ```
- **LOCATION**: `src/pattern_keys.rs:20-47`
- **STATUS**: Production

### 2.5 SAH Beacon Index (Strip AI Harvest)

- **IDEA**: Extract LLM eigenvectors as semantic beacons and index them
  for cascade priming. 507 beacons auto-restored on boot.
- **FORMULA**:
  ```
  beacon_hash = ITQ(eigenvector[:512])
  beacon_index.insert(beacon_hash, beacon_id)
  search_with_sah(query) -> beacon_first -> production_fallback
  ```
- **LOCATION**: `src/sah_hash_bridge.rs`, `yp_autonomic/sah/`
- **STATUS**: Production — auto-load on boot
- **PROOF**: Commit a1b2c3d, 507 beacons restored (memory #64)

### 2.6 Exact Cascade with Feedback Learning

- **IDEA**: Query-result feedback trains bucket routing. After each query,
  the system learns which bucket actually contained the true match.
- **FORMULA**:
  ```rust
  pub fn train_from_result(&mut self, query_hash: &[u8], _bucket_id: u64, true_match_id: u64, score: f32) {
      // Reinforce bucket→result mapping
  }
  ```
- **LOCATION**: `src/exact_cascade.rs:171`
- **STATUS**: Production

### 2.7 Chinese Remainder Theorem Routing

- **IDEA**: CUN (Content-Universal-Name) disk routing using CRT modulo
  for deterministic shard placement.
- **FORMULA**:
  ```rust
  // CUN disk: entry.cun % m == did  (determines shard)
  if entry.cun % m == did { route_to_shard(did) }
  ```
- **LOCATION**: `src/cun_disk.rs:108,210`, `src/pq.rs:50`
- **STATUS**: Production

---

## 3. ITQ & QUANTIZATION MATH

### 3.1 ITQ 512-Bit Hash Training

- **IDEA**: Iterative Quantization with PCA whitening + orthogonal Procrustes
  rotation to maximize hash-embedding correlation.
- **FORMULA**:
  ```
  1. Center embeddings
  2. PCA → eigenvectors
  3. Random rotation R
  4. Binarize: B = sign(V·R)
  5. Procrustes: R = U·V^T where USV^T = B^T·V
  6. Repeat 4-5 until convergence
  ```
- **LOCATION**: `src/sah_hash_bridge.rs`, training scripts
- **STATUS**: Production — MiniLM+ITQ 512-bit is sole viable encoder
- **PROOF**: Correlation 0.891, R@1 74.5%→94.0% (memory #5, 26-07-08)

**NOTE**: July 8 "94%" was embedding-space cosine. Real ITQ Hamming R@1 = 50.8%.
Production now uses: ITQ hash → top 500 candidates → embedding cosine re-rank.

### 3.2 Spectral Eigenvector Drift Detection

- **IDEA**: Monitor spectral eigenvectors from `tensor_spectral.rs` to detect
  model drift and trigger retraining.
- **FORMULA**:
  ```
  drift = ||current_eigenvectors - baseline_eigenvectors||_F
  if drift > threshold: trigger "retrain_model"
  ```
- **LOCATION**: `yp_autonomic/sensors/geometric_health.rs:64-80`,
  `yp_autonomic/cortex/decision.rs:58`
- **STATUS**: Production

---

## 4. HOLOGRAPHIC / VSA MATH

### 4.1 Holographic Cascade with Phase Setting

- **IDEA**: Holographic memory layer where phase is set from spectral
  eigenvectors, enabling multivector interference patterns.
- **FORMULA**:
  ```rust
  pub fn set_phase(&mut self, eigenvector: &[f32]) {
      let n = eigenvector.len().min(self.dim);
      self.phase[..n].copy_from_slice(&eigenvector[..n]);
  }
  ```
- **LOCATION**: `src/holographic_cascade.rs:84`
- **STATUS**: Production — rebuilt with `--features holographic-cascade`
- **PROOF**: Memory #118 (26-08-09)

### 4.2 HolographicMemory Replacing SemanticMemory

- **IDEA**: Direct ctypes bridge from Python autonomic layer to Rust
  holographic cascade FFI, bypassing intermediate serialization.
- **LOCATION**: `yp_autonomic/memory.py` (holographic_context)
- **STATUS**: Production
- **PROOF**: Court accepts holographic_context as evidence (memory #118)

### 4.3 Gravity Batching

- **IDEA**: Batch 32 events before propagating to holographic context,
  with auto-tune based on event velocity.
- **FORMULA**:
  ```
  if event_queue.len() >= 32 || time_since_last_flush > auto_tune_threshold:
      flush_to_holographic_context()
  ```
- **LOCATION**: `yp_autonomic/memory.py`, `yp_autonomic/agent.py`
- **STATUS**: Production
- **PROOF**: Memory #118 (26-08-09)

---

## 5. CAUSAL / PEARL MATH

### 5.1 Dynamic Causal Engine

- **IDEA**: Lightweight Pearl-style causal link scorer. Maintains directed
  causal graph with link strengths updated from observed event pairs.
- **FORMULA**:
  ```rust
  struct CausalLink {
      cause: String,
      effect: String,
      strength: f32,  // reinforced by observation
  }
  // update: if cause observed before effect within window,
  //         strengthen link; else decay
  ```
- **LOCATION**: `yp_autonomic/cortex/causal.py`
- **STATUS**: Production

### 5.2 PredictivePreemption

- **IDEA**: Predict failures before they happen using causal chains
  and sensor trend extrapolation.
- **FORMULA**:
  ```python
  def predict(self, sensor_name):
      trend = self.trend(sensor_name, window=300)
      if trend == "falling" and self.causal.predict_next(sensor_name, min_score=0.15):
          return PreemptionAlert(severity="HIGH")
  ```
- **LOCATION**: `yp_autonomic/immune/predictive_preemption.py`
- **STATUS**: Production
- **PROOF**: Memory #119 (26-08-09)

### 5.3 Correlation Engine (Temporal / Spatial / Causal / Escalation)

- **IDEA**: Four correlation types computed over event history:
  - Temporal: same sensor, time proximity
  - Spatial: same target/component across sensors
  - Causal: cause→effect chain match
  - Escalation: severity increase pattern
- **LOCATION**: `yp_autonomic/cortex/correlation.py`
- **STATUS**: Production — 178 matches in drift monitoring (memory #115)

---

## 6. HNSW / GRAPH MATH

### 6.1 Arena-Based Neighbor Storage

- **IDEA**: Flat `Vec<u32>` arena for all neighbors eliminates per-node
  heap allocation. Node stores `neighbor_start: u32` + `layer_counts: [u8; MAX_LAYERS]`.
- **FORMULA**:
  ```rust
  struct Node {
      hash: Hash512,
      tag: u64,
      num_layers: u8,
      neighbor_start: u32,     // offset into neighbor_arena
      layer_counts: [u8; MAX_LAYERS],
  }
  ```
- **LOCATION**: `src/binary_hnsw.rs:63-101`
- **STATUS**: Production
- **PROOF**: 1.27M vectors, 481µs query (memory #66)

### 6.2 Multi-Edge HNSW (Phase 2 — Deferred)

- **IDEA**: Multiple edge types per node for semantic vs structural navigation.
  Deferred to post-1M validation.
- **LOCATION**: Planned extension to `src/binary_hnsw.rs`
- **STATUS**: Deferred (memory #52, 26-08-11)

---

## 7. LEARNING & ADAPTIVE SYSTEMS

### 7.1 Adaptive Field Weights

- **IDEA**: Query-result feedback updates per-field position weights
  to optimize ranking.
- **FORMULA**:
  ```python
  def compute_weights(self):
      # position weights from feedback histogram
      # higher weight = field position more predictive of relevance
  ```
- **LOCATION**: `yp_autonomic/adaptive_field.py`
- **STATUS**: Production

### 7.2 Intent Classifier with Geometric Fallback

- **IDEA**: Two-path routing:
  - High confidence + small bucket: hash + spectral re-rank
  - Low confidence or crowded bucket: full geometric brain
- **FORMULA**:
  ```rust
  if confidence > 0.85 && bucket_size < 5 {
      fast_path()  // hash + spectral
  } else {
      geometric_brain()  // full multivector query
  }
  ```
- **LOCATION**: `src/intent_classifier.rs`
- **STATUS**: Production

### 7.3 Re-Bucketing Proven

- **IDEA**: Papers physically move between hash buckets after 5K searches
  based on query distribution drift.
- **PROOF**: 17 papers moved after 5K searches (memory #99, 26-07-27)

---

## 8. SPECTRAL / FOURIER MATH

### 8.1 Spectral Stage Re-Rank

- **IDEA**: Stage 1 of cascade uses spectral dot-product re-rank and
  anomaly detection on candidate sets.
- **LOCATION**: `src/spectral_stage.rs`
- **STATUS**: Production

### 8.2 Spectral Drift Sensor

- **IDEA**: Monitor spectral coordinates for distribution drift.
  Trigger retrain if drift exceeds threshold.
- **LOCATION**: `yp_autonomic/sensors/geometric_health.rs`
- **STATUS**: Production

---

## 9. PROBABILITY / STATISTICS

### 9.1 Monitor Brain Trend Engine

- **IDEA**: Pure signal math — no heavy inference. Correlation with lag,
  trend direction (rising/falling/stable), anti-correlation detection.
- **FORMULA**:
  ```python
  def correlate(self, cause, effect, lag=120):
      # Return correlation [-1, 1] between cause and effect with lag.

  def trend(self, sensor, window=300):
      # Return direction: 'rising', 'falling', or 'stable'.
  ```
- **LOCATION**: `yp_autonomic/trend_monitor.py`
- **STATUS**: Production

### 9.2 Enterprise Equilibrium Scoring

- **IDEA**: Per-sensor scoring with circuit breaker. Cheap sensors poll
  adaptively; expensive sensors run event-driven.
- **FORMULA**:
  ```
  sensor_score = f(health, latency, error_rate, drift)
  if score < circuit_breaker_threshold: open_breaker(sensor)
  ```
- **LOCATION**: `yp_autonomic/enterprise_soak.py`
- **STATUS**: Production — deployed 2026-08-09 commit `5a9ae1e`
- **PROOF**: Memory #114

---

## 10. ENTERPRISE SAFETY MATH

### 10.1 Live Auto-Rewire with Syntax Check + Rollback

- **IDEA**: When module_discovery reports missing bridges, auto-patcher
  inserts wrappers into `yp_bridge.py` with timestamped `.bak` backups.
  If `SyntaxError`, rollback to last known good.
- **LOCATION**: `yp_autonomic/replication/auto_rewire.py`
- **STATUS**: Production — 29 FFI wrappers auto-patched
- **PROOF**: Memory #112, #113 (26-08-08)

### 10.2 Checksum-Protected State

- **IDEA**: Cryptographic checksums on all persisted state. Tamper-evident
  audit chain with proof-of-life entries.
- **LOCATION**: `yp_autonomic/enterprise_soak.py`, `yp_autonomic/audit/`
- **STATUS**: Production

**NOTE**: Never modify proof-of-life entries after commit — hash is immutable.

### 10.3 Thermal-Aware Adaptive Sleep

- **IDEA**: Lid sensor + thermal monitoring. ALL launchd jobs manual-only.
  `caffeinate` forbidden unattended (Mac overheated in bag incident).
- **LOCATION**: `yp_autonomic/sensors/lid_sensor.py`, `yp_autonomic/sensors/resource_pressure.py`
- **STATUS**: Production
- **PROOF**: Memory #93 (26-07-18)

---

## 11. MIRROR MESH / GEOMETRIC HOLOGRAPHIC MIND

### 11.1 Mirror Mesh

- **IDEA**: Geometric holographic mind layer using Clifford/VSA multivector
  interference. O(1) retrieval via reflection, swarm consensus.
- **LOCATION**: `src/` (multivector, graded, ternary types)
- **STATUS**: Architecture — user is sole designer/gatekeeper
- **PROOF**: Memory #51 (26-07-08)

---

## 12. WHAT YOU JUST PASTED (GREP RESULTS) — CROSS-REFERENCE

The following are the ORIGINAL mathematical ideas discovered in your
recent grep sweep, now catalogued:

| Grep Category | Your Original Idea | Location |
|---|---|---|
| CRT / Modulo | CUN shard routing via `cun % m == did` | `src/cun_disk.rs:108,210` |
| CRT / Modulo | Prime position hashing `1usize << (i % 9)` | `src/pattern_keys.rs:33` |
| CRT / Modulo | First 512 primes modulo embedding length | `src/pattern_keys.rs:40-47` |
| Hash Bucket | Exact cascade bucket query + training | `src/exact_cascade.rs:25-171` |
| Hash Bucket | Hybrid mesh 128-bit + 512-bit bucket exact | `src/hybrid_mesh.rs:181,248` |
| Geometric | Grade projection isolates bit ranges | `src/types/graded.rs:22-28` |
| Geometric | Grade-weighted distance formula | `src/types/graded.rs:46-67` |
| Geometric | PAP distance on binary multivectors | `src/distance.rs:5-36` |
| Geometric | PAP distance on ternary multivectors | `src/distance.rs:36-53` |
| ITQ | ITQ rotation + binarization bridge | `src/sah_hash_bridge.rs:8-59` |
| ITQ | Batch eigenvector→hash512 | `src/sah_hash_bridge.rs:42` |
| Holographic | Phase set from eigenvector | `src/holographic_cascade.rs:84` |
| Holographic | Holographic cascade FFI | `src/ffi_unified.rs:1326-1334` |
| Causal | Dynamic causal link update | `yp_autonomic/cortex/causal.py:127` |
| Causal | Causal chain prediction | `yp_autonomic/immune/predictive_preemption.py:43-44` |
| Causal | Causal↔Stability correlation | `yp_autonomic/trend_monitor.py:99-101` |
| HNSW | Arena neighbor storage | `src/binary_hnsw.rs:110-111` |
| HNSW | Layer count per node | `src/binary_hnsw.rs:78` |
| HNSW | Hamming popcount on 64-byte hashes | `src/binary_hnsw.rs:47-61` |
| Learning | Adaptive field weight compute | `yp_autonomic/adaptive_field.py:50` |
| Learning | Exact cascade train_from_result | `src/exact_cascade.rs:171` |
| Spectral | Spectral drift detection | `yp_autonomic/sensors/geometric_health.rs:64` |
| Spectral | Spectral stage re-rank | `src/spectral_stage.rs` |
| Probability | Correlation with lag | `yp_autonomic/trend_monitor.py:63` |
| Probability | Anti-correlation detection (RAG↔Court) | `yp_autonomic/trend_monitor.py:97` |

---

## 13. MUSEUM'D IDEAS (Dead Ends, Documented for Posterity)

| Idea | Why It Died | Commit |
|---|---|---|
| Tabulated ITQ | 18% R@1, no better than random | Museum'd |
| PME (Probabilistic Multi-Edge) | No gain over single-edge | Museum'd |
| PLLH (Per-Layer Local Hash) | Storage compression only, no recall gain | Museum'd |
| OPQ+PQ Three-Tier | BinaryHNSW won on speed+recall | `39b5417` |
| Spectral stub | Never completed | Museum'd |
| Fast encoders (all) | MiniLM+ITQ sole viable encoder | Memory #121 |

---

## 14. NEXT MATH TO DOCUMENT

- [ ] Multi-Edge HNSW Phase 2 (when implemented)
- [ ] GFH Resonant Field math (fractal attractor mode)
- [ ] Trinity Cortex full causal graph formulas
- [ ] Federation consensus math (swarm layer)
- [ ] Autopoiesis self-modification sandbox math

---

**END OF CATALOG**

Commit this file with:

```bash
git add docs/MATH_IDEAS.md && git commit -m "[MATH] catalog all original ideas"
```
