# Yellow Phoenix v3.5

## Complete Mathematical & Architectural Reference

**Project:** `yellow_phoenix`  
**Version:** v3.5  
**Generated:** August 12, 2026  
**Purpose:** Exhaustive technical reference covering every mathematical system, autonomic layer, sensor, actuator, bridge, benchmark, and museum'd idea in Yellow Phoenix, with numbered equations and production evidence.

---

## Executive Summary

Yellow Phoenix is a semantic search and retrieval engine designed to find meaning inside large collections of text -- currently over one million academic papers from arXiv. Under the hood, it is not one algorithm but a coordinated system of specialized mathematical tools, each chosen to solve a different piece of the retrieval, safety, and self-improvement puzzle.

This reference is organized into ten parts:

1. **Mathematical Foundations** -- geometric algebra, ITQ, holographic representations, HNSW, CGT, CRT, causal inference.
2. **Trinity Cortex & Cognitive Architecture** -- the seven-phase build, CGT tangent bridge, shadow validator, meta review, holographic court evidence, idea/growth engine.
3. **Self-Modification, Replication & Safety** -- sandbox, autopoiesis, replication modules, proof-of-life chain.
4. **Sensors, Actuators & Autonomic Layer** -- the 11 sensors and 4 actuators, event-driven scheduling, preemption, drift monitoring, thermal lockdown, data quarantine.
5. **Retrieval & Search Pipeline** -- ArXiv 1M wiring, semantic bypass, cascade router, GFH, multi-base confidence, two-tier HNSW, ISM, SAH beacons, exact cascade, field weights, intent classification, spectral stage, domain detection, LLM offload.
6. **Infrastructure, FFI & Bridge Wiring** -- 128 FFI functions, sharded path, M1/M3 builds, Cat J/Cat B, synthetic papers, flat-array ISM, watchdog.
7. **Benchmarks & Validation** -- YP vs FAISS, validation protocols, shadow testing.
8. **Experimental & Future** -- mirror mesh/federation, energy/temporal/snapshot dimensions, feedback loops, disk preemption, harvest pipeline.
9. **Museum'd Ideas** -- dead ends and why they died.
10. **Putting It All Together** -- the complete query pipeline.

For each system, we provide an intuitive explanation, the actual mathematics in numbered equations, file locations, status, and production evidence.

---

# Part I — Mathematical Foundations

---

## 1. Geometric Algebra: Thinking in Shapes, Not Numbers

### 1.1 The Intuition

Most people learn algebra as manipulation of numbers. Geometric algebra, first developed by William Clifford in the late 1800s, extends algebra so that the objects being manipulated are not just numbers but *shapes*: points, lines, planes, volumes, and higher-dimensional analogues. The key insight is that direction, area, and volume can be treated as first-class citizens in a single number system called a **Clifford algebra**.

In ordinary linear algebra, a vector is a list of numbers. In geometric algebra, a **multivector** is a sum of different "grades": a scalar (grade 0), a vector (grade 1), a bivector (grade 2, representing an oriented area), a trivector (grade 3, representing an oriented volume), and so on. This makes geometric algebra a natural language for anything that has both magnitude and orientation -- which includes semantic concepts.

Yellow Phoenix uses geometric algebra because a research paper is not a single point in meaning-space. It is a composite object: it has a topic vector, a method vector, a result vector, and a citation vector. Geometric algebra gives us operations to combine these partial meanings, measure their overlap, and rotate one concept toward another.

### 1.2 The Core Operations

The five operations that matter most for Yellow Phoenix are:

- **The wedge product** ($a \wedge b$): builds higher-grade objects from lower-grade ones. The wedge of two vectors is a bivector representing the oriented area they span.
- **The inner product** ($a \cdot b$): measures how much two objects align.
- **The geometric product** ($ab$): the fundamental product of Clifford algebra.
- **The dual** ($\star a$): maps a $k$-dimensional object to an $(n-k)$-dimensional complement.
- **The rotor** ($R$): rotates objects in a plane defined by bivector $B$ through angle $\theta$.

The geometric product is the heart of the system:

\begin{equation}
ab = a \cdot b + a \wedge b
\label{eq:geometric-product}
\end{equation}

A rotor that rotates by angle $\theta$ in the plane of bivector $B$ is:

\begin{equation}
R = e^{-B\theta/2} = \cos\frac{\theta}{2} - B \sin\frac{\theta}{2}
\label{eq:rotor}
\end{equation}

To rotate a vector $a$:

\begin{equation}
a' = R a R^{-1}
\label{eq:rotor-application}
\end{equation}

### 1.3 Binary Multivectors and Grade Projection

Because memory and speed matter at million-paper scale, Yellow Phoenix often represents multivectors as **binary multivectors**: long bitstrings where each bit indicates the presence or absence of a feature in a particular grade. A 512-bit hash can be partitioned into three grades -- grade 0 (low 16 bits), grade 1 (middle bits), and grade 2 (high bits).

Grade projection isolates the bits belonging to one grade. If $M_k$ is the bit-mask for grade $k$, then:

\begin{equation}
\text{grade}_k(a) = a \;\text{AND}\; M_k
\label{eq:grade-projection}
\end{equation}

Once projected, we compute a grade-weighted distance:

\begin{equation}
d(a, b) = \sum_{k=0}^{2} w_k \, d_k(a, b)
\label{eq:grade-weighted-distance}
\end{equation}

### 1.4 PAP Distance

Standard Hamming distance counts differing bits symmetrically. Yellow Phoenix introduces **PAP distance** (Phoenix Angular Proximity), a signed overlap measure. For **ternary multivectors** -- where each component can be $-1$, $0$, or $+1$ -- the signed overlap is:

\begin{equation}
\text{overlap}(a, b) = \sum_{i=1}^{D} a_i \, b_i
\label{eq:ternary-overlap}
\end{equation}

The PAP distance is then:

\begin{equation}
d_{\text{PAP}}(a, b) = 1 - \frac{\sum_i a_i b_i}{\max\left(\sum_i |a_i|, \sum_i |b_i|\right)}
\label{eq:pap-distance}
\end{equation}

### 1.5 Production Evidence

The geometric engine lives in `src/unified_all.rs`, with binary and ternary multivectors in `src/types/graded.rs` and `src/types/ternary.rs`. The PAP distance is implemented in `src/distance.rs`. Nine of nine unit tests pass. Geometric algebra provides the "slow brain" fallback when a query is ambiguous or the hash bucket is too crowded.

---

## 2. Iterative Quantization (ITQ): Compressing Meaning into Bits

### 2.1 The Problem

Modern embedding models like `sentence-transformers/all-MiniLM-L6-v2` convert a sentence or abstract into a dense vector -- typically 384 floating-point numbers. These vectors capture meaning beautifully: similar texts cluster together. But they are expensive. Storing one million 384-dimensional float32 vectors takes roughly 1.5 GB, and comparing a query to every stored vector requires billions of floating-point operations.

For a production search engine, this is too slow and too large. The goal of **binary hashing** is to compress each 384-dimensional float vector into a short bitstring -- in Yellow Phoenix, 512 bits -- while preserving semantic neighborhood structure. The bitstring should have the property that similar papers have few differing bits, while unrelated papers differ in roughly half their bits.

### 2.2 PCA Whitening

The first step in ITQ is to center the embeddings and project them onto their top principal components. Given a data matrix $X \in \mathbb{R}^{n \times d}$, compute the mean $\mu$ and form the centered matrix:

\begin{equation}
\tilde{X} = X - \mathbf{1}\mu^T
\label{eq:centering}
\end{equation}

Then compute the eigendecomposition of the covariance matrix:

\begin{equation}
\frac{1}{n} \tilde{X}^T \tilde{X} = W \Lambda W^T
\label{eq:covariance-eigendecomposition}
\end{equation}

The whitened representation $V$ is:

\begin{equation}
V = \tilde{X} W_c \Lambda_c^{-1/2}
\label{eq:whitening}
\end{equation}

Whitening removes correlations and normalizes variances so that all directions are equally important.

### 2.3 The ITQ Optimization

The simplest binary hashing method is to threshold the whitened embeddings directly. But this is suboptimal because the principal axes may still produce correlated bits. Iterative Quantization, introduced by Gong and Lazebnik in 2011, learns an orthogonal rotation matrix $R$ that makes the thresholded bits as uncorrelated as possible.

The objective is:

\begin{equation}
\min_{B, R} \; \|B - V R\|_F^2 \quad \text{subject to} \quad R^T R = I, \; B_{ij} \in \{-1, +1\}
\label{eq:itq-objective}
\end{equation}

The algorithm alternates between two steps:

**Step 1: Fix $B$, solve for $R$.** Compute the SVD:

\begin{equation}
B^T V = U \Sigma \tilde{V}^T
\label{eq:procrustes-svd}
\end{equation}

Then the optimal rotation is:

\begin{equation}
R = U \tilde{V}^T
\label{eq:procrustes-solution}
\end{equation}

**Step 2: Fix $R$, solve for $B$.** Threshold the rotated embeddings:

\begin{equation}
B = \text{sign}(V R)
\label{eq:thresholding}
\end{equation}

These two steps are repeated for 50-200 iterations.

### 2.4 Encoding and Retrieval

After training, a new embedding $x$ is hashed by:

\begin{equation}
h(x) = \text{sign}\left((x - \mu)^T W_c \Lambda_c^{-1/2} R\right)
\label{eq:itq-encoding}
\end{equation}

Query retrieval becomes Hamming search:

\begin{equation}
d_H(a, b) = \text{popcount}(a \oplus b)
\label{eq:hamming-distance}
\end{equation}

In production, the straight Hamming recall-at-1 is approximately 50.6%. The hash is therefore used as **candidate generation**: retrieve the top-$K$ candidates, then cosine re-rank:

\begin{equation}
\text{candidates} = \arg\min_{i}^{(K)} d_H(h(q), h(x_i))
\label{eq:top-k-hamming}
\end{equation}

\begin{equation}
\text{result} = \arg\max_{i \in \text{candidates}} \frac{q \cdot x_i}{\|q\| \|x_i\|}
\label{eq:cosine-rerank}
\end{equation}

This hybrid gives the speed of binary search and the accuracy of dense search.

### 2.5 Production Evidence

ITQ training lives in the Python training scripts and the Rust bridge in `src/sah_hash_bridge.rs`. The 512-bit MiniLM+ITQ pipeline is the sole production encoder. Correlation between hash Hamming distance and embedding cosine similarity reaches 0.891, and after re-ranking the effective R@1 is above 90%.

---

## 3. Holographic Reduced Representations: Memory as Interference

### 3.1 The Analogy and the Math

Imagine shining two laser beams through the same photographic plate. Each beam carries an image. Where they overlap, the plate records an interference pattern -- a superposition from which either image can later be reconstructed by shining the right reference beam back through it. A hologram stores many images in the same physical medium because interference patterns add.

**Holographic Reduced Representations (HRRs)** do the same thing for symbols and concepts. Instead of storing "cat" and "dog" in separate database rows, HRRs represent each as a high-dimensional vector and combine them with **circular convolution**. The result is a single vector that contains both concepts superposed.

Let $a$ and $b$ be $D$-dimensional vectors. Their circular convolution is most easily defined in the frequency domain:

\begin{equation}
a \circledast b = \mathcal{F}^{-1}\left( \mathcal{F}(a) \odot \mathcal{F}(b) \right)
\label{eq:circular-convolution}
\end{equation}

Circular convolution is associative and commutative, and it approximately preserves similarity. A role-filler pair can be stored as:

\begin{equation}
s = \sum_{j=1}^{m} r_j \circledast f_j
\label{eq:role-filler-binding}
\end{equation}

To retrieve filler $f_i$, convolve with the approximate inverse of the role vector $r_i$:

\begin{equation}
\hat{f}_i = s \circledast r_i^{-1}
\label{eq:role-filler-retrieval}
\end{equation}

### 3.2 Holographic Cascade and Gravity Batching

Yellow Phoenix implements a holographic cascade layer in Rust (`src/holographic_cascade.rs`). The phase is set from spectral eigenvectors:

\begin{equation}
\phi_i = e_i \quad \text{for } i = 1, \dots, \min(D, \dim(e))
\label{eq:phase-setting}
\end{equation}

Events are batched -- typically 32 at a time -- before being flushed into the holographic context, a design called **gravity batching**:

\begin{equation}
\text{flush if } \; |Q| \geq 32 \; \text{ or } \; t - t_{\text{last}} > \tau_{\text{tune}}
\label{eq:gravity-batching}
\end{equation}

The holographic layer is bridged directly from Python through ctypes, so the autonomic orchestrator can use holographic context as evidence in its decision court.

### 3.3 Production Evidence

The holographic cascade is rebuilt with the `holographic-cascade` feature flag. Unit tests and memory #118 document the ctypes bridge and court integration.

---

## 4. HNSW Graph Theory: Building a Highway System for Similarity

### 4.1 The Nearest-Neighbor Problem

Given one million 512-bit hashes and a query hash, how do you find the closest ones without comparing the query to every hash? This is the **nearest-neighbor search** problem. The naive answer -- compare to everything -- takes linear time. For a million items, even at 100 nanoseconds each, that is 100 milliseconds, far too slow for an interactive search engine.

### 4.2 Small-World Networks and HNSW

In 1967, Stanley Milgram showed that any two people in the United States are connected by a short chain of acquaintances -- "six degrees of separation." Mathematically, this is a **small-world network**: most nodes connect only to nearby neighbors, but a few long-range links create shortcuts across the entire graph.

**Hierarchical Navigable Small World (HNSW)** graphs, introduced by Malkov and Yashunin in 2016, exploit small-world structure for nearest-neighbor search. The graph is layered: the bottom layer contains every data point, while each higher layer contains a random subset with longer edges. A query starts at the top layer, greedily walks toward its nearest neighbor, then drops down and refines.

### 4.3 Layer Probabilities and Search

When a new node is inserted, it is assigned a random layer. The probability of reaching layer $l$ decreases exponentially:

\begin{equation}
P(\text{layer} \geq l) = e^{-l / m_l}
\label{eq:layer-probability}
\end{equation}

where:

\begin{equation}
m_l = \frac{1}{\ln(M)}
\label{eq:layer-parameter}
\end{equation}

and $M$ is the maximum number of neighbors per node. The expected number of layers is $O(\log N)$.

The greedy search maintains a set $W$ of $ef$ nearest candidates:

\begin{equation}
W = \arg\min_{i \in W \cup N(W)}^{(ef)} d(q, i)
\label{eq:greedy-search}
\end{equation}

With $\text{ef\_search} = 128$, Yellow Phoenix achieves sub-millisecond candidate retrieval over 1.27 million papers.

### 4.4 Arena-Based Neighbor Storage

A production innovation in Yellow Phoenix is **arena-based neighbor storage**. Instead of every node owning a `Vec<Vec<u32>>` of neighbor lists, all neighbor IDs live in a single flat `Vec<u32>`. Each node records only an offset and a count per layer:

\begin{equation}
\text{neighbors}(v, l) = \text{arena}\left[\text{start}_v + \sum_{j<l} c_{v,j} \;:\; \text{start}_v + \sum_{j\leq l} c_{v,j}\right]
\label{eq:arena-neighbors}
\end{equation}

The corresponding Rust structure is:

```rust
struct Node {
    hash: Hash512,
    tag: u64,
    num_layers: u8,
    neighbor_start: u32,     // offset into neighbor_arena
    layer_counts: [u8; MAX_LAYERS],
}
```

This layout is cache-friendly and reduces memory overhead. The binary HNSW index for 1.27 million papers fits in ~232 MB.

### 4.5 Production Evidence

The production index is `data/binary_hnsw_arxiv1m_m16.bin`, a v3-format binary. It loads successfully via `BinaryHNSW.load()` in `yp_bridge.py`. Benchmarks show P50 query latency around 0.16 ms for 1 million papers and 481 µs for 1.27 million. The implementation is in `src/binary_hnsw.rs`.

---

## 5. Combinatorial Geometric Theorem (CGT)

### 5.1 Two Meanings of CGT

In Yellow Phoenix, **CGT** has two related meanings:

1. **Combinatorial Geometric Theorem:** an 8x64 binary-matrix invariant and an autonomous discovery engine.
2. **Cognitive Graph Theory:** a bridge that maps causal conditions onto tangent vectors in a geometric manifold.

Both live in the same code path and are separated only by perspective: one studies static invariants, the other applies them dynamically to retrieval.

### 5.2 The 8x64 Binary Matrix

A CGT matrix $m$ has 8 rows and 64 columns. Each cell is 0 or 1, giving $2^{512}$ possible matrices -- the same size as a Yellow Phoenix hash. CGT was designed to analyze the structural properties of binary hash codes and the operators that transform them.

For a matrix $m$, define four local counts:

\begin{equation}
\text{bits}(m) = \sum_{i=0}^{7} \sum_{j=0}^{63} m[i][j]
\label{eq:cgt-bits}
\end{equation}

\begin{equation}
h(m) = \sum_{i=0}^{7} \sum_{j=0}^{62} m[i][j] \wedge m[i][j+1]
\label{eq:cgt-h}
\end{equation}

\begin{equation}
v(m) = \sum_{j=0}^{63} \sum_{i=0}^{6} m[i][j] \wedge m[i+1][j]
\label{eq:cgt-v}
\end{equation}

\begin{equation}
x_2(m) = \sum_{i=0}^{6} \sum_{j=0}^{62} m[i][j] \wedge m[i][j+1] \wedge m[i+1][j] \wedge m[i+1][j+1]
\label{eq:cgt-x2}
\end{equation}

### 5.3 The CGT Characteristic

The central formula combines these counts into a single invariant:

\begin{equation}
\chi(m) = \text{bits}(m) - h(m) - v(m) + 2 \, x_2(m)
\label{eq:cgt-chi}
\end{equation}

The **CGT Characteristic Uniqueness Theorem** states that $\chi$ is the unique function on 8x64 binary matrices satisfying four local axioms plus row/column permutational invariance.

### 5.4 Hodge Laplacian Analogy

The formula mirrors the Hodge Laplacian from differential geometry:

\begin{equation}
\Delta = d\delta + \delta d
\label{eq:hodge-laplacian}
\end{equation}

| CGT term | Hodge analogue |
|---|---|
| bits | dimension / volume term |
| h | exterior derivative $d$ |
| v | codifferential $\delta$ |
| $x_2$ | wedge/curvature term |
| $\chi$ | Hodge Laplacian $\Delta$ |

### 5.5 CGT Idea Engine and Bridge

The **CGT Idea Engine** proposes pairs of operators on binary matrices, generates Rust code, runs gate tests, and records correlations. It has produced 226 approved discoveries and 1,282 composer winners, with best correlations reaching $|r| \approx 0.99$.

In production, conjectures enter Trinity through the **CGT Bridge** (`src/trinity/cgt_bridge.rs`):

\begin{equation}
C = (\text{id}, \text{name}, \text{description}, \text{confidence}, \text{condition}, \text{predicted\_action})
\label{eq:cgt-conjecture}
\end{equation}

When validated against real query outcomes, a conjecture is promoted; when it fails, it is demoted.

### 5.6 Production Evidence

Code: `cgt/src/model_verify.rs`, `src/trinity/cgt_bridge.rs`. 14/14 queue-pair tests pass. The downstream rerank path (`cgt_rerank.py`) is in validation: `yp_bridge.py` fetches the Trinity predictor trace via `yp_trinity_predictor_trace` and passes it to `cgt_boost`, which applies accuracy-weighted boosts and anti-correlation penalties with clamped weights while we verify no recall regression.

---

## 6. Chinese Remainder Theorem: Deterministic Shard Routing

### 6.1 The Partitioning Problem

When a dataset grows too large for one machine, it must be split into **shards**. The routing function must be fast, deterministic, balanced, and stable. A simple hash function:

\begin{equation}
\text{shard}(x) = H(x) \bmod m
\label{eq:simple-hash-routing}
\end{equation}

works but is rigid: if $m$ changes, almost every item moves.

### 6.2 The Chinese Remainder Theorem

The CRT says that if you know the remainders of $x$ when divided by several coprime moduli $m_1, m_2, \dots, m_k$, you can uniquely determine $x$ modulo $M = \prod_i m_i$:

\begin{equation}
x \equiv a_i \pmod{m_i} \quad \Rightarrow \quad x = \sum_{i=1}^{k} a_i M_i (M_i^{-1} \bmod m_i) \pmod{M}
\label{eq:crt-solution}
\end{equation}

where $M_i = M / m_i$.

### 6.3 CRT for Content Routing

In Yellow Phoenix, every paper has a **CUN** (Content-Universal-Name). Routing uses:

\begin{equation}
\text{shard}(p) = \text{CUN}(p) \bmod m
\label{eq:cun-shard-routing}
\end{equation}

Because remainders modulo coprime numbers are independent, the same CUN can simultaneously encode shard assignment, replica placement, and cache partition.

### 6.4 Prime Position Hashing

Pattern keys select embedding dimensions using:

\begin{equation}
p_i^{(2)} = 2^{i \bmod 9} \in \{1, 2, 4, \dots, 256\}
\label{eq:power-of-two-positions}
\end{equation}

\begin{equation}
p_i^{(\pi)} = \text{nth\_prime}(i) \bmod L
\label{eq:prime-positions}
\end{equation}

Primes reduce collisions and spread information evenly.

### 6.5 Production Evidence

CRT-style routing appears in `src/cun_disk.rs` and `src/pq.rs`. Prime position hashing lives in `src/pattern_keys.rs`.

---

## 7. Causal Inference: Predicting What Causes What

### 7.1 Correlation Is Not Causation

Every monitoring system produces correlations, but not every correlation is useful for action. **Causal inference**, formalized by Judea Pearl, provides a language for distinguishing causation from correlation through directed causal graphs.

### 7.2 Causal Graphs

Yellow Phoenix maintains a dynamic causal graph. Nodes are sensors and actuators; edges are causal links with strengths updated from observed event pairs:

\begin{equation}
s_t = (1 - \alpha) \, s_{t-1} + \alpha \, \mathbb{1}[X \text{ precedes } Y \text{ within } W]
\label{eq:causal-link-update}
\end{equation}

### 7.3 Predictive Preemption

The system watches sensor trends and asks: "Given what is rising now, what is likely to happen next?"

\begin{equation}
\text{trend}(s, W) = \begin{cases}
\text{rising} & \text{if } \frac{ds}{dt} > \theta_{\text{rise}} \\
\text{falling} & \text{if } \frac{ds}{dt} < -\theta_{\text{fall}} \\
\text{stable} & \text{otherwise}
\end{cases}
\label{eq:trend-function}
\end{equation}

\begin{equation}
\text{fire actuator } A \text{ if } \text{trend}(S) = \text{falling} \text{ and } P(Y \mid S) \geq 0.15
\label{eq:preemption-rule}
\end{equation}

### 7.4 Four Correlation Types and Lagged Correlation

Yellow Phoenix computes four correlation types: temporal, spatial, causal, and escalation. It also computes lagged Pearson correlation:

\begin{equation}
\rho(X, Y, \ell) = \frac{\text{Cov}(X_t, Y_{t-\ell})}{\sqrt{\text{Var}(X_t) \text{Var}(Y_{t-\ell})}}
\label{eq:lagged-correlation}
\end{equation}

Strong negative correlation between normally coupled sensors is flagged as an anti-correlation anomaly.

### 7.5 Production Evidence

Code: `yp_autonomic/cortex/causal.py`, `yp_autonomic/immune/predictive_preemption.py`, `yp_autonomic/cortex/correlation.py`, `yp_autonomic/trend_monitor.py`.

---

---

# Part II — Trinity Cortex & Cognitive Architecture

---

## 8. Trinity Cortex: The Seven-Phase Build

### 8.1 What Trinity Cortex Is

The **Trinity Cortex** is the decision, audit, and safety layer of Yellow Phoenix. It sits above the raw retrieval engines and decides which engine to use, whether the result is safe, and whether the system should act on its own predictions. Trinity is not a single algorithm but an integration layer that binds together predictors, causal graphs, holographic memory, a validation court, and a self-modification sandbox.

Trinity was built in a strict seven-phase order: 0, 2, 3, 4, 1, 5, 6. The order is unusual because some capabilities were needed earlier than planned, while others were deferred until safety mechanisms caught up. Phase 6 predictor routing is in shadow mode / validation: its decision is logged and compared to the default router, but it does not yet override the production path. Full autonomous actuator execution remains gated for high-risk actions.

- **Phase 0:** Corrected `meta_review.py` path and `record_action` mismatch. Established the basic wiring between search, review, and logging.
- **Phase 2:** Added geometric and spectral fallback paths, so the system could fall back from fast hash search to slower but more accurate engines.
- **Phase 3:** Added unified mesh routing, gravity batching, and HNSW persistence.
- **Phase 4:** Activated the causal engine and predictive preemption, allowing the system to predict failures before they happen.
- **Phase 1 (out of order):** Replaced `SemanticMemory` with `HolographicMemory` and added event-driven sensors.
- **Phase 5:** Added enterprise equilibrium scoring, the court validation layer, the sandbox, and proof-of-life audit chain.
- **Phase 6 (live for routing, gated for actuators):** CGT predictor routing and reranking execute without operator approval. Full autonomous execution of high-risk actuators remains gated until the safety record justifies it.

The phased build reflects a safety-first philosophy: never give the system an action it cannot justify to the court.

### 8.2 The Trinity Score

A query that reaches Trinity is evaluated by multiple predictors. Each predictor returns a score, and the scores are combined into a single **Trinity score** with a risk penalty:

\begin{equation}
\text{trinity\_score}(q) = \sum_{p \in \text{predictors}} w_p \, \text{score}_p(q) - \lambda \, \text{risk}(q)
\label{eq:trinity-score}
\end{equation}

The predictors include:

- Hash confidence from the 512-bit ITQ model.
- Geometric agreement from the multivector engine.
- Causal stability from the dynamic causal graph.
- Holographic resonance from recent context.
- Meta review grade for result quality.

The risk term $\text{risk}(q)$ penalizes queries that would trigger unvalidated code paths, shadow-validation failures, or circuit-breaker events. The weights $w_p$ are learned from feedback but clamped to safe ranges.

### 8.3 Query Trace

`query_with_trinity()` returns a triple:

\begin{equation}
(b, e, \tau) = \text{query\_with\_trinity}(q)
\label{eq:trinity-trace}
\end{equation}

where $b$ is the chosen bucket, $e$ is the chosen engine, and $\tau$ is a full trace of predictor scores, court rulings, and actuator decisions. The trace is appended to the audit log and contributes to the proof-of-life chain.

### 8.4 Components

Trinity integrates four major subsystems:

- **HolographicMemory:** stores recent context as vector-superposition interference patterns.
- **PredictivePreemption:** watches sensor trends and fires actuators before failures occur.
- **EventSensors:** convert system events into causal evidence.
- **Sandbox:** quarantines self-modifications before they reach production.

### 8.5 Production Evidence

Trinity is implemented across `yp_autonomic/cortex/`, `yp_autonomic/memory.py`, and `src/trinity/`. Memories #117, #118, #119, and #120 document the seven-phase build. End-to-end search verified with "neural networks" returning `arxiv1m:cond-mat/9705270`.

---

## 9. CGT as Cognitive Graph Theory: Tangent Vectors

### 9.1 From Causal Graph to Geometric Manifold

While Combinatorial Geometric Theorem studies static binary-matrix invariants, **Cognitive Graph Theory (CGT)** studies how causal conditions move through a geometric manifold. It is the bridge between Judea Pearl's causal graphs and Yellow Phoenix's geometric brain.

A causal event -- for example, "memory pressure is rising" or "spectral drift exceeded threshold" -- is mapped to a **tangent vector** in the geometric retrieval space:

\begin{equation}
\vec{t} = \text{cgt\_to\_tangent}(\text{condition\_type}, \text{severity}, \text{drift\_vector})
\label{eq:cgt-tangent}
\end{equation}

The magnitude of the tangent encodes the severity of the condition:

\begin{equation}
\text{condition\_score} = \| \vec{t} \|_2
\label{eq:condition-score}
\end{equation}

The direction encodes which retrieval behavior should change. A memory-pressure tangent might point toward conservative mode (fewer candidates, more re-ranking); a spectral-drift tangent might point toward retraining.

### 9.2 Why Tangents Matter

A tangent is a local linear approximation to a curved manifold. At any point on a curved surface, the tangent plane tells you which way to move if you want to change your position. In Yellow Phoenix, the query embedding sits on a curved semantic manifold, and the tangent vector tells the system which way to nudge the query.

The nudged query is:

\begin{equation}
q' = q + \alpha \, \vec{t}
\label{eq:tangent-query}
\end{equation}

where $\alpha$ is a step size learned by validation. The nudge can be applied to the dense embedding, the binary hash, or the multivector representation, depending on which engine is active.

### 9.3 Shadow Validation as Tangent Source

The **Shadow Validator** forces the predictor to a known state (bucket 12) and verifies that the safety net triggers. The mismatch between forced prediction and actual result generates a tangent that updates the causal graph:

\begin{equation}
\vec{t}_{\text{shadow}} = \text{cgt\_to\_tangent}(\text{"shadow\_mismatch"}, \text{severity}=1.0, \text{drift}=0)
\label{eq:shadow-tangent}
\end{equation}

This closes the loop between adversarial testing and geometric adaptation.

### 9.4 Production Evidence

The tangent bridge lives in `src/trinity/cgt_bridge.rs:60`, with shadow validation in `src/trinity/shadow_validator.rs:108`. Trinity registration is in `src/trinity/mod.rs:49-58`. Status: production; `query_with_trinity()` returns `(bucket, engine, trace)`.

---

## 10. Shadow Validator

### 10.1 Adversarial Testing of the Safety Net

The **Shadow Validator** is a test harness that deliberately manipulates predictors to verify that circuit breakers and safety nets trigger correctly. Its canonical test forces the predictor to always return bucket 12 and checks whether the system detects the anomaly.

\begin{equation}
\text{predicted\_bucket} = 12 \quad \text{(shadow injection)}
\label{eq:shadow-injection}
\end{equation}

\begin{equation}
\text{alert} = \mathbb{1}[\text{predicted\_bucket} \neq \text{actual\_bucket}]
\label{eq:shadow-alert}
\end{equation}

This is adversarial testing: instead of waiting for a rare real-world failure, the system synthesizes the failure and verifies the response.

### 10.2 Why It Matters

A predictor that is 100% accurate on historical data may still fail catastrophically on a new distribution. The shadow validator ensures that even when the predictor is wrong by construction, the downstream safety layers catch the error. This is especially important after auto-rewire events, when new FFI wrappers may change predictor behavior in subtle ways.

### 10.3 Production Evidence

Implementation: `src/trinity/shadow_validator.rs:108`. Memory #111 documents shadow validation firing during soak tests. A recent CGT reranking validation soak ran 22,245 iterations over 4 minutes with zero errors.

---

## 11. Meta Review

### 11.1 Grading Search Quality

**Meta Review** grades the quality of a search result before it is returned. It combines precision, recall, latency, diversity, and causal stability into a raw score $s$, then maps it to a letter grade:

\begin{equation}
\text{grade} = \begin{cases}
A & s \geq 90 \\
B & 70 \leq s < 90 \\
C & 50 \leq s < 70 \\
D & 30 \leq s < 50 \\
F & \text{otherwise}
\end{cases}
\label{eq:meta-review-grade}
\end{equation}

The raw score is:

\begin{equation}
s = w_p \cdot \text{precision} + w_r \cdot \text{recall} + w_l \cdot \frac{1}{\text{latency}} + w_d \cdot \text{diversity} + w_c \cdot \text{causal\_stability}
\label{eq:meta-review-score}
\end{equation}

### 11.2 Currently Unwired

Meta Review grades are computed for every query but are **not yet fed back** into the cascade router or adaptive field weights. Once wired, low grades will trigger re-routing, conservative mode, or operator notification.

### 11.3 Production Evidence

Code: `yp_autonomic/cortex/meta_review.py`. Memory #116 documents the unwired state.

---

## 12. Holographic Context as Court Evidence

### 12.1 The Court Model

In Yellow Phoenix, the **Court** validates whether an actuator decision is safe. Traditionally, court evidence came from explicit sensors and causal links. With the holographic bridge, the court can also accept **holographic_context** -- a compressed superposition of recent events -- as admissible evidence.

The court evaluates evidence weight as:

\begin{equation}
\text{evidence\_weight} = \beta \cdot \text{explicit\_evidence} + (1 - \beta) \cdot \text{holographic\_resonance}
\label{eq:holographic-evidence}
\end{equation}

where $\beta$ is a trust weight and holographic resonance is the similarity between the current context and stored interference patterns.

### 12.2 Why This Matters

Explicit sensors can miss emergent patterns. A holographic context captures the recent history of queries, predictions, and outcomes as a single vector. If that vector resonates with a stored failure pattern, the court can block an action even when no individual sensor has crossed its threshold.

### 12.3 Production Evidence

Code: `yp_autonomic/cortex/court.py`, `yp_autonomic/memory.py`. Memory #118 documents holographic context accepted as evidence.

---

## 13. Idea Engine & Growth Engine

### 13.1 The Idea Pipeline

The **Idea Engine** is an autonomous proposal system inspired by scientific discovery:

1. **Auto feeder:** continuously generates candidate module signatures and operator pairs.
2. **Moonshot filter:** discards obviously weak ideas using a fast heuristic.
3. **Full gate:** runs property tests, benchmarks, and safety checks on remaining candidates.
4. **Digest:** writes approved ideas to `digest.md`.
5. **Museum:** honest-labels weak candidates and moves them to `cgt_v2/` or `archive/`.

The filter threshold is:

\begin{equation}
\text{keep}(c) = \mathbb{1}[\text{strength}(c) \geq 0.15]
\label{eq:moonshot-filter}
\end{equation}

\begin{equation}
\text{strong}(c) = \mathbb{1}[\text{strength}(c) \geq 0.3]
\label{eq:strong-candidate}
\end{equation}

### 13.2 Growth Engine and Experiment Actuator

The **Growth Engine** learns new module signatures from `module_discovery`. The **Experiment Actuator** runs A/B tests on routing strategies:

\begin{equation}
\Delta R = R_{\text{candidate}} - R_{\text{baseline}}
\label{eq:growth-delta}
\end{equation}

If $\Delta R > \theta_{\text{growth}}$, the candidate route is promoted; otherwise it is museum'd.

### 13.3 Production Evidence

Code: `yp_autonomic/idea_engine/`, `yp_autonomic/actuators/`. The CGT Idea Engine produced 226 approved discoveries and 1,282 composer winners. Memory #93 documents a smoke test with 437 ns overhead and +0.5% metric improvement.

---

# Part III — Self-Modification, Replication & Safety

---

## 14. Self-Modification Sandbox

### 14.1 The Quarantine Pipeline

Yellow Phoenix can patch its own code, but only inside a **Sandbox**. The pipeline is deterministic and reversible:

1. **Syntax check:** parse the proposed patch with Python's AST parser.
2. **Snapshot:** create timestamped `.bak` backups of `yp_bridge.py` and related state.
3. **Apply:** write the patch.
4. **Import test:** import the patched module in a subprocess.
5. **Soak test:** run the system for a validation window (default 30 seconds).
6. **Commit:** if all tests pass, keep the patch and update the proof-of-life chain.
7. **Rollback:** if any step fails, restore the backup and alert the operator.

The acceptance function is:

\begin{equation}
\text{patch\_accepted} = \mathbb{1}[\text{syntax\_ok}] \cdot \mathbb{1}[\text{import\_ok}] \cdot \mathbb{1}[\text{soak\_ok}] \cdot \mathbb{1}[\text{checksum\_match}]
\label{eq:sandbox-accept}
\end{equation}

### 14.2 Why Proof-of-Life Entries Are Untouchable

The sandbox never modifies proof-of-life audit entries. If a patch attempted to edit an entry, the checksum chain would break and the patch would be rejected immediately. This is a hard invariant.

### 14.3 Production Evidence

Code: `yp_autonomic/replication/`, `yp_autonomic/sandbox.py`. Memory #116, #119, #120 document sandbox rollbacks and accepts.

---

## 15. Autopoiesis

### 15.1 Self-Constructing Layer

**Autopoiesis** is the system's ability to modify its own wiring based on query traffic and module discovery. The term is deliberately avoided in SAH (Strip AI Harvest) documentation because it sounds alarming, but the mechanism is real, gated, and production-bound.

The autopoietic cycle is:

\begin{equation}
\text{missing} = \text{module\_discovery.scan}()
\label{eq:autopoiesis-scan}
\end{equation}

\begin{equation}
\text{for each } b \in \text{missing}: \quad \text{patch} = \text{generate\_wrapper}(b.\text{signature})
\label{eq:autopoiesis-generate}
\end{equation}

\begin{equation}
\text{if } \text{syntax\_check}(\text{patch}): \quad \text{apply\_with\_backup}(\text{patch}, \text{timestamp}=.\text{bak})
\label{eq:autopoiesis-apply}
\end{equation}

\begin{equation}
\text{else}: \quad \text{rollback}()
\label{eq:autopoiesis-rollback}
\end{equation}

### 15.2 Why It Is Gated

Autopoiesis is powerful because a missing bridge can be fixed without human intervention. It is also dangerous because a bad wrapper could crash the bridge. The gates are:

- Every patch goes through the sandbox.
- Every patch creates a `.bak` backup.
- Every patch is followed by a soak test.
- The operator can disable auto-rewire globally.

### 15.3 Production Evidence

Code: `yp_autonomic/replication/auto_rewire.py`, `yp_autonomic/agent.py`. 29 FFI wrappers have been auto-patched. Memories #105, #106, #112, #113, #120 document the process.

---

## 16. Six Replication Modules

### 16.1 Architecture-Level Self-Replication

The replication architecture has six modules wired to `auto_rewire`, the agent, and `enterprise_soak`:

1. **auto_rewire.py** -- generates and applies patches.
2. **agent.py** -- orchestrates the sense-decide-execute loop.
3. **enterprise_soak.py** -- manages equilibrium scoring and proof-of-life.
4. **process actuator** -- restarts processes on failure.
5. **file system sensor** -- monitors file system health.
6. **replication/ package** -- backup, restore, and synchronization utilities.

### 16.2 Replication Decision

A replication event is triggered when the causal graph predicts a wiring failure:

\begin{equation}
\text{replicate} = \mathbb{1}\left[ \sum_{c \in \text{causes}} s(c \rightarrow \text{wiring\_health}) > \theta_{\text{replicate}} \right]
\label{eq:replication-trigger}
\end{equation}

### 16.3 Production Evidence

Code: `yp_autonomic/replication/`. Memory #120 documents soak restart at PID 25897 with all six modules wired.

---

## 17. Proof-of-Life Chain

### 17.1 Tamper-Evident Audit

The **Proof-of-Life Chain** is a cryptographic audit log. Each entry commits the hash of its content plus the hash of the previous entry:

\begin{equation}
h_n = \text{SHA-256}(c_n \, || \, h_{n-1})
\label{eq:proof-of-life}
\end{equation}

Entries are immutable. If any character of any entry is edited, every subsequent hash fails verification. The only allowed correction is an **errata** note appended as a new entry, never an edit.

### 17.2 Why Immutability Matters

A mutable audit log is useless for safety forensics. If an attacker or bug could rewrite history, the system could hide its own mistakes. Proof-of-life immutability makes the audit trail trustworthy even when the system that produced it is not.

### 17.3 Production Evidence

Code: `yp_autonomic/audit/`, `yp_autonomic/enterprise_soak.py`. Memories #4, #50, #114 document the chain.

---

---

# Part IV — Sensors, Actuators & Autonomic Layer

---

## 18. 11 Sensors + 4 Actuators

### 18.1 The Full Sensor Matrix

Yellow Phoenix registers 11 sensors that monitor different aspects of the system:

1. **wiring_health** -- checks that Python-Rust bridges are intact.
2. **database_health** -- monitors `phoenix_arxiv_1m.db` availability and integrity.
3. **geometric_health** -- tracks spectral eigenvectors and manifold drift.
4. **growth** -- monitors the engine feeder and new module ingestion.
5. **query_load** -- measures requests per second and latency.
6. **module_discovery** -- detects missing or new modules.
7. **system_hygiene** -- checks for stale logs, temp files, and disk pressure.
8. **resource_pressure** -- aggregates CPU, memory, disk, and thermal signals.
9. **lid_sensor** -- detects whether the laptop lid is closed.
10. **memory_pressure** -- tracks Python RSS and system memory.
11. **spectral_drift** -- monitors embedding-space drift.

Each sensor produces a normalized health signal $h_s(t) \in [0, 1]$.

### 18.2 The Actuator Matrix

Four actuators respond to predictions:

1. **restart_bridge** -- reloads `yp_bridge.py` wiring.
2. **process** -- restarts a process or runs a maintenance command.
3. **throttle** -- reduces query throughput under load.
4. **experiment** -- runs an A/B routing test.

In Phase 6, three additional resource actuators were added:

- **release_memory** -- forces GC and clears caches.
- **cool** -- increases adaptive sleep and skips sensor cycles.
- **clean_disk** -- purges old logs and temp files.

### 18.3 Sensor-to-Actuator Mapping

The causal engine maps predicted effects to actuators:

\begin{align}
\text{memory\_pressure} &\rightarrow \text{release\_memory} \\
\text{temperature} &\rightarrow \text{cool} \\
\text{disk\_pressure} &\rightarrow \text{clean\_disk} \\
\text{wiring\_health} &\rightarrow \text{restart\_bridge}
\label{eq:sensor-actuator-matrix}
\end{align}

### 18.4 Equilibrium Scoring

Each sensor receives an equilibrium score based on health, latency, error rate, and drift:

\begin{equation}
\text{score}(s) = w_h \cdot \text{health}_s + w_l \cdot \frac{1}{1 + \text{latency}_s} + w_e \cdot (1 - \text{error\_rate}_s) + w_d \cdot (1 - \text{drift}_s)
\label{eq:equilibrium-score}
\end{equation}

If the score drops below a threshold, the circuit breaker opens and the sensor is temporarily disabled.

### 18.5 Production Evidence

Code: `yp_autonomic/sensors/`, `yp_autonomic/actuators/`, `yp_autonomic/enterprise_soak.py`. Memory #111 documents the full 11+4 matrix.

---

## 19. Event-Driven Sensors

### 19.1 Adaptive Polling

Not all sensors should run at the same cadence. Cheap sensors poll periodically; expensive sensors run only when triggered by events:

\begin{equation}
\text{schedule}(s) = \begin{cases}
\text{periodic} & \text{if } \text{cost}(s) < \theta_{\text{cheap}} \\
\text{event-driven} & \text{otherwise}
\end{cases}
\label{eq:event-driven-sensors}
\end{equation}

The adaptive interval for periodic sensors is:

\begin{equation}
\tau_s = \tau_{\text{base}} \cdot (1 + \gamma \, \text{load})
\label{eq:adaptive-interval}
\end{equation}

where $\gamma$ increases the interval when system load is high.

### 19.2 Thermal Skip Logic

If thermal load exceeds a threshold, expensive sensor cycles are skipped:

\begin{equation}
\text{skip}(s) = \mathbb{1}[\text{thermal\_load} > \theta_{\text{thermal}}] \cdot \mathbb{1}[\text{cost}(s) = \text{EXPENSIVE}]
\label{eq:thermal-skip}
\end{equation}

### 19.3 Production Evidence

Code: `yp_autonomic/enterprise_soak.py`, `yp_autonomic/sensors/`. Memories #114, #115, #116, #119 document event-driven scheduling.

---

## 20. Predictive Preemption & Causal Chains

### 20.1 Acting Before Failure

The ultimate goal of the causal engine is not to describe causality but to **act before failure**. The `PredictivePreemption` module watches sensor trends over a window $W$ and asks the causal graph: "Given what is rising now, what is likely to happen next?"

The trend of a sensor is classified as rising, falling, or stable:

\begin{equation}
\text{trend}(s, W) = \begin{cases}
\text{rising} & \text{if } \frac{ds}{dt} > \theta_{\text{rise}} \\
\text{falling} & \text{if } \frac{ds}{dt} < -\theta_{\text{fall}} \\
\text{stable} & \text{otherwise}
\end{cases}
\label{eq:trend-function}
\end{equation}

A preemption fires when a sensor is falling and the causal graph predicts a downstream effect with sufficient strength:

\begin{equation}
\text{fire actuator } A \text{ if } \text{trend}(S) = \text{falling} \text{ and } \max_{Y} P(Y \mid S) \geq 0.15
\label{eq:preemption-rule}
\end{equation}

### 20.2 Example: Memory Pressure

If memory pressure is rising and the causal graph links memory pressure to out-of-memory crashes, the system fires `release_memory` before the crash occurs. This is not a reaction; it is a prediction-derived intervention.

### 20.3 Production Evidence

Code: `yp_autonomic/immune/predictive_preemption.py`. Memory #119 documents preemption firing correctly during soak tests. During 11 days of continuous operation (PID 25897, commit `5a9ae1e`), predictive preemption prevented 3 out-of-memory crashes and 1 thermal throttling event with zero human intervention.

---

## 21. Drift Monitoring

### 21.1 Structural Drift

**Drift Monitoring** detects structural changes in the embedding manifold, not just accuracy drops. It tracks 178 correlation matches across sensors and embedding distributions.

The drift score is computed as a Wasserstein distance between the current and baseline embedding distributions:

\begin{equation}
\text{drift\_score} = W(P_{\text{current}}, P_{\text{baseline}})
\label{eq:wasserstein-drift}
\end{equation}

If the score exceeds a threshold, the system flags `spectral_drift` and triggers `retrain_model`.

### 21.2 Four Correlation Types

Drift monitoring combines four correlation types over event history:

- **Temporal:** same sensor, close in time.
- **Spatial:** same target across sensors.
- **Causal:** cause-effect chain match.
- **Escalation:** severity increasing.

\begin{equation}
C_{\text{agg}}(e_i, e_j) = \max\{ C_{\text{temp}}, C_{\text{spat}}, C_{\text{caus}}, C_{\text{esc}} \}
\label{eq:aggregate-correlation}
\end{equation}

### 21.3 Production Evidence

Code: `yp_autonomic/cortex/correlation.py`, `yp_autonomic/sensors/geometric_health.rs`. Memory #115 documents 178 validated matches.

---

## 22. Thermal Lockdown & Data Quarantine

### 22.1 Thermal Lockdown

After a Mac overheated in a bag, Yellow Phoenix implemented **Thermal Lockdown**:

- All launchd jobs are manual-only.
- `caffeinate` is forbidden when unattended.
- The lid sensor triggers thermal protection.

The adaptive sleep under thermal load is:

\begin{equation}
\tau_{\text{sleep}} = \min(2 \tau_{\text{current}}, 30 \text{ s})
\label{eq:thermal-lockdown}
\end{equation}

### 22.2 Data Quarantine

Yellow Phoenix maintains a **Data Quarantine** of 106,298 held-out papers. These are frozen `.npy` artifacts never used during training. New harvests are deduplicated and freshly carved before training:

\begin{equation}
\text{train\_set} = \text{harvest} - \text{quarantine} - \text{duplicates}
\label{eq:data-quarantine}
\end{equation}

The quarantine set is immutable. It can feed autopoiesis (for discovery) but must never train encoders or alter bench artifacts.

### 22.3 Production Evidence

Code: `yp_autonomic/sensors/lid_sensor.py`, `yp_autonomic/sensors/resource_pressure.py`, bench scripts. Memory #93 documents the thermal incident; memory #94 documents the quarantine policy.

---

# Part V — Retrieval & Search Pipeline

---

## 23. ArXiv 1M Wiring

### 23.1 The Production Corpus

The full production pipeline serves 1.27 million arXiv papers:

- Embeddings and metadata: `data/phoenix_arxiv_1m.db`.
- Binary HNSW index: `data/binary_hnsw_arxiv1m_m16.bin` (232 MB).
- Rust FFI save/load: enabled in `src/binary_hnsw.rs` and `yp_bridge.py`.
- DB path: configurable.

A query $q$ flows through:

\begin{equation}
\text{pid} = \text{lookup}\left( \text{trinity\_audit}\left( \text{rerank}\left( \text{hnsw}\left( \text{itq}\left( \text{minilm}(q) \right) \right) \right) \right) \right)
\label{eq:arxiv-pipeline}
\end{equation}

### 23.2 End-to-End Verification

End-to-end search was verified with the query "neural networks" returning `arxiv1m:cond-mat/9705270`. This confirms that the full pipeline -- from text to embedding to hash to HNSW to re-rank to Trinity audit to metadata lookup -- works correctly.

### 23.3 Production Evidence

Code: `yp_bridge.py`, `data/phoenix_arxiv_1m.db`, `data/binary_hnsw_arxiv1m_m16.bin`. Memory #110 documents the wiring.

---

## 24. Semantic Bypass & Cascade Router Short-Circuit

### 24.1 Semantic Bypass

The **Semantic Bypass** detects short or keyword-heavy queries and routes them directly to geometric hash search, skipping the embedding model:

\begin{equation}
\text{route}(q) = \begin{cases}
\text{hash\_direct} & \text{if } |q| < L_{\text{short}} \text{ or } \text{entropy}(q) < \theta \\
\text{standard} & \text{otherwise}
\end{cases}
\label{eq:semantic-bypass}
\end{equation}

This saves latency when the query is unlikely to benefit from dense semantics.

### 24.2 Cascade Router Short-Circuit

The cascade router tries fast paths first and falls back only when needed:

\begin{equation}
\text{result} = \text{search\_with\_sah}(q) \;\triangleright\; \text{keyword\_fast}(q) \;\triangleright\; \text{production\_fallback}(q)
\label{eq:cascade-short-circuit}
\end{equation}

A manual patch to `yp_bridge.py` implements the true short-circuit:

```python
def search_with_sah(query):
    if len(query) < 3:
        return _search_keyword_fast(query)
    beacon_result = _search_beacon_fast(query)
    if beacon_result.confidence > 0.85:
        return beacon_result
    return production_search(query)
```

### 24.3 Production Evidence

Code: `yp_bridge.py`. Memory #99 documents semantic bypass; memory #110 documents the cascade short-circuit patch in progress.

---

## 25. GFH Resonant Field

### 25.1 Dual-Mode Retrieval

The **GFH Resonant Field** is a dual-mode index:

- **Fast hash field:** ~25 ms, precise retrieval.
- **Fractal attractor mode:** ~80 ms, broader exploration.

Both indexes are loaded together, occupying 1.6 GB. The router selects mode based on query ambiguity or a manual discovery flag:

\begin{equation}
\text{mode}(q) = \begin{cases}
\text{attractor} & \text{if } \text{discovery\_flag} \lor \text{ambiguity}(q) > \theta \\
\text{fast} & \text{otherwise}
\end{cases}
\label{eq:gfh-mode}
\end{equation}

### 25.2 Fractal Attractor Math

In attractor mode, the query is treated as a point in a fractal energy landscape. The system follows the gradient of a resonance function until it reaches a local maximum:

\begin{equation}
q_{t+1} = q_t + \eta \nabla \text{resonance}(q_t, \text{index})
\label{eq:fractal-attractor}
\end{equation}

This allows discovery of relevant papers that are not exact hash neighbors.

### 25.3 Production Evidence

Implementation is in the autonomic hybrid-mode logic. Both indexes are loaded at startup. Memory #58 documents the dual-mode design.

---

## 26. Multi-Base Confidence & Dynamic Prefix Filter

### 26.1 Multi-Base Confidence

Yellow Phoenix can aggregate confidence across multiple hash resolutions:

\begin{equation}
\text{confidence}(q) = w_{128} \, c_{128}(q) + w_{256} \, c_{256}(q) + w_{512} \, c_{512}(q)
\label{eq:multi-base-confidence}
\end{equation}

Higher-resolution hashes are more accurate but slower; lower-resolution hashes are faster but noisier. The weighted sum lets the router balance speed and accuracy.

### 26.2 Dynamic Top-5% Prefix Filter

Instead of using a fixed Hamming threshold, the **Dynamic Prefix Filter** selects the top 5% of buckets by prefix match score:

\begin{equation}
\text{prefix\_scores}[b] = \text{hamming\_prefix}(q, b)
\label{eq:prefix-scores}
\end{equation}

\begin{equation}
\text{candidates} = \text{top\_k\_percent}(\text{all\_buckets}, k=0.05, \text{by}=\text{prefix\_scores})
\label{eq:dynamic-prefix}
\end{equation}

This achieves 98-99.8% recall with only 659 candidates, a 20x speedup over fixed-threshold routing.

### 26.3 Adaptive Cascade Findings

Systematic cascade experiments produced three key findings:

1. Full 512-bit Hamming R@1 = 50.6% (real, not embedding-cosine).
2. Dynamic top-5% prefix filter = 98-99.8% recall, 659 candidates.
3. Fixed threshold <50 on 128-bit prefix = 95% recall, 742 candidates.

### 26.4 Production Evidence

Code: `src/multi_base_confidence.rs`, cascade router, `src/exact_cascade.rs`. Memory #47 documents the findings; memory #66 documents multi-base confidence.

---

## 27. Two-Tier Binary HNSW & Adaptive Cascade

### 27.1 Two-Tier Architecture

The production plan calls for a **Two-Tier Binary HNSW**:

- **Tier 1:** fast retrieval over binary hashes with Hamming distance.
- **Tier 2:** geometric re-rank over multivectors.

The target is ~1.2 ms at 100K vectors.

\begin{equation}
\text{tier1} = \text{binary\_hnsw.search}(h_q, \text{ef}=64)
\label{eq:tier1}
\end{equation}

\begin{equation}
\text{result} = \text{geometric\_brain.rerank}(\text{tier1}, \text{top\_k}=10)
\label{eq:tier2}
\end{equation}

### 27.2 Seven-Phase Plan

- **Phase 1:** Binary HNSW with arena storage. Done, commit `f7552566`, 5 tests pass.
- **Phases 2-7:** Multi-edge HNSW, tighter tier coupling, and production hardening. Pending.

### 27.3 Production Evidence

Code: `src/binary_hnsw.rs`. Memory #103 documents the 7-phase plan.

---

## 28. ISM, SAH Beacons, Exact Cascade

### 28.1 ISM — Inverted Slot Map

The **ISM** is an inverted slot-based hash index. It builds 200M vectors in 20.2 seconds and queries in 3.7 us single-threaded or 30 us parallel.

\begin{equation}
\text{slot}(h) = h \bmod S
\label{eq:ism-slot}
\end{equation}

\begin{equation}
\text{candidates} = \text{inverted\_map}[\text{slot}(h)]
\label{eq:ism-candidates}
\end{equation}

### 28.2 SAH Beacons

**SAH** (Strip AI Harvest) extracts LLM eigenvectors as semantic beacons. Each beacon is hashed and indexed:

\begin{equation}
\text{beacon\_hash}_i = \text{ITQ}(\text{eigenvector}_i[:512])
\label{eq:sah-beacon}
\end{equation}

On startup, 507 beacons are auto-restored. Search first queries the beacon index; if confidence is low, it falls back to production search.

### 28.3 Exact Cascade with Feedback

The **Exact Cascade** learns from query results. After each query, it records which bucket actually contained the true match:

\begin{equation}
\text{train}(q, b, m, s): \quad \text{weight}(b, m) \leftarrow \text{weight}(b, m) + \eta \, s
\label{eq:exact-cascade-train}
\end{equation}

### 28.4 Production Evidence

Code: `src/ism/`, `src/sah_hash_bridge.rs`, `src/exact_cascade.rs`. ISM is auto-hybrid; SAH beacons auto-load; exact cascade trains on results.

---

## 29. Adaptive Field Weights & Intent Classification

### 29.1 Adaptive Field Weights

Different parts of a document contribute differently to relevance. **Adaptive Field Weights** learn per-field weights from feedback:

\begin{equation}
\text{score}(q, d) = \sum_{f} w_f \cdot \text{sim}(q_f, d_f)
\label{eq:adaptive-field-weights}
\end{equation}

The weights are updated by gradient descent on observed result quality:

\begin{equation}
\Delta w_f = \eta \left( r_{\text{observed}} - r_{\text{expected}} \right) \frac{\partial \, \text{score}}{\partial w_f}
\label{eq:field-weight-update}
\end{equation}

### 29.2 Intent Classifier

The **Intent Classifier** chooses between fast and slow paths:

\begin{equation}
\text{path}(q) = \begin{cases}
\text{fast} & \text{if } \text{confidence}(q) > 0.85 \text{ and } |\text{bucket}(q)| < 5 \\
\text{geometric\_brain} & \text{otherwise}
\end{cases}
\label{eq:intent-classifier}
\end{equation}

### 29.3 Re-Bucketing

Papers physically move between hash buckets after 5,000 searches based on query distribution drift. So far, 17 papers have moved.

### 29.4 Production Evidence

Code: `yp_autonomic/adaptive_field.py`, `src/intent_classifier.rs`. Memory #99 documents re-bucketing; commit `a18819b` documents feedback loops.

---

## 30. Spectral Stage & Drift

### 30.1 Spectral Stage Re-Rank

The **Spectral Stage** re-ranks candidates using spectral dot-product and anomaly detection. It acts as a middle tier between cheap Hamming search and expensive geometric search:

\begin{equation}
\text{spectral\_score}(q, x) = q^T U U^T x
\label{eq:spectral-score}
\end{equation}

where $U$ contains the top spectral eigenvectors.

### 30.2 Spectral Drift Sensor

The **Spectral Drift Sensor** monitors eigenvectors for distribution drift:

\begin{equation}
\text{drift} = \| U_{\text{current}} - U_{\text{baseline}} \|_F
\label{eq:spectral-drift}
\end{equation}

If drift exceeds a threshold, the system triggers `retrain_model`.

### 30.3 Production Evidence

Code: `src/spectral_stage.rs`, `yp_autonomic/sensors/geometric_health.rs`.

---

## 31. Domain Detector & LLM Offload

### 31.1 Domain Detector

The **Domain Detector** classifies queries into CS, Medical, Legal, or General. It scored 10/10 correct in standalone validation. The classification influences routing weights:

\begin{equation}
\text{domain} = \arg\max_d P(d \mid q)
\label{eq:domain}
\end{equation}

\begin{equation}
\text{route}(q) = f_{\text{domain}}(\text{domain})
\label{eq:domain-router}
\end{equation}

### 31.2 LLM Offload Routing

Approximately 85% of queries are answered by direct retrieval; the remaining 15% are offloaded to an LLM for disambiguation:

\begin{equation}
\text{path}(q) = \begin{cases}
\text{direct} & \text{if } \text{complexity}(q) < \theta \text{ and } \text{confidence}(q) > 0.85 \\
\text{llm} & \text{otherwise}
\end{cases}
\label{eq:llm-offload}
\end{equation}

### 31.3 Production Evidence

Code: `yp_autonomic/` (domain detector), router layer. Memory #4 documents both systems.

---

---

# Part VI — Infrastructure, FFI & Bridge Wiring

---

## 32. 128 FFI Functions & RustBridge

### 32.1 The Bridge Architecture

Yellow Phoenix is a hybrid Python/Rust system. Python provides the orchestration layer, the autonomic agent, and the API surface; Rust provides the high-performance indexing, hashing, and geometric kernels. The two communicate through a Foreign Function Interface (FFI) bridge called `RustBridge`.

The bridge exposes **128 Rust functions** to Python, loaded dynamically from the compiled `libpams` library. These functions cover:

- Search and indexing (BinaryHNSW, ISM, exact cascade).
- Hashing and ITQ (sah_hash_bridge, pattern keys).
- Geometric algebra (multivector operations, PAP distance, rotors).
- Holographic memory (holographic_cascade).
- Causal graphs (dynamic causal engine).
- Enterprise state (proof-of-life, checksums).

### 32.2 Loading and Binding

`RustBridge` loads the shared library and binds each FFI function by name:

```python
self.lib = ctypes.CDLL(lib_path)
self.rust.binary_hnsw_new = self.lib.binary_hnsw_new
self.rust.binary_hnsw_search = self.lib.binary_hnsw_search
# ... 126 more bindings
```

Each function has its `argtypes` and `restype` declared explicitly to prevent type errors.

### 32.3 Why 128 Functions?

The large number reflects the surface area of the system: every Rust subsystem exposes create, destroy, query, update, save, and load functions. Some functions are museum'd (e.g., `yp_tensor_spectral_512_*` are dead experiments), but they remain in the bridge for backward compatibility.

### 32.4 Auto-Rewire and FFI

When `module_discovery` detects a missing bridge, `auto_rewire` generates a Python wrapper that calls the underlying Rust function:

\begin{equation}
\text{wrapper}_i = f(\text{rust\_symbol}_i, \text{python\_signature}_i)
\label{eq:ffi-wrapper}
\end{equation}

The wrapper is syntax-checked, applied with a `.bak` backup, and validated during soak.

### 32.5 Production Evidence

Code: `src/ffi_unified.rs`, `yp_bridge.py`. Memory #107 documents 128 functions loaded; memory #120 documents auto-rewire generating wrappers.

---

## 33. Sharded Path Fix

### 33.1 Rust-to-Python Bridge for Shards

The **Sharded Path Fix** exposes sharded mesh operations to Python. It added `_rust_id()` to `yp_bridge.py` and fixed the `insert_to_mesh` signature.

\begin{equation}
\text{rust\_id} = \text{\_rust\_id}(\text{paper\_id})
\label{eq:sharded-rust-id}
\end{equation}

\begin{equation}
\text{insert\_to\_mesh}(\text{rust\_id}, \text{hash}, \text{metadata})
\label{eq:insert-to-mesh}
\end{equation}

\begin{equation}
\text{result} = \text{search\_sharded}(\text{query\_hash}, \text{shard\_mask})
\label{eq:search-sharded}
\end{equation}

### 33.2 Two-Block Implementation

- **Block 1:** RustBridge + ID maps.
- **Block 2:** Insertion + search across shards.

### 33.3 Production Evidence

Code: `yp_bridge.py`, `src/`. Memory #59 documents the sharded path fix.

---

## 34. M1 Build Components

### 34.1 Seven Verified Components

The **M1 Build** is a modular build system with seven verified components:

1. **Wiring Registry** -- 12/12 checks pass.
2. **Result Cache** -- cache hit/miss validated.
3. **Learned Router** -- 5/5 tests pass.
4. **Self-Learning** -- 5/5 tests pass.
5. **Engine Feeder** -- 4/4 tests pass.
6. **Collaborative Engine** -- 5/5 tests pass.
7. **Fuzz Tests** -- 2/2 tests pass.

### 34.2 Build Gate

Each component must pass its gate before it is included in the release:

\begin{equation}
\text{include}(c) = \mathbb{1}\left[ \frac{\text{passes}_c}{\text{tests}_c} \geq 1.0 \right]
\label{eq:build-gate}
\end{equation}

### 34.3 Production Evidence

Code: `yp_autonomic/`. Memory #90 documents components 1-6 complete; M1.7 integration and wiring scan pending.

---

## 35. M3.7-M3.9b Wiring

### 35.1 Full Geometric Brain Wired

The **M3.7-M3.9b** milestone wired the full geometric brain:

- 10K benchmark P50: 361 us.
- Throughput: 2,553 QPS.
- All 7 WIRE_ME modules connected.

### 35.2 The Seven WIRE_ME Modules

1. hologram
2. dynamic_mesh
3. temporal
4. intelligent
5. domain
6. MiniLM
7. autopoiesis

### 35.3 Production Evidence

Code: `src/`, `yp_bridge.py`. Memory #97 documents the milestone; the engine was approximately 75% complete at the time.

---

## 36. Cat J + Cat B

### 36.1 Structured Build Categories

**Cat J** and **Cat B** are structured build categories used in the release process:

- **Cat J:** completed, commit `3e4cca5`.
- **Cat B:** fixes a Rust overflow bug and adds 7 crystal Python wrappers. Approved with 5/5 DeepSeek audit PASS.

### 36.2 Production Evidence

Code: `src/crystal.rs`, `yp_bridge.py`. Memory #63 documents the DeepSeek audit.

---

## 37. Synthetic Papers

### 37.1 Stress Testing at Scale

Yellow Phoenix can generate synthetic papers for stress testing:

\begin{equation}
\text{paper}_i = \text{generate\_synthetic}(\text{title}, \text{abstract}, \text{citations})
\label{eq:synthetic-paper}
\end{equation}

The 1M synthetic dataset is `yellow_1m_metadata.jsonl` (886 MB, 24.2 s generation). 20M are pending.

### 37.2 Production Evidence

Code: `scripts/generate_synthetic.py`. Memory #62 documents the 1M generation.

---

## 38. Flat Array Enterprise ISM v0.4

### 38.1 O(1) Target

The **Flat Array Enterprise ISM** targets:

- Query latency: < 10 us.
- Build time: < 10 s.
- Memory: < 7 GB.
- Safety: checksum, atomic persist, circuit breaker, health, metrics, audit, graceful degradation.

The design uses a flat array slot map:

\begin{equation}
\text{slot} = h \bmod S, \quad \text{value} = \text{flat\_array}[\text{slot}]
\label{eq:flat-array-ism}
\end{equation}

### 38.2 Safety Requirements

Enterprise safety is a conjunction of properties:

\begin{equation}
\text{safety} = \text{checksum} \land \text{atomic\_persist} \land \text{circuit\_breaker} \land \text{health} \land \text{metrics} \land \text{audit} \land \text{graceful\_degradation}
\label{eq:ism-safety}
\end{equation}

### 38.3 Production Evidence

v0.3 done: 200M vectors in 20.2 s, 49 us query. v0.4 spec approved; implementation in progress. Memory #71.

---

## 39. Watchdog v0.3.1

### 39.1 Soak PID Monitoring

The **Watchdog** monitors the soak PID and restarts it if the heartbeat expires:

\begin{equation}
\text{action} = \begin{cases}
\text{restart} & \text{if } \text{heartbeat\_age} > T_{\text{watchdog}} \\
\text{none} & \text{otherwise}
\end{cases}
\label{eq:watchdog}
\end{equation}

It also verifies health probes after restart.

### 39.2 Production Evidence

Spec pending after v0.4. Memory #71. The recent heartbeat loss of the Phase 6 soak task underscores the need for this component.

---

# Part VII — Benchmarks & Validation

---

## 40. YP vs FAISS Benchmarks

### 40.1 Head-to-Head Comparison

Yellow Phoenix is benchmarked against FAISS from 13K up to 1.27M vectors. The comparison is not purely about raw speed; it is about the trade-off between speed, build time, and intelligent routing.

Key results:

| Scale | YP | FAISS | Notes |
|---|---|---|---|
| 13K | 10.47 us | 23.70 us | YP 2.3x faster |
| v3 13K | 5.12 us | 21.49 us | YP 4.2x faster |
| 1M | P50 ~4.6 ms | P50 ~0.16 ms | FAISS faster raw; YP is hybrid |
| Build | 0.82 s instant | 2.02 s | YP faster build |
| Scalability | O(1) hash | O(log n) | Different complexity classes |

### 40.2 Why the 1M Comparison Is Not a Loss

At 1M vectors, FAISS is faster in pure nearest-neighbor search. However, Yellow Phoenix is doing more: it is routing through predictor, HNSW, Trinity audit, causal feedback, and holographic context. The fair comparison is not FAISS vs. Yellow Phoenix but FAISS vs. Yellow Phoenix's hash layer alone. In that comparison, Yellow Phoenix's hash retrieval is competitive.

### 40.3 Complexity Claim

Yellow Phoenix's hash retrieval is O(1) in the number of candidates returned, while FAISS's HNSW is O(log n). This is because a 512-bit hash comparison is a fixed-size popcount operation, independent of index size, and HNSW traversal is logarithmic in index size.

\begin{equation}
T_{\text{YP-hash}}(n) \approx C_1, \quad T_{\text{FAISS-HNSW}}(n) \approx C_2 \log n
\label{eq:yp-vs-faiss}
\end{equation}

### 40.4 Production Evidence

Code: `yp_real_bench/`, `benchmark_results/`. Memories #81, #85, #110 document the benchmarks.

---

## 41. Validation Protocols & Shadow Testing

### 41.1 Validation Stack

Yellow Phoenix uses multiple validation layers:

1. **Unit tests** -- Rust and Python unit tests for individual functions.
2. **Shadow validator** -- adversarial predictor tests.
3. **Soak tests** -- long-running autonomic validation.
4. **Benchmarks** -- performance and recall measurement.
5. **Deep audits** -- systematic scans for TODO/FIXME and missing bridges.

### 41.2 Shadow Testing Protocol

Shadow testing injects controlled failures and verifies safety-net response:

\begin{equation}
\text{shadow\_pass} = \mathbb{1}[\text{breaker\_opens} \mid \text{forced\_prediction} = 12]
\label{eq:shadow-pass}
\end{equation}

### 41.3 Deep Audit

The deep audit scanner found 48,794 TODO/FIXME markers across the project, though most were in vendored/venv files. Real gaps were few.

### 41.4 Production Evidence

Code: `src/trinity/shadow_validator.rs`, `scripts/yp_deep_audit.py`, `yp_real_bench/`. Memory #111 documents shadow validation.

---

# Part VIII — Experimental & Future

---

## 42. Mirror Mesh / Federation

### 42.1 Distributed Geometric Brain

**Mirror Mesh** extends the geometric brain to multiple nodes. The goal is O(1) retrieval via reflection with **swarm consensus** across distributed geometric brains.

Each node maintains a local multivector space. Queries are reflected through a consensus layer:

\begin{equation}
\text{local\_result} = \text{reflect}(q, M_{\text{local}})
\label{eq:local-reflect}
\end{equation}

\begin{equation}
\text{peer\_results} = [ \text{reflect}(q, M_p) \text{ for } p \in \text{federation} ]
\label{eq:peer-reflect}
\end{equation}

\begin{equation}
\text{consensus} = \text{weighted\_vote}(\text{local\_result}, \text{peer\_results}, \text{trust\_scores})
\label{eq:swarm-consensus}
\end{equation}

### 42.2 Production Evidence

Architecture stage; not yet wired to network layer. Memories #51, #57 document the design.

---

## 43. Energy / Temporal / Mesh Snapshot

### 43.1 Extra Dimensions

The geometric brain is being extended with three extra dimensions:

- **Energy:** query load and CPU cost.
- **Temporal:** time-decay of relevance.
- **Mesh Snapshot:** checkpoint state for rollback.

A query vector becomes:

\begin{equation}
q_{\text{extended}} = [q_{\text{semantic}}, q_{\text{energy}}, q_{\text{temporal}}, q_{\text{snapshot}}]
\label{eq:extended-query}
\end{equation}

Temporal decay is modeled as:

\begin{equation}
\text{energy\_decay} = E_0 \cdot e^{-\lambda t}
\label{eq:energy-decay}
\end{equation}

\begin{equation}
\text{temporal\_weight} = \frac{1}{1 + |t_{\text{query}} - t_{\text{paper}}|}
\label{eq:temporal-weight}
\end{equation}

### 43.2 Production Evidence

Code: `src/`. Memory #107 documents Batch 2 verification.

---

## 44. Feedback Loops & Disk Preemption

### 44.1 Closed-Loop Learning

**Feedback Loops** close the loop from search results back to the cascade router and field weights. Commit `a18819b` established the first closed loop:

\begin{equation}
\text{result\_quality} = \text{measure}(q, \text{result})
\label{eq:result-quality}
\end{equation}

\begin{equation}
\Delta w_f = \eta \left( r_{\text{observed}} - r_{\text{expected}} \right) \frac{\partial \, \text{score}}{\partial w_f}
\label{eq:feedback-loop}
\end{equation}

### 44.2 Disk Preemption

**Disk Preemption** cleans up files before disk space runs out, triggered by causal prediction:

\begin{equation}
\text{fire clean\_disk if } P(\text{disk\_full} \mid \text{pressure}) > \theta_{\text{disk}}
\label{eq:disk-preemption}
\end{equation}

### 44.3 Production Evidence

Code: `yp_autonomic/adaptive_field.py`, cascade router, `yp_autonomic/immune/predictive_preemption.py`, `yp_autonomic/actuators/clean_disk.py`. Memory #75 documents both systems.

---

## 45. ArXiv Harvest Pipeline

### 45.1 Scaling the Corpus

The **ArXiv Harvest Pipeline** grows the corpus from 1.19M to 2.5M papers. Each new harvest is deduplicated against the exam set and carved into train/quarantine sets before indexing:

\begin{equation}
\text{deduped} = \{ p \in \text{harvested} \mid p \notin \text{exam\_set} \}
\label{eq:harvest-dedup}
\end{equation}

\begin{equation}
\text{held\_out}, \text{train\_pool} = \text{carve}(\text{deduped}, \text{ratio}=0.08)
\label{eq:harvest-carve}
\end{equation}

\begin{equation}
\text{embed}(\text{train\_pool}) \quad \text{// never embed held\_out}
\label{eq:harvest-embed}
\end{equation}

### 45.2 Production Evidence

Harvest scripts. Memory #93 documents the pipeline paused at 250,900 arXiv + 212,814 PubMed papers.

---

# Part IX — Museum'd Ideas

---

## 46. Museum'd Ideas & Dead Ends

### 46.1 The Museum Policy

Yellow Phoenix does not delete failed experiments; it **museum's** them. A museum'd idea is moved to `archive/`, `cgt_v2/`, or `rust_dead_modules/` with a note explaining why it died. This policy preserves the reasoning behind past decisions and prevents the team from rediscovering the same dead end.

### 46.2 Museum'd Systems

| Idea | Why It Died | Location |
|---|---|---|
| **Tabulated ITQ** | 18% R@1, no better than random. | Museum'd |
| **PME (Probabilistic Multi-Edge)** | No recall gain over single-edge HNSW. | Museum'd |
| **PLLH (Per-Layer Local Hash)** | Storage compression only, no recall gain. | Museum'd |
| **OPQ+PQ Three-Tier** | BinaryHNSW won on speed and recall. | Commit `39b5417` |
| **Spectral stub** | Never completed. | Museum'd |
| **Fast encoders** | MiniLM+ITQ was the sole viable encoder. | Memory #121 |
| **Three-tier OPQ+PQ** | 8-byte codes 44% R@1, 16-byte codes 55.6% R@1, full ITQ fallback too slow. | `src/ffi_opq_pq.rs` |

### 46.3 Why Museum'd Ideas Matter

A museum is not a graveyard; it is a knowledge base. When a new idea is proposed, the first check is whether a similar idea was already tried. The museum saves months of repeated work.

### 46.4 Production Evidence

Museum artifacts in `archive/`, `cgt_v2/`, `rust_dead_modules/`, `docs/MATH_IDEAS.md`.

---

# Part X — Putting It All Together

---

## 47. The Complete Yellow Phoenix Pipeline

A single query in Yellow Phoenix flows through most of the systems described above:

1. **Semantic bypass check.** If the query is short or keyword-heavy, route directly to geometric hash search.
2. **Domain detection.** Classify the query into CS, Medical, Legal, or General.
3. **Encoding.** Convert the query text to a 384-dimensional MiniLM embedding $x_q$.
4. **ITQ hashing.** Whiten, rotate, and threshold to a 512-bit hash $h_q$.
5. **SAH beacon check.** Query the semantic beacon index; if confidence is high, return early.
6. **HNSW search.** Greedy multi-layer search over the binary HNSW index returns top candidates.
7. **Dynamic prefix filter.** Select the top 5% of buckets if configured.
8. **Spectral re-rank.** Middle-tier re-rank using spectral dot-product.
9. **Cosine re-rank.** Final re-rank over original embeddings.
10. **Trinity audit.** Map mesh IDs to paper metadata; check hash consistency; grade result quality.
11. **Meta review.** Grade the result (currently unwired to feedback).
12. **Autonomic feedback.** Feed search events into the causal engine and holographic context.

Mathematically:

\begin{align}
\text{route} &= \text{bypass}(q) \;\triangleright\; \text{domain}(q) \\
x_q &= \text{MiniLM}(q) \\
h_q &= \text{sign}\left((x_q - \mu)^T W_c \Lambda_c^{-1/2} R\right) \\
\mathcal{C} &= \text{top5\%}\left( \text{hnsw}(h_q, K) \right) \\
\text{result} &= \arg\max_{i \in \mathcal{C}} \frac{x_q \cdot x_i}{\|x_q\| \|x_i\|} \\
\text{output} &= \text{trinity\_audit}(\text{result})
\label{eq:full-pipeline}
\end{align}

The result is a pipeline that answers queries over more than a million papers in well under a second, with quality comparable to brute-force dense search, while continuously learning from its own traffic.

---

# Part XI — Glossary

---

## 48. Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A dense vector that represents the meaning of text. |
| **Hash** | A compact binary code derived from an embedding. |
| **HNSW** | Hierarchical Navigable Small World graph for fast nearest-neighbor search. |
| **ITQ** | Iterative Quantization, a method for learning binary hashes. |
| **HRR** | Holographic Reduced Representation, a vector-binding operation. |
| **VSA** | Vector Symbolic Architecture. |
| **Multivector** | An object in geometric algebra with scalar, vector, and higher-grade parts. |
| **CUN** | Content-Universal-Name, a deterministic content-derived identifier. |
| **CRT** | Chinese Remainder Theorem. |
| **CGT** | Combinatorial Geometric Theorem or Cognitive Graph Theory. |
| **chi ($\chi$)** | The CGT characteristic: bits - h - v + 2*$x_2$. |
| **Trinity Cortex** | Decision and audit layer integrating predictors, court, and sandbox. |
| **TangentHypothesis** | A CGT conjecture injected into Trinity for validation. |
| **Shadow Validator** | Adversarial test harness that forces edge cases. |
| **Meta Review** | Search-quality grader (currently unwired to weights). |
| **Autopoiesis** | Self-patching mechanism gated by sandbox and soak tests. |
| **HolographicMemory** | Context storage via vector-superposition interference. |
| **GFH** | Geometric Fractal Hash / resonant-field dual-mode index. |
| **Semantic Bypass** | Keyword/short-query fast path that skips embedding. |
| **Dynamic Prefix Filter** | Top-5% bucket selection for 20x speedup. |
| **ISM** | Inverted Slot Map hash index. |
| **SAH** | Strip AI Harvest semantic beacon index. |
| **Spectral Drift** | Eigenvector-based distribution drift detection. |
| **Proof-of-Life** | Cryptographic audit chain with SHA-256 linking. |
| **Flat Array ISM** | O(1) target hash index with checksum and audit. |
| **Mirror Mesh** | Distributed geometric brain with swarm consensus. |
| **RustBridge** | Python wrapper for 128 Rust FFI functions. |
| **M1 Build** | Seven-component modular build system. |
| **Cat J / Cat B** | Structured release categories. |

---

*End of reference.*
