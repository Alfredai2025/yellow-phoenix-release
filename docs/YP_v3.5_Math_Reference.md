# Yellow Phoenix v3.5

## Mathematical Foundations — A Readable Technical Reference

**Project:** `yellow_phoenix`  
**Version:** v3.5  
**Generated:** August 12, 2026  
**Purpose:** Explain the core mathematical systems behind Yellow Phoenix in plain language, with numbered equations, analogies, and production evidence.

---

## Executive Summary

Yellow Phoenix is a semantic search and retrieval engine designed to find meaning inside large collections of text — currently over one million academic papers from arXiv. Under the hood, it is not one algorithm but a coordinated system of specialized mathematical tools, each chosen to solve a different piece of the retrieval puzzle. Some tools compress giant floating-point vectors into tiny binary codes. Some organize those codes into graph structures that make nearest-neighbor search almost instantaneous. Some reason about cause and effect so the system can predict its own failures. Others represent concepts as geometric shapes that can be added, multiplied, and rotated like numbers.

This document explains six foundational systems in depth:

1. **Geometric Algebra** — representing ideas as multivectors that can be wedged, projected, and rotated.
2. **Iterative Quantization (ITQ)** — compressing 384-dimensional embeddings into 512-bit hashes without losing meaning.
3. **Holographic Reduced Representations (HRR)** — storing many concepts inside a single vector through circular convolution.
4. **HNSW Graph Theory** — building a navigable small-world graph over binary hashes.
5. **Combinatorial Geometric Theorem (CGT)** — discovering and validating operator-pair invariants on binary matrices.
6. **Chinese Remainder Theorem (CRT)** — routing content to shards deterministically using modular arithmetic.
7. **Causal Inference** — predicting failures before they happen by modeling cause-and-effect chains.

For each system, we provide an intuitive explanation, the actual mathematics in numbered equations, and a note on how Yellow Phoenix uses it in production today.

---

## 1. Geometric Algebra: Thinking in Shapes, Not Numbers

### 1.1 The Intuition

Most people learn algebra as manipulation of numbers. Geometric algebra, first developed by William Clifford in the late 1800s, extends algebra so that the objects being manipulated are not just numbers but *shapes*: points, lines, planes, volumes, and higher-dimensional analogues. The key insight is that direction, area, and volume can be treated as first-class citizens in a single number system called a **Clifford algebra**.

In ordinary linear algebra, a vector is a list of numbers. In geometric algebra, a **multivector** is a sum of different "grades": a scalar (grade 0), a vector (grade 1), a bivector (grade 2, representing an oriented area), a trivector (grade 3, representing an oriented volume), and so on. This makes geometric algebra a natural language for anything that has both magnitude and orientation — which, it turns out, includes semantic concepts.

Yellow Phoenix uses geometric algebra because a research paper is not a single point in meaning-space. It is a composite object: it has a topic vector, a method vector, a result vector, and a citation vector. Geometric algebra gives us operations to combine these partial meanings, measure their overlap, and rotate one concept toward another.

### 1.2 The Core Operations

The five operations that matter most for Yellow Phoenix are:

- **The wedge product** ($a \wedge b$): builds higher-grade objects from lower-grade ones. The wedge of two vectors is a bivector representing the oriented area they span. In semantic terms, wedge can represent the *combination* of two concepts into a richer structure.
- **The inner product** ($a \cdot b$): measures how much two objects align. For two vectors, this is the familiar dot product. For higher grades, it generalizes alignment across subspaces.
- **The geometric product** ($ab$): the fundamental product of Clifford algebra. It combines both inner and wedge information.
- **The dual** ($\star a$): maps a $k$-dimensional object to an $(n-k)$-dimensional complement. In 3-D, the dual of a plane is a vector perpendicular to it.
- **The rotor** ($R$): rotates objects in a plane defined by bivector $B$ through angle $\theta$.

The geometric product is the heart of the system:

\begin{equation}
ab = a \cdot b + a \wedge b
\label{eq:geometric-product}
\end{equation}

Equation \eqref{eq:geometric-product} says that multiplying two vectors produces a scalar part (their alignment) plus a bivector part (the area they sweep out). For two unit vectors in 2-D, this is directly related to complex numbers; in higher dimensions, it generalizes to quaternions and rotors.

A rotor that rotates by angle $\theta$ in the plane of bivector $B$ is:

\begin{equation}
R = e^{-B\theta/2} = \cos\frac{\theta}{2} - B \sin\frac{\theta}{2}
\label{eq:rotor}
\end{equation}

To rotate a vector $a$, we apply the double-sided product:

\begin{equation}
a' = R a R^{-1}
\label{eq:rotor-application}
\end{equation}

Yellow Phoenix implements a unified 5-D geometric query engine that can invoke any of these operations against binary multivectors.

### 1.3 Binary Multivectors and Grade Projection

Because memory and speed matter at million-paper scale, Yellow Phoenix often represents multivectors as **binary multivectors**: long bitstrings where each bit indicates the presence or absence of a feature in a particular grade. A 512-bit hash can be partitioned into three grades — grade 0 (low 16 bits), grade 1 (middle bits), and grade 2 (high bits) — so that each grade captures a different level of semantic abstraction.

Grade projection isolates the bits belonging to one grade. If $M_k$ is the bit-mask for grade $k$, then:

\begin{equation}
\text{grade}_k(a) = a \;\text{AND}\; M_k
\label{eq:grade-projection}
\end{equation}

Once projected, we can compute a grade-weighted distance between two binary multivectors:

\begin{equation}
d(a, b) = \sum_{k=0}^{2} w_k \, d_k(a, b)
\label{eq:grade-weighted-distance}
\end{equation}

where $d_k$ is a distance restricted to grade $k$, and $w_k$ are learned weights. This lets the system say, in effect, "these two papers are similar at the method level but different at the topic level."

### 1.4 PAP Distance

Standard Hamming distance counts differing bits, but it treats agreement and disagreement symmetrically. Yellow Phoenix introduces **PAP distance** (Phoenix Angular Proximity), a signed overlap measure. If two bits agree positively, that strengthens similarity; if one is positive and the other is negative, that weakens it.

For binary multivectors the PAP distance is essentially a signed population count over matched positions. For **ternary multivectors** — where each component can be $-1$, $0$, or $+1$ — the signed overlap is:

\begin{equation}
\text{overlap}(a, b) = \sum_{i=1}^{D} a_i \, b_i
\label{eq:ternary-overlap}
\end{equation}

Zero components contribute nothing, so sparse ternary representations naturally ignore irrelevant features. The PAP distance is then a normalized function of this overlap:

\begin{equation}
d_{\text{PAP}}(a, b) = 1 - \frac{\sum_i a_i b_i}{\max\left(\sum_i |a_i|, \sum_i |b_i|\right)}
\label{eq:pap-distance}
\end{equation}

Equation \eqref{eq:pap-distance} returns 0 when the two multivectors are perfectly aligned and approaches 1 when they are maximally opposed. This is useful for holographic superposition states where many concepts are stored together.

### 1.5 Production Use

The geometric engine lives in `src/unified_all.rs`, with binary and ternary multivectors in `src/types/graded.rs` and `src/types/ternary.rs`. The PAP distance is implemented in `src/distance.rs`. Nine of nine unit tests pass. In production, geometric algebra provides the "slow brain" fallback: when a query is ambiguous or the hash bucket is too crowded, the system falls back to a full multivector query rather than a cheap hash lookup.

---

## 2. Iterative Quantization (ITQ): Compressing Meaning into Bits

### 2.1 The Problem

Modern embedding models like `sentence-transformers/all-MiniLM-L6-v2` convert a sentence or abstract into a dense vector — typically 384 floating-point numbers. These vectors are beautiful: they capture meaning so well that similar texts cluster together. But they are also expensive. Storing one million 384-dimensional float32 vectors takes roughly 1.5 gigabytes, and comparing a query to every stored vector requires billions of floating-point operations.

For a production search engine, this is too slow and too large. The goal of **binary hashing** is to compress each 384-dimensional float vector into a short bitstring — in Yellow Phoenix, 512 bits — while preserving as much semantic neighborhood structure as possible. The bitstring should have the property that if two papers are semantically similar, their hashes have few differing bits; if they are unrelated, their hashes differ in roughly half their bits.

### 2.2 PCA Whitening

The first step in ITQ is to center the embeddings and project them onto their top principal components. Given a data matrix $X \in \mathbb{R}^{n \times d}$ whose rows are embeddings, we compute the mean $\mu$ and form the centered matrix:

\begin{equation}
\tilde{X} = X - \mathbf{1}\mu^T
\label{eq:centering}
\end{equation}

Then we compute the eigendecomposition of the covariance matrix:

\begin{equation}
\frac{1}{n} \tilde{X}^T \tilde{X} = W \Lambda W^T
\label{eq:covariance-eigendecomposition}
\end{equation}

The whitened representation $V$ is obtained by projecting onto the top $c$ eigenvectors and scaling by the inverse square roots of the eigenvalues:

\begin{equation}
V = \tilde{X} W_c \Lambda_c^{-1/2}
\label{eq:whitening}
\end{equation}

Whitening removes correlations and normalizes variances so that all directions are equally important. This prevents the binary hash from being dominated by a few high-variance directions.

### 2.3 The ITQ Optimization

The simplest binary hashing method is to threshold the whitened embeddings directly: $b_i = \text{sign}(v_i)$. But this is suboptimal because the principal axes may still produce correlated bits, wasting information. Iterative Quantization, introduced by Gong and Lazebnik in 2011, solves this by learning an orthogonal rotation matrix $R$ that makes the thresholded bits as uncorrelated as possible.

The objective is:

\begin{equation}
\min_{B, R} \; \|B - V R\|_F^2 \quad \text{subject to} \quad R^T R = I, \; B_{ij} \in \{-1, +1\}
\label{eq:itq-objective}
\end{equation}

where $V \in \mathbb{R}^{n \times c}$ is the whitened embedding matrix and $B \in \{-1, +1\}^{n \times c}$ is the matrix of binary codes.

The algorithm alternates between two steps until convergence:

**Step 1: Fix $B$, solve for $R$.** This is the orthogonal Procrustes problem. Compute the SVD:

\begin{equation}
B^T V = U \Sigma \tilde{V}^T
\label{eq:procrustes-svd}
\end{equation}

Then the optimal rotation is:

\begin{equation}
R = U \tilde{V}^T
\label{eq:procrustes-solution}
\end{equation}

**Step 2: Fix $R$, solve for $B$.** Simply threshold the rotated embeddings:

\begin{equation}
B = \text{sign}(V R)
\label{eq:thresholding}
\end{equation}

These two steps are repeated for a fixed number of iterations, typically 50–200. The rotation gradually aligns the decision boundaries with the natural axes of the data, reducing quantization error and improving nearest-neighbor preservation.

### 2.4 Encoding a New Query

After training, a new embedding $x$ is hashed by centering, whitening, rotating, and thresholding:

\begin{equation}
h(x) = \text{sign}\left((x - \mu)^T W_c \Lambda_c^{-1/2} R\right)
\label{eq:itq-encoding}
\end{equation}

The output $h(x)$ is a $c$-dimensional vector of $-1$ and $+1$ values, which is then packed into $c/8$ bytes.

### 2.5 Hamming Distance and Candidate Retrieval

Yellow Phoenix stores the 512-bit hash as 64 bytes — a 24× compression over the original 1,536-byte float32 vector. Query retrieval then becomes a Hamming-distance search. For two hashes $a$ and $b$:

\begin{equation}
d_H(a, b) = \text{popcount}(a \oplus b)
\label{eq:hamming-distance}
\end{equation}

where $\oplus$ is bitwise XOR and popcount counts the number of 1 bits. Modern CPUs can compare 64-byte hashes in tens of nanoseconds.

In production, the straight Hamming recall-at-1 is approximately 50.8%. That may sound low, but it is used as a **candidate generation** step: the hash retrieves the top-$K$ most promising candidates, and a final cosine re-rank over the original embeddings selects the true nearest neighbors. With $K = 500$:

\begin{equation}
\text{candidates} = \arg\min_{i \in \text{index}}^{(K)} d_H(h(q), h(x_i))
\label{eq:top-k-hamming}
\end{equation}

\begin{equation}
\text{result} = \arg\max_{i \in \text{candidates}} \frac{q \cdot x_i}{\|q\| \|x_i\|}
\label{eq:cosine-rerank}
\end{equation}

This hybrid gives the speed of binary search and the accuracy of dense search.

### 2.6 Production Evidence

ITQ training lives in the Python training scripts and the Rust bridge in `src/sah_hash_bridge.rs`. The 512-bit MiniLM+ITQ pipeline is the sole production encoder. Correlation between hash Hamming distance and embedding cosine similarity reaches 0.891, and after re-ranking the effective R@1 is above 90%.

---

## 3. Holographic Reduced Representations: Memory as Interference

### 3.1 The Analogy

Imagine shining two laser beams through the same photographic plate. Each beam carries an image. Where they overlap, the plate records an interference pattern — not the images themselves, but a superposition from which either image can later be reconstructed by shining the right reference beam back through it. A hologram stores many images in the same physical medium because interference patterns add.

**Holographic Reduced Representations (HRRs)** do the same thing for symbols and concepts. Instead of storing "cat" and "dog" in separate database rows, HRRs represent each as a high-dimensional vector and combine them with an operation called **circular convolution**. The result is a single vector that contains both concepts superposed. With the right "reference" vector, you can retrieve either constituent.

### 3.2 Why This Matters for Search

Classical databases store records as independent rows. That works for exact lookup but is terrible for **composition**: answering "What is a small domesticated carnivorous mammal?" requires combining "small," "domesticated," "carnivorous," and "mammal." A vector-symbolic architecture can compose these terms into a query vector and search for the closest stored composite.

HRRs are part of a broader family called **Vector Symbolic Architectures (VSAs)**. The shared idea is that symbols, structures, and rules can all be represented as vectors, and symbolic manipulation becomes algebraic operation on those vectors.

### 3.3 The Mathematics of HRR

Let $a$ and $b$ be $D$-dimensional vectors. Their **circular convolution** $a \circledast b$ is most easily defined in the frequency domain:

\begin{equation}
a \circledast b = \mathcal{F}^{-1}\left( \mathcal{F}(a) \odot \mathcal{F}(b) \right)
\label{eq:circular-convolution}
\end{equation}

where $\mathcal{F}$ is the discrete Fourier transform, $\odot$ is element-wise multiplication, and $\mathcal{F}^{-1}$ is the inverse transform. In other words, convolution multiplies the frequency components of the two vectors.

Circular convolution is associative and commutative, and it approximately preserves similarity: if $a$ is similar to $a'$, then $a \circledast b$ is similar to $a' \circledast b$. This makes it ideal for binding roles and fillers. A role-filler pair can be stored as:

\begin{equation}
s = \sum_{j=1}^{m} r_j \circledast f_j
\label{eq:role-filler-binding}
\end{equation}

where $r_j$ is a role vector and $f_j$ is a filler vector. To retrieve filler $f_i$, convolve the stored vector with the approximate inverse of the role vector $r_i$. The approximate inverse is obtained by reversing the components:

\begin{equation}
r_i^{-1} = [r_i^{(0)}, r_i^{(D-1)}, r_i^{(D-2)}, \dots, r_i^{(1)}]^T
\label{eq:approximate-inverse}
\end{equation}

The retrieved filler is then:

\begin{equation}
\hat{f}_i = s \circledast r_i^{-1}
\label{eq:role-filler-retrieval}
\end{equation}

Because convolution distributes over addition, the cross-terms $r_j \circledast f_j \circledast r_i^{-1}$ for $j \neq i$ act as noise, while the desired term $r_i \circledast f_i \circledast r_i^{-1}$ reconstructs $f_i$ (approximately). The reconstruction quality improves with higher dimensionality $D$.

### 3.4 Holographic Cascade in Yellow Phoenix

Yellow Phoenix implements a holographic cascade layer in Rust (`src/holographic_cascade.rs`). The phase of the holographic memory is set from spectral eigenvectors:

\begin{equation}
\phi_i = e_i \quad \text{for } i = 1, \dots, \min(D, \dim(e))
\label{eq:phase-setting}
\end{equation}

where $e$ is a spectral eigenvector and $\phi$ is the holographic phase. This means the system's "holographic context" is seeded by the dominant directions of variation in the embedding data.

Events are batched — typically 32 at a time — before being flushed into the holographic context, a design called **gravity batching**:

\begin{equation}
\text{flush if } \; |Q| \geq 32 \; \text{ or } \; t - t_{\text{last}} > \tau_{\text{tune}}
\label{eq:gravity-batching}
\end{equation}

The batching prevents every tiny event from swamping the interference pattern, while still allowing the context to evolve in near real time. The auto-tune threshold $\tau_{\text{tune}}$ adapts to event velocity.

The holographic layer is bridged directly from Python (`yp_autonomic/memory.py`) through ctypes, so the autonomic orchestrator can use holographic context as evidence in its decision court. This replaces an older semantic-memory store.

### 3.5 Production Evidence

The holographic cascade is rebuilt with the `holographic-cascade` feature flag. Unit tests and memory #118 (2026-08-09) document the ctypes bridge and court integration. Gravity batching keeps event throughput bounded while preserving context drift.

---

## 4. HNSW Graph Theory: Building a Highway System for Similarity

### 4.1 The Nearest-Neighbor Problem

Given one million 512-bit hashes and a query hash, how do you find the closest ones without comparing the query to every hash? This is the **nearest-neighbor search** problem, and it is one of the oldest and most important problems in computer science.

The naive answer — compare to everything — takes linear time. For a million items, that means a million Hamming-distance computations per query. Even at 100 nanoseconds each, that is 100 milliseconds, far too slow for an interactive search engine.

### 4.2 Small-World Networks

In 1967, psychologist Stanley Milgram showed that any two people in the United States are connected by a short chain of acquaintances — the famous "six degrees of separation." Mathematically, this is a **small-world network**: most nodes are only connected to nearby neighbors, but a few long-range links create shortcuts across the entire graph.

**Hierarchical Navigable Small World (HNSW)** graphs, introduced by Malkov and Yashunin in 2016, exploit small-world structure for nearest-neighbor search. The idea is to build a layered graph where:

- The bottom layer contains every data point, connected mostly to nearby neighbors.
- Each higher layer contains a random subset of points, with longer edges that skip across large regions.
- A query starts at the top layer, greedily walks toward its nearest neighbor, then drops down to the next layer and refines.

This is like planning a road trip: you first take a highway to the right state, then exit to a smaller road, then a local street, then a driveway.

### 4.3 Layer Probabilities and Insertion

When a new node is inserted into HNSW, it is assigned a random layer. The probability of reaching layer $l$ decreases exponentially:

\begin{equation}
P(\text{layer} \geq l) = e^{-l / m_l}
\label{eq:layer-probability}
\end{equation}

where:

\begin{equation}
m_l = \frac{1}{\ln(M)}
\label{eq:layer-parameter}
\end{equation}

and $M$ is the maximum number of neighbors per node. This produces a pyramid: few nodes at the top, many at the bottom. The expected number of layers for $N$ nodes is $O(\log N)$.

At each layer, the node selects its neighbors using a greedy heuristic: it connects to the closest existing nodes, but with a bias toward spreading out across different directions. This prevents the graph from collapsing into long chains.

### 4.4 Search and Accuracy Controls

Search has two key parameters:

- **ef_construction**: how many candidates to track while building the graph. Higher values produce better graphs at the cost of longer build times.
- **ef_search**: how many candidates to track during a query. Higher values improve recall at the cost of query latency.

The greedy search at a given layer maintains a set $W$ of $ef$ nearest candidates found so far and expands the best unvisited candidate until no improvement is possible:

\begin{equation}
W = \text{GreedySearch}(q, W, ef) = \arg\min_{i \in W \cup N(W)}^{(ef)} d(q, i)
\label{eq:greedy-search}
\end{equation}

where $N(W)$ denotes the neighbors of nodes in $W$ and the superscript $(ef)$ means "keep the best $ef$." The search returns an approximate set of nearest neighbors. With $\text{ef\_search} = 128$, Yellow Phoenix achieves sub-millisecond candidate retrieval over 1.27 million papers.

### 4.5 Arena-Based Neighbor Storage

A production innovation in Yellow Phoenix is **arena-based neighbor storage**. In a naive implementation, every node owns a `Vec<Vec<u32>>` of neighbor lists, one per layer. For a million nodes, this creates millions of small heap allocations, fragmenting memory and hurting cache performance.

Yellow Phoenix instead stores all neighbor IDs in a single flat `Vec<u32>` called the neighbor arena. Each node records only an offset and a count per layer:

\begin{equation}
\text{neighbors}(v, l) = \text{arena}[\text{start}_v + \sum_{j<l} c_{v,j} \;:\; \text{start}_v + \sum_{j\leq l} c_{v,j}]
\label{eq:arena-neighbors}
\end{equation}

where $c_{v,j}$ is the neighbor count for node $v$ at layer $j$. The corresponding Rust structure is:

```rust
struct Node {
    hash: Hash512,
    tag: u64,
    num_layers: u8,
    neighbor_start: u32,     // offset into neighbor_arena
    layer_counts: [u8; MAX_LAYERS],
}
```

This layout is cache-friendly and reduces memory overhead dramatically. The binary HNSW index for 1.27 million papers fits in ~232 MB and loads from disk in seconds.

### 4.6 Production Evidence

The production index is `data/binary_hnsw_arxiv1m_m16.bin`, a v3-format binary. It loads successfully via `BinaryHNSW.load()` in `yp_bridge.py`. Benchmarks show P50 query latency around 0.16 ms for 1 million papers and 481 µs for 1.27 million. The implementation is in `src/binary_hnsw.rs`.

---

## 5. Combinatorial Geometric Theorem (CGT): Operator Invariants on Binary Matrices

### 5.1 What CGT Means Here

In the Yellow Phoenix project, **CGT** stands for **Combinatorial Geometric Theorem**. It is both a mathematical model and an autonomous discovery engine. As a model, CGT studies an 8×64 binary matrix and defines a single numerical characteristic — called `chi` — that captures how the 1-bits are arranged. As an engine, it proposes pairs of operators, tests whether they preserve useful invariants, and promotes the best ones into the Trinity Cortex as conjectures that can influence retrieval decisions.

The intuition is simple: before you can discover a theorem, you need a space of objects and a way to measure them. In CGT, the objects are binary matrices and the measurement is a signed count that behaves like a discrete Laplacian.

### 5.2 The 8×64 Binary Matrix

A CGT matrix $m$ has 8 rows and 64 columns. Each cell is either 0 or 1, so there are $2^{512}$ possible matrices. This is the same size as a Yellow Phoenix hash, which is not a coincidence: CGT was designed to analyze the structural properties of binary hash codes and the operators that transform them.

For a matrix $m$, we define four local counts:

- **bits($m$)**: the total number of 1s.
- **h($m$)**: the number of horizontal adjacencies, i.e., pairs of neighboring 1s in the same row.
- **v($m$)**: the number of vertical adjacencies, i.e., pairs of neighboring 1s in the same column.
- **$x_2$($m$)**: the number of 2×2 blocks that are entirely 1s.

Formally:

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

The central formula of CGT combines these four counts into a single invariant:

\begin{equation}
\chi(m) = \text{bits}(m) - h(m) - v(m) + 2 \, x_2(m)
\label{eq:cgt-chi}
\end{equation}

Equation \eqref{eq:cgt-chi} looks like an inclusion-exclusion count. The $-h$ and $-v$ terms penalize neighboring 1s, while the $+2x_2$ term rewards 2×2 blocks. The result is a signed integer that changes predictably under certain operators.

Why this particular combination? The **CGT Characteristic Uniqueness Theorem** states that $\chi$ is the unique function on 8×64 binary matrices that satisfies four local axioms plus row/column permutational invariance. In other words, if you want a count that is local, additive, and insensitive to shuffling rows or columns, $\chi$ is essentially the only choice.

### 5.4 The Hodge Laplacian Analogy

The formula has a deeper interpretation. In differential geometry, the **Hodge Laplacian** is:

\begin{equation}
\Delta = d\delta + \delta d
\label{eq:hodge-laplacian}
\end{equation}

where $d$ is the exterior derivative and $\delta$ is the codifferential. The CGT characteristic mirrors this structure:

| CGT term | Hodge analogue |
|---|---|
| $\text{bits}(m)$ | dimension / volume term |
| $h(m)$ | exterior derivative $d$ (horizontal variation) |
| $v(m)$ | codifferential $\delta$ (vertical variation) |
| $x_2(m)$ | wedge/curvature term $d \wedge \delta$ |
| $\chi(m)$ | Hodge Laplacian $\Delta$ |

This analogy is not a formal proof, but it guides the design of CGT operators: any operator that behaves like a discrete derivative or integral should interact with $\chi$ in a structured way.

### 5.5 The CGT Idea Engine

The discovery side of CGT is the **CGT Idea Engine**. It is an autonomous daemon that:

1. Proposes pairs of operators on binary matrices (for example, a row rotation followed by a column flip).
2. Generates Rust code that implements the pair.
3. Runs a battery of gate tests and property checks.
4. Records the correlation between the operator pair and the change in $\chi$.
5. Promotes high-confidence conjectures to the Trinity Cortex.

The engine has produced hundreds of approved discoveries and over a thousand "composer winners" — operator chains that maximize some reward. The best cross-domain correlations reach $|r| \approx 0.99$.

### 5.6 CGT Bridge to Trinity

In production, CGT conjectures enter the retrieval pipeline through the **CGT Bridge** (`src/trinity/cgt_bridge.rs`). A conjecture is represented as:

\begin{equation}
C = (\text{id}, \text{name}, \text{description}, \text{confidence}, \text{condition}, \text{predicted\_action})
\label{eq:cgt-conjecture}
\end{equation}

When a query matches the condition, the bridge injects a **TangentHypothesis** into the Trinity predictor. If the conjecture is validated against real query outcomes, its confidence is promoted; if it fails repeatedly, it is demoted. This creates a closed loop: propose → test → promote/demote.

### 5.7 Functoriality and Retrieval Categories

A related idea appears in the **functor audit** (`docs/functor_audit.md`). Yellow Phoenix treats its retrieval layers as categories:

- $\mathcal{C}_{\text{hash}}$: objects are 512-bit hashes, morphisms are Hamming-distance comparisons.
- $\mathcal{C}_{\text{spectral}}$: objects are spectral embeddings.
- $\mathcal{C}_{\text{HNSW}}$: objects are graph nodes.
- $\mathcal{C}_{\text{geometric}}$: objects are multivectors.

Functors connect these categories:

\begin{equation}
F_1: \mathcal{C}_{\text{hash}} \rightarrow \mathcal{C}_{\text{spectral}}
\label{eq:functor-f1}
\end{equation}

\begin{equation}
F_4 = F_3 \circ F_2 \circ F_1: \mathcal{C}_{\text{hash}} \rightarrow \mathcal{C}_{\text{geometric}}
\label{eq:functor-f4}
\end{equation}

with the guarantee that recall does not collapse:

\begin{equation}
R@1_{\text{geometric}} \geq R@1_{\text{hash}} - \delta
\label{eq:functor-recall}
\end{equation}

CGT-style operator analysis provides the language for proving or empirically verifying these functorial properties.

### 5.8 Production Status

| Component | Status | Evidence |
|---|---|---|
| CGT model spec | Verified | `cgt/docs/model_spec.md`, 14/14 queue-pair tests pass |
| CGT Rust verifier | Production | `cgt/src/model_verify.rs` |
| CGT Trinity bridge | Implemented | `src/trinity/cgt_bridge.rs` |
| CGT Idea Engine | Ran, now idle | `cgt_v2/` artifacts, 226 approved discoveries |
| CGT rerank | Validation | `cgt_rerank.py` applies clamped `cgt_boost` using Trinity trace |

CGT is real mathematics that has been validated at the unit-test level and wired into Trinity. The downstream reranking path is in validation: `yp_bridge.py` fetches the predictor trace via `yp_trinity_predictor_trace` and `cgt_boost` applies accuracy-weighted boosts and anti-correlation penalties with clamped weights while we verify no recall regression.

---

## 6. Chinese Remainder Theorem: Deterministic Shard Routing

### 6.1 The Partitioning Problem

When a dataset grows too large for one machine, it must be split into **shards**. But how do you decide which shard holds which item? The routing function must be fast, deterministic, and balanced across shards. It should also be stable: adding or removing one shard should not force a complete reshuffling of the data.

A simple answer is a hash function:

\begin{equation}
\text{shard}(x) = H(x) \bmod m
\label{eq:simple-hash-routing}
\end{equation}

This works but is rigid. If the number of shards $m$ changes, almost every item moves. More sophisticated systems use **consistent hashing**, which minimizes movement when the ring changes.

Yellow Phoenix uses a different, mathematically elegant approach based on the **Chinese Remainder Theorem (CRT)**.

### 6.2 What the Chinese Remainder Theorem Says

The Chinese Remainder Theorem is one of the oldest theorems in number theory. In its simplest form, it says that if you know the remainders of a number $x$ when divided by several coprime moduli $m_1, m_2, \dots, m_k$, then you can uniquely determine $x$ modulo the product $M = m_1 m_2 \cdots m_k$.

Formally, if the moduli are pairwise coprime, the system of congruences:

\begin{equation}
x \equiv a_1 \pmod{m_1}, \quad x \equiv a_2 \pmod{m_2}, \quad \dots, \quad x \equiv a_k \pmod{m_k}
\label{eq:crt-system}
\end{equation}

has a unique solution modulo $M = \prod_i m_i$. The solution can be constructed as:

\begin{equation}
x = \sum_{i=1}^{k} a_i \, M_i \, \left(M_i^{-1} \bmod m_i\right) \pmod{M}
\label{eq:crt-solution}
\end{equation}

where $M_i = M / m_i$ and $M_i^{-1} \bmod m_i$ is the modular inverse of $M_i$.

For example, suppose:

\begin{equation}
x \equiv 2 \pmod{3}, \quad x \equiv 3 \pmod{5}, \quad x \equiv 2 \pmod{7}
\label{eq:crt-example}
\end{equation}

Because 3, 5, and 7 are coprime, there is exactly one solution modulo $105$, namely $x \equiv 23 \pmod{105}$. The theorem tells us that a single integer can carry multiple independent "views" simultaneously, one for each modulus.

### 6.3 CRT for Content Routing

In Yellow Phoenix, every paper has a **CUN** — a Content-Universal-Name, essentially a large integer derived from the paper's content. To route a paper to a shard, the system computes:

\begin{equation}
\text{shard}(p) = \text{CUN}(p) \bmod m
\label{eq:cun-shard-routing}
\end{equation}

The CRT perspective becomes powerful when multiple independent routing decisions are needed. Because remainders modulo coprime numbers are independent, the same CUN can simultaneously encode:

- A shard assignment: $\text{CUN} \bmod m_{\text{shard}}$
- A replica placement: $\text{CUN} \bmod m_{\text{replica}}$
- A cache partition: $\text{CUN} \bmod m_{\text{cache}}$

without any of these decisions interfering with the others. This is the practical benefit of CRT: one identifier, many deterministic views.

### 6.4 Prime Position Hashing

A related idea appears in **pattern keys**. When selecting which dimensions of an embedding to use for a hash, Yellow Phoenix uses:

- Power-of-two positions:

\begin{equation}
p_i^{(2)} = 2^{i \bmod 9} \in \{1, 2, 4, 8, 16, 32, 64, 128, 256\}
\label{eq:power-of-two-positions}
\end{equation}

- Prime positions:

\begin{equation}
p_i^{(\pi)} = \text{nth\_prime}(i) \bmod L
\label{eq:prime-positions}
\end{equation}

where $L$ is the embedding length and the first 512 primes are used. Primes are chosen because they interact well with modular arithmetic. When many indices are selected modulo the embedding length, prime offsets reduce collisions and spread information evenly.

### 6.5 Production Evidence

CRT-style routing appears in `src/cun_disk.rs` and `src/pq.rs`. The exact test `entry.cun % m == did` determines shard routing. Prime position hashing lives in `src/pattern_keys.rs`. These systems are production code, not museum pieces.

---

## 7. Causal Inference: Predicting What Causes What

### 7.1 Correlation Is Not Causation

Every monitoring system produces correlations: memory usage rises when query traffic rises; temperature rises when the laptop lid is closed; disk pressure rises when logs accumulate. But not every correlation is useful for action. If memory usage and temperature rise together because both are caused by high load, then cooling the machine will not fix the memory problem.

**Causal inference**, formalized by Judea Pearl, provides a language for distinguishing causation from correlation. A causal model is a directed graph where nodes are variables and edges represent direct causal influences. Once you know the causal graph, you can ask counterfactual questions: "What would happen if I intervened on X?"

### 7.2 Causal Graphs in Yellow Phoenix

Yellow Phoenix maintains a lightweight dynamic causal graph. Nodes are sensors and actuators: `memory_pressure`, `temperature`, `disk_pressure`, `release_memory`, `cool`, `clean_disk`. Edges are causal links with strengths updated from observed event pairs.

The update rule is simple but effective. If $s_{t-1}$ is the current strength of the link $X \rightarrow Y$, then after observing evidence within a time window $W$:

\begin{equation}
s_t = (1 - \alpha) \, s_{t-1} + \alpha \, \mathbb{1}[X \text{ precedes } Y \text{ within } W]
\label{eq:causal-link-update}
\end{equation}

where $\alpha$ is a learning rate and $\mathbb{1}[\cdot]$ is an indicator function. This is online exponential smoothing applied to causal evidence. If $X$ is observed shortly before $Y$, the link strengthens; otherwise it decays.

### 7.3 Predictive Preemption

The ultimate goal is not to describe causality but to **act before failure**. Yellow Phoenix's `PredictivePreemption` module watches sensor trends over a 300-second window and asks the causal engine: "Given what is rising now, what is likely to happen next?"

The trend function classifies a sensor's recent values into one of three categories:

\begin{equation}
\text{trend}(s, W) = \begin{cases}
\text{rising} & \text{if } \frac{ds}{dt} > \theta_{\text{rise}} \text{ over } W \\
\text{falling} & \text{if } \frac{ds}{dt} < -\theta_{\text{fall}} \text{ over } W \\
\text{stable} & \text{otherwise}
\end{cases}
\label{eq:trend-function}
\end{equation}

The decision rule for preemption is:

\begin{equation}
\text{fire actuator } A \text{ if } \text{trend}(S) = \text{falling} \text{ and } \max_{Y} P(Y \mid S) \geq 0.15
\label{eq:preemption-rule}
\end{equation}

The mapping from predicted effect to actuator is:

\begin{align}
\text{memory\_pressure} &\rightarrow \text{release\_memory} \\
\text{temperature} &\rightarrow \text{cool} \\
\text{disk\_pressure} &\rightarrow \text{clean\_disk}
\label{eq:effect-to-action}
\end{align}

For example, if memory pressure is rising and the causal graph says memory pressure strongly precedes out-of-memory crashes, the system fires the `release_memory` actuator before the crash occurs.

### 7.4 Four Types of Correlation

To enrich causal reasoning, Yellow Phoenix computes four correlation types over event history. Let $e_i$ and $e_j$ be two events with sensor names $z_i, z_j$, targets $t_i, t_j$, severities $\sigma_i, \sigma_j$, and timestamps $\tau_i, \tau_j$.

**Temporal correlation:**

\begin{equation}
C_{\text{temp}}(e_i, e_j) = \mathbb{1}[z_i = z_j] \cdot \exp\left(-\frac{|\tau_i - \tau_j|}{\lambda_t}\right)
\label{eq:temporal-correlation}
\end{equation}

**Spatial correlation:**

\begin{equation}
C_{\text{spat}}(e_i, e_j) = \mathbb{1}[t_i = t_j] \cdot \exp\left(-\frac{|\tau_i - \tau_j|}{\lambda_t}\right)
\label{eq:spatial-correlation}
\end{equation}

**Causal correlation:**

\begin{equation}
C_{\text{caus}}(e_i, e_j) = s(z_i \rightarrow z_j) \cdot \mathbb{1}[\tau_i < \tau_j]
\label{eq:causal-correlation}
\end{equation}

**Escalation correlation:**

\begin{equation}
C_{\text{esc}}(e_i, e_j) = \mathbb{1}[z_i = z_j] \cdot \mathbb{1}[\sigma_j > \sigma_i] \cdot \mathbb{1}[\tau_j > \tau_i]
\label{eq:escalation-correlation}
\end{equation}

These are combined into an aggregate risk signal used by the orchestrator's decision court.

### 7.5 Lagged Correlation and Anti-Correlation

The monitor brain also computes lagged Pearson correlation between sensors:

\begin{equation}
\rho(X, Y, \ell) = \frac{\text{Cov}(X_t, Y_{t-\ell})}{\sqrt{\text{Var}(X_t) \text{Var}(Y_{t-\ell})}}
\label{eq:lagged-correlation}
\end{equation}

where $\ell$ is the lag in seconds (default 120). Strong negative correlation between normally coupled sensors — for example, between the RAG retrieval score and the Court safety score — is flagged as an anti-correlation anomaly:

\begin{equation}
\text{alert} = \mathbb{1}[\rho(X, Y, \ell) < -\theta_{\text{anti}}]
\label{eq:anti-correlation-alert}
\end{equation}

### 7.6 Production Evidence

The dynamic causal engine is in `yp_autonomic/cortex/causal.py`. Predictive preemption is in `yp_autonomic/immune/predictive_preemption.py`. The four correlation types are implemented in `yp_autonomic/cortex/correlation.py`. Lagged correlation is in `yp_autonomic/trend_monitor.py`. Memory #119 documents preemption firing correctly during soak tests. During 11 days of continuous operation (PID 25897, commit `5a9ae1e`), predictive preemption prevented 3 out-of-memory crashes and 1 thermal throttling event with zero human intervention.

---

## 8. Enterprise Equilibrium and Safety Math

### 8.1 Sensor Scoring

Not every sensor should run at the same frequency. Cheap sensors poll continuously; expensive sensors run only when triggered. Yellow Phoenix assigns each sensor an equilibrium score based on health, latency, error rate, and drift:

\begin{equation}
\text{score}(s) = w_h \cdot \text{health}_s + w_l \cdot \frac{1}{1 + \text{latency}_s} + w_e \cdot (1 - \text{error\_rate}_s) + w_d \cdot (1 - \text{drift}_s)
\label{eq:equilibrium-score}
\end{equation}

If the score drops below a circuit-breaker threshold, the sensor is temporarily disabled:

\begin{equation}
\text{breaker}(s) = \begin{cases}
\text{open} & \text{if } \text{score}(s) < \theta_{\text{break}} \\
\text{closed} & \text{otherwise}
\end{cases}
\label{eq:circuit-breaker}
\end{equation}

### 8.2 Checksum-Protected State

All persisted state is protected by cryptographic checksums. For a state file with content $c$:

\begin{equation}
\text{checksum}(c) = \text{SHA-256}(c)
\label{eq:checksum}
\end{equation}

Tamper-evident audit entries include proof-of-life hashes that must never be modified after commit.

### 8.3 Thermal-Aware Sleep

Adaptive sleep duration is increased under thermal load:

\begin{equation}
\tau_{\text{sleep}} = \min(2 \tau_{\text{current}}, 30 \text{ s})
\label{eq:thermal-sleep}
\end{equation}

and sensor cycles are skipped for a cooldown period after a thermal event.

---

---

# Part II — Cognitive & Autonomic Architecture

---

## 9. Trinity Cortex

### 9.1 The Seven-Phase Build

The **Trinity Cortex** is the decision and audit layer of Yellow Phoenix. It was built in seven phases: 0, 2, 3, 4, 1, 5, 6. Phase 6 predictor routing is in shadow mode / validation: its decision is logged and compared to the default router, but it does not yet override the production path. Full autonomous actuator execution remains gated for high-risk actions. Each phase added a new capability:

- **Phase 0:** Basic search and hash routing.
- **Phase 2:** Geometric and spectral fallback paths.
- **Phase 3:** Unified mesh and HNSW persistence.
- **Phase 4:** Causal graph and predictive preemption.
- **Phase 1 (out of order):** Holographic memory and event sensors.
- **Phase 5:** Enterprise equilibrium, court validation, and proof-of-life.
- **Phase 6 (live for routing, gated for actuators):** CGT predictor routing and reranking execute without operator approval. Full autonomous execution of high-risk actuators remains gated.

The out-of-order build reflects a real-world discovery process: some capabilities were needed earlier than planned, and others were deferred until safety mechanisms caught up.

### 9.2 Components

Trinity integrates four major subsystems:

- **HolographicMemory:** stores context as vector-superposition interference patterns.
- **PredictivePreemption:** fires actuators before failures occur.
- **EventSensors:** convert system events into causal evidence.
- **Sandbox:** quarantines self-modifications before they reach production.

A query that reaches Trinity is scored by multiple predictors, and the result is only returned if it passes the **Court** safety check. The combined score can be written as:

\begin{equation}
\text{trinity\_score}(q) = \sum_{p \in \text{predictors}} w_p \, \text{score}_p(q) - \lambda \, \text{risk}(q)
\label{eq:trinity-score}
\end{equation}

where $w_p$ are predictor weights and $\lambda \, \text{risk}(q)$ is a penalty for dangerous or unvalidated behavior.

### 9.3 Production Evidence

Trinity is implemented across `yp_autonomic/cortex/`, `yp_autonomic/memory.py`, and `src/trinity/`. The mesh ID mapping `(pid, title, score)` is live in `yp_bridge.py`, and end-to-end search returns correct arXiv papers.

---

## 10. CGT as Cognitive Graph Theory: Tangent Vectors

### 10.1 From Binary Matrix to Tangent Vector

In addition to Combinatorial Geometric Theorem, **CGT** also refers to the **Cognitive Graph Theory** bridge in Trinity. It maps causal conditions -- such as sensor severity and drift -- onto a tangent vector in a geometric manifold. The mapping is:

\begin{equation}
\vec{t} = \text{cgt\_to\_tangent}(\text{condition}, \text{severity}, \text{drift})
\label{eq:cgt-tangent}
\end{equation}

The tangent vector $\vec{t}$ points in the direction the system should "nudge" its retrieval behavior. A high-severity memory condition might produce a tangent that pushes the query toward conservative mode; a drift condition might push it toward retraining.

### 10.2 Why Tangents Matter

A tangent is a local linear approximation to a curved manifold. In the same way that a derivative tells you which way a function is going at a point, a tangent vector tells Trinity which way the system's behavior should move. This is the bridge between the discrete causal graph ("memory pressure caused an OOM") and the continuous geometric brain ("shift the query rotor by 0.1 radians").

The tangent is added to the query embedding or multivector before search:

\begin{equation}
q' = q + \alpha \, \vec{t}
\label{eq:tangent-query}
\end{equation}

with step size $\alpha$ tuned by validation.

### 10.3 Production Evidence

The tangent bridge lives in `src/trinity/cgt_bridge.rs:60`. It is the production path by which causal conditions influence geometric retrieval.

---

## 11. Shadow Validator

### 11.1 Forcing Edge Cases

The **Shadow Validator** is a test harness that deliberately forces the predictor into specific states to verify safety mechanisms. Its canonical test is forcing the predictor to bucket 12 and confirming that the circuit breaker triggers correctly.

The validation loop is:

\begin{equation}
\text{assert}\left( \text{breaker\_opens}(\text{predict}(q_{\text{forced}}) = 12) \right)
\label{eq:shadow-validator}
\end{equation}

This is adversarial testing: instead of waiting for a rare failure, the system synthesizes it.

### 11.2 Production Evidence

Implemented in `src/trinity/shadow_validator.rs:108`. Shadow validation runs during soak tests and after auto-rewire events. A recent CGT reranking validation soak ran 22,245 iterations over 4 minutes with zero errors.

---

## 12. Meta Review

### 12.1 Grading Search Quality

**Meta Review** grades the quality of a search result before it is returned. It assigns a letter grade based on signal agreement, confidence, and risk:

\begin{equation}
\text{grade} = \begin{cases}
A & \text{if } s \geq 90 \\
B & \text{if } 70 \leq s < 90 \\
C & \text{if } 50 \leq s < 70 \\
D & \text{if } 30 \leq s < 50 \\
F & \text{otherwise}
\end{cases}
\label{eq:meta-review-grade}
\end{equation}

where the raw score $s$ combines predictor confidence, causal stability, and geometric agreement.

### 12.2 Currently Unwired

Meta Review grades are computed but **not yet fed back** into the retrieval weights. Once wired, low grades will trigger re-routing or conservative mode.

### 12.3 Production Evidence

Implementation: `yp_autonomic/cortex/meta_review.py`. It runs on every Trinity query.

---

## 13. Self-Modification Sandbox & Replication

### 13.1 The Quarantine Pipeline

Yellow Phoenix can patch its own code, but only inside a **Sandbox**. The pipeline is:

1. **Syntax check:** parse the proposed patch.
2. **Apply:** write a timestamped `.bak` backup.
3. **Soak test:** run the system for a validation window.
4. **Commit:** if no errors, keep the patch.
5. **Rollback:** if any step fails, restore the backup.

The decision function is:

\begin{equation}
\text{patch\_accepted} = \mathbb{1}[\text{syntax\_ok}] \cdot \mathbb{1}[\text{soak\_ok}] \cdot \mathbb{1}[\text{checksum\_match}]
\label{eq:sandbox-accept}
\end{equation}

### 13.2 Six Replication Modules

The replication architecture has six modules wired to `auto_rewire`, the agent, and `enterprise_soak`. They handle code patching, state checksums, backup rotation, and cross-instance synchronization.

### 13.3 Production Evidence

Code: `yp_autonomic/replication/`, `yp_autonomic/replication/auto_rewire.py`, `yp_autonomic/sandbox.py`. 29 FFI wrappers were auto-patched and validated.

---

## 14. Autopoiesis

### 14.1 Self-Patching and Recompilation

**Autopoiesis** is the system's ability to modify `yp_bridge.py` and recompile Rust stubs when a missing bridge is detected. The term is deliberately avoided in SAH (Strip AI Harvest) documentation because it sounds alarming, but the mechanism is real and gated.

The autopoietic cycle is:

\begin{equation}
\text{detect gap} \rightarrow \text{generate wrapper} \rightarrow \text{syntax check} \rightarrow \text{soak} \rightarrow \text{commit}
\label{eq:autopoiesis-cycle}
\end{equation}

Each step is logged and checksum-protected.

### 14.2 Production Evidence

Implementation: `yp_autonomic/replication/auto_rewire.py`. The system has patched itself 29 times under operator supervision.

---

## 15. Holographic Context as Court Evidence

### 15.1 From Memory to Admissible Evidence

In Yellow Phoenix, the **Court** validates whether a decision is safe. Traditionally, court evidence came from explicit sensors and causal links. With the holographic bridge, the court can also accept **holographic_context** -- a compressed superposition of recent events -- as admissible evidence.

The court evaluates:

\begin{equation}
\text{evidence\_weight} = \beta \cdot \text{explicit\_evidence} + (1 - \beta) \cdot \text{holographic\_resonance}
\label{eq:holographic-evidence}
\end{equation}

where $\beta$ is a trust weight and holographic resonance is the similarity between the current context and stored interference patterns.

### 15.2 Production Evidence

Code: `yp_autonomic/cortex/court.py`, `yp_autonomic/memory.py`. Memory #118 documents holographic context accepted as evidence.

---

## 16. Idea Engine & Growth Engine

### 16.1 The Idea Pipeline

The **Idea Engine** is an autonomous proposal system:

1. **Auto feeder:** generates candidate module signatures.
2. **Moonshot filter:** discards obviously weak ideas.
3. **Full gate:** runs property tests and benchmarks.
4. **Digest:** writes approved ideas to `digest.md`.

Weak candidates are museum'd rather than deleted.

### 16.2 Growth Engine

The **Growth Engine** learns new module signatures and runs A/B routing tests through the **Experiment Actuator**. A new module is accepted if it improves a target metric:

\begin{equation}
\Delta R = R_{\text{new}} - R_{\text{baseline}} > \theta_{\text{growth}}
\label{eq:growth-accept}
\end{equation}

### 16.3 Production Evidence

Code: `yp_autonomic/idea_engine/`, `yp_autonomic/actuators/`. The CGT Idea Engine produced 226 approved discoveries and 1,282 composer winners in `cgt_v2/`.

---

# Part III — Retrieval & Search Pipeline

---

## 17. GFH Resonant Field

### 17.1 Fast vs. Discovery Mode

The **GFH Resonant Field** is a dual-mode index:

- **Fast mode:** hash-based lookup, ~25 ms.
- **Fractal attractor mode:** geometric discovery mode, ~80 ms.

Both indexes together occupy 1.6 GB. The router selects mode based on query ambiguity:

\begin{equation}
\text{mode}(q) = \begin{cases}
\text{fast} & \text{if } \text{confidence}(q) > \theta_{\text{fast}} \\
\text{attractor} & \text{otherwise}
\end{cases}
\label{eq:gfh-mode}
\end{equation}

### 17.2 Production Evidence

Implementation is in the autonomic layer. Both indexes are loaded into memory during startup.

---

## 18. Semantic Bypass & Cascade Router Short-Circuit

### 18.1 Bypass for Short Queries

The **Semantic Bypass** detects short or keyword-heavy queries and routes them directly to the geometric hash layer, skipping the embedding model. This saves latency when the query is unlikely to benefit from dense semantics.

\begin{equation}
\text{route}(q) = \begin{cases}
\text{hash\_direct} & \text{if } |q| < L_{\text{short}} \text{ or } \text{keyword\_ratio}(q) > \rho \\
\text{embedding} & \text{otherwise}
\end{cases}
\label{eq:semantic-bypass}
\end{equation}

### 18.2 Cascade Router Short-Circuit

The cascade router tries fast paths first and falls back only when needed:

\begin{equation}
\text{result} = \text{search\_with\_sah}(q) \;\triangleright\; \text{keyword\_fast}(q) \;\triangleright\; \text{production\_fallback}(q)
\label{eq:cascade-short-circuit}
\end{equation}

where $\triangleright$ means "try the left path; if insufficient, use the right."

### 18.3 Production Evidence

Code: `yp_bridge.py`. The SAH beacon path and keyword mesh path both short-circuit the full pipeline.

---

## 19. Multi-Base Confidence & Dynamic Prefix Filter

### 19.1 Multi-Base Confidence

Yellow Phoenix can aggregate confidence across multiple hash resolutions:

\begin{equation}
\text{confidence}(q) = \sum_{b \in \{128, 256, 512\}} w_b \, \text{confidence}_b(q)
\label{eq:multi-base-confidence}
\end{equation}

Higher resolution hashes are more accurate but slower; lower resolution hashes are faster but noisier. The weighted sum lets the router balance speed and accuracy.

### 19.2 Dynamic Top-5% Prefix Filter

Instead of using a fixed Hamming threshold, the **Dynamic Prefix Filter** selects the top 5% of buckets by estimated relevance. This achieves 98-99.8% recall with only 659 candidates, a 20x speedup over fixed-threshold routing.

\begin{equation}
\text{candidates} = \arg\min_{i}^{(0.05 \, N)} \hat{d}(q, x_i)
\label{eq:dynamic-prefix}
\end{equation}

### 19.3 Production Evidence

Code: `src/multi_base_confidence.rs` and the cascade router. Key benchmark: 512-bit Hamming R@1 = 50.6% (real, not the July 8 embedding-cosine 94%).

---

## 20. Two-Tier Binary HNSW & Adaptive Cascade

### 20.1 Two-Tier Architecture

The production plan calls for a **Two-Tier Binary HNSW**:

- **Tier 1:** fast retrieval over binary hashes.
- **Tier 2:** geometric re-rank over multivectors.

Phase 1 (binary HNSW with arena storage) is complete. Later phases will tighten the coupling between tiers.

### 20.2 Adaptive Cascade

The **Adaptive Cascade** systematically tunes thresholds. The key finding is that 512-bit Hamming R@1 is 50.6%, so the cascade uses Hamming for candidate generation and cosine for final ranking.

\begin{equation}
\text{cascade}(q) = \text{rerank}_{\text{cosine}}\left( \text{hnsw}_{\text{Hamming}}(q, K) \right)
\label{eq:adaptive-cascade}
\end{equation}

### 20.3 Production Evidence

Code: `src/binary_hnsw.rs`, `src/exact_cascade.rs`. Phase 1 is live; the 7-phase plan tracks remaining work.

---

## 21. ArXiv 1M Wiring & Domain Detection

### 21.1 The 1.27M Paper Pipeline

The full production pipeline serves 1.27 million arXiv papers:

- Embeddings: stored in `phoenix_arxiv_1m.db`.
- Binary index: `data/binary_hnsw_arxiv1m_m16.bin` (232 MB).
- Metadata: pid, title, abstract, categories.

A query $q$ is routed through the pipeline and returns a paper ID:

\begin{equation}
\text{pid} = \text{lookup}\left( \text{trinity\_audit}\left( \text{rerank}\left( \text{hnsw}\left( \text{itq}\left( \text{minilm}(q) \right) \right) \right) \right) \right)
\label{eq:arxiv-pipeline}
\end{equation}

### 21.2 Domain Detector

The **Domain Detector** classifies queries into CS, Medical, Legal, or General. It scored 10/10 correct in validation. The classification influences routing weights:

\begin{equation}
\text{route}(q) = f_{\text{domain}}\left( \arg\max_d P(d \mid q) \right)
\label{eq:domain-detector}
\end{equation}

### 21.3 LLM Offload Routing

Approximately 85% of queries are answered by direct retrieval; the remaining 15% are offloaded to an LLM for disambiguation.

### 21.4 Production Evidence

Code: `yp_bridge.py`, `data/`, domain detector in `yp_autonomic/`. End-to-end search verified with "neural networks" -> `arxiv1m:cond-mat/9705270`.

---

## 22. ISM, SAH Beacons, Exact Cascade

### 22.1 ISM — Inverted Slot Map

The **ISM** is an inverted slot-based hash index. It builds 200M vectors in 20.2 seconds and queries in 3.7 us single-threaded or 30 us parallel.

\begin{equation}
\text{slot}(h) = h \bmod S
\label{eq:ism-slot}
\end{equation}

\begin{equation}
\text{candidates} = \text{inverted\_map}[\text{slot}(h)]
\label{eq:ism-candidates}
\end{equation}

### 22.2 SAH Beacons

**SAH** (Strip AI Harvest) extracts LLM eigenvectors as semantic beacons. Each beacon is hashed and indexed:

\begin{equation}
\text{beacon\_hash}_i = \text{ITQ}(\text{eigenvector}_i[:512])
\label{eq:sah-beacon}
\end{equation}

On startup, 507 beacons are auto-restored. Search first queries the beacon index; if confidence is low, it falls back to production search.

### 22.3 Exact Cascade with Feedback

The **Exact Cascade** learns from query results. After each query, it records which bucket actually contained the true match:

\begin{equation}
\text{train}(q, b, m, s): \quad \text{weight}(b, m) \leftarrow \text{weight}(b, m) + \eta \, s
\label{eq:exact-cascade-train}
\end{equation}

where $b$ is the bucket, $m$ is the match ID, and $s$ is the score.

### 22.4 Production Evidence

Code: `src/ism/`, `src/sah_hash_bridge.rs`, `src/exact_cascade.rs`. ISM is auto-hybrid; SAH beacons auto-load; exact cascade trains on results.

---

## 23. Adaptive Field Weights & Intent Classification

### 23.1 Adaptive Field Weights

Different parts of a document contribute differently to relevance. **Adaptive Field Weights** learn per-field weights from feedback:

\begin{equation}
\text{score}(q, d) = \sum_{f} w_f \cdot \text{sim}(q_f, d_f)
\label{eq:adaptive-field-weights}
\end{equation}

where $w_f$ is updated based on which fields predict clicks or downstream satisfaction.

### 23.2 Intent Classifier

The **Intent Classifier** chooses between fast and slow paths:

\begin{equation}
\text{path}(q) = \begin{cases}
\text{fast} & \text{if } \text{confidence}(q) > 0.85 \text{ and } |\text{bucket}(q)| < 5 \\
\text{geometric\_brain} & \text{otherwise}
\end{cases}
\label{eq:intent-classifier}
\end{equation}

### 23.3 Re-Bucketing

Papers physically move between hash buckets after 5,000 searches based on query distribution drift. So far, 17 papers have moved.

### 23.4 Production Evidence

Code: `yp_autonomic/adaptive_field.py`, `src/intent_classifier.rs`. Re-bucketing proven in memory #99.

---

## 24. Spectral Stage & Drift

### 24.1 Spectral Stage Re-Rank

The **Spectral Stage** re-ranks candidates using spectral dot-product and anomaly detection. It acts as a middle tier between cheap Hamming search and expensive geometric search.

\begin{equation}
\text{spectral\_score}(q, x) = q^T U U^T x
\label{eq:spectral-score}
\end{equation}

where $U$ contains the top spectral eigenvectors.

### 24.2 Spectral Drift Sensor

The **Spectral Drift Sensor** monitors eigenvectors for distribution drift:

\begin{equation}
\text{drift} = \| U_{\text{current}} - U_{\text{baseline}} \|_F
\label{eq:spectral-drift}
\end{equation}

If drift exceeds a threshold, the system triggers retraining.

### 24.3 Production Evidence

Code: `src/spectral_stage.rs`, `yp_autonomic/sensors/geometric_health.rs`. Drift sensor feeds the causal engine.

---

# Part IV — Enterprise Safety & Sensors

---

## 25. Proof-of-Life Chain

### 25.1 Tamper-Evident Audit

The **Proof-of-Life Chain** is a cryptographic audit log where each entry is hashed with SHA-256 and linked to the previous entry. For entry $n$ with content $c_n$:

\begin{equation}
h_n = \text{SHA-256}(c_n \, || \, h_{n-1})
\label{eq:proof-of-life}
\end{equation}

Entries are immutable. If one is edited, every subsequent hash breaks.

### 25.2 Production Evidence

Code: `yp_autonomic/audit/`, `yp_autonomic/enterprise_soak.py`.

---

## 26. Thermal Lockdown

### 26.1 Thermal-Aware Operations

After a Mac overheated in a bag, Yellow Phoenix implemented **Thermal Lockdown**:

- All launchd jobs are manual-only.
- `caffeinate` is forbidden when unattended.
- Lid sensor triggers thermal protection.

The adaptive sleep under load is:

\begin{equation}
\tau_{\text{sleep}} = \min(2 \tau_{\text{current}}, 30 \text{ s})
\label{eq:thermal-lockdown}
\end{equation}

### 26.2 Production Evidence

Code: `yp_autonomic/sensors/lid_sensor.py`, `yp_autonomic/sensors/resource_pressure.py`. Memory #93 documents the incident.

---

## 27. Data Quarantine

### 27.1 Held-Out Exam Set

Yellow Phoenix maintains a **Data Quarantine** of 106,298 held-out papers. These are frozen `.npy` artifacts never used during training. New harvests are deduplicated and freshly carved before training.

\begin{equation}
\text{train\_set} = \text{harvest} - \text{quarantine} - \text{duplicates}
\label{eq:data-quarantine}
\end{equation}

### 27.2 Production Evidence

Artifacts in `data/`, managed by bench scripts.

---

## 28. 11 Sensors + 4 Actuators

### 28.1 Sensor Matrix

Yellow Phoenix registers 11 sensors:

1. wiring health
2. database health
3. geometric health
4. growth health
5. query load
6. module discovery
7. system hygiene
8. resource pressure
9. lid state
10. memory pressure
11. spectral drift

### 28.2 Actuator Matrix

Four actuators respond to predictions:

1. **release_memory** — force GC and clear caches.
2. **cool** — increase sleep and skip sensor cycles.
3. **clean_disk** — purge old logs and temp files.
4. **notify** — alert the operator.

The mapping from predicted effect to actuator is:

\begin{align}
\text{memory\_pressure} &\rightarrow \text{release\_memory} \\
\text{temperature} &\rightarrow \text{cool} \\
\text{disk\_pressure} &\rightarrow \text{clean\_disk}
\label{eq:sensor-actuator-matrix}
\end{align}

### 28.3 Production Evidence

Code: `yp_autonomic/sensors/`, `yp_autonomic/actuators/`.

---

## 29. Drift Monitoring

### 29.1 Structural Drift

**Drift Monitoring** detects structural changes in the embedding manifold, not just accuracy drops. It found 178 matches in correlation tracking.

\begin{equation}
\text{drift\_alert} = \mathbb{1}\left[ \text{KL}(P_{\text{current}} \,||\, P_{\text{baseline}}) > \theta_{\text{drift}} \right]
\label{eq:drift-monitoring}
\end{equation}

### 29.2 Production Evidence

Code: `yp_autonomic/cortex/correlation.py`, memory #115.

---

## 30. Event-Driven Sensors

### 30.1 Adaptive Polling

Not all sensors should run at the same cadence. Cheap sensors poll periodically; expensive sensors run only on events:

\begin{equation}
\text{schedule}(s) = \begin{cases}
\text{periodic} & \text{if } \text{cost}(s) < \theta_{\text{cheap}} \\
\text{event-driven} & \text{otherwise}
\end{cases}
\label{eq:event-driven-sensors}
\end{equation}

This prevents thermal overload and reduces CPU waste.

### 30.2 Production Evidence

Code: `yp_autonomic/enterprise_soak.py`.

---

## 31. Live Auto-Rewire

### 31.1 Self-Patching with Rollback

**Live Auto-Rewire** inserts missing FFI wrappers into `yp_bridge.py` at runtime. Each patch creates a timestamped `.bak` backup. If the patch produces a `SyntaxError`, the system rolls back automatically.

\begin{equation}
\text{state}_{t+1} = \begin{cases}
\text{patched} & \text{if } \text{compile}(\text{patch}) \text{ succeeds} \\
\text{backup}_{t-1} & \text{otherwise}
\end{cases}
\label{eq:auto-rewire}
\end{equation}

### 31.2 Production Evidence

Code: `yp_autonomic/replication/auto_rewire.py`. 29 wrappers auto-patched; rollback tested.

---

# Part V — Infrastructure & Build

---

## 32. Flat Array Enterprise ISM v0.4

### 32.1 O(1) Target

The **Flat Array Enterprise ISM** targets:

- Query latency: < 10 us
- Build time: < 10 s
- Memory: < 7 GB
- Plus checksum, atomic persist, circuit breaker, and audit.

The design uses a flat array slot map:

\begin{equation}
\text{slot} = h \bmod S, \quad \text{value} = \text{flat\_array}[\text{slot}]
\label{eq:flat-array-ism}
\end{equation}

### 32.2 Production Evidence

Spec approved; implementation in progress.

---

## 33. Watchdog v0.3.1

### 33.1 Soak PID Monitoring

The **Watchdog** monitors the soak PID and restarts it if the heartbeat expires:

\begin{equation}
\text{action} = \begin{cases}
\text{restart} & \text{if } \text{heartbeat\_age} > T_{\text{watchdog}} \\
\text{none} & \text{otherwise}
\end{cases}
\label{eq:watchdog}
\end{equation}

### 33.2 Production Evidence

Spec pending; intended to prevent the exact heartbeat loss that just occurred.

---

## 34. M1 Build Components

### 34.1 Seven Verified Components

The **M1 Build** verifies seven components:

1. Registry
2. Cache
3. Router
4. Self-Learning
5. Feeder
6. Collaborative
7. Fuzz

Each component passes a gate before release.

### 34.2 Production Evidence

Code: `yp_autonomic/`.

---

## 35. Sharded Path Fix

### 35.1 Rust-to-Python Bridge

The **Sharded Path Fix** exposes sharded mesh operations to Python:

\begin{equation}
\text{rust\_id} = \text{\_rust\_id}(\text{shard}, \text{local\_id})
\label{eq:sharded-rust-id}
\end{equation}

\begin{equation}
\text{result} = \text{search\_sharded}(q, \text{shard\_mask})
\label{eq:search-sharded}
\end{equation}

### 35.2 Production Evidence

Code: `yp_bridge.py`.

---

## 36. M3.7-M3.9b Wiring

### 36.1 Performance Milestone

The **M3.7-M3.9b** wiring milestone achieved:

- 10K benchmark P50: 361 us
- 2,553 QPS
- All 7 WIRE_ME modules connected.

### 36.2 Production Evidence

Code: `src/`, `yp_bridge.py`.

---

## 37. 128 FFI Functions

### 37.1 Full Bridge

The Rust-to-Python bridge exposes **128 FFI functions** from `src/ffi_unified.rs`. They cover search, indexing, hashing, holographic memory, causal graphs, and enterprise state.

### 37.2 Production Evidence

Code: `src/ffi_unified.rs`, loaded in `yp_bridge.py` via `RustBridge`.

---

## 38. Cat J + Cat B

### 38.1 Structured Build Categories

**Cat J** and **Cat B** are structured build categories used in the release process. A DeepSeek audit scored them 5/5 PASS.

### 38.2 Production Evidence

Code: `src/crystal.rs`.

---

## 39. Synthetic Papers

### 39.1 1M Synthetic Dataset

Yellow Phoenix can generate synthetic papers for scale testing:

- File: `yellow_1m_metadata.jsonl` (886 MB, 24.2 s generation).
- 1 million records generated.
- 20 million pending.

### 39.2 Production Evidence

Code: `scripts/generate_synthetic.py`.

---

# Part VI — Experimental & Future

---

## 40. Mirror Mesh / Federation

### 40.1 Distributed Geometric Brain

**Mirror Mesh** is a distributed geometric brain with swarm consensus. The goal is O(1) retrieval via reflection across a federation of nodes.

Each node maintains a local multivector space. Queries are reflected through a consensus layer:

\begin{equation}
\text{result} = \text{consensus}\left( \bigcup_{n \in \text{nodes}} \text{reflect}(q, M_n) \right)
\label{eq:mirror-mesh}
\end{equation}

### 40.2 Production Evidence

Architecture stage; user is sole gatekeeper.

---

## 41. Energy / Temporal / Mesh Snapshot

### 41.1 Extra Dimensions

The geometric brain is being extended with three extra dimensions:

- **Energy:** query load and CPU cost.
- **Temporal:** time-decay of relevance.
- **Mesh Snapshot:** checkpoint state for rollback.

A query vector becomes:

\begin{equation}
q_{\text{extended}} = [q_{\text{semantic}}, q_{\text{energy}}, q_{\text{temporal}}, q_{\text{snapshot}}]
\label{eq:extended-query}
\end{equation}

### 41.2 Production Evidence

Code: `src/`.

---

## 42. Feedback Loops

### 42.1 Closed-Loop Learning

**Feedback Loops** close the loop from search results back to the cascade router and field weights. Commit `a18819b` established the first closed loop.

\begin{equation}
\Delta w_f = \eta \left( r_{\text{observed}} - r_{\text{expected}} \right) \frac{\partial \, \text{score}}{\partial w_f}
\label{eq:feedback-loop}
\end{equation}

### 42.2 Production Evidence

Code: `yp_autonomic/`.

---

## 43. Disk Preemption

### 43.1 Preemptive Cleanup

**Disk Preemption** cleans up files before disk space runs out, rather than reacting after a failure. The actuator is `clean_disk`, triggered by predictive preemption.

\begin{equation}
\text{fire clean\_disk if } P(\text{disk\_full} \mid \text{pressure}) > 0.15
\label{eq:disk-preemption}
\end{equation}

### 43.2 Production Evidence

Code: `yp_autonomic/immune/`, `yp_autonomic/actuators/clean_disk.py`.

---

## 44. YP vs FAISS Benchmarks

### 44.1 Head-to-Head Comparison

Yellow Phoenix is benchmarked against FAISS from 13K up to 1.27M vectors. The claim is O(1) Yellow Phoenix behavior versus O(log n) FAISS scaling.

\begin{equation}
T_{\text{YP}}(n) \approx C_1, \quad T_{\text{FAISS}}(n) \approx C_2 \log n
\label{eq:yp-vs-faiss}
\end{equation}

### 44.2 Production Evidence

Code: `yp_real_bench/`, `benchmark_results/`.

---

## 45. ArXiv Harvest Pipeline

### 45.1 Scaling the Corpus

The **ArXiv Harvest Pipeline** grows the corpus from 1.19M to 2.5M papers. Each new harvest is deduplicated and carved into train/quarantine sets before indexing.

\begin{equation}
\text{new\_index} = \text{dedup}(\text{harvest}) \cup \text{old\_index}
\label{eq:arxiv-harvest}
\end{equation}

### 45.2 Production Evidence

Harvest scripts in the project root and `scripts/`.

---

## 46. Putting It All Together: The Yellow Phoenix Search Pipeline

A single query in Yellow Phoenix flows through most of the systems described above:

1. **Encoding.** The query text $q$ is converted by MiniLM into a 384-dimensional embedding $x_q$.
2. **ITQ hashing.** The embedding is whitened, rotated by the learned ITQ matrix, and thresholded to a 512-bit hash $h_q = \text{sign}(V R)$.
3. **HNSW search.** The hash is inserted into the navigable small-world graph, and a greedy multi-layer search returns the top 500 candidate hashes.
4. **Re-rank.** The original embeddings of the 500 candidates are compared to the query embedding by cosine similarity.
5. **Trinity audit.** The top results pass through an audit layer that maps mesh IDs back to paper metadata and checks hash consistency.
6. **Autonomic feedback.** Search events feed into the causal engine and holographic context, allowing the system to learn from its own traffic.

Mathematically, the pipeline is:

\begin{align}
x_q &= \text{MiniLM}(q) \\
h_q &= \text{sign}\left((x_q - \mu)^T W_c \Lambda_c^{-1/2} R\right) \\
\mathcal{C} &= \arg\min_{i}^{(500)} d_H(h_q, h_i) \\
\text{result} &= \arg\max_{i \in \mathcal{C}} \frac{x_q \cdot x_i}{\|x_q\| \|x_i\|}
\label{eq:full-pipeline}
\end{align}

The result is a pipeline that answers queries over more than a million papers in well under a second, with quality comparable to a brute-force dense search.

---

## 47. Appendix: Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A dense vector that represents the meaning of text. |
| **Hash** | A compact binary code derived from an embedding. |
| **HNSW** | Hierarchical Navigable Small World graph for fast nearest-neighbor search. |
| **ITQ** | Iterative Quantization, a method for learning binary hashes. |
| **HRR** | Holographic Reduced Representation, a vector-binding operation. |
| **VSA** | Vector Symbolic Architecture, a family of vector-based symbolic AI systems. |
| **Multivector** | An object in geometric algebra with scalar, vector, and higher-grade parts. |
| **CUN** | Content-Universal-Name, a deterministic content-derived identifier. |
| **CRT** | Chinese Remainder Theorem, used for modular routing. |
| **Causal graph** | A directed graph representing cause-and-effect relationships. |
| **CGT** | Combinatorial Geometric Theorem, a binary-matrix invariant and discovery engine. |
| **chi ($\chi$)** | The CGT characteristic: bits − h − v + 2·$x_2$. |
| **TangentHypothesis** | A CGT conjecture injected into Trinity for validation. |
| **Functor** | A structure-preserving map between retrieval categories. |
| **Trinity Cortex** | Decision and audit layer integrating predictors, court, and sandbox. |
| **Shadow Validator** | Adversarial test harness that forces edge cases. |
| **Meta Review** | Search-quality grader (currently unwired to weights). |
| **Autopoiesis** | Self-patching mechanism gated by sandbox and soak tests. |
| **GFH** | Geometric Fractal Hash / resonant-field dual-mode index. |
| **Semantic Bypass** | Keyword/short-query fast path that skips embedding. |
| **Dynamic Prefix Filter** | Top-5% bucket selection for 20x speedup. |
| **ISM** | Inverted Slot Map hash index. |
| **SAH** | Strip AI Harvest semantic beacon index. |
| **Spectral Drift** | Eigenvector-based distribution drift detection. |
| **Proof-of-Life** | Cryptographic audit chain with SHA-256 linking. |
| **Flat Array ISM** | O(1) target hash index with checksum and audit. |
| **Mirror Mesh** | Distributed geometric brain with swarm consensus. |

---

*End of reference.*
