// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! PQHNSW: HNSW graph over product-quantized codes (the crown engine, Way 2).
//! Distance is ADC: per-query tables over per-block codebooks; candidate
//! distance = sum of table lookups. Mirrors the proven BinaryHNSW algorithm
//! (exp-decay layers, ef-beam search, M-neighbor pruning at insert).
//! Methodology: HNSW (Malkov & Yashunin 2020) + PQ/OPQ (Jégou 2011, Ge 2013)
//! — published literature, independent implementation, no third-party code.
//!
//! Serialization: magic "YPHQ", v1. Codes stored block-major per node.
//! Query side: caller builds QueryCtx (rotate + tables) via PqCodec.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

/// Per-block codebooks + rotation, shared by build and query.
pub struct PqCodec {
    pub nb: usize,          // blocks
    pub k: usize,           // codebook size (256)
    pub bl: usize,          // dims per block
    pub books: Vec<f32>,    // nb * k * bl
    pub v: Vec<f32>,        // d x d rotation (row-major)
    pub mu: Vec<f32>,       // d
    pub d: usize,
}

/// Per-query context: rotated query + ADC tables + ITQ hash (the compass).
pub struct QueryCtx {
    pub tables: Vec<f32>,   // nb * k
    pub qhash: [u8; 64],
    pub qsk: [f32; 3],      // query's top-3 PCA dims
}

impl PqCodec {
    /// Rotate a raw query and build its ADC tables.
    pub fn encode_query(&self, q: &[f32]) -> QueryCtx {
        let d = self.d;
        let mut qr = vec![0f32; d];
        for i in 0..d { qr[i] = q[i] - self.mu[i]; }
        let mut qrot = vec![0f32; d];
        for b in 0..d {
            let mut s = 0f32;
            for i in 0..d { s += qr[i] * self.v[i * d + b]; }
            qrot[b] = s;
        }
        let mut tables = vec![0f32; self.nb * self.k];
        for b in 0..self.nb {
            for c in 0..self.k {
                let mut s = 0f32;
                for dd in 0..self.bl {
                    let diff = self.books[(b * self.k + c) * self.bl + dd]
                        - qrot[b * self.bl + dd];
                    s += diff * diff;
                }
                tables[b * self.k + c] = s;
            }
        }
        QueryCtx { tables, qhash: [0u8; 64], qsk: [0f32; 3] }
    }

    /// Quantize a rotated vector to PQ codes.
    pub fn quantize(&self, qrot: &[f32]) -> Vec<u8> {
        let mut code = vec![0u8; self.nb];
        for b in 0..self.nb {
            let mut best = f32::MAX;
            let mut bi = 0usize;
            for c in 0..self.k {
                let mut s = 0f32;
                for dd in 0..self.bl {
                    let diff = self.books[(b * self.k + c) * self.bl + dd]
                        - qrot[b * self.bl + dd];
                    s += diff * diff;
                }
                if s < best { best = s; bi = c; }
            }
            code[b] = bi as u8;
        }
        code
    }

    /// Rotate a raw vector (no tables).
    pub fn rotate(&self, q: &[f32]) -> Vec<f32> {
        let d = self.d;
        let mut qr = vec![0f32; d];
        for i in 0..d { qr[i] = q[i] - self.mu[i]; }
        let mut qrot = vec![0f32; d];
        for b in 0..d {
            let mut s = 0f32;
            for i in 0..d { s += qr[i] * self.v[i * d + b]; }
            qrot[b] = s;
        }
        qrot
    }

    /// ADC distance from ctx to a code (nb bytes, block-major).
    #[inline]
    pub fn adc(&self, ctx: &QueryCtx, code: &[u8]) -> f32 {
        let mut s = 0f32;
        for b in 0..self.nb {
            s += ctx.tables[b * self.k + code[b] as usize];
        }
        s
    }
}

pub struct PqNode {
    pub id: u64,
    pub code: Vec<u8>,       // nb bytes (PQ)
    pub hash: [u8; 64],      // ITQ-512 (fusion compass)
    pub sketch: [f32; 3],    // top-3 PCA dims (M3.3 spectral compass, true space)
    pub edges: Vec<Vec<u32>>, // per layer
    pub level: usize,
}

pub struct PqHnsw {
    pub nodes: Vec<PqNode>,
    pub m: usize,
    pub ef_construction: usize,
    pub enter_point: Option<u32>,
    pub max_level: usize,
    pub codec: PqCodec,
    pub itq_wt: Vec<f32>,   // 512 x 128 transposed ITQ matrix
    pub itq_mu: Vec<f32>,   // 128
    pub fuse_w: f32,        // hamming weight (ADC units per bit)
    pub sketch_w: f32,      // M3.3 spectral sketch weight
}

fn hamming64(a: &[u8; 64], b: &[u8; 64]) -> u32 {
    let mut s = 0u32;
    for i in 0..64 {
        s += (a[i] ^ b[i]).count_ones();
    }
    s
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b))
}
fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0])
}

impl PqHnsw {
    pub fn new(m: usize, ef_construction: usize, codec: PqCodec,
               itq_wt: Vec<f32>, itq_mu: Vec<f32>, fuse_w: f32, sketch_w: f32) -> Self {
        assert!(itq_wt.len() == 512 * 128 && itq_mu.len() == 128);
        PqHnsw { nodes: Vec::new(), m, ef_construction, enter_point: None, max_level: 0,
                 codec, itq_wt, itq_mu, fuse_w, sketch_w }
    }

    fn random_level(&self, rng: &mut u64) -> usize {
        // xorshift, exp decay with classic multiplier 1/ln(m)
        *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
        let u = (*rng >> 11) as f64 / (1u64 << 53) as f64;
        let mult = 1.0 / (self.m.max(2) as f64).ln();
        (-u.ln() * mult) as usize
    }

    pub fn node_label(&self, idx: u32) -> u64 {
        self.nodes[idx as usize].id
    }

    /// Full query ctx: ADC tables + ITQ hash + spectral sketch (compasses).
    pub fn encode_query(&self, q: &[f32]) -> QueryCtx {
        let mut ctx = self.codec.encode_query(q);
        let qrot = self.codec.rotate(q);
        ctx.qsk = [qrot[0], qrot[1], qrot[2]];
        let mut qm = [0f32; 128];
        for i in 0..128 { qm[i] = q[i] - self.itq_mu[i]; }
        let mut hb = [0u8; 64];
        for b in 0..512 {
            let mut s = 0f32;
            for d in 0..128 {
                s += qm[d] * self.itq_wt[b * 128 + d];
            }
            if s >= 0.0 { hb[b / 8] |= 1 << (7 - (b % 8)); }
        }
        ctx.qhash = hb;
        ctx
    }

    /// Fused distance: ADC + fuse_w*hamming + sketch_w*L2(top3 PCA)
    /// (user's fusion + Yellow M3.3 spectral compass, in-graph).
    #[inline]
    pub fn fused(&self, ctx: &QueryCtx, node: &PqNode) -> f32 {
        let d0 = ctx.qsk[0] - node.sketch[0];
        let d1 = ctx.qsk[1] - node.sketch[1];
        let d2 = ctx.qsk[2] - node.sketch[2];
        self.codec.adc(ctx, &node.code)
            + self.fuse_w * hamming64(&ctx.qhash, &node.hash) as f32
            + self.sketch_w * (d0 * d0 + d1 * d1 + d2 * d2).sqrt()
    }

    /// Search: ef-beam from top layer down, then bottom layer for results.
    pub fn search_with_ef(&self, ctx: &QueryCtx, k: usize, ef: usize) -> Vec<(f32, u32)> {
        if self.nodes.is_empty() { return Vec::new(); }
        let ef = ef.max(k);
        let mut cand: Vec<(f32, u32)> = Vec::new();
        if let Some(ep) = self.enter_point {
            let mut ep = ep;
            for lvl in (1..=self.max_level).rev() {
                ep = self.greedy_layer(ctx, ep, lvl);
            }
            cand = self.beam_layer(ctx, ep, 0, ef);
        }
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cand.truncate(k);
        cand
    }

    /// Search starting from a given entry node (compass warm-start experiment).
    pub fn search_with_ef_from(&self, ctx: &QueryCtx, k: usize, ef: usize, entry: u32) -> Vec<(f32, u32)> {
        if self.nodes.is_empty() { return Vec::new(); }
        let ef = ef.max(k);
        let mut ep = entry.min(self.nodes.len() as u32 - 1);
        for lvl in (1..=self.max_level).rev() {
            ep = self.greedy_layer(ctx, ep, lvl);
        }
        let mut cand = self.beam_layer(ctx, ep, 0, ef);
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cand.truncate(k);
        cand
    }

    /// SDC code-to-code distance (symmetric, book-to-book; no query needed).
    pub fn sdc(&self, a: &[u8], b: &[u8]) -> f32 {
        let (nb, k, bl, books) = (self.codec.nb, self.codec.k, self.codec.bl, &self.codec.books);
        let mut s = 0f32;
        for j in 0..nb {
            let oa = (j * k + a[j] as usize) * bl;
            let ob = (j * k + b[j] as usize) * bl;
            for dd in 0..bl {
                let d = books[oa + dd] - books[ob + dd];
                s += d * d;
            }
        }
        s
    }

    /// Vamana-style diverse-neighbor pruning on layer-0 edges (SDC distances).
    /// Same rule as the binary graph: cut nb iff sdc(nb, kept_p) < alpha * sdc(node, nb).
    pub fn prune_diverse(&mut self, alpha: f32) -> usize {
        let mut removed = 0usize;
        for ni in 0..self.nodes.len() {
            let nbs = self.nodes[ni].edges.get(0).cloned().unwrap_or_default();
            if nbs.len() <= 2 { continue; }
            let mut by: Vec<(f32, u32)> = nbs.iter()
                .map(|&nb| (self.sdc(&self.nodes[ni].code.clone(), &self.nodes[nb as usize].code), nb))
                .collect();
            by.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
            let mut kept: Vec<u32> = Vec::with_capacity(by.len());
            for (d, nb) in by {
                let thresh = if d <= 0.0 { f32::EPSILON } else { alpha * d };
                let ndc = self.nodes[nb as usize].code.clone();
                let redundant = kept.iter().any(|&p| self.sdc(&ndc, &self.nodes[p as usize].code) < thresh);
                if !redundant { kept.push(nb); }
            }
            if kept.len() < nbs.len() {
                removed += nbs.len() - kept.len();
                self.nodes[ni].edges[0] = kept;
            }
        }
        removed
    }

    fn greedy_layer(&self, ctx: &QueryCtx, mut ep: u32, lvl: usize) -> u32 {
        loop {
            let mut best_d = self.fused(ctx, &self.nodes[ep as usize]);
            let mut best_nb = ep;
            for &nb in &self.nodes[ep as usize].edges.get(lvl).cloned().unwrap_or_default() {
                let d = self.fused(ctx, &self.nodes[nb as usize]);
                if d < best_d { best_d = d; best_nb = nb; }
            }
            if best_nb == ep { return ep; }
            ep = best_nb;
        }
    }

    fn beam_layer(&self, ctx: &QueryCtx, ep: u32, lvl: usize, ef: usize) -> Vec<(f32, u32)> {
        #[cfg(debug_assertions)]
        {}
        crate::pq_hnsw::BEAM_EXPANSIONS.with(|c| c.set(c.get() + 1));
        use std::collections::BinaryHeap;
        let mut visited = vec![false; self.nodes.len()];
        let mut results: BinaryHeap<(ordered::OrdF32, u32)> = BinaryHeap::new();
        let mut queue: BinaryHeap<(ordered::OrdF32, u32)> = BinaryHeap::new();
        let d0 = self.fused(ctx, &self.nodes[ep as usize]);
        queue.push((ordered::OrdF32(-d0), ep));
        results.push((ordered::OrdF32(d0), ep));
        visited[ep as usize] = true;
        while let Some((neg_d, u)) = queue.pop() {
            let d = -neg_d.0;
            let worst = results.peek().map(|r| r.0 .0).unwrap_or(f32::INFINITY);
            if d > worst && results.len() >= ef { break; }
            for &nb in &self.nodes[u as usize].edges.get(lvl).cloned().unwrap_or_default() {
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let nd = self.fused(ctx, &self.nodes[nb as usize]);
                let worst2 = results.peek().map(|r| r.0 .0).unwrap_or(f32::INFINITY);
                if nd < worst2 || results.len() < ef {
                    queue.push((ordered::OrdF32(-nd), nb));
                    results.push((ordered::OrdF32(nd), nb));
                    if results.len() > ef { results.pop(); }
                }
            }
        }
        results.into_iter().map(|(od, i)| (od.0, i)).collect()
    }

    /// Insert a point given its RAW vector (build-time; the graph stores
    /// codes). Insertion ctx = true rotated vector; neighbor-edge pruning
    /// ranks by ADC to the neighbor's raw vector.
    pub fn insert(&mut self, id: u64, raw: &[f32], raws: &[f32]) {
        let qrot = self.codec.rotate(raw);
        let sketch = [qrot[0], qrot[1], qrot[2]];
        let code = self.codec.quantize(&qrot);
        let ctx = self.encode_query(raw);
        // ITQ hash of the new point
        let mut qm = [0f32; 128];
        for i in 0..128 { qm[i] = raw[i] - self.itq_mu[i]; }
        let mut hash = [0u8; 64];
        for b in 0..512 {
            let mut s = 0f32;
            for d in 0..128 { s += qm[d] * self.itq_wt[b * 128 + d]; }
            if s >= 0.0 { hash[b / 8] |= 1 << (7 - (b % 8)); }
        }
        self.insert_with_ctx(id, code, hash, sketch, &ctx, raws);
    }

    pub fn insert_with_ctx(&mut self, id: u64, code: Vec<u8>, hash: [u8; 64], sketch: [f32; 3], ctx: &QueryCtx, raws: &[f32]) {
        let mut rng = 0x9E3779B97F4A7C15u64 ^ id.wrapping_mul(0x2545F4914F6CDD1D);
        let level = self.random_level(&mut rng);
        let idx = self.nodes.len() as u32;
        // push first so neighbor-edge pruning can reference idx
        self.nodes.push(PqNode { id, code, hash, sketch, edges: vec![Vec::new(); level + 1], level });
        let mut ep = self.enter_point;
        for lvl in (level + 1..=self.max_level).rev() {
            if let Some(e) = ep { ep = Some(self.greedy_layer(ctx, e, lvl)); }
        }
        let mut cur_ep = ep;
        for lvl in (0..=level.min(self.max_level)).rev() {
            let (cands, best_ep) = if let Some(e) = cur_ep {
                let c = self.beam_layer(ctx, e, lvl, self.ef_construction);
                let be = c.iter()
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|&(_, i)| i).unwrap_or(e);
                (c, be)
            } else { (Vec::new(), 0) };
            cur_ep = Some(best_ep);
            let mut sel: Vec<(f32, u32)> = cands;
            sel.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            sel.truncate(self.m);
            for &(_, nb) in &sel {
                self.nodes[idx as usize].edges[lvl].push(nb);
            }
            for &(_, nb) in &sel {
                let edges = &mut self.nodes[nb as usize].edges;
                while edges.len() <= lvl { edges.push(Vec::new()); }
                edges[lvl].push(idx);
                if edges[lvl].len() > self.m {
                    let d = self.codec.d;
                    let nraw = &raws[nb as usize * d..nb as usize * d + d];
                    let nctx = self.encode_query(nraw);
                    let edge_snapshot: Vec<u32> = self.nodes[nb as usize].edges[lvl].clone();
                    let mut scored: Vec<(f32, u32)> = edge_snapshot.iter()
                        .map(|&e| (self.fused(&nctx, &self.nodes[e as usize]), e)).collect();
                    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    scored.truncate(self.m);
                    self.nodes[nb as usize].edges[lvl] = scored.into_iter().map(|(_, e)| e).collect();
                }
            }
        }
        if level > self.max_level || self.enter_point.is_none() {
            self.max_level = self.max_level.max(level);
            self.enter_point = Some(idx);
        }
    }

    // ---- serialization ----
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"YPHQ")?;
        w.write_all(&1u16.to_le_bytes())?;
        w.write_all(&(self.codec.nb as u8).to_le_bytes())?;
        w.write_all(&(self.codec.bl as u8).to_le_bytes())?;
        w.write_all(&0u16.to_le_bytes())?;
        w.write_all(&(self.codec.k as u32).to_le_bytes())?;
        w.write_all(&(self.codec.d as u32).to_le_bytes())?;
        // codec payload: books, v, mu
        w.write_all(&self.codec.books.len().to_le_bytes())?;
        for x in &self.codec.books { w.write_all(&x.to_le_bytes())?; }
        for x in &self.codec.v { w.write_all(&x.to_le_bytes())?; }
        for x in &self.codec.mu { w.write_all(&x.to_le_bytes())?; }
        for x in &self.itq_wt { w.write_all(&x.to_le_bytes())?; }
        for x in &self.itq_mu { w.write_all(&x.to_le_bytes())?; }
        // graph
        w.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        w.write_all(&(self.m as u32).to_le_bytes())?;
        w.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        w.write_all(&self.fuse_w.to_le_bytes())?;
        w.write_all(&self.sketch_w.to_le_bytes())?;
        w.write_all(&(self.max_level as u32).to_le_bytes())?;
        w.write_all(&self.enter_point.unwrap_or(u32::MAX).to_le_bytes())?;
        for n in &self.nodes {
            w.write_all(&n.id.to_le_bytes())?;
            w.write_all(&(n.level as u32).to_le_bytes())?;
            w.write_all(&(n.edges.len() as u32).to_le_bytes())?;
            for e in &n.code { w.write_all(&[*e])?; }
            w.write_all(&n.hash)?;
            for x in &n.sketch { w.write_all(&x.to_le_bytes())?; }
            for layer in &n.edges {
                w.write_all(&(layer.len() as u32).to_le_bytes())?;
                for &e in layer { w.write_all(&e.to_le_bytes())?; }
            }
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"YPHQ", "bad magic");
        let _ver = read_u16(&mut r)?;
        let nb = read_u8(&mut r)? as usize;
        let bl = read_u8(&mut r)? as usize;
        let _ = read_u16(&mut r)?;
        let k = read_u32(&mut r)? as usize;
        let d = read_u32(&mut r)? as usize;
        let blen = read_u64(&mut r)? as usize;
        let mut books = vec![0f32; blen];
        for x in books.iter_mut() { *x = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b }); }
        let mut v = vec![0f32; d * d];
        for x in v.iter_mut() { *x = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b }); }
        let mut mu = vec![0f32; d];
        for x in mu.iter_mut() { *x = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b }); }
        let mut itq_wt = vec![0f32; 512 * 128];
        for x in itq_wt.iter_mut() { *x = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b }); }
        let mut itq_mu = vec![0f32; 128];
        for x in itq_mu.iter_mut() { *x = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b }); }
        let n = read_u64(&mut r)? as usize;
        let m = read_u32(&mut r)? as usize;
        let ef = read_u32(&mut r)? as usize;
        let fuse_w = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b });
        let sketch_w = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b });
        let max_level = read_u32(&mut r)? as usize;
        let ep = read_u32(&mut r)?;
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            let id = read_u64(&mut r)?;
            let level = read_u32(&mut r)? as usize;
            let n_layers = read_u32(&mut r)? as usize;
            let mut code = vec![0u8; nb];
            r.read_exact(&mut code)?;
            let mut hash = [0u8; 64];
            r.read_exact(&mut hash)?;
            let mut sketch = [0f32; 3];
            for x in sketch.iter_mut() { *x = f32::from_le_bytes({ let mut b=[0u8;4]; r.read_exact(&mut b)?; b }); }
            let mut edges = Vec::with_capacity(n_layers);
            for _ in 0..n_layers {
                let cnt = read_u32(&mut r)? as usize;
                let mut layer = Vec::with_capacity(cnt);
                for _ in 0..cnt { layer.push(read_u32(&mut r)?); }
                edges.push(layer);
            }
            nodes.push(PqNode { id, code, hash, sketch, edges, level });
        }
        Ok(PqHnsw {
            nodes,
            m,
            ef_construction: ef,
            enter_point: if ep == u32::MAX { None } else { Some(ep) },
            max_level,
            codec: PqCodec { nb, k, bl, books, v, mu, d },
            itq_wt,
            itq_mu,
            fuse_w,
            sketch_w,
        })
    }
}

thread_local! {
    pub static BEAM_EXPANSIONS: std::cell::Cell<u64> = std::cell::Cell::new(0);
}

mod ordered {
    use std::cmp::Ordering;
    #[derive(PartialEq)]
    pub struct OrdF32(pub f32);
    impl Eq for OrdF32 {}
    impl PartialOrd for OrdF32 {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for OrdF32 {
        fn cmp(&self, o: &Self) -> Ordering {
            self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal)
        }
    }
}
