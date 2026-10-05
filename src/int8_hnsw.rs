// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Int8HNSW: HNSW graph over 128-byte residual-int8 codes with coarse centroid table.
//! Distance (honest, query): rq = clip(round(q - C[node.cell]), -127,127); d = sum (rq - code)^2
//! with per-query cell-change caching. Distance (build/pre-centered): sum (code_a - code_b)^2.
//! Serialization: magic "YPI8" v1.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub const D: usize = 128;

/// Total-order wrapper for f32 heap keys (mirrors pq_hnsw::ordered).
mod ordered {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct OrdF32(pub f32);
    impl Eq for OrdF32 {}
    impl PartialOrd for OrdF32 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }
    impl Ord for OrdF32 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}

pub struct I8Node {
    pub id: u64,
    pub cell: u16,
    pub code: [i8; D],
    pub sq: i32,                   // sum(code^2) (legacy i32 path / prune)
    pub xsq: f32,                  // sum((C[cell]+code)^2): reconstruction norm for recon-L2
    pub edges: Vec<Vec<u32>>,      // primary
    pub alt_edges: Vec<Vec<u32>>,  // alternative (next-m candidates)
    pub level: usize,
}

pub struct I8Hnsw {
    pub gateways: Vec<u32>,        // per coarse cell: node nearest centroid (min code norm)
    pub use_gateway: bool,         // T7 compass: start search at gateways[own_cell]
    pub pre_centered: bool,
    pub nodes: Vec<I8Node>,
    stamp: Vec<u32>,
    epoch: u32,
    pub coarse: Vec<f32>,   // KC x D
    pub kc: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub enter_point: Option<u32>,
    pub max_level: usize,
}

/// v1 distance view: per-candidate-cell clipped residual (i8) with per-cell cache,
/// L2 via maddubs SIMD identity (bit-exact). cache_res = clip(round(q - C[cache_cell])).
pub struct I8Ctx {
    pub q: [f32; D],
    pub own_cell: usize,
    pub v_sq: f32,
    cache_cell: u32,
    pub cache_res: [i8; D],
    cache_sq: i32,
}

impl I8Ctx {
    #[inline]
    pub fn d(&mut self, coarse: &[f32], node: &I8Node) -> i32 {
        if self.cache_cell != node.cell as u32 {
            let base = node.cell as usize * D;
            let mut sq = 0i32;
            for i in 0..D {
                let r = (self.q[i] - coarse[base + i]).round().clamp(-127.0, 127.0) as i8;
                sq += r as i32 * r as i32;
                self.cache_res[i] = r;
            }
            self.cache_sq = sq;
            self.cache_cell = node.cell as u32;
        }
        crate::simd_kernels::i8_l2_128(&self.cache_res, &node.code, self.cache_sq, node.sq)
    }
}

impl I8Hnsw {
    pub fn new(m: usize, ef_construction: usize, coarse: Vec<f32>, kc: usize) -> Self {
        Self { gateways: Vec::new(), use_gateway: false, pre_centered: false, nodes: Vec::new(), stamp: Vec::new(), epoch: 0, coarse, kc, m, ef_construction, enter_point: None, max_level: 0 }
    }

    pub fn residual(&self, q: &[f32], cell: usize) -> [i8; D] {
        let base = cell * D;
        let mut r = [0i8; D];
        for i in 0..D {
            let v = (q[i] - self.coarse[base + i]).round().clamp(-127.0, 127.0);
            r[i] = v as i8;
        }
        r
    }

    /// Reconstruct a node's approx raw vector: C[cell] + code.
    pub fn recon(&self, cell: u16, code: &[i8; D]) -> [f32; D] {
        let base = cell as usize * D;
        let mut r = [0f32; D];
        for i in 0..D { r[i] = self.coarse[base + i] + code[i] as f32; }
        r
    }

    /// Residual of v vs a cell, written to out; returns sum(residual^2).
    #[inline]
    pub fn residual_buf(&self, v: &[f32; D], cell: usize, out: &mut [i8; D]) -> i32 {
        let base = cell * D;
        let mut sq = 0i32;
        for i in 0..D {
            let r = (v[i] - self.coarse[base + i]).round().clamp(-127.0, 127.0) as i8;
            sq += r as i32 * r as i32;
            out[i] = r;
        }
        sq
    }

    /// Build-side distance (v1): same clipped-residual path as queries — consistency by construction.
    pub fn dist_v(view: &mut I8Ctx, coarse: &[f32], node: &I8Node) -> i32 {
        view.d(coarse, node)
    }

    pub fn encode_query(&self, q: &[f32]) -> I8Ctx {
        let mut best = 0usize; let mut bd = f32::MAX;
        for c in 0..self.kc {
            let base = c * D;
            let mut s = 0f32;
            for i in 0..D { let d = q[i] - self.coarse[base + i]; s += d * d; }
            if s < bd { bd = s; best = c; }
        }
        let mut v_sq = 0f32;
        for i in 0..D { v_sq += q[i] * q[i]; }
        I8Ctx { q: q.try_into().unwrap(), own_cell: best, v_sq, cache_cell: u32::MAX, cache_res: [0i8; D], cache_sq: 0 }
    }

    #[inline]
    pub fn dist_honest(&self, ctx: &mut I8Ctx, node: &I8Node) -> i32 {
        ctx.d(&self.coarse, node)
    }

    /// SDC over raw codes.
    pub fn sdc_i8(a: &[i8; D], b: &[i8; D]) -> i32 {
        let mut s = 0i32;
        for i in 0..D {
            let d = a[i] as i32 - b[i] as i32;
            s += d * d;
        }
        s
    }

    /// SDC (code-to-code, pre-centered) for pruning/build-side.
    pub fn sdc(a: &I8Node, b: &I8Node) -> i32 {
        Self::sdc_i8(&a.code, &b.code)
    }

    /// Insert with PROVEN PqHnsw mechanics: per-layer efC beam from descended position,
    /// forward = closest m, backward push + local rescore-truncate to m (no global eviction).
    pub fn insert(&mut self, id: u64, cell: u16, code: [i8; D]) -> u32 {
        let mut rng = 0x9E3779B97F4A7C15u64 ^ id.wrapping_mul(0x2545F4914F6CDD1D);
        let level = self.random_level(&mut rng);
        let idx = self.nodes.len() as u32;
        let sq = crate::simd_kernels::i8_sq_128(&code);
        let cb = cell as usize * D;
        let mut xsq = 0f32;
        for i in 0..D { let x = self.coarse[cb + i] + code[i] as f32; xsq += x * x; }
        self.nodes.push(I8Node { id, cell, code, sq, xsq, edges: vec![Vec::new(); level + 1], alt_edges: vec![Vec::new(); level + 1], level });
        self.stamp.push(0);
        let rv = self.recon(cell, &code);
        let mut v_sq = 0f32;
        for i in 0..D { v_sq += rv[i] * rv[i]; }
        let mut view = I8Ctx { q: rv, own_cell: cell as usize, v_sq, cache_cell: u32::MAX, cache_res: [0i8; D], cache_sq: 0 };
        let mut ep = self.enter_point;
        for lvl in (level + 1..=self.max_level).rev() {
            if let Some(e) = ep { ep = Some(self.greedy(lvl, e, &mut view)); }
        }
        let mut cur_ep = ep;
        for lvl in (0..=level.min(self.max_level)).rev() {
            let (cands, best_ep) = if let Some(e) = cur_ep {
                let c = self.beam(lvl, e, &mut view, idx, self.ef_construction);
                let be = c.0.iter().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).map(|&(_, i)| i).unwrap_or(e);
                (c.0, be)
            } else { (Vec::new(), 0) };
            cur_ep = Some(best_ep);
            let mut sel = cands;
            sel.sort_unstable();
            let primary: Vec<u32> = sel.iter().take(self.m).map(|&(_, i)| i).collect();
            let alt: Vec<u32> = sel.iter().skip(self.m).take(self.m).map(|&(_, i)| i).collect();
            for &nb in &primary {
                if (nb as usize) < self.nodes.len() && nb != idx {
                    self.nodes[idx as usize].edges[lvl].push(nb);
                }
            }
            for &nb in &alt {
                if (nb as usize) < self.nodes.len() && nb != idx {
                    self.nodes[idx as usize].alt_edges[lvl].push(nb);
                }
            }
            for (is_alt, nbrs) in [(false, &primary), (true, &alt)] {
                for &nb in nbrs {
                    if (nb as usize) >= self.nodes.len() || nb == idx { continue; }
                    {
                        let edges = if is_alt { &mut self.nodes[nb as usize].alt_edges } else { &mut self.nodes[nb as usize].edges };
                        while edges.len() <= lvl { edges.push(Vec::new()); }
                        edges[lvl].push(idx);
                    }
                    let over = if is_alt { self.nodes[nb as usize].alt_edges.get(lvl).map(|e| e.len()).unwrap_or(0) } else { self.nodes[nb as usize].edges.get(lvl).map(|e| e.len()).unwrap_or(0) } > self.m;
                    if over {
                        let snap: Vec<u32> = if is_alt { self.nodes[nb as usize].alt_edges[lvl].clone() } else { self.nodes[nb as usize].edges[lvl].clone() };
                        let ncode = self.nodes[nb as usize].code;
                        let mut scored: Vec<(i32, u32)> = snap.iter()
                            .map(|&e| (Self::sdc_i8(&ncode, &self.nodes[e as usize].code), e))
                            .collect();
                        scored.sort_unstable();
                        scored.truncate(self.m);
                        let tgt = if is_alt { &mut self.nodes[nb as usize].alt_edges } else { &mut self.nodes[nb as usize].edges };
                        tgt[lvl] = scored.into_iter().map(|(_, e)| e).collect();
                    }
                }
            }
        }
        if level > self.max_level {
            self.max_level = level;
            self.enter_point = Some(idx);
        }
        idx
    }

    fn random_level(&self, rng: &mut u64) -> usize {
        let m_l = 1.0 / (self.m as f64).ln();
        let mut l = 0;
        let mut x: u64 = *rng;
        loop {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17; *rng = x;
            if (x as f64 / u64::MAX as f64) < m_l { l += 1; } else { break; }
        }
        l.min(15)
    }

    /// Greedy descent with reconstruction-L2 (build: view.v = recon of target).
    fn greedy(&mut self, l: usize, mut ep: u32, view: &mut I8Ctx) -> u32 {
        loop {
            let mut best = ep;
            let mut bd = Self::dist_v(view, &self.coarse, &self.nodes[ep as usize]);
            if let Some(layer) = self.nodes[ep as usize].edges.get(l) { for &nb in layer {
                let d = Self::dist_v(view, &self.coarse, &self.nodes[nb as usize]);
                if d < bd { bd = d; best = nb; } }
            }
            if let Some(layer) = self.nodes[ep as usize].alt_edges.get(l) { for &nb in layer {
                let d = Self::dist_v(view, &self.coarse, &self.nodes[nb as usize]);
                if d < bd { bd = d; best = nb; } }
            }
            if best == ep { return ep; }
            ep = best;
        }
    }

    /// ef-beam using SDC against the target (build-time).
    fn beam(&mut self, l: usize, ep: u32, view: &mut I8Ctx, _target: u32, ef: usize) -> (Vec<(i32, u32)>, u32) {
        use std::collections::BinaryHeap;
        let vis_epoch = { self.epoch += 1; self.epoch };
        let mut results: BinaryHeap<(i32, u32)> = BinaryHeap::new();
        let mut queue: BinaryHeap<(std::cmp::Reverse<(i32, u32)>, u32)> = BinaryHeap::new();
        let d0 = Self::dist_v(view, &self.coarse, &self.nodes[ep as usize]);
        self.stamp[ep as usize] = vis_epoch;
        queue.push((std::cmp::Reverse((d0, ep)), ep));
        results.push((d0, ep));
        while let Some((std::cmp::Reverse((d, u)), _)) = queue.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(i32::MAX);
            if d > worst && results.len() >= ef { break; }
            if let Some(layer) = self.nodes[u as usize].edges.get(l) { for &nb in layer {
                if (nb as usize) >= self.nodes.len() || self.stamp[nb as usize] == vis_epoch { continue; }
                self.stamp[nb as usize] = vis_epoch;
                let nd = Self::dist_v(view, &self.coarse, &self.nodes[nb as usize]);
                let worst2 = results.peek().map(|r| r.0).unwrap_or(i32::MAX);
                if nd < worst2 || results.len() < ef {
                    queue.push((std::cmp::Reverse((nd, nb)), nb));
                    results.push((nd, nb));
                    if results.len() > ef { results.pop(); }
                } }
            }
            if let Some(layer) = self.nodes[u as usize].alt_edges.get(l) { for &nb in layer {
                if (nb as usize) >= self.nodes.len() || self.stamp[nb as usize] == vis_epoch { continue; }
                self.stamp[nb as usize] = vis_epoch;
                let nd = Self::dist_v(view, &self.coarse, &self.nodes[nb as usize]);
                let worst2 = results.peek().map(|r| r.0).unwrap_or(i32::MAX);
                if nd < worst2 || results.len() < ef {
                    queue.push((std::cmp::Reverse((nd, nb)), nb));
                    results.push((nd, nb));
                    if results.len() > ef { results.pop(); }
                } }
            }
        }
        let nearest = results.iter().min_by_key(|(d, _)| *d).map(|(_, i)| *i).unwrap_or(ep);
        (results.into_iter().collect(), nearest)
    }

    /// Query-time ef-beam with honest distance.
    pub fn search_with_ef(&mut self, ctx: &mut I8Ctx, k: usize, ef: usize) -> Vec<(i32, u32)> {
        if self.nodes.is_empty() { return Vec::new(); }
        let ef = ef.max(k);
        let mut out: Vec<(i32, u32)> = Vec::new();
        let ep_opt = if self.use_gateway && ctx.own_cell < self.gateways.len()
            && self.gateways[ctx.own_cell] != u32::MAX {
            Some(self.gateways[ctx.own_cell])
        } else { self.enter_point };
        if let Some(ep) = ep_opt {
            let mut cur = ep;
            for l in (1..=self.max_level).rev() {
                cur = self.greedy_query(l, cur, ctx);
            }
            out = self.beam_query(0, cur, ctx, ef);
        }
        out.sort_by_key(|x| x.0);
        out.truncate(k);
        out
    }

    fn greedy_query(&mut self, l: usize, mut ep: u32, ctx: &mut I8Ctx) -> u32 {
        loop {
            let mut best = ep;
            let mut bd = self.dist_honest(ctx, &self.nodes[ep as usize]);
            if let Some(layer) = self.nodes[ep as usize].edges.get(l) { for &nb in layer {
                let d = self.dist_honest(ctx, &self.nodes[nb as usize]);
                if d < bd { bd = d; best = nb; } }
            }
            if let Some(layer) = self.nodes[ep as usize].alt_edges.get(l) { for &nb in layer {
                let d = self.dist_honest(ctx, &self.nodes[nb as usize]);
                if d < bd { bd = d; best = nb; } }
            }
            if best == ep { return ep; }
            ep = best;
        }
    }

    fn beam_query(&mut self, l: usize, ep: u32, ctx: &mut I8Ctx, ef: usize) -> Vec<(i32, u32)> {
        use std::collections::BinaryHeap;
        let vis_epoch = { self.epoch += 1; self.epoch };
        let mut results: BinaryHeap<(i32, u32)> = BinaryHeap::new();
        let mut queue: BinaryHeap<(std::cmp::Reverse<(i32, u32)>, u32)> = BinaryHeap::new();
        let d0 = self.dist_honest(ctx, &self.nodes[ep as usize]);
        self.stamp[ep as usize] = vis_epoch;
        queue.push((std::cmp::Reverse((d0, ep)), ep));
        results.push((d0, ep));
        while let Some((std::cmp::Reverse((d, u)), _)) = queue.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(i32::MAX);
            if d > worst && results.len() >= ef { break; }
            if let Some(layer) = self.nodes[u as usize].edges.get(l) { for &nb in layer {
                if (nb as usize) >= self.nodes.len() || self.stamp[nb as usize] == vis_epoch { continue; }
                self.stamp[nb as usize] = vis_epoch;
                let nd = self.dist_honest(ctx, &self.nodes[nb as usize]);
                let worst2 = results.peek().map(|r| r.0).unwrap_or(i32::MAX);
                if nd < worst2 || results.len() < ef {
                    queue.push((std::cmp::Reverse((nd, nb)), nb));
                    results.push((nd, nb));
                    if results.len() > ef { results.pop(); }
                } }
            }
            if let Some(layer) = self.nodes[u as usize].alt_edges.get(l) { for &nb in layer {
                if (nb as usize) >= self.nodes.len() || self.stamp[nb as usize] == vis_epoch { continue; }
                self.stamp[nb as usize] = vis_epoch;
                let nd = self.dist_honest(ctx, &self.nodes[nb as usize]);
                let worst2 = results.peek().map(|r| r.0).unwrap_or(i32::MAX);
                if nd < worst2 || results.len() < ef {
                    queue.push((std::cmp::Reverse((nd, nb)), nb));
                    results.push((nd, nb));
                    if results.len() > ef { results.pop(); }
                } }
            }
        }
        results.into_iter().collect()
    }

    /// Vamana-style prune on layer-0 edges using SDC.
    pub fn prune_diverse(&mut self, alpha: f32) -> usize {
        let mut removed = 0usize;
        for ni in 0..self.nodes.len() {
            let nbs = self.nodes[ni].edges.get(0).cloned().unwrap_or_default(); // primary only
            if nbs.len() <= 2 { continue; }
            let mut by: Vec<(i32, u32)> = nbs.iter()
                .map(|&nb| (Self::sdc(&self.nodes[ni], &self.nodes[nb as usize]), nb))
                .collect();
            by.sort_unstable();
            let mut kept: Vec<u32> = Vec::with_capacity(by.len());
            for (d, nb) in by {
                let thresh = if d == 0 { 1 } else { (alpha * (d as f32)) as i32 };
                let redundant = kept.iter().any(|&p| Self::sdc(&self.nodes[nb as usize], &self.nodes[p as usize]) < thresh);
                if !redundant { kept.push(nb); }
            }
            if kept.len() < nbs.len() {
                removed += nbs.len() - kept.len();
                self.nodes[ni].edges[0] = kept;
            }
        }
        removed
    }

    pub fn node_label(&self, idx: u32) -> u64 { self.nodes[idx as usize].id }

    /// T7: per-cell gateway = node whose reconstruction is nearest its cell centroid
    /// (||x̂ - C|| = ||code||, so min code-norm). One O(N) pass.
    pub fn rebuild_gateways(&mut self) {
        self.gateways = vec![u32::MAX; self.kc];
        let mut best = vec![i32::MAX; self.kc];
        for (i, nd) in self.nodes.iter().enumerate() {
            let c = nd.cell as usize;
            if c < self.kc && nd.sq < best[c] {
                best[c] = nd.sq;
                self.gateways[c] = i as u32;
            }
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"YPI8")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
        w.write_all(&(self.m as u32).to_le_bytes())?;
        w.write_all(&(self.ef_construction as u32).to_le_bytes())?;
        w.write_all(&self.enter_point.unwrap_or(u32::MAX).to_le_bytes())?;
        w.write_all(&(self.max_level as u32).to_le_bytes())?;
        w.write_all(&(self.kc as u32).to_le_bytes())?;
        for &c in &self.coarse { w.write_all(&c.to_le_bytes())?; }
        for nd in &self.nodes {
            w.write_all(&nd.id.to_le_bytes())?;
            w.write_all(&nd.cell.to_le_bytes())?;
            w.write_all(&(nd.level as u32).to_le_bytes())?;
            for &b in &nd.code { w.write_all(&(b as i8).to_le_bytes())?; }
            w.write_all(&(nd.edges.len() as u32).to_le_bytes())?;
            for layer in &nd.edges {
                w.write_all(&(layer.len() as u32).to_le_bytes())?;
                for &e in layer { w.write_all(&e.to_le_bytes())?; }
            }
            w.write_all(&(nd.alt_edges.len() as u32).to_le_bytes())?;
            for layer in &nd.alt_edges {
                w.write_all(&(layer.len() as u32).to_le_bytes())?;
                for &e in layer { w.write_all(&e.to_le_bytes())?; }
            }
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 4]; r.read_exact(&mut magic)?;
        assert_eq!(&magic, b"YPI8", "bad magic");
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?; let _v = u32::from_le_bytes(b4);
        let mut b8 = [0u8; 8]; r.read_exact(&mut b8)?; let n = u64::from_le_bytes(b8) as usize;
        r.read_exact(&mut b4)?; let m = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4)?; let efc = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4)?; let ep = u32::from_le_bytes(b4);
        r.read_exact(&mut b4)?; let max_level = u32::from_le_bytes(b4) as usize;
        r.read_exact(&mut b4)?; let kc = u32::from_le_bytes(b4) as usize;
        let mut coarse = vec![0f32; kc * D];
        for c in coarse.iter_mut() { r.read_exact(&mut b4)?; *c = f32::from_le_bytes(b4); }
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            r.read_exact(&mut b8)?; let id = u64::from_le_bytes(b8);
            let mut b2 = [0u8; 2]; r.read_exact(&mut b2)?; let cell = u16::from_le_bytes(b2);
            r.read_exact(&mut b4)?; let level = u32::from_le_bytes(b4) as usize;
            let mut code = [0i8; D];
            let mut b1 = [0u8; 1];
            for cb in code.iter_mut() { r.read_exact(&mut b1)?; *cb = b1[0] as i8; }
            r.read_exact(&mut b4)?; let nl = u32::from_le_bytes(b4) as usize;
            let mut edges = Vec::with_capacity(nl);
            for _ in 0..nl {
                r.read_exact(&mut b4)?; let el = u32::from_le_bytes(b4) as usize;
                let mut layer = Vec::with_capacity(el);
                for _ in 0..el { r.read_exact(&mut b4)?; layer.push(u32::from_le_bytes(b4)); }
                edges.push(layer);
            }
            let mut alt_edges = Vec::new();
            r.read_exact(&mut b4)?; let nal = u32::from_le_bytes(b4) as usize;
            for _ in 0..nal {
                r.read_exact(&mut b4)?; let el = u32::from_le_bytes(b4) as usize;
                let mut layer = Vec::with_capacity(el);
                for _ in 0..el { r.read_exact(&mut b4)?; layer.push(u32::from_le_bytes(b4)); }
                alt_edges.push(layer);
            }
            let sq = crate::simd_kernels::i8_sq_128(&code);
            let cb = cell as usize * D;
            let mut xsq = 0f32;
            for i in 0..D { let x = coarse[cb + i] + code[i] as f32; xsq += x * x; }
            nodes.push(I8Node { id, cell, code, sq, xsq, edges, alt_edges, level });
        }
        let stamp = vec![0u32; nodes.len()];
        let mut g = Self { gateways: Vec::new(), use_gateway: false, pre_centered: false, nodes, stamp, epoch: 0, coarse, kc, m, ef_construction: efc,
                  enter_point: if ep == u32::MAX { None } else { Some(ep) }, max_level };
        g.rebuild_gateways();
        Ok(g)
    }
}
