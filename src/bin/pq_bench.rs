// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bench PQHNSW: raw query -> ADC search -> top-K candidates -> exact float
//! verify -> recall@10 vs GT. Same metric as the two-stage bench.
//! Usage: pq_bench <graph.bin> <payload.f32> <queries.f32> <gt.bin> <out.json> [ef] [K]
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use pams::pq_hnsw::PqHnsw;

fn load_f32(path: &str) -> Vec<f32> {
    let mut f = File::open(path).expect("open f32");
    let mut v = Vec::new();
    f.read_to_end(&mut v).expect("read f32");
    v.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[inline]
fn l2(a: &[f32], b: &[f32]) -> f32 {
    pams::simd_kernels::l2_sq_f32(a, b)
}

/// Compact Vogel-spiral compass over the sketch plane (2D).
struct SpiralNav2 { ax: Vec<f32>, ay: Vec<f32>, node: Vec<u32>, c: f32, win: usize }
impl SpiralNav2 {
    fn build(coords: &[f32], n: usize, n_anchors: usize) -> Self {
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for i in 0..n {
            xmin = xmin.min(coords[i*2]); xmax = xmax.max(coords[i*2]);
            ymin = ymin.min(coords[i*2+1]); ymax = ymax.max(coords[i*2+1]);
        }
        let r_max = ((xmax-xmin).hypot(ymax-ymin)) / 2.0 + 1e-6;
        let c = r_max / (n_anchors as f32).sqrt();
        const G: usize = 128;
        let mut grid: Vec<Vec<u32>> = vec![Vec::new(); G*G];
        for i in 0..n {
            let cx = (((coords[i*2]-xmin)/(xmax-xmin+1e-9)) * (G as f32-1.0)) as usize;
            let cy = (((coords[i*2+1]-ymin)/(ymax-ymin+1e-9)) * (G as f32-1.0)) as usize;
            grid[cy*G+cx].push(i as u32);
        }
        let theta_g = 2.3999632f32;
        let (mut ax, mut ay, mut node) = (Vec::new(), Vec::new(), vec![0u32; n_anchors]);
        for k in 0..n_anchors {
            let r = c * ((k+1) as f32).sqrt();
            let th = (k as f32) * theta_g;
            let (px, py) = (r*th.cos(), r*th.sin());
            ax.push(px); ay.push(py);
            let cx0 = (((px-xmin)/(xmax-xmin+1e-9)) * (G as f32-1.0)) as usize;
            let cy0 = (((py-ymin)/(ymax-ymin+1e-9)) * (G as f32-1.0)) as usize;
            let mut best = u32::MAX; let mut bd = f32::MAX;
            for dy in 0..3 { for dx in 0..3 {
                let xx = cx0+dx; let yy = cy0+dy;
                if xx >= G || yy >= G { continue; }
                for &ni in &grid[yy*G+xx] {
                    let d = (coords[ni as usize*2]-px).hypot(coords[ni as usize*2+1]-py);
                    if d < bd { bd = d; best = ni; }
                }
            }}
            node[k] = if best == u32::MAX { 0 } else { best };
        }
        SpiralNav2 { ax, ay, node, c, win: 24 }
    }
    fn nearest(&self, qx: f32, qy: f32) -> u32 {
        let r = qx.hypot(qy);
        let k0 = ((r / self.c).powi(2)) as usize;
        let lo = k0.saturating_sub(self.win);
        let hi = (k0 + self.win).min(self.ax.len());
        let mut best = self.node[lo]; let mut bd = f32::MAX;
        for k in lo..hi {
            let d = (self.ax[k]-qx).hypot(self.ay[k]-qy);
            if d < bd { bd = d; best = self.node[k]; }
        }
        best
    }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let mut g = PqHnsw::load(&a[1]).expect("load graph");
    let payload = load_f32(&a[2]);
    let n_vecs = payload.len() / 128;
    let qf = load_f32(&a[3]);
    let nq = qf.len() / 128;
    let mut gtf = File::open(&a[4]).expect("open gt");
    let mut gt = vec![[0u64; 10]; nq];
    for row in gt.iter_mut() {
        for id in row.iter_mut() {
            *id = read_u64(&mut gtf).expect("gt");
        }
    }
    let ef: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(128);
    // optional prune + save: a[8]=alpha a[9]=savepath
    let prune_alpha: f32 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    if prune_alpha > 0.0 {
        let removed = g.prune_diverse(prune_alpha);
        eprintln!("PQ prune alpha={}: removed {} edges", prune_alpha, removed);
        if let Some(p) = a.get(9) { if !p.is_empty() { g.save(p).expect("save pruned"); eprintln!("saved {}", p); } }
    }
    // compass navigator over node sketch (dims 0..2)
    let compass = a.get(10).map(|s| s == "compass").unwrap_or(false);
    let nav: Option<SpiralNav2> = if compass {
        let coords: Vec<f32> = g.nodes.iter().flat_map(|n| [n.sketch[0], n.sketch[1]]).collect();
        Some(SpiralNav2::build(&coords, g.nodes.len(), 1024))
    } else { None };
    // graph health dump
    {
        let mut deg0 = 0usize; let mut nonempty0 = 0usize; let mut maxl = 0usize;
        for nd in &g.nodes {
            if !nd.edges.is_empty() && !nd.edges[0].is_empty() { nonempty0 += 1; deg0 += nd.edges[0].len(); }
            maxl = maxl.max(nd.level);
        }
        eprintln!("GRAPH HEALTH: nodes={} avg_l0_degree={:.2} nonempty_l0={} ep={:?} ep_level={} max_level={}",
            g.nodes.len(), deg0 as f64 / g.nodes.len().max(1) as f64, nonempty0,
            g.enter_point, g.enter_point.map(|e| g.nodes[e as usize].level).unwrap_or(0), g.max_level);
    }
    let k: usize = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(200);

    let mut lat: Vec<u128> = Vec::with_capacity(nq);
    let mut recall = 0f64;
    for i in 0..50.min(nq) {
        let q = &qf[i * 128..(i + 1) * 128];
        let ctx = g.codec.encode_query(q);
        let _ = g.search_with_ef(&ctx, k, ef);
    }
    for i in 0..nq {
        let q = &qf[i * 128..(i + 1) * 128];
        let t0 = std::time::Instant::now();
        let ctx = g.codec.encode_query(q);
        let top = if let Some(nv) = &nav {
            let entry = nv.nearest(ctx.qsk[0], ctx.qsk[1]);
            g.search_with_ef_from(&ctx, k, ef, entry)
        } else {
            g.search_with_ef(&ctx, k, ef)
        };
        let cand: Vec<u64> = top.iter().map(|&(_, idx)| g.node_label(idx)).collect();
        let mut best: Vec<(f32, u64)> = cand.iter()
            .map(|&id| (l2(&payload[id as usize * 128..id as usize * 128 + 128], q), id))
            .collect();
        best.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        best.truncate(10);
        lat.push(t0.elapsed().as_micros());
        recall += best.iter().filter(|(_, id)| gt[i].contains(id)).count() as f64 / 10.0;
    }
    lat.sort_unstable();
    let p50 = lat[nq / 2] as f64;
    let p95 = lat[nq * 95 / 100] as f64;
    let qps = 1e6 / (lat.iter().map(|&x| x as f64).sum::<f64>() / nq as f64);
    let exps = pams::pq_hnsw::BEAM_EXPANSIONS.with(|c| c.get());
    let arm = if compass { "pruned+compass" } else if prune_alpha > 0.0 { "pruned" } else { "base" };
    let out = format!(
        "{{\"kind\":\"pqhnsw\",\"arm\":\"{arm}\",\"alpha\":{prune_alpha},\"ef\":{ef},\"K\":{k},\"recall_at_10\":{:.6},\"p50_us\":{:.2},\"p95_us\":{:.2},\"qps\":{:.1},\"beam_expansions_per_q\":{:.1},\"n_queries\":{nq},\"n_vecs\":{n_vecs}}}",
        recall / nq as f64, p50, p95, qps, exps as f64 / nq as f64);
    let mut fo = File::create(&a[5]).expect("create out");
    fo.write_all(out.as_bytes()).unwrap();
    println!("{}", out);
}
