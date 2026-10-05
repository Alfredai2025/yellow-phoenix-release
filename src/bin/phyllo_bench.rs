// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer
//! Phyllotactic entry-point A/B bench: baseline (global EP) vs golden-angle entry.
//! Panel-improved protocol: per-query baseline hops -> difficulty terciles ->
//! phyllo must not regress in ANY bin. Metrics: recall@10 (after float verify),
//! p50 latency, hops (search_layer calls).
//! Usage: phyllo_bench <graph.bin> <payload.f32> <qcodes.ism> <qcoords.f32> <nodecoords.f32> <gt.bin> <queries.f32> <out.json>
use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use pams::binary_hnsw::{BinaryHNSW, HASH512_BYTES, HOPS, set_fusion_int, set_fusion_q, clear_fusion};
use pams::phyllotactic::PhyllotacticNavigator;

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

/// Vogel-spiral compass: anchors at plane positions (c*sqrt(k), k*137.5deg);
/// query looks up nearest anchor by estimating k0=(r/c)^2 and scanning a window.
struct SpiralNav {
    ax: Vec<f32>, ay: Vec<f32>, node: Vec<u32>, c: f32, opp: Vec<u32>, dens: Vec<f32>, win: usize,
}
impl SpiralNav {
    fn build(coords: &[f32], n: usize, n_anchors: usize) -> Self {
        Self::build_win(coords, n, n_anchors, 24)
    }
    fn build_win(coords: &[f32], n: usize, n_anchors: usize, win: usize) -> Self {
        let theta_g = 2.3999632f32; // golden angle
        // data extent
        let (mut xmin, mut xmax, mut ymin, mut ymax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for i in 0..n {
            let (x, y) = (coords[i * 2], coords[i * 2 + 1]);
            xmin = xmin.min(x); xmax = xmax.max(x);
            ymin = ymin.min(y); ymax = ymax.max(y);
        }
        let r_max = ((xmax - xmin).hypot(ymax - ymin)) / 2.0 + 1e-6;
        let c = r_max / (n_anchors as f32).sqrt();
        // anchors on the Vogel spiral
        let mut ax = Vec::with_capacity(n_anchors);
        let mut ay = Vec::with_capacity(n_anchors);
        let mut node = vec![0u32; n_anchors];
        let mut dens = vec![0f32; n_anchors];
        // coarse grid for nearest-node lookup
        const G: usize = 128;
        let mut grid: Vec<Vec<u32>> = vec![Vec::new(); G * G];
        let gx = |x: f32| (((x - xmin) / (xmax - xmin + 1e-9)) * (G as f32 - 1.0)) as usize;
        for i in 0..n {
            let cx = (((coords[i * 2] - xmin) / (xmax - xmin + 1e-9)) * (G as f32 - 1.0)) as usize;
            let cy = (((coords[i * 2 + 1] - ymin) / (ymax - ymin + 1e-9)) * (G as f32 - 1.0)) as usize;
            grid[cy * G + cx].push(i as u32);
        }
        for k in 0..n_anchors {
            let r = c * ((k + 1) as f32).sqrt();
            let th = (k as f32) * theta_g;
            let (px, py) = (r * th.cos(), r * th.sin());
            ax.push(px); ay.push(py);
            // nearest node: probe 3x3 cells around the anchor
            let cx = (((px - xmin) / (xmax - xmin + 1e-9)) * (G as f32 - 1.0)) as usize;
            let cy = (((py - ymin) / (ymax - ymin + 1e-9)) * (G as f32 - 1.0)) as usize;
            let mut best = u32::MAX; let mut bd = f32::MAX;
            for dy in 0..3 { for dx in 0..3 {
                let xx = cx + dx; let yy = cy + dy;
                if xx >= G || yy >= G { continue; }
                for &ni in &grid[yy * G + xx] {
                    let d = (coords[ni as usize * 2] - px).hypot(coords[ni as usize * 2 + 1] - py);
                    if d < bd { bd = d; best = ni; }
                }
            }}
            node[k] = if best == u32::MAX { 0 } else { best };
            // density = nodes in the anchor's own grid cell (+ 4-neighbors)
            let mut dsum = 0f32;
            for dy in 0..2 { for dx in 0..2 {
                let xx = cx + dx; let yy = cy + dy;
                if xx < G && yy < G { dsum += grid[yy * G + xx].len() as f32; }
            }}
            dens[k] = dsum;
        }
        // antipodal partner per anchor (same radius, angle + pi)
        let mut opp = vec![0u32; n_anchors];
        for k in 0..n_anchors {
            let tx = -ax[k]; let ty = -ay[k];
            let mut best = 0u32; let mut bd = f32::MAX;
            for j in 0..n_anchors {
                let d = (ax[j] - tx).hypot(ay[j] - ty);
                if d < bd { bd = d; best = j as u32; }
            }
            opp[k] = best;
        }
        SpiralNav { ax, ay, node, c, opp, dens, win }
    }
    /// density-weighted nearest: score = dens / (1 + d^2)
    fn nearest_dens(&self, qx: f32, qy: f32) -> u32 {
        let r = qx.hypot(qy);
        let k0 = ((r / self.c).powi(2)) as usize;
        let lo = k0.saturating_sub(24);
        let hi = (k0 + 24).min(self.ax.len());
        let mut best = self.node[lo]; let mut bs = f32::MIN;
        for k in lo..hi {
            let d = (self.ax[k] - qx).hypot(self.ay[k] - qy);
            let s = self.dens[k] / (1.0 + d * d);
            if s > bs { bs = s; best = self.node[k]; }
        }
        best
    }
    #[inline]
    fn nearest_opp(&self, qx: f32, qy: f32) -> (u32, u32) {
        let a = self.nearest_anchor(qx, qy);
        (self.node[a], self.node[self.opp[a] as usize])
    }
    #[inline]
    /// returns ANCHOR index of the nearest spiral anchor
    fn nearest_anchor(&self, qx: f32, qy: f32) -> usize {
        let r = qx.hypot(qy);
        let k0 = ((r / self.c).powi(2)) as usize;
        let lo = k0.saturating_sub(self.win);
        let hi = (k0 + self.win).min(self.ax.len());
        let mut best = lo; let mut bd = f32::MAX;
        for k in lo..hi {
            let d = (self.ax[k] - qx).hypot(self.ay[k] - qy);
            if d < bd { bd = d; best = k; }
        }
        best
    }
    #[inline]
    fn nearest(&self, qx: f32, qy: f32) -> u32 {
        self.node[self.nearest_anchor(qx, qy)]
    }
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let g = BinaryHNSW::load(&a[1]).expect("load graph");
    let payload = load_f32(&a[2]);
    let n_vecs = payload.len() / 128;
    let mut r = BufReader::with_capacity(8 << 20, File::open(&a[3]).expect("open qcodes"));
    let nq = read_u64(&mut r).expect("n") as usize;
    let mut qcodes = vec![0u8; nq * HASH512_BYTES];
    r.read_exact(&mut qcodes).expect("qcodes");
    let qcoords = load_f32(&a[4]);
    assert!(qcoords.len() == nq * 2);
    let node_coords = load_f32(&a[5]);
    assert!(node_coords.len() == n_vecs * 2);
    let mut gtf = File::open(&a[6]).expect("gt");
    let mut gt = vec![[0u64; 10]; nq];
    for row in gt.iter_mut() {
        for id in row.iter_mut() { *id = read_u64(&mut gtf).expect("gt"); }
    }
    // raw queries for verify (from the hdf5 test set, same order as qcodes)
    let qraw = load_f32(&a[7]);
    assert!(qraw.len() == nq * 128, "queries.f32 size");

    for i in 0..100u32.min(n_vecs as u32) {
        assert_eq!(g.node_label(i), i as u64, "graph not in insertion order");
    }
    let angles: Vec<(usize, f64)> = (0..n_vecs)
        .map(|i| (i, (node_coords[i * 2 + 1] as f64).atan2(node_coords[i * 2] as f64)))
        .collect();
    let nav = PhyllotacticNavigator::build(&angles);
    let win: usize = std::env::var("SP_WIN").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let spiral = SpiralNav::build_win(&node_coords, n_vecs, 1024, win);
    eprintln!("navigators: phyllo {} anchors, spiral {} anchors / {} nodes", nav.len(), spiral.ax.len(), n_vecs);

    let cells: Vec<(usize, usize)> = if a.len() > 10 {
        a[10].split(',').map(|s| { let mut it = s.split(':'); (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap()) }).collect()
    } else {
        vec![(64usize, 128usize), (128, 200), (256, 500), (1000, 1000)]
    };
    let n_arms: u32 = if a.len() > 11 && a[11] == "flat" { 7 } else if a.len() > 9 { 6 } else { 5 };
    let fusion_w: i64 = if a.len() > 9 { ((a[9].parse::<f32>().expect("fusion_w")) * 64.0) as i64 } else { 0 };
    // integer sketch: coords * 1024; den = (rms_radius*1024)^2
    let sketch_i32: Vec<i32> = node_coords.iter().map(|&c| (c * 1024.0) as i32).collect();
    let mut sketch_i32 = sketch_i32;
    if a.len() > 12 && a[12] == "shuffle" {
        // NEGATIVE CONTROL: randomize sketch geometry; fusion delta should vanish if signal is real
        let mut seed = 0xDEADBEEFu64;
        for i in (1..sketch_i32.len()).rev() {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let j = (seed as usize) % (i + 1);
            sketch_i32.swap(i, j);
        }
        eprintln!("NEGATIVE CONTROL: sketch coords shuffled");
    }
    let rms1024 = ((0..n_vecs).map(|i| (node_coords[i*2]*node_coords[i*2] + node_coords[i*2+1]*node_coords[i*2+1]).sqrt()).sum::<f32>() / n_vecs as f32 * 1024.0) as i64;
    let fusion_den: i64 = rms1024 * rms1024;
    if fusion_w > 0 {
        set_fusion_int(sketch_i32.clone(), fusion_den.max(1), fusion_w);
    }
    let mut summary = Vec::new();
    for &(ef, k) in &cells {
        for i in 0..30.min(nq) {
            let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
            let _ = g.search_with_ef(q, k, ef);
        }
        let mut arms: Vec<(Vec<f64>, Vec<u64>, Vec<u128>)> = Vec::new();
        for arm_id in 0..n_arms {
            let mut recs = vec![0f64; nq];
            let mut hops_v = vec![0u64; nq];
            let mut lats = vec![0u128; nq];
            for i in 0..nq {
                let q: &[u8; HASH512_BYTES] = qcodes[i * HASH512_BYTES..(i + 1) * HASH512_BYTES].try_into().unwrap();
                let t0 = std::time::Instant::now();
                HOPS.with(|c| c.set(0));
                let top = match arm_id {
                    0 => g.search_with_ef(q, k, ef),
                    1 => {
                        let qc = [qcoords[i * 2], qcoords[i * 2 + 1]];
                        let entry = nav.nearest_entry(qc).unwrap_or(0) as u32;
                        g.search_with_ef_from(q, k, ef, entry)
                    }
                    2 => {
                        let entry = spiral.nearest(qcoords[i * 2], qcoords[i * 2 + 1]);
                        g.search_with_ef_from(q, k, ef, entry)
                    }
                    4 => {
                        let entry = spiral.nearest_dens(qcoords[i * 2], qcoords[i * 2 + 1]);
                        g.search_with_ef_from(q, k, ef, entry)
                    }
                    5 => {
                        // in-graph fusion: hamming + w * sketch_d2 (integer fixed point)
                        set_fusion_q([(qcoords[i * 2] * 1024.0) as i32, (qcoords[i * 2 + 1] * 1024.0) as i32]);
                        let entry = spiral.nearest(qcoords[i * 2], qcoords[i * 2 + 1]);
                        g.search_with_ef_from(q, k, ef, entry)
                    }
                    6 => {
                        // flat search: beam starts at layer 0 from compass entry, no upper descent
                        let entry = spiral.nearest(qcoords[i * 2], qcoords[i * 2 + 1]);
                        g.search_flat_from(q, k, ef, entry)
                    }
                    _ => {
                        let (e1, e2) = spiral.nearest_opp(qcoords[i * 2], qcoords[i * 2 + 1]);
                        let mut t1 = g.search_with_ef_from(q, k, ef, e1);
                        let t2 = g.search_with_ef_from(q, k, ef, e2);
                        t1.extend(t2);
                        t1.sort_by_key(|&(_, idx, _)| idx);
                        t1.dedup_by_key(|&mut (_, idx, _)| idx);
                        t1.sort_by_key(|&(d, _, _)| d);
                        t1.truncate(k);
                        t1
                    }
                };
                hops_v[i] = HOPS.with(|c| c.get());
                lats[i] = t0.elapsed().as_micros();
                let cand: Vec<u64> = top.iter().map(|&(_, idx, _)| g.node_label(idx)).collect();
                let qq = &qraw[i * 128..(i + 1) * 128];
                let mut scored: Vec<(f32, u64)> = cand.iter()
                    .map(|&id| (l2(&payload[id as usize * 128..id as usize * 128 + 128], qq), id))
                    .collect();
                scored.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
                scored.truncate(10);
                recs[i] = scored.iter().filter(|(_, id)| gt[i].contains(id)).count() as f64 / 10.0;
            }
            arms.push((recs, hops_v, lats));
        }
        let (b, p, sp, s2, sd) = (&arms[0], &arms[1], &arms[2], &arms[3], &arms[4]);
        let fus = if n_arms > 5 { &arms[5] } else { &arms[4] };
        let flt = if n_arms > 6 { &arms[6] } else { &arms[4] };
        let med = |v: &[u128]| { let mut c = v.to_vec(); c.sort_unstable(); c[c.len() / 2] as f64 };
        let bin_report = |name: &str, ord: &[usize]| {
            let bh: f64 = ord.iter().map(|&i| b.1[i] as f64).sum::<f64>() / ord.len() as f64;
            let ph: f64 = ord.iter().map(|&i| p.1[i] as f64).sum::<f64>() / ord.len() as f64;
            let br: f64 = ord.iter().map(|&i| b.0[i]).sum::<f64>() / ord.len() as f64;
            let pr: f64 = ord.iter().map(|&i| p.0[i]).sum::<f64>() / ord.len() as f64;
            format!("\"{name}_hops\":{bh:.0},\"{name}_hops_ph\":{ph:.0},\"{name}_rec\":{br:.4},\"{name}_rec_ph\":{pr:.4}")
        };
        let mut order: Vec<usize> = (0..nq).collect();
        order.sort_by_key(|&i| b.1[i]);
        let t = nq / 3;
        let line = format!(
            "\"ef{ef}_K{k}\":{{\"base_rec\":{:.4},\"ph_rec\":{:.4},\"sp_rec\":{:.4},\"sp2_rec\":{:.4},\"spd_rec\":{:.4},\"fus_rec\":{:.4},\"flt_rec\":{:.4},\"base_p50\":{:.0},\"ph_p50\":{:.0},\"sp_p50\":{:.0},\"sp2_p50\":{:.0},\"spd_p50\":{:.0},\"fus_p50\":{:.0},\"flt_p50\":{:.0},\"base_hops\":{:.0},\"ph_hops\":{:.0},\"sp_hops\":{:.0},\"sp2_hops\":{:.0},\"spd_hops\":{:.0},\"fus_hops\":{:.0},\"flt_hops\":{:.0}}}",
            b.0.iter().sum::<f64>() / nq as f64, p.0.iter().sum::<f64>() / nq as f64,
            sp.0.iter().sum::<f64>() / nq as f64, s2.0.iter().sum::<f64>() / nq as f64,
            sd.0.iter().sum::<f64>() / nq as f64, fus.0.iter().sum::<f64>() / nq as f64,
            flt.0.iter().sum::<f64>() / nq as f64,
            med(&b.2), med(&p.2), med(&sp.2), med(&s2.2), med(&sd.2), med(&fus.2), med(&flt.2),
            b.1.iter().sum::<u64>() as f64 / nq as f64,
            p.1.iter().sum::<u64>() as f64 / nq as f64,
            sp.1.iter().sum::<u64>() as f64 / nq as f64, s2.1.iter().sum::<u64>() as f64 / nq as f64,
            sd.1.iter().sum::<u64>() as f64 / nq as f64, fus.1.iter().sum::<u64>() as f64 / nq as f64,
            flt.1.iter().sum::<u64>() as f64 / nq as f64);
        eprintln!("{line}");
        summary.push(line);
    }
    let out = format!("{{{}}}", summary.join(","));
    let mut fo = File::create(&a[8]).unwrap();
    fo.write_all(out.as_bytes()).unwrap();
    println!("{}", out);
}
