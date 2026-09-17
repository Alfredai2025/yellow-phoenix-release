//! Criterion.rs benchmark for HybridMesh insert + query (the "insert
//! benchmark" behind the Aug-16 layered-search note).
//!
//! History: this bench used `yp_insert_to_mesh_bytes` / `yp_query_mesh_bytes`
//! (ffi_unified); those symbols were removed 2026-08-22 (b1f856df) which left
//! `cargo bench --bench yp_query` failing to compile. Rewritten 2026-09-12
//! against the native HybridMesh API that every mesh FFI path wraps.
//!
//! Measures, per scale N in {10k, 100k, 1M} dual (128+512-bit) records:
//!   - insert throughput (records/s, no edges)
//!   - build_edges wall time
//!   - query_auto top-10 latency (graph path, after build_edges)
//!   - query_hybrid top-10 latency (linear coarse + fine re-rank)
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use pams::hybrid_mesh::{HybridMesh, PAP_128_BYTES, PAP_512_BYTES};
use rand::{RngCore, SeedableRng};
use std::hint::black_box as std_black_box;
use std::time::Duration;

const SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];
const N_QUERIES: usize = 256;
const SEED: u64 = 0x5950_2026_0912; // fixed — reproducible runs

struct Rng(rand::rngs::StdRng);
impl Rng {
    fn new() -> Self { Self(rand::rngs::StdRng::seed_from_u64(SEED)) }
    fn pap(&mut self) -> ([u8; PAP_128_BYTES], [u8; PAP_512_BYTES]) {
        let mut a = [0u8; PAP_128_BYTES];
        let mut b = [0u8; PAP_512_BYTES];
        self.0.fill_bytes(&mut a);
        self.0.fill_bytes(&mut b);
        (a, b)
    }
}

fn dataset(n: usize) -> (Vec<([u8; PAP_128_BYTES], [u8; PAP_512_BYTES])>, Vec<([u8; PAP_128_BYTES], [u8; PAP_512_BYTES])>) {
    let mut rng = Rng::new();
    let records = (0..n).map(|_| rng.pap()).collect();
    let queries = (0..N_QUERIES).map(|_| rng.pap()).collect();
    (records, queries)
}

fn bench_insert(c: &mut Criterion) {
    for &n in &SIZES {
        let (records, _) = dataset(n);
        let mut group = c.benchmark_group(format!("mesh_insert_{n}"));
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(6));
        group.bench_function("insert_dual", |b| {
            b.iter(|| {
                let mut mesh = HybridMesh::new(n, n);
                for (i, (p128, p512)) in records.iter().enumerate() {
                    mesh.insert_dual(i as u64, p128, p512);
                }
                std_black_box(&mesh);
            })
        });
        group.finish();
    }
}

fn bench_build_and_query(c: &mut Criterion) {
    for &n in &SIZES {
        let (records, queries) = dataset(n);
        // Setup once per size: insert + build_edges (not part of measurement).
        let mut mesh = HybridMesh::new(n, n);
        for (i, (p128, p512)) in records.iter().enumerate() {
            mesh.insert_dual(i as u64, p128, p512);
        }
        let t0 = std::time::Instant::now();
        mesh.build_edges(8);
        println!("build_edges({n}) = {:?} ({:.0} rec/s)", t0.elapsed(),
                 n as f64 / t0.elapsed().as_secs_f64());

        let mut group = c.benchmark_group(format!("mesh_query_{n}"));
        group.sample_size(100);
        group.measurement_time(Duration::from_secs(8));

        group.bench_function("query_auto_top10", |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let (p128, p512) = &queries[idx % queries.len()];
                let r = mesh.query_auto(p128, p512, 10);
                idx += 1;
                black_box(r.len())
            })
        });
        group.bench_function("query_hybrid_top10", |b| {
            let mut idx = 0usize;
            b.iter(|| {
                let (p128, p512) = &queries[idx % queries.len()];
                let r = mesh.query(p128, p512, 10);
                idx += 1;
                black_box(r.len())
            })
        });
        group.finish();
    }
}

criterion_group!(benches, bench_insert, bench_build_and_query);
criterion_main!(benches);
