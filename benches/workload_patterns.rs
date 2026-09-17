use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn bench_workloads(c: &mut Criterion) {
    let n = 100_000;
    let mut pipeline = QueryPipeline::new_default();
    let mut papers = Vec::with_capacity(n);
    for i in 0..n {
        let b = [i as u8, (i>>8) as u8, (i>>16) as u8, (i>>24) as u8, 0,0,0,0,0,0,0,0,0,0,0,0];
        let p = BinaryMultivector::from_bytes(b);
        pipeline.add_paper(&p);
        papers.push(p);
    }
    let mut g = c.benchmark_group("workload_100k");
    g.bench_function("self", |b| b.iter(|| pipeline.query(black_box(&papers[50000]))));
    let neighbor = BinaryMultivector::from_bytes([0u8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]);
    g.bench_function("neighbor", |b| b.iter(|| pipeline.query(black_box(&neighbor))));
    let random = BinaryMultivector::from_bytes([0xABu8; 16]);
    g.bench_function("random", |b| b.iter(|| pipeline.query(black_box(&random))));
    let adv = BinaryMultivector::from_bytes([0xFFu8; 16]);
    g.bench_function("adversarial", |b| b.iter(|| pipeline.query(black_box(&adv))));
    g.finish();
}
criterion_group!(benches, bench_workloads);
criterion_main!(benches);
