use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn bench_cold_start(c: &mut Criterion) {
    let n = 100_000;
    let mut papers = Vec::with_capacity(n);
    for i in 0..n {
        let b = [i as u8, (i>>8) as u8, 0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        papers.push(BinaryMultivector::from_bytes(b));
    }
    c.bench_function("cold_start_100k", |b| b.iter(|| {
        let mut p = QueryPipeline::new_default();
        for paper in &papers { p.add_paper(paper); }
        black_box(p.query(&BinaryMultivector::from_bytes([1u8; 16])))
    }));
}
criterion_group!(benches, bench_cold_start);
criterion_main!(benches);
