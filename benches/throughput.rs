use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn bench_throughput(c: &mut Criterion) {
    let n = 100_000;
    let mut pipeline = QueryPipeline::new_default();
    for i in 0..n {
        let b = [i as u8, (i>>8) as u8, 0,0,0,0,0,0,0,0,0,0,0,0,0,0];
        pipeline.add_paper(&BinaryMultivector::from_bytes(b));
    }
    let q = BinaryMultivector::from_bytes([1u8; 16]);
    c.bench_function("sustained_100q", |b| b.iter(|| {
        for _ in 0..100 { black_box(pipeline.query(&q)); }
    }));
    let mut ctr = 0u8;
    c.bench_function("mixed_10pct_ingest", |b| b.iter(|| {
        black_box(pipeline.query(&q));
        ctr = ctr.wrapping_add(1);
        if ctr % 10 == 0 {
            pipeline.add_paper(&BinaryMultivector::from_bytes([ctr; 16]));
        }
    }));
}
criterion_group!(benches, bench_throughput);
criterion_main!(benches);
