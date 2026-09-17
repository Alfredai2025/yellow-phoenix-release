use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn bench_scale(c: &mut Criterion) {
    let scales = [1_000, 10_000, 100_000, 1_000_000];
    for &n in &scales {
        let mut pipeline = QueryPipeline::new_default();
        for i in 0..n {
            let b = [i as u8, (i>>8) as u8, (i>>16) as u8, (i>>24) as u8, 0,0,0,0,0,0,0,0,0,0,0,0];
            pipeline.add_paper(&BinaryMultivector::from_bytes(b));
        }
        let query = BinaryMultivector::from_bytes([1u8; 16]);
        c.bench_function(&format!("scale_{}k", n/1000), |b| {
            b.iter(|| pipeline.query(black_box(&query)))
        });
    }
}
criterion_group!(benches, bench_scale);
criterion_main!(benches);
