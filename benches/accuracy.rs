use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn bench_accuracy(c: &mut Criterion) {
    let n = 100_000;
    let mut pipeline = QueryPipeline::new_default();
    for i in 0..n {
        let b = [i as u8, (i>>8) as u8, (i>>16) as u8, (i>>24) as u8, 0,0,0,0,0,0,0,0,0,0,0,0];
        pipeline.add_paper(&BinaryMultivector::from_bytes(b));
    }
    let queries = [
        ("self", BinaryMultivector::from_bytes([0u8; 16])),
        ("neighbor", BinaryMultivector::from_bytes([0u8,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])),
        ("random", BinaryMultivector::from_bytes([0xABu8; 16])),
        ("adversarial", BinaryMultivector::from_bytes([0xFFu8; 16])),
    ];
    println!("\n=== TIER HIT RATES (100K papers) ===");
    for (name, q) in &queries {
        let r = pipeline.query(q);
        println!("{:12}: tier={:?}, score={}, matches={}", name, r.tier, r.score, r.matches.len());
    }
    c.bench_function("accuracy_dummy", |b| b.iter(|| black_box(pipeline.query(&queries[1].1))));
}
criterion_group!(benches, bench_accuracy);
criterion_main!(benches);
