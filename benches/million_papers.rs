use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn million_papers(c: &mut Criterion) {
    let mut pipeline = QueryPipeline::new_default();

    // Ingest 1M random papers
    println!("Ingesting 1M papers...");
    let start = std::time::Instant::now();
    for i in 0..1_000_000 {
        let bytes = [
            (i as u8),
            ((i >> 8) as u8),
            ((i >> 16) as u8),
            ((i >> 24) as u8),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let paper = BinaryMultivector::from_bytes(bytes);
        pipeline.add_paper(&paper);
    }
    println!("Ingested in {:?}", start.elapsed());

    let query = BinaryMultivector::from_bytes([1u8; 16]);

    c.bench_function("query_1m_papers", |b| {
        b.iter(|| pipeline.query(black_box(&query)))
    });
}

criterion_group!(benches, million_papers);
criterion_main!(benches);
