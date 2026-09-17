use criterion::{black_box, Criterion, criterion_group, criterion_main, Throughput};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;
use pams::query::LSHIndex;

fn bench_tier1_query(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut pipeline = QueryPipeline::new(0, u32::MAX, u32::MAX);
    let papers: Vec<BinaryMultivector> = (0..N)
        .map(|i| BinaryMultivector::from_seed_text(&format!("paper-{}", i)))
        .collect();
    for p in &papers {
        pipeline.add_paper(p);
    }
    let q = BinaryMultivector::from_seed_text("bench-query-tier1");

    c.bench_function("tier1_query", |b| {
        b.iter(|| {
            let res = pipeline.query(&q);
            black_box(res);
        })
    });
}

fn bench_tier2_ensemble(c: &mut Criterion) {
    const N: usize = 1_000_000;
    let mut pipeline = QueryPipeline::new(0, 0, u32::MAX);
    let papers: Vec<BinaryMultivector> = (0..N)
        .map(|i| BinaryMultivector::from_seed_text(&format!("paper-{}", i)))
        .collect();
    for p in &papers {
        pipeline.add_paper(p);
    }
    let q = BinaryMultivector::from_seed_text("bench-query-tier2");

    c.bench_function("tier2_ensemble", |b| {
        b.iter(|| {
            let res = pipeline.query(&q);
            black_box(res);
        })
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    const N: usize = 10_000;
    let mut pipeline = QueryPipeline::new(0, 0, 0);
    let papers: Vec<BinaryMultivector> = (0..N)
        .map(|i| BinaryMultivector::from_seed_text(&format!("paper-{}", i)))
        .collect();
    for p in &papers {
        pipeline.add_paper(p);
    }
    let q = BinaryMultivector::from_seed_text("bench-query-full");

    c.bench_function("full_pipeline", |b| {
        b.iter(|| {
            let res = pipeline.query(&q);
            black_box(res);
        })
    });
}

fn bench_ingestion(c: &mut Criterion) {
    let sample = BinaryMultivector::from_seed_text("incoming-paper");
    c.bench_function("ingestion", |b| {
        b.iter_with_setup(
            || {
                let mut pipeline = QueryPipeline::new(0, 0, 0);
                for i in 0..1000 {
                    pipeline.add_paper(&BinaryMultivector::from_seed_text(
                        &format!("prefill-{}", i),
                    ));
                }
                pipeline
            },
            |mut pipeline| {
                pipeline.add_paper(&sample);
            },
        )
    });
}

fn bench_lsh_query(c: &mut Criterion) {
    let sizes = [10_000usize, 100_000, 1_000_000];
    let mut group = c.benchmark_group("lsh_query");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(2));
    group.throughput(Throughput::Elements(1));

    let q = BinaryMultivector::from_seed_text("bench-query");

    for &n in &sizes {
        let mut idx = LSHIndex::new(256, 8);
        for i in 0..n {
            idx.add(BinaryMultivector::from_seed_text(&format!("paper-{}", i)));
        }

        let id_top1 = format!("size_{}_top1", n);
        group.bench_function(&id_top1, |b| {
            b.iter(|| {
                let res = idx.query(&q, 1, 0.0);
                black_box(res);
            })
        });

        let id_top5 = format!("size_{}_top5", n);
        group.bench_function(&id_top5, |b| {
            b.iter(|| {
                let res = idx.query(&q, 5, 0.0);
                black_box(res);
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tier1_query,
    bench_tier2_ensemble,
    bench_full_pipeline,
    bench_ingestion,
    bench_lsh_query
);
criterion_main!(benches);
