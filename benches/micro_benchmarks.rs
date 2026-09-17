use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::direct_hash::DirectIndex;
use pams::multi_base_crystal::MultiBaseCrystal;
use pams::dynamic_mesh::DynamicMesh;
use pams::multi_base_dynamic::MultiBaseDynamic;
use pams::cascade_index::CascadeIndex;
use pams::intelligent::IntelligentRouter;
use pams::crystal::CrystalMesh;
use pams::types::multivector::BinaryMultivector;

fn make_pap(seed: u64) -> [u8; 64] {
    let mut pap = [0u8; 64];
    for i in 0..64 {
        pap[i] = ((seed.wrapping_mul(7919 + i as u64)) % 256) as u8;
    }
    pap
}

fn make_binary(seed: u64) -> BinaryMultivector {
    let mut chunks = [0u64; 2];
    for i in 0..128 {
        if ((seed.wrapping_mul(i as u64 + 1)) % 7) < 3 {
            chunks[i / 64] |= 1u64 << (i % 64);
        }
    }
    BinaryMultivector(chunks)
}

fn bench_direct_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("direct_hash");
    
    let mut index = DirectIndex::new();
    for i in 0..10000 {
        let pap = make_pap(i);
        index.insert(i as u64, &pap);
    }
    
    let query_pap = make_pap(5000);
    
    group.bench_function("single_lookup", |b| {
        b.iter(|| black_box(index.query(&query_pap)))
    });
    
    group.bench_function("batch_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(index.query(&make_pap(i * 100)));
            }
        })
    });
    
    group.finish();
}

fn bench_multi_base_crystal(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_base_crystal");
    
    let bases = vec![3u32, 5, 7, 11];
    let pap = make_pap(42);
    
    group.bench_function("compute_coord_base3", |b| {
        b.iter(|| black_box(MultiBaseCrystal::compute_coord(&pap, 3)))
    });
    
    group.bench_function("compute_coord_base11", |b| {
        b.iter(|| black_box(MultiBaseCrystal::compute_coord(&pap, 11)))
    });
    
    group.bench_function("compute_all_bases", |b| {
        b.iter(|| {
            for &base in &bases {
                black_box(MultiBaseCrystal::compute_coord(&pap, base));
            }
        })
    });
    
    group.finish();
}

fn bench_dynamic_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("dynamic_mesh");
    
    let mut mesh = DynamicMesh::new(5, 0.5, 0.99, 0.3, 0.1, 0.1, 0.01);
    
    // Pre-populate
    for i in 0..1000 {
        let pap = make_pap(i);
        mesh.insert(&pap, i);
    }
    
    let query_pap = make_pap(500);
    
    group.bench_function("insert", |b| {
        let mut counter = 1000u64;
        b.iter(|| {
            let pap = make_pap(counter);
            mesh.insert(&pap, counter);
            counter += 1;
        })
    });
    
    group.bench_function("query_spatial_radius1", |b| {
        b.iter(|| black_box(mesh.query_spatial(MultiBaseCrystal::compute_coord(&query_pap, 5), 1)))
    });
    
    group.bench_function("query_spatial_radius2", |b| {
        b.iter(|| black_box(mesh.query_spatial(MultiBaseCrystal::compute_coord(&query_pap, 5), 2)))
    });
    
    group.finish();
}

fn bench_multi_base_dynamic(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_base_dynamic");
    
    let mut hierarchy = MultiBaseDynamic::new(
        vec![3u32, 5, 7],
        0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
        vec![0.1, 0.1, 0.1],
    );
    
    // Pre-populate
    for i in 0..1000 {
        let pap = make_pap(i);
        hierarchy.insert(&pap, i);
    }
    
    let query_pap = make_pap(500);
    
    group.bench_function("insert", |b| {
        let mut counter = 1000u64;
        b.iter(|| {
            let pap = make_pap(counter);
            hierarchy.insert(&pap, counter);
            counter += 1;
        })
    });
    
    group.bench_function("query_hierarchical_top5", |b| {
        b.iter(|| black_box(hierarchy.query_hierarchical(&query_pap, 5, 1, 1)))
    });
    
    group.finish();
}

fn bench_intelligent_router(c: &mut Criterion) {
    let mut group = c.benchmark_group("intelligent_router");

    // Train PQ first
    let mut train = Vec::with_capacity(8000);
    for i in 0..1000 {
        for d in 0..8 {
            train.push(((i * 17 + d * 31) % 100) as f32 / 100.0);
        }
    }
    let mut cascade = CascadeIndex::new(64, 1, 2, 4, 8);
    cascade.train_pq(&train, 1000);

    let direct = DirectIndex::new();
    let multi_dynamic = MultiBaseDynamic::new(
        vec![3u32, 5, 7],
        0.5, 0.99, 0.3, 0.1, 0.1, 0.01,
        vec![0.1, 0.1, 0.1],
    );
    let crystal = CrystalMesh::new();
    let mut router = IntelligentRouter::new(direct, multi_dynamic, crystal, cascade);

    // Pre-populate
    for i in 0..1000 {
        let pap = make_pap(i);
        let binary = make_binary(i);
        let vec: Vec<f32> = (0..8).map(|d| ((i * 17 + d * 31) % 100) as f32 / 100.0).collect();
        router.add(&pap, binary, &vec, &vec);
    }

    let query_pap = make_pap(500);
    let query_binary = make_binary(500);
    let query_float: Vec<f32> = (0..8).map(|d| ((500 * 17 + d * 31) % 100) as f32 / 100.0).collect();

    group.bench_function("query_full_path", |b| {
        b.iter(|| black_box(router.query(&query_pap, &query_binary, &query_float, 5, 1, 2)))
    });

    group.bench_function("query_fast", |b| {
        b.iter(|| black_box(router.query_fast(&query_pap, &query_binary, &query_float, 5, 1)))
    });

    group.finish();
}
criterion_group!(benches, bench_direct_hash, bench_multi_base_crystal, bench_dynamic_mesh, bench_multi_base_dynamic, bench_intelligent_router);
criterion_main!(benches);
