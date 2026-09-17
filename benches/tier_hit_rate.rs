use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pams::core::resonance::QueryPipeline;
use pams::types::multivector::BinaryMultivector;

fn generate_paper(topic_seed: u64, noise_bits: usize) -> BinaryMultivector {
    let mut bytes = [0u8; 16];
    for i in 0..16 {
        bytes[i] = ((topic_seed.wrapping_mul(0x9E3779B97F4A7C15) >> (i * 8)) as u8);
    }
    let mut rng = topic_seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..noise_bits {
        rng = rng.wrapping_mul(0x5851F42D4C957F2D)
            .wrapping_add(0x14057B7EF767814F);
        let bit = (rng % 128) as usize;
        bytes[bit / 8] ^= 1 << (bit % 8);
    }
    BinaryMultivector::from_bytes(bytes)
}

fn percentile(sorted: &[u32], pct: usize) -> u32 {
    let idx = (sorted.len() * pct / 100).min(sorted.len() - 1);
    sorted[idx]
}

fn bench_tier_hit_rate(c: &mut Criterion) {
    let n_topics = 100;
    let papers_per_topic = 1_000;
    let n_papers = n_topics * papers_per_topic;
    let noise_bits = 8;

    println!(
        "\n=== INGESTING {} PAPERS ({} topics × {} each, {} noise bits) ===",
        n_papers, n_topics, papers_per_topic, noise_bits
    );

    let mut all_papers = Vec::with_capacity(n_papers);
    for topic in 0..n_topics {
        for _ in 0..papers_per_topic {
            all_papers.push(generate_paper(topic as u64, noise_bits));
        }
    }

    // Build measurement pipelines with thresholds forced to return at each tier.
    println!("Building tier-score pipelines...");
    let mut p_tier0 = QueryPipeline::new(u32::MAX, u32::MAX, u32::MAX);
    let mut p_tier1 = QueryPipeline::new(0, u32::MAX, u32::MAX);
    let mut p_tier2 = QueryPipeline::new(0, 0, u32::MAX);
    let mut p_exact = QueryPipeline::new_default();

    for paper in &all_papers {
        p_tier0.add_paper(paper);
        p_tier1.add_paper(paper);
        p_tier2.add_paper(paper);
        p_exact.add_paper(paper);
    }

    println!("=== AUTO-TUNING THRESHOLDS ===");
    let sample_size = 1_000;
    let step = n_papers / sample_size;

    let mut s0 = Vec::with_capacity(sample_size);
    let mut s1 = Vec::with_capacity(sample_size);
    let mut s2 = Vec::with_capacity(sample_size);
    let mut exact_best = Vec::with_capacity(sample_size);

    for i in (0..n_papers).step_by(step) {
        let q = &all_papers[i];
        s0.push(p_tier0.query(q).score as u32);
        s1.push(p_tier1.query(q).score as u32);
        s2.push(p_tier2.query(q).score as u32);
        exact_best.push(p_exact.query(q).score as u32);
    }

    s0.sort();
    s1.sort();
    s2.sort();
    exact_best.sort();

    let t0 = percentile(&s0, 95);
    let t1 = percentile(&s1, 95);
    let t2 = percentile(&s2, 95);

    println!("  Tier 0 95th percentile score: {}", t0);
    println!("  Tier 1 95th percentile score: {}", t1);
    println!("  Tier 2 95th percentile score: {}", t2);
    println!("  Exact 95th percentile best:   {}", percentile(&exact_best, 95));

    // Build tuned pipeline
    println!("Building tuned pipeline...");
    let mut pipeline = QueryPipeline::new(t0, t1, t2);
    for paper in &all_papers {
        pipeline.add_paper(paper);
    }

    println!("=== TIER HIT RATE TEST (1000 queries) ===");
    let mut tier0_hits = 0u32;
    let mut tier1_hits = 0u32;
    let mut tier2_hits = 0u32;
    let mut tier3_hits = 0u32;
    let mut total_time = 0u64;

    for i in (0..n_papers).step_by(n_papers / 1000) {
        let q = &all_papers[i];
        let start = std::time::Instant::now();
        let r = pipeline.query(q);
        total_time += start.elapsed().as_nanos() as u64;

        match r.tier {
            0 => tier0_hits += 1,
            1 => tier1_hits += 1,
            2 => tier2_hits += 1,
            _ => tier3_hits += 1,
        }
    }

    let total = (tier0_hits + tier1_hits + tier2_hits + tier3_hits) as f32;
    println!("  Tier 0 (geometric): {:5} hits ({:.1}%)", tier0_hits, tier0_hits as f32 / total * 100.0);
    println!("  Tier 1:             {:5} hits ({:.1}%)", tier1_hits, tier1_hits as f32 / total * 100.0);
    println!("  Tier 2:             {:5} hits ({:.1}%)", tier2_hits, tier2_hits as f32 / total * 100.0);
    println!("  Tier 3 (exact):     {:5} hits ({:.1}%)", tier3_hits, tier3_hits as f32 / total * 100.0);
    println!("  Avg latency:        {} ns", total_time / 1000);

    println!("=== RECALL TEST (100 random queries) ===");
    let mut correct_top1 = 0u32;
    let mut recall_at_10 = 0f32;

    for i in 0..100 {
        let q = &all_papers[i * (n_papers / 100)];
        let fast = pipeline.query(q);

        let mut exact: Vec<(usize, u32)> = all_papers
            .iter()
            .enumerate()
            .map(|(j, p)| (j, q.hamming_distance(p)))
            .collect();
        exact.sort_by_key(|(_, d)| *d);

        if let Some((idx, _)) = fast.matches.first() {
            if *idx == exact[0].0 {
                correct_top1 += 1;
            }
        }

        let exact_set: std::collections::HashSet<usize> =
            exact.iter().take(10).map(|(i, _)| *i).collect();
        let fast_set: std::collections::HashSet<usize> =
            fast.matches.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = exact_set.intersection(&fast_set).count();
        recall_at_10 += overlap as f32 / 10.0;
    }

    println!("  Top-1 accuracy: {:.1}%", correct_top1 as f32);
    println!("  Recall@10:      {:.2}%", recall_at_10);

    let q = all_papers[0];
    c.bench_function("tier_hit_self_query", |b| {
        b.iter(|| black_box(pipeline.query(&q)))
    });
}

criterion_group!(benches, bench_tier_hit_rate);
criterion_main!(benches);
