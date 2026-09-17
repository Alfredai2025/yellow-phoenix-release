use pams::intelligent_shard_manager::IntelligentShardManager;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

fn generate_records_file(n: usize, path: &str) {
    let mut rng = fastrand::Rng::new();
    let file = File::create(path).expect("create temp file");
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    let mut hash = [0u8; 32];
    for i in 0..n as u64 {
        writer.write_all(&i.to_le_bytes()).unwrap();
        rng.fill(&mut hash[..]);
        writer.write_all(&hash).unwrap();
    }
    writer.flush().unwrap();
}

fn benchmark(n: usize) {
    println!("\n🔄 ISM Rust benchmark (mmap file build): {} papers", n);
    let tmp = PathBuf::from(format!("/tmp/ism_{}m_records.bin", n / 1_000_000));

    let t_gen = Instant::now();
    generate_records_file(n, tmp.to_str().unwrap());
    println!("  generated {} in {:.2}s", tmp.display(), t_gen.elapsed().as_secs_f64());

    let mode = std::env::args().nth(2).unwrap_or_else(|| "seq".to_string());
    let mut mgr = IntelligentShardManager::new();
    let t0 = Instant::now();
    if mode == "par" {
        mgr.build_parallel_flat_from_file(tmp.to_str().unwrap(), n, 32)
            .unwrap();
    } else {
        mgr.build_flat_from_file_sequential(tmp.to_str().unwrap(), n, 32)
            .unwrap();
    }
    let build_s = t0.elapsed().as_secs_f64();
    println!("  mode         : {}", mode);

    let mut latencies: Vec<f64> = Vec::with_capacity(1000);
    let mut rng = fastrand::Rng::new();
    for _ in 0..1000 {
        let id = rng.usize(0..n) as u64;
        let tq = Instant::now();
        let _ = mgr.query_parallel(id);
        latencies.push(tq.elapsed().as_secs_f64() * 1_000_000.0);
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];

    println!("  build_s      : {:.2}", build_s);
    println!("  insert_rate  : {:.0}/s", n as f64 / build_s);
    println!("  p50_us       : {:.2}", p50);
    println!("  p99_us       : {:.2}", p99);
    println!("  shards       : {}", mgr.shards.len());
    println!("  total_papers : {}", mgr.total_count);

    std::fs::remove_file(&tmp).ok();
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    benchmark(n);
}
