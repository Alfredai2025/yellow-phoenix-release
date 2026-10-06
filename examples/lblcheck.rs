fn main() {
    let g = pams::binary_hnsw::BinaryHNSW::load(&std::env::args().nth(1).unwrap()).unwrap();
    let sample = g.len().min(100_000);
    let mut non_id = 0u32;
    for i in 0..sample as u32 {
        if g.node_label(i) != i as u64 { non_id += 1; }
    }
    let first: Vec<u64> = (0..g.len().min(8) as u32).map(|i| g.node_label(i)).collect();
    println!("nodes={} first_labels={:?} non_identity_in_first_{}={}", g.len(), first, sample, non_id);
}
