//! Decisive encoder test: on the Mac, with the real model + real row 0,
//! does `yp_itq_encode` produce the graph hash (c40a51...) like Python, or
//! the device's 23508a...? Run:
//!   YP_GENESIS_PHRASE=YP_DEV_BUILD_2026_08_19 cargo test --release itq_row0 -- --nocapture

use std::os::unix::fs::FileExt;

fn f16_to_f32_exact(h: u16) -> f32 {
    let mag = f32::from_bits(((h & 0x7fff) as u32) << 13) * f32::from_bits(0x7780_0000);
    if h & 0x8000 != 0 { -mag } else { mag }
}

#[test]
fn itq_row0_matches_graph() {
    let itq = "/Users/mac/yellow_phoenix_mobile/YPPhone/Resources/itq512_W_mean.bin";
    let f16 = "/Users/mac/yellow_phoenix/data/floats_5m.f16";
    let hnsw = "/Users/mac/yellow_phoenix_mobile/../yellow_phoenix/data/real_5m_hnsw.bin";

    // NOTE: the Mac's real_5m_hnsw.bin is the STALE gear-hash build, so we
    // compare against the PHONE node-0 hash pulled earlier instead:
    let phone_node0_hash_prefix: [u8; 8] = [0xc4, 0x0a, 0x51, 0x1f, 0xcb, 0x39, 0x6e, 0xa6];
    let _ = hnsw;

    let rc = pams::ffi_itq::yp_itq_init(
        std::ffi::CString::new(itq).unwrap().as_ptr());
    assert_eq!(rc, 0, "yp_itq_init failed");

    // read f16 row 0
    let f = std::fs::File::open(f16).unwrap();
    let mut raw = [0u8; 384 * 2];
    f.read_exact_at(&mut raw, 32).unwrap();
    let mut x = [0f32; 384];
    for d in 0..384 {
        x[d] = f16_to_f32_exact(u16::from_le_bytes([raw[d * 2], raw[d * 2 + 1]]));
    }

    let mut hash = [0u8; 64];
    let rc = pams::ffi_itq::yp_itq_encode(x.as_ptr(), 384, hash.as_mut_ptr());
    assert_eq!(rc, 0, "yp_itq_encode failed");

    println!("Rust yp_itq_encode(row0)[:8] = {:?}", &hash[..8]);
    println!("expected (graph)      [:8] = {:?}", phone_node0_hash_prefix);
    assert_eq!(&hash[..8], &phone_node0_hash_prefix, "device mismatch reproduces on Mac");
}
