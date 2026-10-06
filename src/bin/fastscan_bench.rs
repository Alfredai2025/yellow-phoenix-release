// M2 brick: 4-bit PQ fast-scan kernel microbench (Apple Silicon NEON).
// Design per council review (SiliconFlow, 2026-10-06): vqtbl1q_s8 LUT + int16
// accumulate; 32-code blocks, low nibble = subq 2j, high nibble = subq 2j+1.
// Layout: M/2 blocks of 32B per 32 codes. Zero copied code (Andre 2016 / faiss
// MIT mechanics re-implemented from the papers).
//
// Gate: >20M query-code distances/sec single-thread (M3 Pro).
// Usage: fastscan_bench [n_vectors] [n_queries]
#![allow(clippy::missing_safety_doc)]

use std::arch::aarch64::*;
use std::time::Instant;

const M: usize = 64; // subquantizers (4-bit each) -> 32 bytes/code
const BBS: usize = 32; // codes per block
const BLOCK_BYTES: usize = BBS; // 32 codes x 2 nibbles = 32 bytes (1 byte per code)
const NB: usize = M / 2; // blocks per 32-code group

/// Pack (n, M) nibble codes into block layout.
/// Block j of a 32-code group: byte k = code[base+k, 2j] | code[base+k, 2j+1] << 4.
fn pack(codes: &[u8], n: usize) -> Vec<u8> {
    assert_eq!(codes.len(), n * M);
    let groups = n / BBS;
    let mut packed = vec![0u8; groups * NB * BLOCK_BYTES];
    for g in 0..groups {
        for j in 0..NB {
            for k in 0..BBS {
                let lo = codes[(g * BBS + k) * M + 2 * j];
                let hi = codes[(g * BBS + k) * M + 2 * j + 1];
                packed[(g * NB + j) * BLOCK_BYTES + k] = lo | (hi << 4);
            }
        }
    }
    packed
}

/// Per-query int8 LUTs: lut[m][c] for c in 0..16. Values biased per-subq so the
/// u8 accumulation cannot wrap the u16 lane before reduction (distances are
/// compared within a query only, so a per-subq affine bias is ranking-neutral
/// at int8 precision — the Andre-style float correction lives at decode time).
#[inline]
unsafe fn scan_group(
    blocks: &[u8],          // NB * 32 bytes for this 32-code group
    luts: &[*const u8; NB], // per-block int8x16 LUT pointers (subq 2j then 2j+1)
) -> [u16; BBS] {
    let mut acc = [vdupq_n_u16(0); 4]; // 4 x 8 lanes = 32 u16 accumulators
    let mask = vdupq_n_u8(0x0F);
    for j in 0..NB {
        let b0 = vld1q_u8(blocks.as_ptr().add(j * BLOCK_BYTES));
        let b1 = vld1q_u8(blocks.as_ptr().add(j * BLOCK_BYTES + 16));
        let lut_lo = vld1q_s8(luts[j] as *const i8);
        let lut_hi = vld1q_s8(luts[j].add(16) as *const i8);
        // low nibbles -> subq 2j LUT; high nibbles -> subq 2j+1 LUT
        let lo0 = vandq_u8(b0, mask);
        let hi0 = vshrq_n_u8(b0, 4);
        let lo1 = vandq_u8(b1, mask);
        let hi1 = vshrq_n_u8(b1, 4);
        let d0 = vqtbl1q_s8(lut_lo, lo0);
        let d1 = vqtbl1q_s8(lut_hi, hi0);
        let d2 = vqtbl1q_s8(lut_lo, lo1);
        let d3 = vqtbl1q_s8(lut_hi, hi1);
        // pairwise u8->u16 accumulate (table bits are unsigned-biased values)
        acc[0] = vpadalq_u8(acc[0], vreinterpretq_u8_s8(d0));
        acc[1] = vpadalq_u8(acc[1], vreinterpretq_u8_s8(d1));
        acc[2] = vpadalq_u8(acc[2], vreinterpretq_u8_s8(d2));
        acc[3] = vpadalq_u8(acc[3], vreinterpretq_u8_s8(d3));
    }
    let mut out = [0u16; BBS];
    vst1q_u16(out.as_mut_ptr(), acc[0]);
    vst1q_u16(out.as_mut_ptr().add(8), acc[1]);
    vst1q_u16(out.as_mut_ptr().add(16), acc[2]);
    vst1q_u16(out.as_mut_ptr().add(24), acc[3]);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
    let nq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let groups = n / BBS;

    // synthetic codes + LUTs (kernel gate is throughput, not data)
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let codes: Vec<u8> = (0..n * M).map(|_| (next() & 0xF) as u8).collect();
    let packed = pack(&codes, n);
    let lut_bufs: Vec<Vec<u8>> = (0..NB)
        .map(|j| {
            (0..32)
                .map(|i| ((next() >> 32) & 0xFF) as u8)
                .collect()
        })
        .collect();
    let lut_ptrs: Vec<[*const u8; NB]> = (0..nq)
        .map(|_| {
            let mut arr: [*const u8; NB] = [std::ptr::null(); NB];
            for j in 0..NB {
                arr[j] = lut_bufs[(j * 7 + 3) % NB].as_ptr(); // fixed set, hot in L1
            }
            arr
        })
        .collect();

    // ---- fast-scan pass ----
    let mut sink = 0u64;
    let t = Instant::now();
    for q in 0..nq {
        let luts = &lut_ptrs[q];
        for g in 0..groups {
            let d = unsafe { scan_group(&packed[g * NB * BLOCK_BYTES..(g + 1) * NB * BLOCK_BYTES], luts) };
            sink += d[0] as u64; // keep alive
        }
    }
    let dt = t.elapsed().as_secs_f64();
    let dists = nq as f64 * n as f64;
    let dps = dists / dt;
    println!(
        "FASTSCAN n={} q={}: {:.1}s, {:.2}M dist/sec ({:.2} GB/s codes), sink={}",
        n, nq, dt, dps / 1e6, dps * 32.0 / 1e9, sink
    );

    // ---- scalar reference (unpacked LUT loop) for contrast ----
    let lut_u16: Vec<[u16; 16]> = (0..M)
        .map(|m| {
            let mut a = [0u16; 16];
            for c in 0..16 {
                a[c] = (((m * 31 + c * 17 + 7) % 251) + 1) as u16;
            }
            a
        })
        .collect();
    let t = Instant::now();
    let mut sink2 = 0u64;
    let sample = n.min(20_000);
    for _q in 0..nq {
        for i in 0..sample {
            let mut s = 0u32;
            for m in 0..M {
                s += lut_u16[m][codes[i * M + m] as usize] as u32;
            }
            sink2 += s as u64;
        }
    }
    let dt2 = t.elapsed().as_secs_f64();
    let dps2 = nq as f64 * sample as f64 / dt2;
    println!(
        "SCALAR-REF n={}: {:.2}M dist/sec, sink={}",
        sample, dps2 / 1e6, sink2
    );
    println!("GATE >20M dist/sec: {}", if dps > 20e6 { "PASS" } else { "FAIL" });
}
