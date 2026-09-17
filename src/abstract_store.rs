// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

//! YPA1 compressed abstract store reader (device-side).
//!
//! Format (built by scripts/build_abstracts_5m_bin.py):
//!   header: 5B magic+ver ("YPA1\x01") | u64 n_blocks @5 | u64 count @13
//!           | u32 dict_len @21 | u32 block_target @25 | dict @29
//!   blocks: zstd frames (dictionary-compressed) back to back
//!   index:  at EOF, n_blocks x 24B {u64 first_id, u64 offset, u32 nrec, u32 clen}
//!   record 0 of a block: varint ABSOLUTE id, varint len, text
//!   records 1+:          varint id_delta,   varint len, text
//!
//! Read-only, mmap-backed, single-block decode cache. Thread-safe via &self
//! (cache behind Mutex).

use memmap2::Mmap;
use std::fs::File;
use std::sync::Mutex;

const HEADER_LEN: usize = 29; // 4 magic + 1 ver + 8 + 8 + 4 + 4
const INDEX_REC: usize = 24;  // u64 + u64 + u32 + u32
const MAX_BLOCK_RAW: usize = 32 * 1024 * 1024; // safety cap on decompressed size

fn read_varint(raw: &[u8], pos: &mut usize) -> Option<u64> {
    let mut r: u64 = 0;
    let mut shift = 0u32;
    loop {
        let b = *raw.get(*pos)?;
        *pos += 1;
        r |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Some(r);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

fn read_u32(m: &Mmap, off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(m.get(off..off + 4)?.try_into().ok()?))
}
fn read_u64(m: &Mmap, off: usize) -> Option<u64> {
    Some(u64::from_le_bytes(m.get(off..off + 8)?.try_into().ok()?))
}

pub struct AbstractStore {
    map: Mmap,
    n_blocks: u64,
    index_off: usize,
    dict: Vec<u8>,
    /// (block_idx, decompressed raw payload) — most recent block only.
    cache: Mutex<(u64, Vec<u8>)>,
}

impl AbstractStore {
    pub fn open(path: &str) -> std::io::Result<Self> {
        let map = unsafe { Mmap::map(&File::open(path)?)? };
        if map.len() < HEADER_LEN + INDEX_REC || &map[0..4] != b"YPA1" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "not a YPA1 abstract store",
            ));
        }
        let n_blocks = read_u64(&map, 5).unwrap();
        let dict_len = read_u32(&map, 21).unwrap() as usize;
        let dict = map[HEADER_LEN..HEADER_LEN + dict_len].to_vec();
        let index_off = map
            .len()
            .checked_sub(n_blocks as usize * INDEX_REC)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "bad index"))?;
        Ok(Self {
            map,
            n_blocks,
            index_off,
            dict,
            cache: Mutex::new((u64::MAX, Vec::new())),
        })
    }

    pub fn len(&self) -> u64 {
        read_u64(&self.map, 13).unwrap_or(0)
    }

    /// Block meta: (first_id, offset, nrec, clen)
    fn block_meta(&self, bi: u64) -> Option<(u64, usize, u32, u32)> {
        let off = self.index_off + bi as usize * INDEX_REC;
        Some((
            read_u64(&self.map, off)?,
            read_u64(&self.map, off + 8)? as usize,
            read_u32(&self.map, off + 16)?,
            read_u32(&self.map, off + 20)?,
        ))
    }

    /// Find the block that should contain `id` (last block with first_id <= id).
    fn locate_block(&self, id: u64) -> Option<u64> {
        if self.n_blocks == 0 {
            return None;
        }
        let (mut lo, mut hi) = (0u64, self.n_blocks - 1);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            match self.block_meta(mid) {
                Some((first_id, _, _, _)) if first_id <= id => lo = mid,
                _ => hi = mid - 1,
            }
        }
        Some(lo)
    }

    fn block_raw(&self, bi: u64) -> Option<Vec<u8>> {
        {
            let c = self.cache.lock().ok()?;
            if c.0 == bi {
                return Some(c.1.clone());
            }
        }
        let (_first, off, _nrec, clen) = self.block_meta(bi)?;
        let comp = self.map.get(off..off + clen as usize)?;
        // zstd frame with dictionary; decompressed size unknown -> grow buffer.
        let mut dctx = zstd_safe::DCtx::default();
        let mut cap = 1024usize * 1024;
        let raw = loop {
            let mut buf = vec![0u8; cap];
            match dctx.decompress_using_dict(buf.as_mut_slice(), comp, self.dict.as_slice()) {
                Ok(n) => {
                    buf.truncate(n);
                    break buf;
                }
                Err(e) if cap < MAX_BLOCK_RAW => {
                    // grow only on "destination buffer too small"
                    if zstd_safe::get_error_name(e).contains("DstSize") {
                        cap = (cap * 2).min(MAX_BLOCK_RAW);
                    } else {
                        return None;
                    }
                }
                Err(_) => return None,
            }
        };
        let mut c = self.cache.lock().ok()?;
        *c = (bi, raw.clone());
        Some(raw)
    }

    /// Decode the abstract for `id`. Returns None if the id is not present
    /// or on any format error; Some(String) otherwise (possibly empty —
    /// papers without abstracts are stored as empty records).
    pub fn decode(&self, id: u64) -> Option<String> {
        let bi = self.locate_block(id)?;
        let (_first, _off, nrec, _clen) = self.block_meta(bi)?;
        let raw = self.block_raw(bi)?;
        let mut pos = 0usize;
        // record 0: absolute id
        let mut cur = read_varint(&raw, &mut pos)?;
        for k in 0..nrec as usize {
            let len = read_varint(&raw, &mut pos)? as usize;
            let end = pos.checked_add(len)?;
            if end > raw.len() {
                return None;
            }
            if cur == id {
                return String::from_utf8(raw[pos..end].to_vec()).ok();
            }
            pos = end;
            if k + 1 < nrec as usize {
                let delta = read_varint(&raw, &mut pos)?;
                cur = cur.checked_add(delta)?;
            }
        }
        None
    }
}
