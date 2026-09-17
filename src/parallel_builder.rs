// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Marc John Sawyer

use crate::shard::{Shard, ShardError, IsmPap};
use memmap2::Mmap;

pub struct ParallelBuilder {
    n_shards: usize,
}

impl ParallelBuilder {
    pub fn new(n_shards: usize) -> Self {
        ParallelBuilder { n_shards }
    }

    pub fn build_shards_flat(
        &self,
        ids: &[u64],
        hashes: &[u8],
        hash_len: usize,
        ranges: Vec<(usize, usize)>,
    ) -> Result<Vec<Shard>, ShardError> {
        let mut shards = Vec::with_capacity(self.n_shards);
        for (shard_id, (start, end)) in ranges.into_iter().enumerate() {
            let mut shard = Shard::new(shard_id as u32, end - start);
            for idx in start..end {
                let id = ids[idx];
                let hash_start = idx * hash_len;
                let hash_end = hash_start + hash_len;
                let hash_bytes = &hashes[hash_start..hash_end];
                let mut pap: IsmPap = [0u8; 32];
                let copy_len = hash_len.min(32);
                pap[..copy_len].copy_from_slice(&hash_bytes[..copy_len]);
                shard.insert(id, pap)?;
            }
            shards.push(shard);
        }
        Ok(shards)
    }

    pub fn build_shards_flat_from_file(
        &self,
        mmap: &Mmap,
        hash_len: usize,
        ranges: Vec<(usize, usize)>,
    ) -> Result<Vec<Shard>, ShardError> {
        let mut shards = Vec::with_capacity(self.n_shards);
        let record_size = 8 + hash_len;
        
        for (shard_id, (start, end)) in ranges.into_iter().enumerate() {
            let mut shard = Shard::new(shard_id as u32, end - start);
            for idx in start..end {
                let offset = idx * record_size;
                let id = u64::from_le_bytes(mmap[offset..offset + 8].try_into().unwrap());
                let hash_bytes = &mmap[offset + 8..offset + 8 + hash_len];
                let mut pap: IsmPap = [0u8; 32];
                let copy_len = hash_len.min(32);
                pap[..copy_len].copy_from_slice(&hash_bytes[..copy_len]);
                shard.insert(id, pap)?;
            }
            shards.push(shard);
        }
        Ok(shards)
    }

    pub fn build_shards(
        &self,
        chunks: Vec<Vec<(u64, IsmPap)>>,
    ) -> Result<Vec<Shard>, ShardError> {
        let mut shards = Vec::with_capacity(chunks.len());
        for (shard_id, chunk) in chunks.into_iter().enumerate() {
            let mut shard = Shard::new(shard_id as u32, chunk.len());
            for (id, pap) in chunk {
                shard.insert(id, pap)?;
            }
            shards.push(shard);
        }
        Ok(shards)
    }
}
