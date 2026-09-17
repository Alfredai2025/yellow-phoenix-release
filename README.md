
## Binary HNSW (v3.7)

Native Rust HNSW index for 512-bit ITQ hashes.  3.2× faster than FAISS HNSW, 9.4× smaller memory.

### Quick start
```bash
cargo build --release --features living_mesh
python -c "from yp_bridge import BinaryHNSW; h = BinaryHNSW(); print(len(h))"
```

### Benchmarks (100K vectors)
- Build: 18.3 s
- Search P50: 118 µs
- Memory: 18 MB
- Two-tier R@1: 99.8%

See `docs/binary_hnsw_summary.md` for full details.

## A note on naming
Early research-phase commits (May–September 2026) used phenomenological
codenames — "geometric brain", "mind", "consciousness" — for subsystems
that are geometric-algebra index structures and cross-signal monitoring
components. These are engineering components, not claims about machine
consciousness; the tree has since been renamed to neutral terms. Early
commit messages retain the old names and refer to the same components.

## License

Dual-licensed by the sole copyright holder (Marc John Sawyer):

- [AGPL-3.0-or-later](LICENSE) for open-source use and research
- [Commercial license](COMMERCIAL.md) available for closed-source products

External contributions require the [CLA](CLA.md).
Every source file carries an `SPDX-License-Identifier: AGPL-3.0-or-later` header.
