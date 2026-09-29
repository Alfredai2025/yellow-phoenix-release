# Contributing to Yellow Phoenix

Thank you for your interest in the project. Yellow Phoenix is developed
in the open; contributions, issues, and benchmarks from independent
verification are especially welcome.

## Ground rules

- The engine is licensed AGPL-3.0-or-later. Contributions are accepted
  under the same license. Commercial dual licensing is handled separately
  (contact the maintainer via GitHub).
- Honest measurement is the house culture: benchmarks must state corpus,
  scale, hardware, ground-truth method, and raw result files. Negative
  results are first-class contributions.

## Getting started

1. Fork the repository and clone your fork.
2. Build: `cargo build --release` (Rust toolchain, stable channel).
3. Type-check everything (fast): `cargo check --workspace --all-targets`.
4. Python helpers live under `scripts/` and `tests/` (Python 3.10+,
  numpy). Some integration tests additionally require built artifacts
  and model files that are not distributed in this repository; the CI
  workflow runs the self-contained checks.

## Making changes

- Open an issue first for anything non-trivial (design discussion is
  cheap, reverts are expensive).
- Small, focused commits with descriptive messages. One logical change
  per commit.
- Update `CHANGELOG.md` (Unreleased section) with user-visible changes.
- Ensure `cargo check --workspace --all-targets` passes before pushing.

## Reporting bugs

Use the bug-report issue template and include: hardware, corpus scale,
reproduction steps, and the exact commit hash.

## Roadmap process

Larger items are tracked as issues with the `roadmap` label. The
maintainer's standing priorities: recall-per-byte at fixed latency
budgets, edge-device memory ceilings, and reproducibility of every
published number.
