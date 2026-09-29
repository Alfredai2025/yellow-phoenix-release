# Privacy Policy — Yellow Phoenix

**Effective date: 2026-09-29** · Applies to the Yellow Phoenix open-source engine
(this repository and official releases)

## The short version

Yellow Phoenix is an **on-device, offline-first semantic retrieval engine**.
It is designed so that there is nothing to leak:

- **We collect nothing.** The software has no analytics, no telemetry, no
  crash reporting, no usage statistics, no tracking of any kind.
- **No accounts, no servers we run.** There is no Yellow Phoenix service.
  The engine runs entirely on your hardware.
- **Your corpus never leaves your device** unless *you* choose to export or
  sync it. Personal Library and Grow Mode data (documents, papers, embeddings,
  indexes) live in local storage under your control.
- **Network access is explicit and user-initiated.** Harvest connectors
  (arXiv, OpenAlex, OAI-PMH) contact those public services only when you run
  them, using standard polite-pool conventions. Queries you type are sent
  only to the endpoints you configure, and only when you run a harvest.
- **No third-party SDKs, no ads, no monetization pipeline.**

## What the project itself sees

Nothing. This repository distributes source code. Downloading, building, or
running Yellow Phoenix transmits no information to the project maintainers.

## Your responsibilities as a user

- Documents you import into your Personal Library may contain personal data
  (your own papers, student records, etc.). **You are the data controller**
  for that content; Yellow Phoenix is a tool you run locally.
- If you build and distribute a product on top of Yellow Phoenix, you are
  responsible for the privacy posture of *your* product. The engine imposes
  no telemetry on you, which also means it cannot help you comply — that is
  your layer.

## Changes

Material changes to this policy will be recorded in `CHANGELOG.md` and
committed to this repository, which is the canonical source for this policy.

## Contact

Privacy questions: open an issue at
https://github.com/Alfredai2025/yellow-phoenix-release/issues
(or contact the maintainer via the GitHub profile).

---

*This document describes software behavior and project practice. It is not
legal advice.*
