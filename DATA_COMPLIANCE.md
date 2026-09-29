# Data & Regulatory Compliance — Yellow Phoenix

**Effective date: 2026-09-29**

This document explains how Yellow Phoenix relates to GDPR (EU/UK), CCPA/CPRA
(California), and similar privacy regimes, and what data protection posture
the project maintains.

## TL;DR

Yellow Phoenix is an on-device engine with **no telemetry, no accounts, and
no project-operated service**. The project itself is **not a data controller
or processor** for any user's content. The user is the controller of any
personal data they choose to index locally.

## 1. Roles under GDPR

| Party | Role | Why |
|---|---|---|
| Yellow Phoenix project (maintainers) | Neither controller nor processor | We operate no service and receive no personal data from the software |
| End user running the engine | Controller of their Personal Library | They decide what documents/papers to import; data stays on their device |
| Optional connectors (arXiv, OpenAlex, OAI-PMH endpoints) | Independent controllers of their own services | Their terms/privacy policies apply to your use of them |

## 2. What personal data the software touches

- **Locally imported content** (PDFs, papers, notes): stored on-device, in
  files/databases under the user's control. Never transmitted by the engine.
- **Queries**: processed in memory on-device; no query log leaves the device.
- **Harvest operations**: the connectors send request metadata that any HTTP
  client sends (IP address, user-agent; polite-pool conventions add a contact
  `mailto` where configured) to the *source's* servers. Those sources' own
  GDPR/CCPA notices govern that interaction — not this document.

## 3. Data protection by design

- **Minimization**: the engine stores only what the user imports or harvests.
  Nothing is collected "because we might want it."
- **Locality**: embeddings, indexes, and libraries live in on-device storage;
  Personal Library and Grow Mode work fully offline.
- **Deletion = actual deletion**: removing a paper from the library deletes
  its embedding and cached PDF (see Grow Mode swipe-to-delete); no shadow
  copies are retained by the project (we never had them).
- **No profiling, no cross-device identifiers, no advertising.**

## 4. CCPA/CPRA specifics

California residents' rights (know, delete, correct, opt-out of sale/share)
are rights against *businesses that collect personal information*. The Yellow
Phoenix project collects no personal information and sells none. If a user
builds a product on the engine, that user's business must honor CCPA for
*their* collection — the engine neither helps nor hinders; it simply collects
nothing on its own.

## 5. If you embed Yellow Phoenix in your product

You inherit a privacy-clean foundation (nothing to disclose about engine
telemetry, because there is none), but you remain responsible for:
- Your own collection/processing disclosures (GDPR Art. 13/14, CCPA notice),
- Lawful basis for any personal data *you* index into the engine,
- Source-license compliance for harvested datasets (e.g., OpenAlex data is
  ODC-BY; arXiv metadata is subject to arXiv's terms; respect robots/polite
  pools).

## 6. Data sources and their terms

| Source | License / terms (as commonly stated) |
|---|---|
| arXiv metadata & abstracts | arXiv API terms of use; non-commercial scholarly use; bulk harvesting restricted |
| OpenAlex | CC0 / ODC-BY attribution for data |
| OAI-PMH repositories | Per-repository terms; public scholarly metadata; polite harvesting expected |

Attribution for data used in benchmarks and exam records is kept with those
records under `results/`.

## Contact

Compliance questions: https://github.com/Alfredai2025/yellow-phoenix-release/issues

---

*Project practice statement maintained by the maintainers — not legal advice.
For product use, consult counsel for your jurisdictions.*
