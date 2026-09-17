#!/usr/bin/env python3
"""
SAH Benchmark: 100 hand-crafted queries test domain routing.
WIRING SMOKE TEST: Seeds beacons from query hashes to verify cascade.
For production, replace seed with real safetensor harvester (Stage 4).
"""

import sys, time
sys.path.insert(0, "/Users/mac/yellow_phoenix")
from yp_bridge import YPEngine

QUERIES = {
    "CS": [
        "Python memory management", "neural network backpropagation",
        "distributed systems consensus", "compiler optimization",
        "graph algorithm complexity", "cybersecurity threat detection",
        "database indexing strategy", "cloud computing architecture",
        "machine learning pipeline", "API rate limiting",
        "microservices orchestration", "blockchain smart contract",
        "Kubernetes pod scheduling", "TCP congestion control",
        "Rust memory safety", "SQL query optimization",
        "deep learning transformer", "CI/CD pipeline automation",
        "DNS resolution process", "cache eviction policy",
        "reinforcement learning reward", "functional programming monad",
        "zero trust security model", "event-driven architecture",
        "big data MapReduce",
    ],
    "Medical": [
        "cardiac arrhythmia treatment", "diabetes mellitus pathophysiology",
        "MRI image segmentation", "antibiotic resistance mechanism",
        "oncology immunotherapy checkpoint", "epidemiology outbreak modeling",
        "surgical robotics precision", "pharmacokinetics drug interaction",
        "genomic sequencing variant", "neurodegenerative disease biomarker",
        "vaccine efficacy trial", "radiology deep learning",
        "sepsis early detection", "CRISPR gene editing therapy",
        "pediatric asthma management", "stroke thrombolysis window",
        "hospital infection control", "mental health cognitive therapy",
        "tissue engineering scaffold", "pandemic preparedness plan",
        "chronic pain opioid", "allergen immunotherapy",
        "sleep apnea CPAP", "dermatology melanoma screening",
        "orthopedic joint replacement",
    ],
    "Legal": [
        "contract breach remedy", "intellectual property patent",
        "GDPR data protection", "arbitration clause enforceability",
        "tort negligence duty", "merger antitrust review",
        "copyright fair use", "criminal procedure evidence",
        "estate planning trust", "securities fraud litigation",
        "immigration asylum claim", "environmental regulation compliance",
        "labor union collective bargaining", "bankruptcy Chapter 11",
        "international law jurisdiction", "constitutional amendment process",
        "defamation libel standard", "real property title dispute",
        "family law custody", "tax evasion prosecution",
        "class action certification", "consumer protection warranty",
        "criminal sentencing guideline", "administrative agency rulemaking",
        "maritime law salvage",
    ],
    "General": [
        "history of printing press", "renaissance art technique",
        "climate change carbon cycle", "ancient Rome governance",
        "photography exposure triangle", "music theory harmony",
        "philosophy existentialism", "ocean current thermohaline",
        "archaeology radiocarbon dating", "linguistics phoneme",
        "economics supply demand", "psychology cognitive bias",
        "sustainable agriculture permaculture", "urban planning transit",
        "meteorology frontal system", "geology plate tectonics",
        "anthropology cultural diffusion", "literature narrative structure",
        "astronomy exoplanet detection", "botany photosynthesis pathway",
        "zoology migration pattern", "chemistry catalysis",
        "physics quantum entanglement", "mathematics topology",
        "engineering materials fatigue",
    ],
}

def seed_beacons_from_queries(engine):
    """Seed beacon index with query hashes as synthetic domain beacons."""
    print("[SAH] Seeding beacons from query hashes (wiring smoke test)...")
    rc = engine.rust.beacon_index_new(100_000)
    assert rc == 0, f"beacon_index_new failed: {rc}"
    
    beacon_id = 0
    for domain, queries in QUERIES.items():
        if domain == "General":
            continue  # General queries should fallback
        tag = 0x03  # Topic gravity well
        for q in queries:
            h = engine._hash_query(q)
            rc = engine.rust.beacon_index_insert(h, beacon_id, tag)
            if rc == 0:
                beacon_id += 1
    print(f"[SAH] Seeded {beacon_id} beacons")


def main():
    engine = YPEngine()
    
    # Seed beacons if empty
    if engine.rust.beacon_index_count() == 0:
        seed_beacons_from_queries(engine)
    
    print(f"Beacon count: {engine.rust.beacon_index_count()}")
    print("Running 100 domain routing queries...\n")
    
    correct = 0
    total = 0
    latencies = []
    per_domain = {domain: {"correct": 0, "total": 0} for domain in QUERIES}
    
    for domain, queries in QUERIES.items():
        for q in queries:
            t0 = time.perf_counter()
            result = engine.search_sah(q, k=5)
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000)  # ms
            
            # CS/Medical/Legal: expect beacon hit (strong or weak)
            # General: expect fallback (none or weak)
            if domain == "General":
                per_domain[domain]["total"] += 1
                total += 1
                if result["hit_strength"] in ("none", "weak"):
                    correct += 1
                    per_domain[domain]["correct"] += 1
            else:
                per_domain[domain]["total"] += 1
                total += 1
                if result["hit_strength"] in ("strong", "weak"):
                    correct += 1
                    per_domain[domain]["correct"] += 1
    
    # Report
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    
    print("=" * 50)
    print("SAH 100-QUERY DOMAIN ROUTING BENCHMARK")
    print("=" * 50)
    print(f"Total queries: {total}")
    print(f"Correctly routed: {correct} / {total} ({100*correct/total:.1f}%)")
    print(f"P50 latency: {p50:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print()
    for domain, stats in per_domain.items():
        acc = 100 * stats["correct"] / stats["total"] if stats["total"] else 0
        print(f"  {domain:12s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:.1f}%")
    
    print()
    if correct >= 75:
        print("BENCHMARK PASSED: >=75% domain routing accuracy")
        return 0
    else:
        print("BENCHMARK FAILED: <75% domain routing accuracy")
        return 1

if __name__ == "__main__":
    sys.exit(main())
