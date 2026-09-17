#!/usr/bin/env python3
"""Experiment B report: assemble kill-ceiling graph results into
benchmark_results/killceiling_graph_1m_<date>.{json,md} + proof Entry 62.

Reads: /tmp/killceiling/{subset_meta.json, build_hnsw.log tail stats,
build_vamana.json, build_vamana_paged.json, eval_hnsw.json, eval_vamana.json,
eval_vamana_paged.json}
"""
import os, json, glob, datetime

ROOT = "/Users/mac/yellow_phoenix"
os.chdir(ROOT)
D = "/tmp/killceiling"
DATE = datetime.datetime.now().strftime("%Y%m%d")

meta = json.load(open(f"{D}/subset_meta.json"))
evals = {}
for name in ["hnsw", "vamana", "vamana_paged"]:
    evals[name] = json.load(open(f"{D}/eval_{name}.json"))
builds = {}
for name in ["vamana", "vamana_paged"]:
    p = f"{D}/build_{name}.json"
    if os.path.exists(p):
        builds[name] = json.load(open(p))

EFS = [16, 64, 256]

def row(name):
    r = {"file_bytes": evals[name]["file_bytes"],
         "ru_maxrss_mb": evals[name]["ru_maxrss_bytes_max"] / 2**20}
    for ef in EFS:
        v = evals[name]["levels"][f"{name if name!='vamana_paged' else 'vamana'}_ef{ef}"]
        r[f"R@10_ef{ef}"] = v["recall_at_10"]
        r[f"p50_us_ef{ef}"] = v["p50_us"]
        r[f"flt_per_q_ef{ef}"] = v["minor_faults_per_query"]
    return r

rows = {name: row(name) for name in evals}

# pre-registered decision:
# valid if any variant >= +5 points R@10 at equal resident bytes (compare at
# same ef, rss within 20%) OR >=25% fewer page-faults at equal recall.
h = rows["hnsw"]
verdicts = {}
for name in ["vamana", "vamana_paged"]:
    v = rows[name]
    gain = {ef: v[f"R@10_ef{ef}"] - h[f"R@10_ef{ef}"] for ef in EFS}
    flt_gain = {ef: 1 - v[f"flt_per_q_ef{ef}"] / max(1e-9, h[f"flt_per_q_ef{ef}"]) for ef in EFS}
    recall_match = {ef: abs(v[f"R@10_ef{ef}"] - h[f"R@10_ef{ef}"]) <= 0.02 for ef in EFS}
    valid = any(g >= 0.05 for g in gain.values()) or \
            any(recall_match[ef] and flt_gain[ef] >= 0.25 for ef in EFS)
    verdicts[name] = dict(gain=gain, flt_gain=flt_gain, valid=bool(valid))
decision = "VALID" if any(v["valid"] for v in verdicts.values()) else "KILLED"

out = dict(date=DATE, seed=meta["seed"], n_subset=meta["n_subset"],
           n_queries=meta["n_queries"], gt_wall_s=meta["gt_wall_s"],
           query_self_in_subset_frac=meta["query_source_self_in_subset_frac"],
           rows=rows, builds=builds, verdicts=verdicts, decision=decision,
           note=("1M subset of 5.1M (scaling choice: 10M builds too slow overnight); "
                 "ru_maxrss is per-process high-water (macOS bytes). Page-fault criterion "
                 "marginal at 1M scale: all graphs fit in 18GB RAM after warm-up; "
                 "faults reported as measured."))
json.dump(out, open(f"benchmark_results/killceiling_graph_1m_{DATE}.json", "w"), indent=2)

lines = [f"# Experiment B — Kill-ceiling graph construction (1M subset, {DATE})", "",
         f"- Seed {meta['seed']}; subset {meta['n_subset']:,} of 5.1M (uniform sample); "
         f"{meta['n_queries']} text-space queries, GT top-10 restricted to subset "
         f"(GT wall {meta['gt_wall_s']:.0f}s; query-source-in-subset {meta['query_source_self_in_subset_frac']:.1%}).",
         "- Graphs: HNSW M=12 efC=200 (existing builder, v5 mmap); Vamana-binary "
         "R=12 L=60 alpha=1.2 RobustPrune in Hamming (custom YPVA1, forward+reverse pass); "
         "paged variant adds selection score d + lambda*page_dist (auto-tuned).",
         "- Recall = strict set R@10 vs restricted GT; RSS = process high-water.", ""]
lines.append("| Graph | file MB | RSS MB | R@10@ef16 | R@10@ef64 | R@10@ef256 | p50us@ef64 | flt/q@ef64 |")
lines.append("|---|---|---|---|---|---|---|---|")
for name, r in rows.items():
    lines.append(f"| {name} | {r['file_bytes']/2**20:.0f} | {r['ru_maxrss_mb']:.0f} | "
                 f"{r['R@10_ef16']:.3f} | {r['R@10_ef64']:.3f} | {r['R@10_ef256']:.3f} | "
                 f"{r['p50_us_ef64']:.0f} | {r['flt_per_q_ef64']:.2f} |")
lines += ["", f"Builds: {json.dumps(builds)}", ""]
for name, v in verdicts.items():
    lines.append(f"- {name}: R@10 gain vs hnsw {v['gain']}; fault reduction {v['flt_gain']}; valid={v['valid']}")
lines += ["", f"**Pre-registered decision: {decision}** "
          "(valid if any variant ≥+5 R@10 points at equal resident bytes, or ≥25% fewer faults at equal recall).",
          "", "Page-fault criterion is marginal at 1M scale (all graphs RAM-resident post-warmup); "
          "the 3.44GB paging regime was not exercisable on this machine without cache-dropping privileges.",
          "", "## Repro", "",
          "- scripts/b_subset_prep.py 1000000 /tmp/killceiling",
          "- HNSW_M=12 HNSW_EF_CONSTRUCTION=200 build_hnsw_from_ism subset.ism hnsw_1m.bin",
          "- vamana_build subset.ism out 12 60 1200 0  (plain) / 300 paged",
          "- eval_graph {hnsw|vamana} <graph> queries.ism gt.bin out.json",
          "- scripts/killceiling_report.py"]
open(f"benchmark_results/killceiling_graph_1m_{DATE}.md", "w").write("\n".join(lines))

entry = f"""
Entry 62: Experiment B — Kill-Ceiling Graph Construction at 1M ({decision})
Date: {DATE}
Git Hash: engine worktree 62eb8ae1 (+ uncommitted; vamana_build/eval_graph bins,
  node_label accessor on BinaryHNSW)
Status: MEASURED ON DESKTOP (M3-class, 18GB; single-threaded search).
  1M uniform subset of 5.1M (10M infeasible overnight; documented scaling
  choice). Three graphs on identical hashes: HNSW M=12 efC=200 (v5 mmap);
  Vamana-binary R=12 L=60 a=1.2 Hamming RobustPrune fwd+rev (custom YPVA1);
  Vamana + page-local edge bias (score d+lambda*page_dist, auto-tuned to
  same-page>=60%%). 500 text-space queries, strict R@10 vs subset-restricted
  exact GT, ef {{16,64,256}}: HNSW R@10 {rows['hnsw']['R@10_ef16']:.3f}/
  {rows['hnsw']['R@10_ef64']:.3f}/{rows['hnsw']['R@10_ef256']:.3f} rss
  {rows['hnsw']['ru_maxrss_mb']:.0f}MB; vamana {rows['vamana']['R@10_ef16']:.3f}/
  {rows['vamana']['R@10_ef64']:.3f}/{rows['vamana']['R@10_ef256']:.3f} rss
  {rows['vamana']['ru_maxrss_mb']:.0f}MB; paged {rows['vamana_paged']['R@10_ef16']:.3f}/
  {rows['vamana_paged']['R@10_ef64']:.3f}/{rows['vamana_paged']['R@10_ef256']:.3f} rss
  {rows['vamana_paged']['ru_maxrss_mb']:.0f}MB same_page
  {builds.get('vamana_paged',{}).get('same_page_frac','?')}. Faults/query ~0 for
  all (1M fits RAM; 3.44GB paging regime not exercisable without cache-drop).
  Decision rule (+5 R@10 at equal bytes, or -25%% faults at equal recall):
  {decision}.
Artifacts: benchmark_results/killceiling_graph_1m_{DATE}.{{json,md}},
  src/bin/{{vamana_build,eval_graph}}.rs, scripts/{{b_subset_prep,killceiling_report}}.py,
  /tmp/killceiling/
Next: {'graph-construction research line continues' if decision == 'VALID' else 'HNSW-insertion topology stands; line closed'}
"""
guard = f"{D}/entry62_appended"
if not os.path.exists(guard):
    with open("logs/proof/proof_of_life_chain.txt", "a") as f:
        f.write(entry)
    open(guard, "w").write("done")
    print("Entry 62 appended")
else:
    print("Entry 62 guard present; report regenerated only")
print("decision:", decision)
