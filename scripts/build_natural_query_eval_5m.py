#!/usr/bin/env python3
"""Natural-text query eval @5.1M (MiniLM -> ITQ pipeline, semantic quality).

Gap #10: every previous recall number used INDEXED hashes as queries
(perturb / exclude-self / random). This eval measures what a user actually
gets: real natural-language queries encoded by the same MiniLM-L6 model used
for the corpus, passed through the same ITQ-512 transform, and searched.

Method (single streaming pass, reuses build_float_gt_5m machinery):
  - Q natural queries -> MiniLM-L6 float embeddings (384-d)
  - GT    : exact top-10 cosine over all 5,107,508 kept embeddings
  - ham   : exact top-10 over ITQ-512 binary hashes (same pass)
  - metrics: recall@k (ham vs float GT, no self to exclude),
             gt5_in_ham10 (operational: of the 5 best float papers,
             how many appear in the shown top-10)

Output: benchmark_results/natural_query_eval_5m_<date>.json
"""
import os, json, time
import numpy as np

DATA = os.path.expanduser("~/yellow_phoenix/data")
WORK = os.path.join(DATA, "_build_papers5m")
SOURCES = [
    ("data/embeddings_3m.npy", "raw"),
    ("data/embeddings_oai439k.npy", "npy"),
    ("data/embeddings_pubmed_new.npy", "npy"),
    ("data/embeddings_pubmed_new2.npy", "npy"),
    ("data/droplet_embeddings_aligned.npy", "npy"),
]
CHUNK = 50_000
TOPK = 10
POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

QUERIES = [
    # --- biomedical (~44, matches 70% PubMed corpus) ---
    "metformin treatment for type 2 diabetes",
    "immunotherapy for non-small cell lung cancer",
    "CRISPR gene editing for sickle cell disease",
    "gut microbiome and depression",
    "mRNA vaccine immune response mechanisms",
    "hypertension management in elderly patients",
    "amyloid beta plaque formation in alzheimer disease",
    "antibiotic resistance in E coli",
    "obesity and cardiovascular disease risk",
    "melatonin for sleep disorders",
    "statin associated muscle side effects",
    "HER2 targeted therapy for breast cancer",
    "zika virus and microcephaly in pregnancy",
    "naloxone treatment of opioid addiction",
    "vitamin D deficiency and bone health",
    "inflammation and atherosclerosis",
    "deep brain stimulation for parkinson disease",
    "SARS-CoV-2 spike protein structure",
    "insulin resistance and fatty liver disease",
    "stem cell therapy for heart failure",
    "mediterranean diet and cognitive decline",
    "sleep apnea and cardiovascular outcomes",
    "childhood asthma and air pollution",
    "remyelination in multiple sclerosis",
    "serotonin and SSRI mechanism of depression",
    "treatment of drug resistant tuberculosis",
    "malaria vaccine efficacy in children",
    "biomarkers of kidney transplant rejection",
    "wound healing in diabetic foot ulcers",
    "nanoparticle drug delivery for cancer",
    "gut brain axis and anxiety",
    "exercise mitochondrial biogenesis and aging",
    "air pollution and dementia risk",
    "influenza vaccine effectiveness in the elderly",
    "cesarean section infant microbiome and immunity",
    "CAR T cell therapy for leukemia",
    "treatment of fibromyalgia chronic pain",
    "SARS-CoV-2 variant immune escape",
    "probiotics for antibiotic associated diarrhea",
    "hormone therapy risks at menopause",
    "epigenetics of cancer progression",
    "oxidative stress and neurodegeneration",
    "antidepressants during pregnancy safety",
    "hepatitis C antiviral treatment cure rates",
    "dialysis quality of life end stage renal disease",
    # --- physics / CS / math (~16) ---
    "quantum entanglement communication protocols",
    "gravitational waves from black hole mergers",
    "high temperature superconductivity in cuprates",
    "dark matter direct detection experiments",
    "topological insulators surface states",
    "convolutional neural networks for image classification",
    "attention mechanisms in transformer language models",
    "reinforcement learning for robot control",
    "graph neural networks for molecular property prediction",
    "prime number distribution and the Riemann hypothesis",
    "black hole information paradox and holography",
    "surface code quantum error correction",
    "exoplanet atmosphere transit spectroscopy",
    "deep learning for medical image segmentation",
    "cosmic microwave background and inflation",
    "finite element methods for fluid simulation",
    # --- cross-domain (~4) ---
    "machine learning for drug discovery",
    "climate change and spread of vector borne diseases",
    "neural networks for protein structure prediction",
    "blockchain for healthcare data privacy",
]


def open_source(rel, kind):
    path = os.path.join(os.path.expanduser("~/yellow_phoenix"), rel)
    if kind == "raw":
        mm = np.memmap(path, dtype=np.float32, mode="r").reshape(-1, 384)
        assert mm.shape[0] == 3_025_752, f"{rel}: {mm.shape}"
        return mm
    arr = np.load(path)
    assert arr.dtype == np.float32 and arr.shape[1] == 384, f"{rel}: {arr.shape}"
    return arr


def main():
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    print("encoding queries with MiniLM-L6-v2...", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(QUERIES, convert_to_numpy=True).astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)  # model already normalizes; idempotent
    Q = len(QUERIES)
    print(f"  {Q} queries in {time.time()-t0:.0f}s", flush=True)

    m = np.load(os.path.join(DATA, "itq_model_512_fixed.npz"))
    mean = m["mean"].astype(np.float32)
    proj = m["proj"].astype(np.float32)
    qbits = np.packbits(((emb - mean) @ proj) > 0, axis=1)  # Q x 64

    kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
    bounds = np.cumsum([0] + [len(a) for a in kept])
    total = int(bounds[-1])
    print(f"corpus: {total:,} kept papers", flush=True)

    best_f = np.full((Q, TOPK), np.inf, dtype=np.float32)
    best_fid = np.full((Q, TOPK), -1, dtype=np.int64)
    best_h = np.full((Q, TOPK), 65535, dtype=np.uint16)
    best_hid = np.full((Q, TOPK), -1, dtype=np.int64)

    for si, (rel, kind) in enumerate(SOURCES):
        arr = open_source(rel, kind)
        rows = kept[si]
        n = len(rows)
        if n == 0:
            print(f"  src{si}: empty, skipped", flush=True)
            continue
        for lo in range(0, n, CHUNK):
            hi = min(lo + CHUNK, n)
            x = np.ascontiguousarray(arr[rows[lo:hi]], dtype=np.float32)
            ids = bounds[si] + lo + np.arange(hi - lo, dtype=np.int64)

            sn = -(emb @ (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).T)
            idx = np.argpartition(sn, TOPK, axis=1)[:, :TOPK]
            cs = np.take_along_axis(sn, idx, 1)
            ms = np.concatenate([best_f, cs], axis=1)
            mi = np.concatenate([best_fid, ids[idx]], axis=1)
            top = np.argpartition(ms, TOPK, axis=1)[:, :TOPK]
            best_f = np.take_along_axis(ms, top, 1)
            best_fid = np.take_along_axis(mi, top, 1)

            bits = np.packbits(((x - mean) @ proj) > 0, axis=1)
            for q in range(Q):
                d = POPCNT[np.bitwise_xor(bits, qbits[q])].sum(axis=1, dtype=np.uint16)
                j = np.argpartition(d, TOPK)[:TOPK]
                msh = np.concatenate([best_h[q], d[j]])
                mih = np.concatenate([best_hid[q], ids[j]])
                top = np.argpartition(msh, TOPK)[:TOPK]
                best_h[q] = msh[top]
                best_hid[q] = mih[top]
        print(f"  src{si}: {n:,} rows | {time.time()-t0:.0f}s", flush=True)
        del arr

    order = np.argsort(best_f, axis=1)
    gt = np.take_along_axis(best_fid, order, 1)
    horder = np.argsort(best_h, axis=1)
    ham = np.take_along_axis(best_hid, horder, 1)

    def recall(k):
        hits = sum(len(set(ham[i, :k].tolist()) & set(gt[i, :k].tolist()))
                   for i in range(Q))
        return hits / (Q * k)

    def gt5_in_ham10():
        hits = sum(len(set(ham[i].tolist()) & set(gt[i, :5].tolist()))
                   for i in range(Q))
        return hits / (Q * 5)

    # qualitative: titles of float GT top-1 vs hamming top-1 per query
    import sqlite3
    db = sqlite3.connect(os.path.join(DATA, "papers_5m.db"))
    def title(i):
        r = db.execute("SELECT substr(title,1,80) FROM meta WHERE id=?", (int(i),)).fetchone()
        return r[0] if r else None
    samples = [{
        "query": QUERIES[i],
        "float_top1": title(gt[i, 0]),
        "ham_top1": title(ham[i, 0]),
        "ham_top1_rank_in_gt": int(np.where(gt[i] == ham[i, 0])[0][0]) if (gt[i] == ham[i, 0]).any() else None,
    } for i in range(Q)]
    db.close()

    result = {
        "method": "natural-language queries (MiniLM-L6) -> ITQ-512 hamming top-k "
                  "vs exact float cosine GT over 5,107,508 papers",
        "queries": Q, "topk": TOPK,
        "total_corpus": total,
        "overall": {f"recall@{k}": round(recall(k), 4) for k in (1, 5, 10)}
                   | {"gt5_in_ham10": round(gt5_in_ham10(), 4)},
        "samples": samples,
        "runtime_s": round(time.time() - t0, 1),
    }
    out = os.path.join(os.path.expanduser("~/yellow_phoenix"),
                       "benchmark_results", f"natural_query_eval_5m_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["overall"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
