#!/usr/bin/env python3
"""Verify abstracts_5m.bin: spot-decode N random ids, compare text against
source DBs (same resolvers as the builder). Also verifies id coverage.
"""
import os, json, sqlite3, random
import numpy as np
import zstandard as zstd

DATA = os.path.expanduser("~/yellow_phoenix/data")
WORK = os.path.join(DATA, "_build_papers5m")
OUT = os.path.join(DATA, "abstracts_5m.bin")
N = int(os.environ.get("N", "1000"))


def read_varint(buf, pos):
    shift = 0
    result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


class Store:
    def __init__(self, path):
        f = open(path, "rb")
        self.f = f
        magic = f.read(4)
        assert magic == b"YPA1", magic
        ver = f.read(1)[0]
        hdr = f.read(8 + 8 + 4 + 4)
        self.n_blocks = int(np.frombuffer(hdr[0:8], "<u8")[0])
        self.count = int(np.frombuffer(hdr[8:16], "<u8")[0])
        dlen = int(np.frombuffer(hdr[16:20], "<u4")[0])
        self.block_target = int(np.frombuffer(hdr[20:24], "<u4")[0])
        dict_bytes = f.read(dlen)
        self.dctx = zstd.ZstdDecompressor(dict_data=zstd.ZstdCompressionDict(dict_bytes))
        # layout: header | dict | blocks... | index(at EOF, 24B/entry)
        f.seek(0, 2)
        index_end = f.tell()
        f.seek(index_end - self.n_blocks * 24)
        self.index = []
        for _ in range(self.n_blocks):
            e = f.read(24)
            self.index.append((int(np.frombuffer(e[0:8], "<u8")[0]),
                               int(np.frombuffer(e[8:16], "<u8")[0]),
                               int(np.frombuffer(e[16:20], "<u4")[0]),
                               int(np.frombuffer(e[20:24], "<u4")[0])))

    def get(self, pid):
        lo, hi = 0, self.n_blocks - 1
        while lo < hi:  # last block with first_id <= pid
            mid = (lo + hi + 1) // 2
            if self.index[mid][0] <= pid:
                lo = mid
            else:
                hi = mid - 1
        first_id, offset, nrec, clen = self.index[lo]
        assert first_id <= pid < first_id + nrec, (pid, first_id, nrec)
        self.f.seek(offset)
        raw = self.dctx.decompress(self.f.read(clen), max_output_size=self.block_target * 4)
        # record 0: [abs_id][len][text]; records 1+: [id_delta][len][text]
        cur, pos = read_varint(raw, 0)
        for k in range(nrec):
            ln, p2 = read_varint(raw, pos)
            if cur == pid:
                return raw[p2:p2 + ln].decode("utf-8", "replace")
            pos = p2 + ln
            if k < nrec - 1:
                delta, pos = read_varint(raw, pos)
                cur += delta
        raise AssertionError(f"id {pid} not in block")


def main():
    ph = sqlite3.connect(os.path.join(DATA, "phoenix_arxiv_1m.db"))
    dr = sqlite3.connect(os.path.join(DATA, "droplet_papers_106298.db"))
    t3 = sqlite3.connect(os.path.join(DATA, "titles_3m.db"))
    pids = [r[0] for r in t3.execute("SELECT pid FROM papers ORDER BY id")]
    t3.close()
    kept = [np.load(os.path.join(WORK, f"kept_src{i}.npy")) for i in range(5)]
    reg = {}
    for si, path in [(1, "new_registry.jsonl"), (2, "new_registry_pubmed.jsonl"), (3, "new_registry_pubmed2.jsonl")]:
        reg[si] = {r["new_idx"]: r["src_rowid"]
                   for r in (json.loads(l) for l in open(os.path.join(DATA, path)))}

    s = Store(OUT)
    print(f"blocks={s.n_blocks:,} papers={s.count:,} target={s.block_target}")

    def expected(pid):
        base = 0
        for si, a in enumerate(kept):
            if base + len(a) > pid:
                r = int(a[pid - base])
                if si == 0:
                    return ph.execute("SELECT abstract FROM papers WHERE id=?",
                                      (pids[r],)).fetchone()[0]
                if si in (1, 2, 3):
                    return ph.execute("SELECT abstract FROM papers WHERE rowid=?",
                                      (reg[si][r],)).fetchone()[0]
                return dr.execute("SELECT abstract FROM papers WHERE rowid=?",
                                  (r + 1,)).fetchone()[0]
            base += len(a)
        raise ValueError(pid)

    random.seed(11)
    sample = random.sample(range(s.count), N)
    ok = empty_both = 0
    mismatches = []
    for pid in sample:
        got = s.get(pid)
        exp = expected(pid) or ""
        if got == exp:
            ok += 1
        elif not got and not exp.strip():
            empty_both += 1
            ok += 1
        else:
            mismatches.append(pid)
    print(f"decode: {ok}/{N} exact match "
          f"({empty_both} both-empty)")
    if mismatches:
        print("MISMATCH ids:", mismatches[:10])
        pid = mismatches[0]
        print("--- got ---\n", s.get(pid)[:200])
        print("--- expected ---\n", (expected(pid) or "")[:200])
    # coverage: first, last, block boundaries
    assert s.get(0) is not None
    assert s.get(s.count - 1) is not None
    for b in (0, s.n_blocks // 2, s.n_blocks - 1):
        s.get(s.index[b][0])
    print("coverage: first/last/block-boundaries OK")
    print("GATE:", "PASS" if ok == N else "FAIL")


if __name__ == "__main__":
    main()
