#!/usr/bin/env python3
"""extract_claims_layer.py — extract and validate the machine-readable claims
layer embedded as a PDF attachment in the deposited Mac paper.

The deposited PDF (Zenodo DOI 10.5281/zenodo.23920218) embeds
claims_layer.json as an /EmbeddedFile stream. This tool finds that stream
(scanning every PDF stream object, inflating FlateDecode when needed), writes
the payload, and checks its SHA-256.

Usage:
  python3 extract_claims_layer.py yellow_phoenix_mac_paper.pdf \
      --out claims_layer.json \
      --sha256 f272acd3d34d043adff96c53ba3de2c4705a6ce1edcaa9b817443944b4710364
"""
import argparse, hashlib, re, sys, zlib

ap = argparse.ArgumentParser()
ap.add_argument("pdf")
ap.add_argument("--out", default="claims_layer.json")
ap.add_argument("--sha256", default="f272acd3d34d043adff96c53ba3de2c4705a6ce1edcaa9b817443944b4710364")
a = ap.parse_args()

data = open(a.pdf, "rb").read()
payloads = []
for m in re.finditer(rb"stream\r?\n", data):
    start = m.end()
    end = data.find(b"endstream", start)
    if end < 0:
        continue
    raw = data[start:end].rstrip(b"\r\n")
    blob = None
    try:
        blob = zlib.decompress(raw)
    except Exception:
        blob = raw
    if b'"claims_layer"' in blob:
        payloads.append(blob)

# keep only payloads that parse as JSON with a top-level claims_layer key
# (an ObjStm that merely mentions the attachment inside its inflated object
#  dump is not the attachment itself)
import json
parsed = []
for p in payloads:
    try:
        d = json.loads(p.decode("utf-8"))
    except Exception:
        continue
    if isinstance(d, dict) and "claims_layer" in d:
        parsed.append(p)
if not parsed:
    print("FAIL: no stream containing a parseable claims_layer JSON object found")
    sys.exit(1)
uniq = {hashlib.sha256(p).hexdigest() for p in parsed}
if len(uniq) != 1:
    print(f"FAIL: {len(parsed)} JSON candidate streams with differing content")
    sys.exit(1)
payload = parsed[0]

# validate it parses as JSON with the expected top-level shape
doc = json.loads(payload.decode("utf-8"))
assert "claims_layer" in doc, "top-level 'claims_layer' key missing"
claims = doc["claims_layer"]["claims"]
ids = [c["id"] for c in claims]
print(f"extracted claims_layer.json: {len(payload)} bytes, {len(claims)} claims: {ids}")

got = hashlib.sha256(payload).hexdigest()
with open(a.out, "wb") as f:
    f.write(payload)
print(f"wrote {a.out}")
print(f"sha256: {got}")
print(f"target: {a.sha256}")
print("PASS" if got == a.sha256 else "FAIL: sha256 mismatch")
sys.exit(0 if got == a.sha256 else 1)
