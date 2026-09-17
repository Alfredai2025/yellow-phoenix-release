#!/usr/bin/env python3
"""Generate Engines.pdf on the Desktop from the engine audit data."""

import json
from fpdf import FPDF
from pathlib import Path

OUT = Path.home() / "Desktop" / "Engines.pdf"
RESULTS_JSON = Path.home() / "yellow_phoenix" / "logs" / "batch1_experimental_test.json"
BATCH2_JSON = Path.home() / "yellow_phoenix" / "logs" / "batch2_module_wiring_test.json"


def sanitize(text: str) -> str:
    return text.replace(chr(8212), "-").replace(chr(8211), "-").replace("\n", " ")


HEADERS = [sanitize(h) for h in ["#", "Engine", "Completion", "Build Time", "Query P50", "Query P99", "Recall", "Status"]]
COL_WIDTHS = [8, 38, 18, 45, 38, 38, 45, 40]
LINE_HEIGHT = 5

ROWS_RAW = [
    ["1", "BinaryHNSW 512-bit (M=16)", "90%", "1M: ~445s / 10M: ~4294s", "1M: 0.16 ms / 10M: 0.09 ms", "1M: 2.38 ms / 10M: 0.29 ms", "R@1 ~99% at 1M", "Production"],
    ["2", "BinaryHNSW 384-bit (M=16)", "85%", "100K: 13.7s", "100K: 55 us", "-", "R@1 60.5% / R@50 98.6%", "Production-ready"],
    ["3", "HybridMesh384", "90%", "1M: 692s / 100K: ~?", "1M: 0.20 ms / 100K: 0.18 ms", "1M: 1.40 ms", "R@1 99.6%", "Production fallback"],
    ["4", "HybridMesh 512-bit", "50%", "-", "-", "-", "-", "Experimental"],
    ["5", "ShardedMesh", "40%", "-", "-", "-", "-", "Experimental"],
    ["6", "Direct Hash Dict", "95%", "Instant", "0.01 ms", "0.01 ms", "100% exact", "Production exact cache"],
    ["7", "ISM v0.3", "45%", "10M: ~10s (claimed)", "10M: 49 us (claimed)", "-", "100% exact", "Batch 2 wired"],
    ["8", "Flat Array v0.4", "45%", "10M: ~6s (claimed)", "10M: 1.6 us (claimed)", "-", "100% exact", "Batch 2 wired"],
    ["9", "MmapShard", "35%", "-", "-", "-", "-", "Not wired"],
    ["10", "HolographicCascade", "30%", "100K: 0.38s", "100K: 4.78 ms", "100K: 10.56 ms", "R@1 95.5% / R@10 11.7%", "Museum"],
    ["11", "PatternHologram", "25%", "100K: 0.66s", "100K: 33.76 ms", "100K: 72.57 ms", "R@1 1.5% / R@10 0.55%", "Museum"],
    ["12", "OpticalHologram", "30%", "100K: 1.09s", "100K: 244 ms (full)", "100K: 272 ms", "R@1 61% / R@10 29.4%", "Museum"],
    ["13", "CymaticsHologram", "30%", "100K: 1.51s", "100K: 16 us", "100K: 23 us", "R@1 100% / R@10 10.2%", "Museum"],
    ["14", "SpectralHologram", "60%", "100K: 1.11s", "100K: 4.05 ms", "100K: 4.16 ms", "R@1 100% / R@10 100%", "Worked, not shipped"],
    ["15", "TensorSpectral", "35%", "-", "-", "-", "-", "Batch 1 wired"],
    ["16", "SpectralCoords", "35%", "-", "-", "-", "-", "Batch 1 wired"],
    ["17", "CheatSheetCascade", "50%", "-", "-", "-", "-", "Built, fallback only"],
    ["18", "AutoShardIndex", "30%", "100K: 12.4s / 1M: 662s", "100K: 0.31 ms / 1M: 33.4 ms", "-", "-", "Import broken"],
    ["19", "hnswlib Python", "25%", "-", "-", "-", "-", "Import optional"],
    ["20", "GeometricRerank", "20%", "-", "-", "-", "-", "Not wired"],
    ["21", "MultiBaseCrystal", "30%", "-", "-", "-", "-", "Experimental"],
    ["22", "DynamicMesh", "35%", "-", "-", "-", "-", "Batch 2 wired"],
    ["27", "Shard", "35%", "-", "-", "-", "-", "Batch 2 wired"],
    ["28", "CascadeIndex", "35%", "-", "-", "-", "-", "Batch 2 wired"],
    ["23", "CollaborativeEngine", "25%", "-", "-", "-", "-", "Experimental"],
    ["24", "LearnedRouter", "25%", "-", "-", "-", "-", "Batch 1 wired"],
    ["25", "IntentClassifier", "25%", "-", "-", "-", "-", "Experimental"],
    ["26", "DriftDetector", "25%", "-", "-", "-", "-", "Batch 1 wired"],
]
ROWS = [[sanitize(cell) for cell in row] for row in ROWS_RAW]

STATUS_COLORS = {
    "Production": (200, 255, 200),
    "Production-ready": (200, 255, 200),
    "Production fallback": (220, 255, 220),
    "Production exact cache": (200, 255, 200),
    "Experimental": (255, 255, 200),
    "Worked, not shipped": (255, 240, 200),
    "Built, fallback only": (255, 255, 200),
    "Audit-only": (255, 255, 200),
    "Compiled, not queried": (255, 255, 200),
    "Batch 1 wired": (220, 255, 220),
    "Batch 2 wired": (200, 255, 220),
    "Source incomplete": (255, 200, 200),
    "Source restored": (255, 200, 200),
    "Not wired": (255, 200, 200),
    "Import broken": (255, 200, 200),
    "Import optional": (255, 200, 200),
    "Museum": (230, 230, 230),
}

SUMMARY_ROWS_RAW = [
    ["Fastest exact lookup", "Direct Hash", "0.01 ms"],
    ["Fastest claimed exact lookup", "Flat Array", "1.6 us (not wired)"],
    ["Fast ANN at 10M", "BinaryHNSW M=16", "0.09 ms P50"],
    ["Best recall ANN", "SpectralHologram", "100% R@10 at 100K, but 4 ms"],
    ["Best recall + speed balance", "HybridMesh384", "99.6% R@1 at 1M, 0.20 ms"],
    ["Fastest failed engine", "CymaticsHologram", "16 us, but only 10.2% R@10"],
]
SUMMARY_ROWS = [[sanitize(cell) for cell in row] for row in SUMMARY_ROWS_RAW]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Yellow Phoenix - Engine Audit", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, "Date: 2026-08-01. Numbers from actual logs where available.", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", new_x="RIGHT", align="C")


pdf = PDF(orientation="L", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Helvetica", "", 7)

# Main table header
pdf.set_fill_color(220, 220, 220)
pdf.set_font("Helvetica", "B", 7)
for i, h in enumerate(HEADERS):
    pdf.cell(COL_WIDTHS[i], 8, h, border=1, align="C", fill=True)
pdf.ln()

# Main table rows
pdf.set_font("Helvetica", "", 6.5)
for row in ROWS:
    status = row[-1]
    fill_color = STATUS_COLORS.get(status, (255, 255, 255))
    pdf.set_fill_color(*fill_color)
    for i, text in enumerate(row):
        pdf.cell(COL_WIDTHS[i], LINE_HEIGHT, text, border=1, align="L", fill=True)
    pdf.ln()

# Summary section
pdf.add_page()
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 10, "What the numbers say", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

pdf.set_font("Helvetica", "B", 8)
pdf.cell(80, 7, "What you need", border=1, fill=True)
pdf.cell(55, 7, "Best engine", border=1, fill=True)
pdf.cell(100, 7, "Numbers", border=1, fill=True)
pdf.ln()

pdf.set_font("Helvetica", "", 7)
for row in SUMMARY_ROWS:
    pdf.cell(80, 6, row[0], border=1)
    pdf.cell(55, 6, row[1], border=1)
    pdf.cell(100, 6, row[2], border=1)
    pdf.ln()

pdf.ln(8)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 8, "Bottom line", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5, "Only 4 engines have validated numbers worth trusting:\n"
                     "1. BinaryHNSW 512 M=16 - speed king at scale\n"
                     "2. HybridMesh384 - recall king\n"
                     "3. BinaryHNSW 384 - fast but lower R@1\n"
                     "4. Direct Hash - exact cache\n\n"
                     "The other 22 are either unbenchmarked, museum-archived, or broken.")

# Batch 1 experimental wiring results
pdf.add_page()
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 10, "Batch 1 Experimental Wiring Test", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)

try:
    batch1 = json.loads(RESULTS_JSON.read_text())
    pass_count = sum(1 for d in batch1.values() if d["status"] == "PASS")
    pdf.cell(0, 6, f"Result: {pass_count}/10 PASS", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    headers = ["#", "Module", "Status", "Time (s)", "Notes"]
    widths = [10, 45, 25, 30, 120]
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("Helvetica", "B", 7)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], 7, h, border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 6.5)
    for idx, (name, data) in enumerate(batch1.items(), 1):
        fill = (200, 255, 200) if data["status"] == "PASS" else (255, 200, 200)
        pdf.set_fill_color(*fill)
        note = sanitize(data.get("result", "")[:90])
        row = [str(idx), name, data["status"], f"{data['time']:.3f}", note]
        for i, cell in enumerate(row):
            pdf.cell(widths[i], 5, cell, border=1, align="L", fill=True)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 4, "Note: these modules now expose a working FFI surface and can be called from Python without crashing. "
                          "The implementations are minimal wiring stubs; full production logic is still future work.")
except Exception as e:
    pdf.cell(0, 6, f"Could not load Batch 1 results: {e}", new_x="LMARGIN", new_y="NEXT")

# Batch 2 wiring results
pdf.add_page()
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 10, "Batch 2 Module Wiring Test", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)

try:
    batch2 = json.loads(BATCH2_JSON.read_text())
    pass_count2 = sum(1 for d in batch2.values() if d["status"] == "PASS")
    pdf.cell(0, 6, f"Result: {pass_count2}/{len(batch2)} PASS", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    headers2 = ["#", "Module", "Status", "Time (s)", "Notes"]
    widths2 = [10, 45, 25, 30, 120]
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("Helvetica", "B", 7)
    for i, h in enumerate(headers2):
        pdf.cell(widths2[i], 7, h, border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 6.5)
    for idx, (name, data) in enumerate(batch2.items(), 1):
        fill = (200, 255, 200) if data["status"] == "PASS" else (255, 200, 200)
        pdf.set_fill_color(*fill)
        note = sanitize(data.get("result", "")[:90])
        row = [str(idx), name, data["status"], f"{data['time']:.3f}", note]
        for i, cell in enumerate(row):
            pdf.cell(widths2[i], 5, cell, border=1, align="L", fill=True)
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 4, "Note: flat, ISM, dynamic_mesh, shard, and cascade_index now expose working FFI surfaces from Python.")
except Exception as e:
    pdf.cell(0, 6, f"Could not load Batch 2 results: {e}", new_x="LMARGIN", new_y="NEXT")

pdf.output(str(OUT))
print(f"Saved: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
