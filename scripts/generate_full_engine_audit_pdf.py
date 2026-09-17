#!/usr/bin/env python3
"""Generate the full Yellow Phoenix Engine Audit PDF on the Desktop."""

import json
from pathlib import Path
from fpdf import FPDF

OUT = Path.home() / "Desktop" / "Yellow_Phoenix_Engine_Audit_2026-08-01.pdf"
BATCH1_JSON = Path.home() / "yellow_phoenix" / "logs" / "batch1_experimental_test.json"
BATCH2_JSON = Path.home() / "yellow_phoenix" / "logs" / "batch2_module_wiring_test.json"


def sanitize(text: str) -> str:
    return text.replace(chr(8212), "-").replace(chr(8211), "-").replace("\n", " ")


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

ENGINE_HEADERS = ["#", "Engine", "Completion", "Build Time", "Query P50", "Query P99", "Recall", "Status"]
ENGINE_WIDTHS = [8, 38, 18, 45, 38, 38, 45, 40]

ENGINE_ROWS_RAW = [
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
    ["23", "CollaborativeEngine", "25%", "-", "-", "-", "-", "Experimental"],
    ["24", "LearnedRouter", "25%", "-", "-", "-", "-", "Batch 1 wired"],
    ["25", "IntentClassifier", "25%", "-", "-", "-", "-", "Experimental"],
    ["26", "DriftDetector", "25%", "-", "-", "-", "-", "Batch 1 wired"],
    ["27", "Shard", "35%", "-", "-", "-", "-", "Batch 2 wired"],
    ["28", "CascadeIndex", "35%", "-", "-", "-", "-", "Batch 2 wired"],
]
ENGINE_ROWS = [[sanitize(cell) for cell in row] for row in ENGINE_ROWS_RAW]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Yellow Phoenix - Engine Audit", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, "Date: 2026-08-01. Build: target/release/libpams.dylib (115 exported functions)",
                  new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", new_x="RIGHT", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, sanitize(text))
        self.ln(2)


def add_engine_table(pdf):
    pdf.section_title("Engine Comparison")
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("Helvetica", "B", 7)
    for i, h in enumerate(ENGINE_HEADERS):
        pdf.cell(ENGINE_WIDTHS[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 6.5)
    for row in ENGINE_ROWS:
        status = row[-1]
        fill_color = STATUS_COLORS.get(status, (255, 255, 255))
        pdf.set_fill_color(*fill_color)
        for i, text in enumerate(row):
            pdf.cell(ENGINE_WIDTHS[i], 5, text, border=1, align="L", fill=True)
        pdf.ln()
    pdf.ln(3)


def add_batch_table(pdf, title, json_path):
    pdf.add_page()
    pdf.section_title(title)
    try:
        data = json.loads(json_path.read_text())
        pass_count = sum(1 for d in data.values() if d["status"] == "PASS")
        pdf.body_text(f"Result: {pass_count}/{len(data)} PASS")

        headers = ["#", "Module", "Status", "Time (s)", "Notes"]
        widths = [10, 45, 25, 30, 120]
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Helvetica", "B", 7)
        for i, h in enumerate(headers):
            pdf.cell(widths[i], 7, h, border=1, align="C", fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", "", 6.5)
        for idx, (name, d) in enumerate(data.items(), 1):
            fill = (200, 255, 200) if d["status"] == "PASS" else (255, 200, 200)
            pdf.set_fill_color(*fill)
            note = sanitize(d.get("result", "")[:90])
            row = [str(idx), name, d["status"], f"{d['time']:.3f}", note]
            for i, cell in enumerate(row):
                pdf.cell(widths[i], 5, cell, border=1, align="L", fill=True)
            pdf.ln()
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(0, 4, sanitize(
            "Note: these modules expose a working FFI surface and can be called from Python without crashing. "
            "The implementations are minimal wiring stubs; full production logic is still future work."
        ))
    except Exception as e:
        pdf.body_text(f"Could not load results: {e}")


def main():
    pdf = PDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1: summary + engine table
    pdf.add_page()
    pdf.section_title("Executive Summary")
    pdf.body_text(
        "The phoenix-bench-windows-v2 branch builds cleanly with cargo build --release. "
        "The dylib exports 115 functions. Recent wiring work focused on two batches of experimental modules: "
        "Batch 1 wired 8 autopoiesis/experimental modules (10/10 PASS), and Batch 2 wired 5 additional engines "
        "(flat, ISM, dynamic_mesh, shard, cascade_index) with 5/5 PASS. "
        "Production-grade implementations remain future work; these are minimal stubs that prove the FFI boundary."
    )
    add_engine_table(pdf)

    # Page 2: wiring audit
    pdf.add_page()
    pdf.section_title("Python Wiring Audit")
    pdf.body_text(
        "New Rust FFI stubs added: src/ffi_flat.rs, src/ffi_ism.rs, src/ffi_dynamic_mesh.rs, "
        "src/ffi_shard.rs, src/ffi_cascade_index.rs. All are declared in src/lib.rs. "
        "yp_bridge.py now exposes RustBridge methods for each module. "
        "Stateful modules (dynamic_mesh, shard, cascade_index) use a simple in-process handle registry so that "
        "Python receives compact c_int handles instead of raw pointers."
    )
    pdf.section_title("Key Counts")
    pdf.body_text(
        "Exported FFI functions: 115\n"
        "Batch 1 PASS: 10/10\n"
        "Batch 2 PASS: 5/5\n"
        "Production engines with validated benchmarks: 4 (BinaryHNSW 512, HybridMesh384, BinaryHNSW 384, Direct Hash)\n"
        "Engines wired via FFI stubs: 13 (Batch 1 + Batch 2 + cheat_sheet_cascade)\n"
        "Engines still not wired or museum-archived: remainder"
    )
    pdf.section_title("Bottom Line")
    pdf.body_text(
        "The FFI boundary is now wider and more stable. Next steps: replace stubs with real logic for "
        "flat, ISM, dynamic_mesh, shard, and cascade_index; resolve remaining orphaned FFI surfaces; "
        "and add end-to-end benchmarks for the newly wired engines."
    )

    add_batch_table(pdf, "Batch 1 Experimental Wiring Test", BATCH1_JSON)
    add_batch_table(pdf, "Batch 2 Module Wiring Test", BATCH2_JSON)

    pdf.output(str(OUT))
    print(f"Saved: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
