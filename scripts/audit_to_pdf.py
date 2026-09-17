#!/usr/bin/env python3
"""Convert the read-only audit report into a PDF on the Desktop."""
from fpdf import FPDF, XPos, YPos
from pathlib import Path

OUTPUT = Path.home() / "Desktop" / "July 27 Audit.pdf"

AUDIT_TITLE = "Yellow Phoenix — Read-Only Audit Report"
AUDIT_DATE = "July 27, 2026"

SECTIONS = [
    (
        "Executive Summary",
        """The project compiles and passes its small test suite, but it carries a large amount of dead/unwired code. Thirteen root-level Rust modules are on disk but not declared in src/lib.rs, including intelligent.rs, dynamic_mesh.rs, and temporal_evolution.rs. Several FFI surfaces have no Python caller, and some Python functions reference Rust symbols that do not exist.""",
    ),
    (
        "Critical Findings",
        """• src/ffi_unified.rs:1269 — yp_spectral_eigenvectors returns dummy eigenvectors. TODO: replace with real eigenvectors from tensor_spectral.rs.
• src/intelligent.rs — Ghost module; hybrid intelligent router is implemented but never compiled.
• src/dynamic_mesh.rs — Ghost module; not declared in src/lib.rs.
• src/temporal_evolution.rs — Ghost module; temporal snapshot layer not wired into the build.""",
    ),
    (
        "Medium Findings",
        """• yp_bridge.py:195 — _is_landmark_paper always returns False; landmark-paper protection disabled.
• yp_bridge.py:3142 — yp_mesh_decay_gravity referenced only in a comment; no Rust implementation.
• src/ffi.rs — yp_query_mesh exported but never called from Python.
• src/ffi_kernel.rs — 18 exported FFI functions have no Python caller (kernel_new, kernel_predict, hologram_new, hologram_query, ...).
• src/trinity/ffi.rs — 11 yp_trinity_* functions have no Python caller.
• src/async_ffi.rs — 4 exported FFI functions have no Python caller (yp_temporal_submit, yp_temporal_poll, yp_temporal_gc, yp_fast_path_eligible).
• src/ffi_unified.rs — 15 exported FFI functions have no Python caller (yp_query_auto_with_trinity_hint, yp_predictor_*, yp_hologram_*, yp_spectral_query, yp_compute_bucket_128/512, yp_free_u32_array, yp_hopfield_filter).
• yp_bridge.py _propose_bucket_from_gravity — partial fallback: uses real co-occurrence but falls back to top_id % 8 when buddy has no bucket.
• Project-wide — yp_mesh_rebuild_edges referenced nowhere.""",
    ),
    (
        "Low Findings",
        """• src/types/graded.rs:28 — panic! on invalid grade; acceptable guard but no graceful error path.
• yp_bridge.py:3337 — actual_bucket is a TODO placeholder in test/metrics formatting.
• scripts/nightly_retrain.py:39 — TODO: call incremental training script (not implemented).
• src/lib.rs.backup — backup artifact inside src/; duplicate panic_handler / rust_eh_personality.
• Ghost modules (also tracked above): cascade_index.rs, cross_reference.rs, cun_disk.rs, disk_paging.rs, disk_query.rs, dual_identity.rs, exact.rs, multi_base_dynamic.rs, pq.rs, proof_of_life.rs, slot.rs.
• Cargo build/test — 18+ compiler warnings (unused imports/variables/statics/fields), no errors.
• Cargo test — only 5 unit tests run; very low test/doc-test coverage.
• 40 .rs files have no module-level doc comment.
• 449 public functions have no preceding doc comment.""",
    ),
    (
        "Key Counts",
        """Rust unimplemented! : 0
Rust TODO             : 1
Rust FIXME            : 0
Rust panic!           : 1
Python TODO           : 3
Python FIXME          : 0
Python NotImplementedError : 0
Ghost Rust root modules : 13
Exported FFI functions with no Python caller : 49
Rust files without module doc : 40
Public functions without doc : 449
Unit tests passing : 5 / 5
Doc tests : 0""",
    ),
    (
        "Bottom Line",
        """The branch is buildable but heavily under-wired. Recommended next steps:
1. Decide whether to wire intelligent.rs, dynamic_mesh.rs, and temporal_evolution.rs into src/lib.rs or delete them.
2. Implement real eigenvectors for yp_spectral_eigenvectors (currently dummy data).
3. Implement or remove yp_mesh_decay_gravity and yp_mesh_rebuild_edges.
4. Expose or prune orphaned FFI surfaces (trinity, kernel, temporal, predictor) to reduce attack surface and compile-time bloat.

No files were modified during this audit.""",
    ),
]


class AuditPDF(FPDF):
    def header(self):
        self.set_font("ArialUnicode", "B", 16)
        self.cell(0, 10, AUDIT_TITLE, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("ArialUnicode", "", 11)
        self.cell(0, 6, AUDIT_DATE, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)
        y = self.get_y()
        self.line(10, y, 200, y)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("ArialUnicode", "I", 9)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section(self, title, body):
        self.set_font("ArialUnicode", "B", 13)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("ArialUnicode", "", 10)
        for line in body.splitlines():
            self.multi_cell(190, 6, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)


def main():
    pdf = AuditPDF()
    pdf.add_font("ArialUnicode", "", "/Library/Fonts/Arial Unicode.ttf")
    pdf.add_font("ArialUnicode", "B", "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    pdf.add_font("ArialUnicode", "I", "/System/Library/Fonts/Supplemental/Arial Italic.ttf")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for title, body in SECTIONS:
        pdf.section(title, body)

    pdf.output(str(OUTPUT))
    print(f"Saved audit PDF to: {OUTPUT}")


if __name__ == "__main__":
    main()
