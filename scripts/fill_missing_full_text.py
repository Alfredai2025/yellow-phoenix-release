#!/usr/bin/env python3
"""
Fill missing full_text in crystal_mind.db from local .txt and .pdf files.
No downloads, no external APIs, no mesh building.
"""
import glob
import os
import sqlite3
import subprocess
from pathlib import Path

DB_PATH = os.path.expanduser("~/yellow_phoenix/crystal_mind.db")
PDFTOTEXT = "/opt/homebrew/bin/pdftotext"

TXT_DIRS = [
    os.path.expanduser("~/Desktop/droplet_sample_texts"),
    os.path.expanduser("~/yellow_phoenix/BULK_DIR"),
    os.path.expanduser("~/yellow_phoenix/data/knowledge_base/papers"),
]

PDF_DIRS = [
    os.path.expanduser("~/alfred-v2/crystal_data/papers/bulk"),
]


def find_text_file(arxiv_id):
    """Find a .txt file matching arxiv_id in known directories."""
    if not arxiv_id:
        return None
    patterns = [
        f"arxiv_{arxiv_id}.txt",
        f"{arxiv_id}.txt",
        f"*{arxiv_id}*.txt",
    ]
    for txt_dir in TXT_DIRS:
        if not os.path.exists(txt_dir):
            continue
        for pattern in patterns:
            matches = glob.glob(os.path.join(txt_dir, pattern))
            if matches:
                return matches[0]
    return None


def find_pdf_file(arxiv_id):
    """Find a local PDF matching arxiv_id."""
    if not arxiv_id:
        return None
    patterns = [
        f"{arxiv_id}.pdf",
        f"{arxiv_id}*.pdf",
        f"*{arxiv_id}*.pdf",
    ]
    for pdf_dir in PDF_DIRS:
        if not os.path.exists(pdf_dir):
            continue
        for pattern in patterns:
            matches = glob.glob(os.path.join(pdf_dir, pattern))
            if matches:
                return matches[0]
    return None


def extract_text_from_pdf(pdf_path):
    """Extract text from a local PDF using pdftotext."""
    try:
        result = subprocess.run(
            [PDFTOTEXT, "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def update_row(cursor, rowid, text):
    cursor.execute("""
        UPDATE papers
        SET full_text = ?,
            full_text_extracted_at = datetime('now'),
            pipeline_stage = 'extracted'
        WHERE id = ?
    """, (text, rowid))


def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM papers WHERE full_text IS NULL OR LENGTH(full_text) < 100")
    total_missing = cursor.fetchone()[0]
    print(f"Papers missing full_text: {total_missing}")

    cursor.execute("""
        SELECT id, arxiv_id, title, abstract
        FROM papers
        WHERE full_text IS NULL OR LENGTH(full_text) < 100
    """)
    rows = cursor.fetchall()

    filled = 0
    skipped = 0
    not_found = 0

    for rowid, arxiv_id, title, abstract in rows:
        source = None
        text = None

        # 1. Try local .txt files
        txt_path = find_text_file(arxiv_id)
        if txt_path and os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                source = "txt"
            except Exception as e:
                print(f"  Error reading {txt_path}: {e}")

        # 2. Try local PDF files
        if not text:
            pdf_path = find_pdf_file(arxiv_id)
            if pdf_path and os.path.exists(pdf_path):
                text = extract_text_from_pdf(pdf_path)
                if text:
                    source = "pdf"

        if text:
            if len(text) > 500:
                update_row(cursor, rowid, text)
                filled += 1
            else:
                skipped += 1
        else:
            not_found += 1

        if (filled + skipped + not_found) % 100 == 0:
            print(f"  Progress: {filled} filled, {skipped} skipped, {not_found} not found / {total_missing}")
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\nDone:")
    print(f"  Filled:    {filled}")
    print(f"  Skipped:   {skipped}")
    print(f"  Not found: {not_found}")
    print(f"  Still missing: {total_missing - filled}")


if __name__ == "__main__":
    main()
