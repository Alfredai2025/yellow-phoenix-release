#!/usr/bin/env python3
"""
Fill missing full texts from existing sources only.
NO new APIs. NO Semantic Scholar. Uses stdlib urllib instead of requests.
"""

import json
import re
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DB_PATH = Path.home() / "yellow_phoenix" / "crystal_mind.db"
PDF_DIR = Path.home() / "yellow_phoenix" / "pdfs_missing"
PDF_DIR.mkdir(exist_ok=True)
SLEEP = 2.0

USER_AGENT = "YellowPhoenix/1.0 (research index; polite robot)"


def get_db():
    return sqlite3.connect(str(DB_PATH), timeout=30.0)


def safe_filename_part(text):
    """Make a DB id safe for use in a filename."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(text))[:80]


def http_get(url, timeout=30):
    """Simple urllib GET returning bytes or None."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        # Don't spam log for 404s
        if e.code not in (404, 403, 410):
            print(f"    HTTP {e.code} for {url[:80]}")
    except Exception as e:
        print(f"    fetch error: {e}")
    return None


def extract_text_from_pdf(pdf_path):
    """Use pdftotext if available, else return None."""
    txt_path = pdf_path.with_suffix(".txt")
    try:
        result = subprocess.run(
            ["/opt/homebrew/bin/pdftotext", str(pdf_path), str(txt_path)],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0 and txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            txt_path.unlink(missing_ok=True)
            return text
    except FileNotFoundError:
        print("    pdftotext not found")
    except Exception as e:
        print(f"    pdftotext error: {e}")
    txt_path.unlink(missing_ok=True)
    return None


def download_and_extract(url, paper_id, prefix):
    """Download PDF from url, extract text, clean up PDF."""
    data = http_get(url)
    if not data or len(data) < 1024:
        return None

    pdf_path = PDF_DIR / f"{prefix}_{safe_filename_part(paper_id)}.pdf"
    try:
        pdf_path.write_bytes(data)
        text = extract_text_from_pdf(pdf_path)
        if text and len(text) > 200:
            return text
    finally:
        pdf_path.unlink(missing_ok=True)
    return None


def download_arxiv(arxiv_id, paper_id):
    """Download arXiv PDF directly."""
    if not arxiv_id or not arxiv_id.strip():
        return None
    arxiv_id = arxiv_id.strip()
    # Strip version suffix for URL
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    url = f"https://arxiv.org/pdf/{base_id}.pdf"
    return download_and_extract(url, paper_id, "arxiv")


def download_direct_pdf(url, paper_id):
    """Download from direct .pdf URL."""
    if not url or not url.strip().endswith(".pdf"):
        return None
    return download_and_extract(url.strip(), paper_id, "direct")


def try_unpaywall(doi, paper_id):
    """Try Unpaywall API (free, no key)."""
    if not doi or not doi.strip():
        return None
    doi = doi.strip()
    api_url = f"https://api.unpaywall.org/v2/{doi}?email=yellow.phoenix@example.com"
    try:
        data = http_get(api_url, timeout=15)
        if not data:
            return None
        info = json.loads(data.decode("utf-8"))
        pdf_url = info.get("best_oa_location", {}).get("url_for_pdf")
        if pdf_url:
            return download_direct_pdf(pdf_url, paper_id)
    except Exception as e:
        print(f"    Unpaywall error: {e}")
    return None


def try_hal(hal_id, paper_id):
    """Try HAL landing page to find PDF link."""
    if not hal_id:
        return None
    landing_url = f"https://hal.science/{hal_id}"
    try:
        html = http_get(landing_url, timeout=15)
        if not html:
            return None
        text = html.decode("utf-8", errors="ignore")
        # Look for PDF link
        pdf_match = re.search(r'href="([^"]+\.pdf)"', text)
        if pdf_match:
            pdf_url = pdf_match.group(1)
            if pdf_url.startswith("/"):
                pdf_url = "https://hal.science" + pdf_url
            return download_and_extract(pdf_url, paper_id, "hal")
    except Exception as e:
        print(f"    HAL error: {e}")
    return None


def update_db(paper_id, text):
    """Update paper with extracted text."""
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute(
            """UPDATE papers
               SET full_text = ?,
                   full_text_extracted_at = datetime('now'),
                   pipeline_stage = 'extracted'
               WHERE id = ?""",
            (text, paper_id),
        )
        conn.commit()
    finally:
        conn.close()


def main():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, arxiv_id, url, doi, source, title
        FROM papers
        WHERE full_text IS NULL OR LENGTH(full_text) < 100
    """)
    missing = c.fetchall()
    conn.close()

    print(f"Total missing: {len(missing)}")

    stats = {"arxiv": 0, "direct_pdf": 0, "unpaywall": 0, "hal": 0, "failed": 0}

    for i, (paper_id, arxiv_id, url, doi, source, title) in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] ID {paper_id}: {title[:60] if title else ''}...")
        text = None

        # Priority 1: arXiv
        if arxiv_id and arxiv_id.strip():
            print(f"  Trying arXiv: {arxiv_id}")
            text = download_arxiv(arxiv_id.strip(), paper_id)
            if text:
                stats["arxiv"] += 1
                update_db(paper_id, text)
                print(f"  ✓ arXiv success ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # Priority 2: Direct PDF URL
        if url and url.strip().endswith(".pdf"):
            print(f"  Trying direct PDF: {url[:60]}...")
            text = download_direct_pdf(url.strip(), paper_id)
            if text:
                stats["direct_pdf"] += 1
                update_db(paper_id, text)
                print(f"  ✓ Direct PDF success ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # Priority 3: Unpaywall (DOI)
        if doi and doi.strip():
            print(f"  Trying Unpaywall: {doi}")
            text = try_unpaywall(doi.strip(), paper_id)
            if text:
                stats["unpaywall"] += 1
                update_db(paper_id, text)
                print(f"  ✓ Unpaywall success ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # Priority 4: HAL
        if source and "hal" in source.lower():
            hal_id = None
            if url:
                hal_match = re.search(r'(hal-\d+)', url)
                if hal_match:
                    hal_id = hal_match.group(1)
            if not hal_id and arxiv_id and "hal" in arxiv_id.lower():
                hal_id = arxiv_id.strip()
            if hal_id:
                print(f"  Trying HAL: {hal_id}")
                text = try_hal(hal_id, paper_id)
                if text:
                    stats["hal"] += 1
                    update_db(paper_id, text)
                    print(f"  ✓ HAL success ({len(text)} chars)")
                    time.sleep(SLEEP)
                    continue

        # Failed
        stats["failed"] += 1
        print("  ✗ No PDF found")

        if (i + 1) % 50 == 0:
            print(f"\n--- Progress after {i+1} ---")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    print(f"\n{'='*50}")
    print("FINAL STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    filled = stats["arxiv"] + stats["direct_pdf"] + stats["unpaywall"] + stats["hal"]
    print(f"Total filled: {filled}")
    print(f"Total failed: {stats['failed']}")


if __name__ == "__main__":
    main()
