#!/usr/bin/env python3
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


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(text))[:80]


def looks_like_arxiv(arxiv_id):
    """Only attempt arXiv download for real arXiv IDs."""
    if not arxiv_id:
        return False
    aid = arxiv_id.strip()
    # Modern ID: 2212.14787 or 2602.16829v1
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", aid):
        return True
    # Old-style: cs/0011047
    if re.match(r"^[a-z\-]+/\d+$", aid, re.IGNORECASE):
        return True
    # Wrapped with arxiv: ...
    if re.match(r"^arxiv[\s:]+(.+)$", aid, re.IGNORECASE):
        return True
    return False


def http_get(url, timeout=10):
    """GET with short timeout — fails fast, no hanging."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception:
        return None


def extract_text(pdf_path):
    """Try pdftotext (any path), return text or None."""
    txt_path = pdf_path.with_suffix(".txt")
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), str(txt_path)],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            txt_path.unlink(missing_ok=True)
            return text
    except Exception:
        pass
    txt_path.unlink(missing_ok=True)
    return None


def download_and_extract(url, paper_id, prefix):
    data = http_get(url)
    if not data or len(data) < 1024:
        return None
    pdf_path = PDF_DIR / f"{prefix}_{safe_filename(paper_id)}.pdf"
    try:
        pdf_path.write_bytes(data)
        text = extract_text(pdf_path)
        if text and len(text) > 200:
            return text
    finally:
        pdf_path.unlink(missing_ok=True)
    return None


def download_arxiv(arxiv_id, paper_id):
    if not arxiv_id or not arxiv_id.strip():
        return None
    base_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    url = f"https://arxiv.org/pdf/{base_id}.pdf"
    return download_and_extract(url, paper_id, "arxiv")


def download_direct(url, paper_id):
    if not url or not url.strip().endswith(".pdf"):
        return None
    return download_and_extract(url.strip(), paper_id, "direct")


def try_unpaywall(doi, paper_id):
    if not doi or not doi.strip():
        return None
    api_url = f"https://api.unpaywall.org/v2/{doi.strip()}?email=yellow.phoenix@example.com"
    data = http_get(api_url, timeout=8)
    if not data:
        return None
    try:
        info = json.loads(data.decode("utf-8"))
        pdf_url = info.get("best_oa_location", {}).get("url_for_pdf")
        if pdf_url:
            return download_direct(pdf_url, paper_id)
    except Exception:
        pass
    return None


def try_hal(hal_id, paper_id):
    if not hal_id:
        return None
    landing_url = f"https://hal.science/{hal_id}"
    html = http_get(landing_url, timeout=8)
    if not html:
        return None
    text = html.decode("utf-8", errors="ignore")
    pdf_match = re.search(r'href="([^"]+\.pdf)"', text)
    if pdf_match:
        pdf_url = pdf_match.group(1)
        if pdf_url.startswith("/"):
            pdf_url = "https://hal.science" + pdf_url
        return download_and_extract(pdf_url, paper_id, "hal")
    return None


def update_db(paper_id, text):
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
        ORDER BY
            CASE source
                WHEN 'arxiv' THEN 0
                WHEN 'bulk' THEN 1
                WHEN 'phoenix_self' THEN 2
                WHEN 'core' THEN 3
                ELSE 4
            END,
            id
    """)
    missing = c.fetchall()
    conn.close()

    print(f"Total missing: {len(missing)}")

    stats = {"arxiv": 0, "direct": 0, "unpaywall": 0, "hal": 0, "failed": 0}

    for i, (paper_id, arxiv_id, url, doi, source, title) in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] ID {paper_id}: {title[:50] if title else ''}...")
        text = None

        # 1. arXiv (only for real arXiv IDs, not DOIs stored in that column)
        if looks_like_arxiv(arxiv_id):
            print(f"  arXiv: {arxiv_id}")
            text = download_arxiv(arxiv_id.strip(), paper_id)
            if text:
                stats["arxiv"] += 1
                update_db(paper_id, text)
                print(f"  ✓ arXiv ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # 2. Direct PDF
        if url and url.strip().endswith(".pdf"):
            print(f"  Direct: {url[:50]}...")
            text = download_direct(url.strip(), paper_id)
            if text:
                stats["direct"] += 1
                update_db(paper_id, text)
                print(f"  ✓ Direct ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # 3. Unpaywall
        if doi and doi.strip():
            print(f"  Unpaywall: {doi}")
            text = try_unpaywall(doi.strip(), paper_id)
            if text:
                stats["unpaywall"] += 1
                update_db(paper_id, text)
                print(f"  ✓ Unpaywall ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # 4. HAL
        if source and "hal" in source.lower():
            hal_id = None
            if url:
                m = re.search(r'(hal-\d+)', url)
                if m:
                    hal_id = m.group(1)
            if not hal_id and arxiv_id and "hal" in arxiv_id.lower():
                hal_id = arxiv_id.strip()
            if hal_id:
                print(f"  HAL: {hal_id}")
                text = try_hal(hal_id, paper_id)
                if text:
                    stats["hal"] += 1
                    update_db(paper_id, text)
                    print(f"  ✓ HAL ({len(text)} chars)")
                    time.sleep(SLEEP)
                    continue

        # Failed
        stats["failed"] += 1
        print("  ✗ No PDF")

        if (i + 1) % 50 == 0:
            filled = sum(stats[k] for k in stats if k != "failed")
            print(f"\n--- Progress: {i+1} done, {filled} filled ---")

    print(f"\n{'='*40}")
    print("DONE")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    filled = sum(stats[k] for k in stats if k != "failed")
    print(f"Total filled: {filled}")


if __name__ == "__main__":
    main()
