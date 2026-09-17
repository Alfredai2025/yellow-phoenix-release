#!/usr/bin/env python3
"""
Fill missing full texts v3.
- Broad direct-PDF detection (.pdf, /pdf, /download/, type=printable)
- Treat DOI-like arxiv_id values as DOIs for Unpaywall
- Skip invalid arXiv IDs
- HAL landing-page extraction
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


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(text))[:80]


def looks_like_arxiv(arxiv_id):
    if not arxiv_id:
        return False
    aid = arxiv_id.strip()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", aid):
        return True
    if re.match(r"^[a-z\-]+/\d+$", aid, re.IGNORECASE):
        return True
    if re.match(r"^arxiv[\s:]+(.+)$", aid, re.IGNORECASE):
        return True
    return False


def looks_like_doi(text):
    if not text:
        return False
    return bool(re.match(r"^10\.\d{4,}/", text.strip()))


def http_get(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code not in (404, 403, 410):
            print(f"    HTTP {e.code}")
    except Exception as e:
        print(f"    fetch error: {e}")
    return None


def extract_text(pdf_path):
    txt_path = pdf_path.with_suffix(".txt")
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), str(txt_path)],
            capture_output=True, timeout=60,
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


def looks_like_direct_pdf(url):
    if not url:
        return False
    u = url.strip().lower()
    return (
        u.endswith(".pdf")
        or u.endswith("/pdf")
        or "/download/" in u
        or "type=printable" in u
        or "/article/file" in u
        or "/article/download/" in u
        or "/bitstream/" in u
    )


def download_arxiv(arxiv_id, paper_id):
    if not looks_like_arxiv(arxiv_id):
        return None
    base_id = re.sub(r"v\d+$", "", arxiv_id.strip())
    url = f"https://arxiv.org/pdf/{base_id}.pdf"
    return download_and_extract(url, paper_id, "arxiv")


def download_direct(url, paper_id):
    if not looks_like_direct_pdf(url):
        return None
    return download_and_extract(url.strip(), paper_id, "direct")


def try_unpaywall(doi, paper_id):
    if not looks_like_doi(doi):
        return None
    api_url = f"https://api.unpaywall.org/v2/{doi.strip()}?email=yellow.phoenix@example.com"
    data = http_get(api_url, timeout=10)
    if not data:
        return None
    try:
        info = json.loads(data.decode("utf-8"))
        pdf_url = info.get("best_oa_location", {}).get("url_for_pdf")
        if pdf_url:
            return download_direct(pdf_url, paper_id) or download_and_extract(pdf_url, paper_id, "unpaywall")
    except Exception:
        pass
    return None


def try_hal(hal_id, paper_id):
    if not hal_id:
        return None
    landing_url = f"https://hal.science/{hal_id}"
    html = http_get(landing_url, timeout=15)
    if not html:
        return None
    text = html.decode("utf-8", errors="ignore")
    # Multiple patterns for PDF links
    for pattern in [
        r'href="([^"]+\.pdf)"',
        r'data-url="([^"]+\.pdf)"',
        r'"pdfUrl":\s*"([^"]+)"',
    ]:
        m = re.search(pattern, text)
        if m:
            pdf_url = m.group(1)
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
            CASE
                WHEN url IS NOT NULL AND (
                    url LIKE '%.pdf%' OR url LIKE '%/pdf%' OR url LIKE '%/download/%'
                    OR url LIKE '%type=printable%' OR url LIKE '%/article/file%'
                ) THEN 0
                WHEN source LIKE '%hal%' THEN 1
                WHEN source = 'arxiv' OR source = 'bulk' THEN 2
                ELSE 3
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

        # 1. Direct PDF URL (broad detection)
        if url and looks_like_direct_pdf(url):
            print(f"  Direct: {url[:60]}...")
            text = download_direct(url.strip(), paper_id)
            if text:
                stats["direct"] += 1
                update_db(paper_id, text)
                print(f"  ✓ Direct ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # 2. arXiv
        if looks_like_arxiv(arxiv_id):
            print(f"  arXiv: {arxiv_id}")
            text = download_arxiv(arxiv_id.strip(), paper_id)
            if text:
                stats["arxiv"] += 1
                update_db(paper_id, text)
                print(f"  ✓ arXiv ({len(text)} chars)")
                time.sleep(SLEEP)
                continue

        # 3. Unpaywall (DOI column or DOI-like arxiv_id)
        candidate_doi = doi if looks_like_doi(doi) else (arxiv_id if looks_like_doi(arxiv_id) else None)
        if candidate_doi:
            print(f"  Unpaywall: {candidate_doi}")
            text = try_unpaywall(candidate_doi, paper_id)
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
