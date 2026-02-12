#!/usr/bin/env python3
"""Download arXiv papers and extract text for RLM testing.

Usage:
    python scripts/fetch_arxiv.py [--target-chars 4000000] [--output-dir data]

Downloads papers in priority order:
  1. LLM Agents / Tool Use
  2. RAG / Context Windows
  3. General AI

For each paper, tries (in order):
  a) LaTeX source → strip to plain text
  b) PDF → extract text via pymupdf
  c) Abstract only (last resort)

Stops when the cumulative character count reaches --target-chars.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import tarfile
import time
import logging
from pathlib import Path

import arxiv
import pymupdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# arXiv search queries, ordered by priority
# ---------------------------------------------------------------------------
QUERIES = [
    ("agents_tool_use", 'cat:cs.AI AND (ti:"tool use" OR ti:"LLM agent" OR ti:"language model agent" OR ti:"tool calling" OR ti:"function calling" OR ti:"agentic")'),
    ("rag_context", 'cat:cs.CL AND (ti:"retrieval augmented" OR ti:"long context" OR ti:"context window" OR ti:"million token" OR ti:"RAG")'),
    ("ai_general", "cat:cs.AI OR cat:cs.CL OR cat:cs.LG"),
]

MAX_RESULTS_PER_QUERY = 200  # arXiv caps at 300k but we paginate in small batches


# ---------------------------------------------------------------------------
# LaTeX → plain text
# ---------------------------------------------------------------------------

# Patterns to remove from LaTeX source
_LATEX_STRIP = [
    (re.compile(r"\\begin\{(figure|table|equation|align|lstlisting)\*?\}.*?\\end\{\1\*?\}", re.S), ""),
    (re.compile(r"\\(usepackage|documentclass|bibliographystyle|bibliography|label|ref|cite|eqref|url|href)\{[^}]*\}"), ""),
    (re.compile(r"\\(textbf|textit|emph|underline|texttt)\{([^}]*)\}"), r"\2"),
    (re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}"), r"\n\n\2\n"),
    (re.compile(r"\\(title|author|date)\{([^}]*)\}"), r"\n\2\n"),
    (re.compile(r"\\begin\{(abstract)\}(.*?)\\end\{\1\}", re.S), r"\n\2\n"),
    (re.compile(r"\\begin\{(itemize|enumerate)\}"), ""),
    (re.compile(r"\\end\{(itemize|enumerate)\}"), ""),
    (re.compile(r"\\item\s*"), "\n- "),
    (re.compile(r"%.*$", re.M), ""),  # comments
    (re.compile(r"\\[a-zA-Z]+"), ""),  # remaining commands
    (re.compile(r"[{}]"), ""),  # braces
    (re.compile(r"\n{3,}"), "\n\n"),  # excessive newlines
]


def latex_to_text(latex: str) -> str:
    """Best-effort conversion of LaTeX source to plain text."""
    text = latex
    for pattern, replacement in _LATEX_STRIP:
        text = pattern.sub(replacement, text)
    return text.strip()


def extract_tex_from_tar(tar_bytes: bytes) -> str | None:
    """Extract and concatenate .tex files from a tar.gz archive."""
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            tex_contents = []
            for member in tar.getmembers():
                if member.name.endswith(".tex") and member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        tex_contents.append(f.read().decode("utf-8", errors="replace"))
            if tex_contents:
                return "\n\n".join(tex_contents)
    except Exception as e:
        log.debug("Failed to extract tar: %s", e)
    return None


# ---------------------------------------------------------------------------
# PDF → plain text
# ---------------------------------------------------------------------------

def pdf_to_text(pdf_bytes: bytes) -> str | None:
    """Extract text from a PDF using pymupdf."""
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n\n".join(pages)
        return text if len(text) > 500 else None  # skip if almost empty
    except Exception as e:
        log.debug("PDF extraction failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_bytes(url: str) -> bytes | None:
    """Download URL content as bytes."""
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RLM-Prototipo/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        log.debug("Download failed %s: %s", url, e)
    return None


def get_paper_text(result: arxiv.Result) -> tuple[str, str]:
    """Try to get text from a paper. Returns (text, method)."""
    arxiv_id = result.entry_id.split("/")[-1]

    # 1) Try LaTeX source
    source_url = f"https://arxiv.org/e-print/{arxiv_id}"
    log.info("  Trying LaTeX source: %s", source_url)
    tar_bytes = download_bytes(source_url)
    if tar_bytes:
        tex_raw = extract_tex_from_tar(tar_bytes)
        if tex_raw:
            text = latex_to_text(tex_raw)
            if len(text) > 1000:
                return text, "latex"

    # 2) Try PDF
    pdf_url = result.pdf_url
    if pdf_url:
        log.info("  Trying PDF: %s", pdf_url)
        pdf_bytes = download_bytes(pdf_url)
        if pdf_bytes:
            text = pdf_to_text(pdf_bytes)
            if text:
                return text, "pdf"

    # 3) Fallback to abstract
    log.info("  Falling back to abstract")
    return result.summary or "", "abstract"


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a paper title."""
    name = re.sub(r"[^\w\s-]", "", name.lower())
    name = re.sub(r"[\s]+", "_", name.strip())
    return name[:80]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_papers(target_chars: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    paper_count = 0
    seen_ids: set[str] = set()

    # Check existing files in output_dir
    for existing in output_dir.glob("*.txt"):
        total_chars += existing.stat().st_size
        paper_count += 1
    if total_chars > 0:
        log.info("Found %d existing files (%d chars). Continuing from there.", paper_count, total_chars)

    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,
        num_retries=3,
    )

    for query_name, query_str in QUERIES:
        if total_chars >= target_chars:
            break

        log.info("=== Query: %s ===", query_name)
        search = arxiv.Search(
            query=query_str,
            max_results=MAX_RESULTS_PER_QUERY,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        for result in client.results(search):
            if total_chars >= target_chars:
                break

            arxiv_id = result.entry_id.split("/")[-1]
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            title = result.title or arxiv_id
            filename = f"{sanitize_filename(title)}.txt"
            filepath = output_dir / filename

            if filepath.exists():
                log.info("Skipping (exists): %s", filename)
                continue

            log.info("[%d | %d/%d chars] %s", paper_count + 1, total_chars, target_chars, title[:80])

            text, method = get_paper_text(result)
            if not text:
                log.warning("  No text extracted, skipping")
                continue

            # Add metadata header
            header = (
                f"TITLE: {title}\n"
                f"AUTHORS: {', '.join(a.name for a in result.authors[:5])}\n"
                f"DATE: {result.published.strftime('%Y-%m-%d') if result.published else 'unknown'}\n"
                f"URL: {result.entry_id}\n"
                f"EXTRACTION: {method}\n"
                f"{'=' * 60}\n\n"
            )
            full_text = header + text

            filepath.write_text(full_text, encoding="utf-8")
            chars = len(full_text)
            total_chars += chars
            paper_count += 1
            log.info("  Saved %s (%d chars, method=%s)", filename, chars, method)

            # Be nice to arXiv
            time.sleep(1)

    log.info("=" * 60)
    log.info("Done! %d papers, %d total chars (~%d tokens)", paper_count, total_chars, total_chars // 4)


def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv papers for RLM testing")
    parser.add_argument("--target-chars", type=int, default=4_000_000, help="Target character count (default: 4M)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    fetch_papers(target_chars=args.target_chars, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
