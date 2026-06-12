"""Resolve TOC page numbers by rendering DOCX to PDF and locating section headings."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import fitz
from docx import Document
from kkp_docx_fill import (
    TOC_ENTRIES,
    _find_paragraph_index,
    _find_toc_target_paragraph,
    _normalize,
)


def _norm(text: str) -> str:
    return " ".join(_normalize(text).upper().split())


def _find_soffice() -> Path | None:
    candidates = (
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        Path("/usr/local/bin/soffice"),
        Path("/opt/homebrew/bin/soffice"),
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _convert_docx_to_pdf_via_libreoffice(docx_path: Path, pdf_path: Path) -> bool:
    soffice = _find_soffice()
    if soffice is None:
        return False
    out_dir = pdf_path.parent
    try:
        result = subprocess.run(
            [
                str(soffice),
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(out_dir),
                str(docx_path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    generated = out_dir / f"{docx_path.stem}.pdf"
    if generated != pdf_path and generated.is_file():
        generated.replace(pdf_path)
    return pdf_path.is_file()


def _convert_docx_to_pdf_via_pages(docx_path: Path, pdf_path: Path) -> bool:
    pages_app = Path("/Applications/Pages.app")
    if not pages_app.is_dir():
        return False
    docx_posix = str(docx_path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    pdf_posix = str(pdf_path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    script = f"""
tell application "Pages"
    set docPath to POSIX file "{docx_posix}"
    set pdfPath to POSIX file "{pdf_posix}"
    open docPath
    delay 5
    set theDoc to front document
    export theDoc to pdfPath as PDF
    delay 1
    close theDoc saving no
end tell
"""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and pdf_path.is_file()


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path | None = None) -> Path:
    docx_path = docx_path.resolve()
    pdf_path = docx_path.with_suffix(".pdf") if pdf_path is None else pdf_path.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if _convert_docx_to_pdf_via_libreoffice(docx_path, pdf_path):
        return pdf_path
    if _convert_docx_to_pdf_via_pages(docx_path, pdf_path):
        return pdf_path
    raise RuntimeError(
        "Cannot convert DOCX to PDF for TOC page numbers. Install LibreOffice or use macOS Pages."
    )


def _find_body_start_page(pdf: fitz.Document) -> int:
    for index in range(pdf.page_count):
        text = pdf[index].get_text()
        if "ПЕРЕЛІК СКОРОЧЕНЬ" in text and "AI —" in text:
            return index
    raise RuntimeError("Could not locate body start (ПЕРЕЛІК СКОРОЧЕНЬ) in rendered PDF.")


def _find_heading_page(
    pdf: fitz.Document,
    heading: str,
    *,
    start_page: int,
    prefix_fallback: int = 20,
) -> int | None:
    needle = _norm(heading)
    for index in range(start_page, pdf.page_count):
        if needle in _norm(pdf[index].get_text()):
            return index + 1
    token = needle[:prefix_fallback]
    if len(token) < 8:
        return None
    for index in range(start_page, pdf.page_count):
        if token in _norm(pdf[index].get_text()):
            return index + 1
    return None


def resolve_toc_pages(docx_path: Path, *, pdf_path: Path | None = None) -> list[str]:
    """Return 1-based page numbers for each TOC entry in document order."""
    docx_path = docx_path.resolve()
    doc = Document(str(docx_path))
    perelik = _find_paragraph_index(doc, "ПЕРЕЛІК СКОРОЧЕНЬ")
    if perelik is None:
        raise RuntimeError("ПЕРЕЛІК СКОРОЧЕНЬ heading not found in document.")

    with tempfile.TemporaryDirectory(prefix="kkp_toc_") as tmp:
        tmp_pdf = Path(tmp) / "layout.pdf"
        convert_docx_to_pdf(docx_path, tmp_pdf)
        if pdf_path is not None:
            pdf_path = pdf_path.resolve()
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(tmp_pdf.read_bytes())
        pdf = fitz.open(str(tmp_pdf))

    body_start: int | None = None
    for attempt in range(2):
        try:
            body_start = _find_body_start_page(pdf)
            break
        except RuntimeError:
            if attempt == 0:
                pdf.close()
                with tempfile.TemporaryDirectory(prefix="kkp_toc_retry_") as tmp:
                    tmp_pdf = Path(tmp) / "layout.pdf"
                    convert_docx_to_pdf(docx_path, tmp_pdf)
                    pdf = fitz.open(str(tmp_pdf))
            else:
                raise
    assert body_start is not None
    search_from_para = perelik
    page_cursor = body_start
    last_page = body_start + 1
    pages: list[str] = []

    for entry in TOC_ENTRIES:
        target = _find_toc_target_paragraph(doc, entry, start=search_from_para)
        if target is None:
            pages.append(str(last_page))
            continue
        heading = doc.paragraphs[target].text.strip()
        found = _find_heading_page(pdf, heading, start_page=page_cursor)
        if found is None:
            pages.append(str(last_page))
        else:
            page_num = max(last_page, found)
            pages.append(str(page_num))
            last_page = page_num
            page_cursor = page_num - 1
        search_from_para = target + 1

    pdf.close()
    return pages


def resolve_document_page_count(docx_path: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="kkp_pages_") as tmp:
        pdf_path = Path(tmp) / "layout.pdf"
        convert_docx_to_pdf(docx_path, pdf_path)
        pdf = fitz.open(str(pdf_path))
        count = pdf.page_count
        pdf.close()
    return count
