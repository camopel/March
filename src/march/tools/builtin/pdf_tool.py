"""PDF analysis using PyMuPDF for text extraction and page images for vision."""

from __future__ import annotations

import base64
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.pdf_tool")


@tool(name="pdf", description="Analyze a PDF document. Extracts text and optionally renders pages as images.")
async def pdf_tool(
    pdf: str = "",
    pdfs: list = None,
    prompt: str = "Analyze this document.",
    pages: str = "",
    extract_images: bool = False,
) -> str:
    """Analyze one or more PDF documents.

    Args:
        pdf: Single PDF path or URL.
        pdfs: Multiple PDF paths/URLs (up to 10).
        prompt: Prompt describing what to analyze.
        pages: Page range to process, e.g. '1-5', '1,3,5-7'. Defaults to all.
        extract_images: Whether to render pages as images for vision analysis.
    """
    sources = []
    if pdf:
        sources.append(pdf)
    if pdfs:
        sources.extend(pdfs[:10])

    if not sources:
        return "Error: No PDFs provided."

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "Error: PyMuPDF not installed. Run: pip install pymupdf"

    results = []
    for src in sources:
        try:
            text = await _extract_pdf(src, pages, extract_images)
            results.append(text)
        except Exception as e:
            results.append(f"Error processing {src}: {e}")

    return "\n\n---\n\n".join(results)


def _parse_page_ranges(spec: str, total: int) -> list[int]:
    """Parse page range specification into a list of 0-indexed page numbers."""
    if not spec:
        return list(range(total))

    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            s = max(1, int(start))
            e = min(total, int(end))
            pages.update(range(s - 1, e))
        else:
            p = int(part) - 1
            if 0 <= p < total:
                pages.add(p)
    return sorted(pages)


async def _extract_pdf(source: str, pages_spec: str, extract_images: bool) -> str:
    """Extract text (and optionally images) from a PDF."""
    import fitz

    # Handle URL
    if source.startswith(("http://", "https://")):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(source)
                resp.raise_for_status()
                data = resp.content
        except Exception as e:
            return f"Error downloading PDF: {e}"
        doc = fitz.open(stream=data, filetype="pdf")
    else:
        p = Path(source).expanduser().resolve()
        if not p.is_file():
            return f"Error: File not found: {source}"
        doc = fitz.open(str(p))

    total_pages = len(doc)
    page_nums = _parse_page_ranges(pages_spec, total_pages)

    parts = [f"PDF: {source} ({total_pages} pages)"]

    for pn in page_nums:
        page = doc[pn]
        text = page.get_text().strip()
        parts.append(f"\n--- Page {pn + 1} ---\n{text}")

        if extract_images:
            pix = page.get_pixmap(dpi=150)
            img_data = pix.tobytes("png")
            b64 = base64.b64encode(img_data).decode()
            parts.append(f"[Page {pn + 1} image: {len(img_data)} bytes, data:image/png;base64,{b64[:50]}...]")

    doc.close()
    return "\n".join(parts)
