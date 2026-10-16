"""Document preparation: format detection, PPTX/DOCX -> PDF conversion,
page rendering, text extraction, and chunking with source details."""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_EXT = {".pdf"}
OFFICE_EXT = {".pptx", ".ppt", ".odp", ".key", ".docx", ".doc", ".odt", ".rtf"}
PRESENTATION_EXT = {".pptx", ".ppt", ".odp", ".key"}
TEXT_EXT = {".txt", ".md", ".markdown"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_EXT = PDF_EXT | OFFICE_EXT | TEXT_EXT | IMAGE_EXT

SUPPORTED_FORMATS_HELP = (
    "**Accepted files:** PDF (.pdf); PowerPoint (.pptx, .ppt) and other slide decks (.odp) - converted to PDF "
    "with LibreOffice so each slide keeps its original image; Word (.docx, .doc, .odt, .rtf) - also via LibreOffice; "
    "plain text / Markdown (.txt, .md); images (.png, .jpg, .jpeg, .webp - text read with the OCR service). "
    "Scanned PDF pages with no text layer are OCR'd automatically."
)


class UnsupportedFormat(ValueError):
    pass


class ConversionError(RuntimeError):
    pass


@dataclass
class Page:
    number: int                 # 1-based page / slide / section number
    label: str                  # "Slide 33", "Page 2", "Section 3: Grading"
    text: str
    image_path: str | None = None
    ocr: bool = False


@dataclass
class PreparedDoc:
    kind: str                   # slides | pdf | text | image
    pages: list[Page] = field(default_factory=list)
    converted_pdf: str | None = None


# --------------------------------------------------------------------------- LibreOffice

def find_soffice(explicit: str = "") -> str | None:
    candidates = [explicit, os.getenv("SOFFICE_PATH", ""), shutil.which("soffice") or "", shutil.which("libreoffice") or ""]
    if platform.system() == "Darwin":
        candidates.append("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    if platform.system() == "Windows":
        candidates += [r"C:\Program Files\LibreOffice\program\soffice.exe",
                       r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def convert_to_pdf(src: Path, out_dir: Path, soffice: str = "", timeout: int = 600) -> Path:
    """Convert an Office document to PDF with headless LibreOffice."""
    exe = find_soffice(soffice)
    if not exe:
        raise ConversionError(
            "LibreOffice is required to read PowerPoint/Word files but was not found. Install it "
            "(https://www.libreoffice.org/download/) or set SOFFICE_PATH, or export the file to PDF and upload the PDF."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        cmd = [exe, f"-env:UserInstallation=file://{Path(profile).as_posix()}", "--headless",
               "--convert-to", "pdf", "--outdir", str(out_dir), str(src)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ConversionError(f"LibreOffice timed out converting {src.name}") from None
        except subprocess.CalledProcessError as exc:
            raise ConversionError(f"LibreOffice failed to convert {src.name}: {exc.stderr.decode(errors='ignore')[:200]}") from None
    pdf = out_dir / (src.stem + ".pdf")
    if not pdf.exists():
        raise ConversionError(f"LibreOffice did not produce a PDF for {src.name}")
    return pdf


def pptx_notes(src: Path) -> list[str] | None:
    """Speaker notes per slide (None if python-pptx is unavailable or file unreadable)."""
    try:
        from pptx import Presentation
        prs = Presentation(str(src))
    except Exception:  # noqa: BLE001
        return None
    notes = []
    for slide in prs.slides:
        txt = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            txt = slide.notes_slide.notes_text_frame.text.strip()
        notes.append(txt)
    return notes


# --------------------------------------------------------------------------- rendering

def _clean(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def render_pdf(pdf_path: Path, pages_dir: Path, zoom: float = 2.0, presentation: bool = False) -> tuple[str, list[Page]]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    if doc.page_count == 0:
        raise ConversionError("PDF has no pages")
    first = doc[0].rect
    is_slides = presentation or (first.width / max(first.height, 1) >= 1.3)
    noun = "Slide" if is_slides else "Page"
    pages = []
    for i, page in enumerate(doc):
        # cap long side at ~1600px so very large pages do not explode
        z = min(zoom, 1600 / max(page.rect.width, page.rect.height, 1))
        pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), alpha=False)
        img = pages_dir / f"p{i + 1:04d}.png"
        pix.save(str(img))
        pages.append(Page(number=i + 1, label=f"{noun} {i + 1}", text=_clean(page.get_text("text")), image_path=str(img)))
    doc.close()
    return ("slides" if is_slides else "pdf"), pages


def text_sections(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".md", ".markdown"}:
        parts = re.split(r"(?m)^(?=#{1,3}\s)", raw)
    else:
        parts = re.split(r"\n\s*\n\s*\n", raw)  # blank-line separated blocks
    pages = []
    for part in [p for p in parts if p.strip()]:
        n = len(pages) + 1
        heading = part.strip().splitlines()[0].lstrip("# ").strip()[:60]
        pages.append(Page(number=n, label=f"Section {n}: {heading}" if heading else f"Section {n}", text=_clean(part)))
    if not pages:
        pages.append(Page(number=1, label="Section 1", text=""))
    return pages


def prepare(src: Path, work_dir: Path, zoom: float = 2.0, soffice: str = "") -> PreparedDoc:
    """Turn any supported file into pages with text + original page images."""
    ext = src.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise UnsupportedFormat(f"{src.name}: unsupported file type '{ext or '?'}'.")
    pages_dir = work_dir / "pages"
    if ext in PDF_EXT:
        kind, pages = render_pdf(src, pages_dir, zoom)
        return PreparedDoc(kind=kind, pages=pages)
    if ext in OFFICE_EXT:
        pdf = convert_to_pdf(src, work_dir / "converted", soffice)
        kind, pages = render_pdf(pdf, pages_dir, zoom, presentation=ext in PRESENTATION_EXT)
        if ext == ".pptx":
            notes = pptx_notes(src)
            if notes and len(notes) == len(pages):  # only attach when slide counts line up
                for p, n in zip(pages, notes):
                    if n:
                        p.text = (p.text + "\nSpeaker notes: " + _clean(n)).strip()
        return PreparedDoc(kind=kind, pages=pages, converted_pdf=str(pdf))
    if ext in TEXT_EXT:
        return PreparedDoc(kind="text", pages=text_sections(src))
    # image
    pages_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    img = pages_dir / "p0001.png"
    with Image.open(src) as im:
        im.convert("RGB").save(img)
    return PreparedDoc(kind="image", pages=[Page(number=1, label="Image", text="", image_path=str(img))])


# --------------------------------------------------------------------------- chunking

def chunk_pages(pages: list[Page], doc_id: str, doc_name: str, chunk_size: int = 800, overlap: int = 120) -> list[dict]:
    """Split page text into chunks; every chunk keeps doc + page/slide details."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap,
                                              separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    for p in pages:
        if not p.text.strip():
            continue
        for j, piece in enumerate(splitter.split_text(p.text)):
            chunks.append({
                "chunk_id": f"{doc_id}:{p.number}:{j}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": p.number,
                "label": p.label,
                "text": piece,
            })
    return chunks
