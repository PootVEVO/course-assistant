from __future__ import annotations

import io
import resource
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

from course_assistant.config import load_settings
from course_assistant.store import Library
from tests.fakes import FakeClients

SLIDE = (720, 405)

# macOS allows only 256 open files per process by default; every test opens its own
# Chroma database, so raise the soft limit as far as the OS allows.
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _want = 8192 if _hard == resource.RLIM_INFINITY else min(8192, _hard)
    if _soft < _want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_want, _hard))
except (ValueError, OSError):  # pragma: no cover
    pass

# What the fake "vision" models see in each page image of sample_deck.pdf
CAPTIONS = {
    1: "title slide sample course",
    2: "meme boromir one does not simply vibe code a production grade enterprise app",
    3: "line chart stack overflow monthly questions peak decline after chatgpt launch",
    4: "diagram hybrid rag pipeline keyword search vector search reranker",
}


def _meme_png() -> bytes:
    im = Image.new("RGB", (480, 270), (120, 80, 40))
    d = ImageDraw.Draw(im)
    d.text((140, 20), "ONE DOES NOT SIMPLY", fill="white")
    d.text((110, 230), "VIBE CODE A PRODUCTION APP", fill="white")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def make_deck(path: Path, variant: str = "") -> Path:
    doc = pymupdf.open()
    pages = [
        ("MBAX Sample Course", "Sample Course Overview\nInstructor: Test Professor" + variant),
        ("Vibe Coding on “Prod”", ""),
        ("LLMs Changed Things…Forever?", ""),
        ("Hybrid RAG Pipeline with Reranking",
         "Retrieval finds relevant candidate chunks using both keyword search and vector search in Hybrid RAG. "
         "Reranking compares these candidates to the original question and prioritizes which chunks are supplied to the LLM as context."),
        ("Grading", "Final Project (45%). Assignments (20%). Quizzes (15%). Final Exam (15%). Attendance and Participation (5%). "
         "Late assignments lose 10% per calendar day and are not accepted more than five days late."),
    ]
    for title, body in pages:
        p = doc.new_page(width=SLIDE[0], height=SLIDE[1])
        p.insert_text((30, 50), title, fontsize=22)
        if body:
            p.insert_textbox(pymupdf.Rect(30, 80, 690, 390), body, fontsize=13)
        if title.startswith("Vibe Coding"):
            p.insert_image(pymupdf.Rect(120, 80, 600, 380), stream=_meme_png())
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def settings(tmp_path, monkeypatch):
    for var in ("COURSE_API_KEY", "USE_RERANKER", "SUPPORT_CHECK"):
        monkeypatch.delenv(var, raising=False)
    s = load_settings(env_file=tmp_path / "missing.env", data_dir=tmp_path / "data")
    s.evidence_k = 3
    return s


@pytest.fixture
def clients():
    return FakeClients(CAPTIONS)


@pytest.fixture(autouse=True)
def _release_chroma():
    """Close Chroma clients after each test so file handles don't pile up (macOS)."""
    yield
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def lib(settings, clients):
    return Library(settings, clients)


@pytest.fixture
def deck(tmp_path):
    return make_deck(tmp_path / "Week 2 Sample Slides.pdf")
