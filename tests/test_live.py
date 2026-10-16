"""Live checks against the real class services. Skipped unless COURSE_API_KEY is set.

Run:  pytest -m live -v
"""
from __future__ import annotations

import math
import os

import pytest

from course_assistant.clients import Clients
from course_assistant.config import load_settings

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(not (os.getenv("COURSE_API_KEY") or load_settings().secrets),
                                 reason="COURSE_API_KEY not configured")]


@pytest.fixture(scope="module")
def live():
    return Clients(load_settings())


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_text_embeddings(live):
    q = live.text_embedder.embed(["What is retrieval augmented generation?"], "query")[0]
    d1, d2 = live.text_embedder.embed(["RAG retrieves documents and adds them to the prompt.",
                                       "The final exam is completed remotely."], "document")
    assert len(q) > 100 and _cos(q, d1) > _cos(q, d2)


def test_visual_embeddings_and_reranker(live, tmp_path):
    from tests.conftest import make_deck
    from course_assistant.documents import prepare
    pages = prepare(make_deck(tmp_path / "deck.pdf"), tmp_path / "w").pages
    meme = pages[1].image_path
    v_img = live.visual_embedder.embed_images([meme, pages[4].image_path])
    q = live.visual_embedder.embed_query("a meme about vibe coding")
    assert _cos(q, v_img[0]) > _cos(q, v_img[1])
    scores = live.reranker.rerank("meme about vibe coding in production",
                                  [{"text": "Grading percentages", "image": pages[4].image_path},
                                   {"text": "Vibe Coding on Prod", "image": meme}])
    assert scores[1] > scores[0]


def test_vision_llm_json(live):
    out = live.llm.chat([{"role": "user", "content": "Return the JSON object {\"ok\": true} and nothing else."}],
                        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                        max_tokens=50)
    assert "true" in out.lower()


def test_document_parser_ocr(live, tmp_path):
    from PIL import Image, ImageDraw
    p = tmp_path / "t.png"
    im = Image.new("RGB", (600, 120), "white")
    ImageDraw.Draw(im).text((20, 40), "Prompt caching reduces latency", fill="black")
    im = im.resize((1200, 240))
    im.save(p)
    assert "caching" in live.parser.ocr(p).lower()
