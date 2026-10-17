"""Unit tests: individual functions, no model services."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from course_assistant.answering import LLMAnswer, quote_in_text, validate_answer
from course_assistant.clients import (ChatLLM, Reranker, ServiceError, TextEmbedder, VisualEmbedder, extract_json,
                                      redact)
from course_assistant.config import ServiceConfig, load_settings
from course_assistant.documents import Page, chunk_pages, prepare, text_sections, UnsupportedFormat
from course_assistant.quiz import QuizSession, QuizItem
from course_assistant.retrieval import Candidate, _rrf


# ----------------------------------------------------------------- parsing / validation
def test_extract_json_handles_fences_and_think_tags():
    raw = "<think>hmm</think>Here you go:\n```json\n{\"found\": true, \"answer\": \"x\", \"sources\": []}\n```"
    assert extract_json(raw)["answer"] == "x"
    assert extract_json('prefix {"a": 1} suffix') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("quote,ok", [
    ("Final Project (45%)", True),
    ("final   project (45%)", True),                         # whitespace/case-insensitive
    ("Reranking compares these candidates to the question", True),  # near-exact token match
    ("The final exam is worth 80%", False),
    ("worth 80%", False),                                     # short quotes must match exactly
])
def test_quote_in_text(quote, ok):
    text = "Final Project (45%). Reranking compares these candidates to the original question and prioritizes chunks."
    assert quote_in_text(quote, text) is ok


def test_answer_schema_requires_fields():
    with pytest.raises(Exception):
        LLMAnswer.model_validate({"answer": "x"})
    ok = LLMAnswer.model_validate({"found": True, "answer": "x", "sources": [{"source_id": "S1", "quote": "y"}]})
    assert ok.sources[0].evidence_type == "text"


class _StubLib:
    def page_text(self, doc_id, page):
        return ""


def _cand(text, image=None, page=1):
    return Candidate(doc_id="d", doc_name="Doc", page=page, label=f"Slide {page}", image=image, chunks=[text])


def test_validate_answer_flags_bad_and_fake_citations():
    ev = [_cand("Final Project (45%). Quizzes (15%).", page=1), _cand("no image here", page=2)]
    raw = json.dumps({"found": True, "answer": "45%", "sources": [
        {"source_id": "S1", "quote": "Final Project (45%)", "evidence_type": "text"},
        {"source_id": "S7", "quote": "x", "evidence_type": "text"},
        {"source_id": "S2", "quote": "a chart", "evidence_type": "image"},
    ]})
    parsed, resolved, problems = validate_answer(raw, ev, _StubLib(), max_images=4)
    assert parsed.found
    assert [r["verified"] for r in resolved] == [True, False]
    assert any("S7" in p for p in problems) and any("no image" in p for p in problems)


def test_validate_answer_rejects_non_json():
    with pytest.raises(ValueError):
        validate_answer("I think it's 45%", [_cand("x")], _StubLib(), 4)


# ----------------------------------------------------------------- chunking / documents
def test_chunk_pages_preserves_source_details():
    pages = [Page(1, "Slide 1", "short"), Page(2, "Slide 2", "word " * 600), Page(3, "Slide 3", "")]
    chunks = chunk_pages(pages, "abc", "Deck.pdf", chunk_size=300, overlap=50)
    assert {c["page"] for c in chunks} == {1, 2}          # empty page skipped
    assert all(c["doc_name"] == "Deck.pdf" and c["label"].startswith("Slide") for c in chunks)
    assert len([c for c in chunks if c["page"] == 2]) > 1  # long page split
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_markdown_sections(tmp_path):
    p = tmp_path / "notes.md"
    p.write_text("# Intro\nhello\n\n## Grading\nProject 45%\n")
    secs = text_sections(p)
    assert [s.label for s in secs] == ["Section 1: Intro", "Section 2: Grading"]


def test_prepare_rejects_unsupported(tmp_path):
    p = tmp_path / "data.xlsx"
    p.write_bytes(b"x")
    with pytest.raises(UnsupportedFormat):
        prepare(p, tmp_path / "w")


def test_prepare_pdf_renders_every_page(tmp_path, deck):
    prepared = prepare(deck, tmp_path / "w")
    assert prepared.kind == "slides" and len(prepared.pages) == 5
    assert all(Path(p.image_path).exists() for p in prepared.pages)
    assert prepared.pages[1].label == "Slide 2"


# ----------------------------------------------------------------- fusion
def test_rrf_merges_lists_at_page_level():
    class L:
        def page_info(self, d, p):
            return {"image": f"/img/{p}.png"}
    lists = {
        "keyword": [{"doc_id": "a", "doc_name": "A", "page": 1, "label": "Slide 1", "text": "x"},
                    {"doc_id": "a", "doc_name": "A", "page": 1, "label": "Slide 1", "text": "y"},
                    {"doc_id": "a", "doc_name": "A", "page": 2, "label": "Slide 2", "text": "z"}],
        "visual": [{"doc_id": "a", "doc_name": "A", "page": 2, "label": "Slide 2"}],
    }
    c = _rrf(lists, L())
    assert c[("a", 1)].ranks == {"keyword": 1} and c[("a", 1)].chunks == ["x", "y"]
    assert c[("a", 2)].ranks == {"keyword": 2, "visual": 1}
    assert c[("a", 2)].rrf > c[("a", 1)].rrf  # found by two retrievers


# ----------------------------------------------------------------- quiz session
def _session():
    q = {"question": "Q?", "options": ["a", "b", "c", "d"], "correct_option": 2, "explanation": "because",
         "source": {"doc_name": "D", "label": "Slide 1", "quote": "q", "image": None}}
    return QuizSession(questions=[dict(q), dict(q, correct_option=0), dict(q, correct_option=1)])


def test_quiz_solution_hidden_until_answered_or_revealed():
    s = _session()
    assert "correct_option" not in s.public_question(0)
    with pytest.raises(PermissionError):
        s.feedback(0)
    fb = s.submit(0, 2)
    assert fb["is_correct"] and fb["explanation"] == "because"


def test_quiz_first_answer_is_final_and_score_matches_key():
    s = _session()
    s.submit(0, 1)            # wrong
    s.submit(0, 2)            # ignored: locked
    s.submit(1, 0)            # right
    s.reveal(2)               # revealed without answering -> not counted correct
    with pytest.raises(ValueError):
        s.submit(1, 7)
    assert s.answers == {0: 1, 1: 0}
    assert s.score() == {"correct": 1, "answered": 2, "total": 3, "revealed_without_answer": 1}


def test_quiz_item_validation():
    good = dict(question="What is RAG?", options=["a", "b", "c", "d"], correct_option=1, explanation="It is ...",
                source_id="S1", quote="x")
    assert QuizItem.model_validate(good)
    for bad in (dict(good, options=["a", "a", "b", "c"]), dict(good, options=["a", "b"]), dict(good, correct_option=4)):
        with pytest.raises(Exception):
            QuizItem.model_validate(bad)


# ----------------------------------------------------------------- HTTP clients (request formats, errors, secrets)
class _Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data
        self.text = json.dumps(data)

    def json(self):
        return self._data


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, copy.deepcopy(json), headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _cfg(url="http://example.test:9004", key="sekret-key-123"):
    return ServiceConfig(name="svc", base_url=url, model="m", api_key=key, timeout=5)


def test_reranker_payload_and_scores(tmp_path, deck):
    from course_assistant.documents import prepare as prep
    img = prep(deck, tmp_path / "w").pages[1].image_path
    r = Reranker(_cfg())
    r.session = _Session([_Resp(200, {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]})])
    scores = r.rerank("meme", [{"text": "a", "image": None}, {"text": "b", "image": img}], batch=4)
    assert scores == [0.1, 0.9]
    url, body, headers = r.session.calls[0]
    assert url == "http://example.test:9004/rerank" and body["query"] == "meme"
    assert body["documents"][1]["content"][0]["type"] == "image_url"
    assert body["documents"][1]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert headers["Authorization"] == "Bearer sekret-key-123"


def test_text_embedder_falls_back_when_input_type_unsupported():
    e = TextEmbedder(_cfg("http://example.test:9002"))
    e.session = _Session([_Resp(400, {"error": "input_type not supported"}),
                          _Resp(200, {"embeddings": {"float": [[0.1, 0.2]]}})])
    assert e.embed(["hello"], "query") == [[0.1, 0.2]]
    assert e.session.calls[0][0] == "http://example.test:9002/v2/embed"
    assert "input_type" not in e.session.calls[1][1] and e.session.calls[1][1]["texts"] == ["query: hello"]


def test_visual_embedder_uses_chat_style_messages():
    v = VisualEmbedder(_cfg("http://example.test:9003/v1"))
    v.session = _Session([_Resp(200, {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})])
    assert v.embed_query("chart") == [1.0, 0.0]
    url, body, _ = v.session.calls[0]
    assert url == "http://example.test:9003/v1/embeddings" and body["messages"][1]["content"][0]["text"] == "chart"


def test_chat_retries_without_response_format_and_strips_thinking():
    llm = ChatLLM(_cfg("http://example.test:9001/v1"))
    llm.session = _Session([_Resp(400, {"error": "bad response_format"}),
                            _Resp(200, {"choices": [{"message": {"content": "<think>x</think>{\"ok\": 1}"}}]})])
    assert llm.chat([{"role": "user", "content": "hi"}], json_schema={"type": "object"}) == '{"ok": 1}'
    assert "response_format" in llm.session.calls[0][1] and "response_format" not in llm.session.calls[1][1]
    assert llm.session.calls[0][1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_errors_never_leak_the_api_key():
    import requests
    llm = ChatLLM(_cfg("http://example.test:9001/v1"))
    llm.session = _Session([_Resp(500, {"error": "boom sekret-key-123 Bearer sekret-key-123"})] * 2)
    with pytest.raises(ServiceError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "sekret-key-123" not in str(ei.value)
    llm.session = _Session([requests.ConnectionError("conn refused sekret-key-123")] * 2)
    with pytest.raises(ServiceError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "unreachable" in str(ei.value) and "sekret" not in str(ei.value)
    llm.session = _Session([_Resp(401, {"error": "bad key"})])
    with pytest.raises(ServiceError) as ei:
        llm.chat([{"role": "user", "content": "hi"}])
    assert "authentication failed" in str(ei.value)
    assert redact("token Bearer abc123 and abc123", ["abc123"]) == "token Bearer *** and ***"


def test_settings_read_env_and_hide_key(monkeypatch, tmp_path):
    monkeypatch.setenv("COURSE_API_KEY", "k-999")
    monkeypatch.setenv("USE_RERANKER", "false")
    monkeypatch.setenv("CLASS_HOST", "http://h.test")
    s = load_settings(env_file=tmp_path / "none.env", data_dir=tmp_path)
    assert s.reranker.base_url == "http://h.test:9004" and s.use_reranker is False
    assert s.llm.api_key == "k-999" and "k-999" not in repr(s.llm)


def test_quiz_streaks_and_finish():
    s = _session()  # keys: 2, 0, 1
    s.submit(0, 2)
    s.submit(1, 0)
    assert s.streak == 2 and s.best_streak == 2 and not s.finished
    s.reveal(2)       # peeking breaks the streak
    assert s.streak == 0 and s.best_streak == 2 and s.finished


def test_final_message_tiers():
    from course_assistant.app import final_message
    assert "Perfect" in final_message(5, 5, 5) and "streak" in final_message(5, 5, 5)
    assert "Great" in final_message(4, 5, 1)
    assert "Solid" in final_message(3, 5, 0)
    assert "practice" in final_message(1, 5, 0)
