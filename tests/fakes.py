"""Deterministic stand-ins for the class model services, used by the automated tests.

They implement the same interface as course_assistant.clients.* but run offline,
so tests are repeatable and do not need API keys or network access.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from course_assistant.clients import ServiceError

DIM = 256
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "is", "are", "what", "which", "how", "does", "do",
        "for", "about", "with", "this", "that", "it", "be", "by", "as", "find", "show", "me", "slide", "slides",
        "week", "course", "course's", "from", "its", "their", "image", "picture"}


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9%]+", (text or "").lower()) if w not in STOP and len(w) > 1]


def hash_vec(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in words(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v] if any(v) else [1.0 / math.sqrt(DIM)] * DIM


def overlap(a: str, b: str) -> float:
    wa, wb = set(words(a)), set(words(b))
    return len(wa & wb) / (len(wa) or 1)


class Down:
    """Mixin: set .down = True to simulate an unavailable service."""
    down = False
    name = "fake"

    def _check(self):
        if self.down:
            raise ServiceError(self.name, "service unreachable (ConnectionError). Check the endpoint URL and your network connection.")


class FakeTextEmbedder(Down):
    name = "text embedding"

    def __init__(self):
        self.calls = 0

    def embed(self, texts, input_type="document"):
        self._check()
        self.calls += 1
        return [hash_vec(t) for t in texts]


class FakeVisualEmbedder(Down):
    """'Sees' an image through a caption registry keyed by page file name (p0002.png -> caption)."""
    name = "visual embedding"

    def __init__(self, captions: dict[int, str] | None = None):
        self.captions = captions or {}

    def caption(self, path) -> str:
        m = re.search(r"p(\d+)\.png$", str(path))
        return self.captions.get(int(m.group(1)), "") if m else ""

    def embed_images(self, paths):
        self._check()
        return [hash_vec(self.caption(p) or "blank page") for p in paths]

    def embed_query(self, text):
        self._check()
        return hash_vec(text)


class FakeReranker(Down):
    name = "multimodal reranker"

    def __init__(self, visual: FakeVisualEmbedder):
        self.visual = visual
        self.calls = 0

    def rerank(self, query, docs, batch=4):
        self._check()
        self.calls += 1
        return [overlap(query, d.get("text", "") + " " + (self.visual.caption(d["image"]) if d.get("image") else ""))
                for d in docs]


class FakeLLM(Down):
    """Answers by picking the source with the highest word overlap with the question."""
    name = "vision LLM"

    def __init__(self, visual: FakeVisualEmbedder):
        self.visual = visual
        self.calls = 0
        self.garbage_first = False   # return invalid JSON on the first call (tests the repair loop)
        self.bad_citation = False    # cite a non-existent source

    @staticmethod
    def _sources(messages):
        user = messages[1]["content"]
        sources, cur = [], None
        for part in user:
            if part["type"] == "text" and part["text"].startswith("[S"):
                m = re.match(r"\[(S\d+)\] Document: (.*?) \| (.*?)(?: \(image attached below\))?\nText:\n(.*)", part["text"], re.S)
                cur = {"id": m.group(1), "label": m.group(3), "text": m.group(4), "image": False}
                sources.append(cur)
            elif part["type"] == "image_url" and cur is not None:
                cur["image"] = True
        q = next((p["text"] for p in reversed(user) if p["type"] == "text" and p["text"].startswith("Question:")), "")
        return sources, q[len("Question:"):].strip()

    def chat(self, messages, json_schema=None, temperature=0.1, max_tokens=None):
        self._check()
        self.calls += 1
        system = messages[0]["content"] if isinstance(messages[0]["content"], str) else ""
        if self.garbage_first and self.calls == 1:
            return "Sure! Here is the answer without JSON."
        if "fact-checker" in system:
            return json.dumps({"verdict": "supported", "reason": "The cited text states this."})
        if "multiple-choice" in system:
            return self._quiz(messages, system)
        sources, question = self._sources(messages)
        best, best_score, best_kind = None, 0.0, "text"
        for s in sources:
            sc_text = overlap(question, s["text"])
            cap = self._caption_for(s) if s["image"] else ""
            sc_img = overlap(question, cap) if cap else 0.0
            visual_q = bool(re.search(r"meme|image|chart|diagram|picture|graph", question, re.I))
            if max(sc_text, sc_img) > best_score:
                best, best_score = s, max(sc_text, sc_img)
                best_kind = "image" if (sc_img > sc_text or (visual_q and sc_img > 0)) else "text"
        if best is None or best_score < 0.34:
            return json.dumps({"found": False, "answer": "The provided materials do not cover this.", "sources": []})
        if best_kind == "image":
            cap = self._caption_for(best)
            out = {"found": True, "answer": f"The slide shows: {cap}.",
                   "sources": [{"source_id": best["id"], "quote": cap, "evidence_type": "image"}]}
        else:
            sents = [x for x in re.split(r"(?<=[.!?])\s|\n", best["text"].strip()) if x.strip()]
            sentence = max(sents, key=lambda x: overlap(question, x))
            quote = " ".join(sentence.split()[:10])
            out = {"found": True, "answer": sentence, "sources": [{"source_id": best["id"], "quote": quote, "evidence_type": "text"}]}
        if self.bad_citation:
            out["sources"] = [{"source_id": "S9", "quote": "made up", "evidence_type": "text"}]
        return "```json\n" + json.dumps(out) + "\n```"

    def _caption_for(self, src) -> str:
        m = re.search(r"(\d+)$", src["label"])
        return self.visual.captions.get(int(m.group(1)), "") if m else ""

    def _quiz(self, messages, system):
        n = int(re.search(r"Write exactly (\d+) questions", system).group(1))
        sources, _ = self._sources(messages)
        qs = []
        for k in range(n):
            s = sources[k % len(sources)]
            ws = s["text"].split()
            quote = " ".join(ws[:8])
            qs.append({"question": f"According to {s['label']}, which statement appears in the material (#{k + 1})?",
                       "options": [quote, f"Distractor one {k}", f"Distractor two {k}", f"Distractor three {k}"],
                       "correct_option": 0, "explanation": f"The source says: {quote}.", "source_id": s["id"],
                       "quote": quote, "evidence_type": "text"})
        return json.dumps({"questions": qs})


class FakeParser(Down):
    name = "document parser"

    def __init__(self, text: str = "OCR TEXT: scanned handout about prompt caching"):
        self.text = text

    def ocr(self, image_path):
        self._check()
        return self.text


class FakeClients:
    def __init__(self, captions: dict[int, str] | None = None):
        self.visual_embedder = FakeVisualEmbedder(captions)
        self.text_embedder = FakeTextEmbedder()
        self.reranker = FakeReranker(self.visual_embedder)
        self.llm = FakeLLM(self.visual_embedder)
        self.parser = FakeParser()
