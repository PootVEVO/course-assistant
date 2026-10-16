"""Practice quiz generation and scoring.

The answer key is fixed when the quiz is generated and lives only on the server
(in a QuizSession). The UI receives questions and options; the correct option,
explanation and source are revealed only after the student answers or asks.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError, field_validator

from .answering import build_evidence_messages, quote_in_text
from .clients import ServiceError, extract_json
from .retrieval import Candidate, retrieve
from .store import Library

LETTERS = "ABCD"

QUIZ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 4},
                    "correct_option": {"type": "integer", "minimum": 0, "maximum": 3},
                    "explanation": {"type": "string"},
                    "source_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "evidence_type": {"type": "string", "enum": ["text", "image"]},
                },
                "required": ["question", "options", "correct_option", "explanation", "source_id", "quote", "evidence_type"],
            },
        }
    },
    "required": ["questions"],
}

QUIZ_PROMPT = """You write multiple-choice practice questions for an MBA course, using ONLY the numbered sources provided (text and slide images).
Write exactly {n} questions{topic_clause}. Requirements:
- Each question must be answerable from a single source; set "source_id" to that source (S1, S2, ...).
- Exactly 4 options, one clearly correct ("correct_option" is its 0-based index), three plausible but wrong distractors. No "all/none of the above".
- "explanation": 1-3 sentences explaining why the correct option is right, based on the source.
- "quote": copy a short phrase word-for-word from the source text that supports the correct answer ("evidence_type": "text"). Only if the evidence is purely visual, use "evidence_type": "image" and describe what the image shows.
- Cover different sources/ideas; avoid trivia like slide numbers or author names.
Return only JSON: {{"questions": [{{"question", "options", "correct_option", "explanation", "source_id", "quote", "evidence_type"}}]}}"""


class QuizItem(BaseModel):
    question: str = Field(min_length=5)
    options: list[str]
    correct_option: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=5)
    source_id: str = Field(pattern=r"^S\d+$")
    quote: str = ""
    evidence_type: str = "text"

    @field_validator("options")
    @classmethod
    def _four_distinct(cls, v: list[str]) -> list[str]:
        v = [str(o).strip() for o in v]
        if len(v) != 4 or any(not o for o in v) or len({o.lower() for o in v}) != 4:
            raise ValueError("need exactly 4 distinct, non-empty options")
        return v


@dataclass
class QuizSession:
    questions: list[dict] = field(default_factory=list)  # includes the fixed answer key
    answers: dict[int, int] = field(default_factory=dict)  # question index -> chosen option (first answer is final)
    revealed: set[int] = field(default_factory=set)
    meta: dict = field(default_factory=dict)

    def public_question(self, i: int) -> dict:
        q = self.questions[i]
        return {"question": q["question"], "options": q["options"]}

    def submit(self, i: int, choice: int) -> dict:
        if not 0 <= i < len(self.questions):
            raise IndexError("no such question")
        if not 0 <= choice < 4:
            raise ValueError("choice must be 0-3")
        if i in self.answers or i in self.revealed:
            return self.feedback(i)  # answers are locked once given or revealed
        self.answers[i] = choice
        self.revealed.add(i)
        return self.feedback(i)

    def reveal(self, i: int) -> dict:
        self.revealed.add(i)
        return self.feedback(i)

    def feedback(self, i: int) -> dict:
        if i not in self.revealed:
            raise PermissionError("solution is hidden until the question is answered or revealed")
        q = self.questions[i]
        chosen = self.answers.get(i)
        return {"chosen": chosen, "correct_option": q["correct_option"], "is_correct": chosen == q["correct_option"],
                "explanation": q["explanation"], "source": q["source"], "answered": chosen is not None}

    def score(self) -> dict:
        correct = sum(1 for i, c in self.answers.items() if c == self.questions[i]["correct_option"])
        return {"correct": correct, "answered": len(self.answers), "total": len(self.questions),
                "revealed_without_answer": len(self.revealed - set(self.answers))}


def _pick_material(lib: Library, doc_ids: list[str], topic: str, max_pages: int = 8) -> list[Candidate]:
    if topic.strip():
        ret = retrieve(lib, topic, doc_ids, evidence_k=max_pages)
        return ret.evidence
    pages: list[Candidate] = []
    for did in doc_ids:
        doc = lib.get_doc(did)
        if not doc:
            continue
        for p in doc["pages"]:
            txt = lib.page_text(did, p["number"])
            if len(txt) >= 80:
                pages.append(Candidate(doc_id=did, doc_name=doc["name"], page=p["number"], label=p["label"],
                                       image=p.get("image"), chunks=[txt]))
    if len(pages) <= max_pages:
        return pages
    step = len(pages) / max_pages  # spread evenly through the material
    return [pages[int(i * step)] for i in range(max_pages)]


def generate_quiz(lib: Library, doc_ids: list[str], topic: str = "", n: int = 5, seed: int | None = None) -> QuizSession:
    doc_ids = [d for d in (doc_ids or []) if lib.get_doc(d)]
    if not doc_ids:
        raise ValueError("Choose at least one document for the quiz.")
    n = max(1, min(int(n), 10))
    material = _pick_material(lib, doc_ids, topic or "")
    if not material:
        raise ValueError("The selected material has no usable text for a quiz" + (f" about '{topic}'." if topic else "."))

    topic_clause = f" focused on the topic '{topic.strip()}'" if topic.strip() else " covering the main ideas"
    system = QUIZ_PROMPT.format(n=n, topic_clause=topic_clause)
    max_images = min(3, lib.s.max_evidence_images)
    messages, _ = build_evidence_messages("Write the quiz now.", material, max_images, system=system)
    img_ids = {f"S{i}" for i, c in enumerate(material, 1) if c.image and i <= max_images}

    t0 = time.perf_counter()
    items: list[QuizItem] = []
    rejected: list[str] = []
    for attempt in range(2):
        raw = lib.c.llm.chat(messages, json_schema=QUIZ_SCHEMA, temperature=0.4, max_tokens=3500)
        try:
            data = extract_json(raw)
            arr = data.get("questions", []) if isinstance(data, dict) else []
        except ValueError:
            arr = []
        for obj in arr:
            try:
                it = QuizItem.model_validate(obj)
            except ValidationError as exc:
                rejected.append(f"format: {exc.errors()[0]['msg']}")
                continue
            idx = int(it.source_id[1:]) - 1
            if not 0 <= idx < len(material):
                rejected.append("unknown source")
                continue
            c = material[idx]
            supported = (it.evidence_type == "image" and it.source_id in img_ids) or \
                quote_in_text(it.quote, c.excerpt(10_000) + " " + lib.page_text(c.doc_id, c.page))
            if not supported:
                rejected.append("quote not found in source")
                continue
            if any(it.question.strip().lower() == x.question.strip().lower() for x in items):
                continue
            items.append(it)
        if len(items) >= n:
            break
        if attempt == 0:
            messages = messages + [{"role": "user", "content":
                                    f"Only {len(items)} valid questions so far. Write {n} questions again, following every rule "
                                    "(4 distinct options, quote copied exactly from the cited source). Return only JSON."}]
    items = items[:n]
    if not items:
        raise ValueError("The model could not produce valid, source-backed questions. Try another topic or more material.")

    rng = random.Random(seed if seed is not None else time.time_ns())
    questions = []
    for it in items:
        c = material[int(it.source_id[1:]) - 1]
        order = list(range(4))
        rng.shuffle(order)  # avoid the "answer is always A" bias; key is fixed after this
        options = [it.options[k] for k in order]
        correct = order.index(it.correct_option)
        questions.append({
            "question": it.question.strip(), "options": options, "correct_option": correct,
            "explanation": it.explanation.strip(),
            "source": {"doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page, "label": c.label,
                       "image": c.image, "quote": it.quote, "evidence_type": it.evidence_type,
                       "excerpt": c.excerpt(500)},
        })
    return QuizSession(questions=questions, meta={"topic": topic, "doc_ids": doc_ids, "requested": n,
                                                  "rejected": rejected, "seconds": round(time.perf_counter() - t0, 1)})


__all__ = ["QuizSession", "generate_quiz", "LETTERS", "ServiceError"]
