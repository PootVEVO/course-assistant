"""Grounded answering: send the best evidence (text + page images) to the vision
LLM, require structured JSON with separate `answer` and `sources`, validate it,
and check that the cited sources actually support the answer."""
from __future__ import annotations

import logging
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .clients import ServiceError, extract_json, image_part
from .retrieval import Candidate, RetrievalResult, retrieve
from .store import Library

log = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the course materials that are loaded, so I can't answer it without guessing."


# --------------------------------------------------------------------------- schema

class SourceRef(BaseModel):
    source_id: str = Field(pattern=r"^S\d+$")
    quote: str = Field(min_length=1, max_length=600)
    evidence_type: Literal["text", "image"] = "text"


class LLMAnswer(BaseModel):
    found: bool
    answer: str = Field(min_length=1)
    sources: list[SourceRef] = Field(default_factory=list)


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "evidence_type": {"type": "string", "enum": ["text", "image"]},
                },
                "required": ["source_id", "quote", "evidence_type"],
            },
        },
    },
    "required": ["found", "answer", "sources"],
}

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
}

SYSTEM_PROMPT = """You are a course assistant for an MBA class. Answer the student's question using ONLY the numbered sources provided (text excerpts and images of the original slides/pages).
Rules:
- If the sources do not contain the answer, set "found" to false, say briefly that the materials don't cover it, and return an empty "sources" list. Never guess or use outside knowledge.
- When the question is about a picture, meme, diagram, or chart, look at the slide image and describe what it actually shows (text in the image, labels, axes, trends).
- Cite every source you used. "source_id" must be one of the given ids (S1, S2, ...). For text evidence, "quote" must be copied word-for-word from that source's text (a short phrase or sentence). For visual evidence, set "evidence_type" to "image" and "quote" to a short description of what in the image supports the answer.
- Keep the answer concise (2-6 sentences or a short list) and do not put citations or source ids inside "answer".
Return only a JSON object: {"found": bool, "answer": str, "sources": [{"source_id": str, "quote": str, "evidence_type": "text"|"image"}]}"""


# --------------------------------------------------------------------------- helpers

def _norm(t: str) -> str:
    t = t.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[^\w%$.' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def quote_in_text(quote: str, text: str, min_overlap: float = 0.85) -> bool:
    """True if `quote` appears in `text` (exactly after normalization, or near-exactly by tokens)."""
    q, t = _norm(quote).strip(" ."), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    qt = q.split()
    if len(qt) < 3:
        return False
    tt = set(t.split())
    return sum(1 for w in qt if w in tt) / len(qt) >= min_overlap


def build_evidence_messages(question: str, evidence: list[Candidate], max_images: int, system: str = SYSTEM_PROMPT):
    content: list[dict] = []
    imgs = 0
    for i, c in enumerate(evidence, 1):
        header = f"[S{i}] Document: {c.doc_name} | {c.label}"
        has_img = bool(c.image) and imgs < max_images
        content.append({"type": "text", "text": f"{header}{' (image attached below)' if has_img else ''}\nText:\n{c.excerpt(1500) or '(no extractable text)'}"})
        if has_img:
            content.append(image_part(c.image))
            imgs += 1
    content.append({"type": "text", "text": f"Question: {question}"})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}], imgs


def validate_answer(raw: str, evidence: list[Candidate], lib: Library, max_images: int) -> tuple[LLMAnswer, list[dict], list[str]]:
    """Parse + validate model output. Returns (answer, resolved sources, problems)."""
    problems: list[str] = []
    try:
        parsed = LLMAnswer.model_validate(extract_json(raw))
    except (ValueError, ValidationError) as exc:
        raise ValueError(f"invalid answer format: {str(exc)[:300]}") from None
    resolved: list[dict] = []
    img_ids = {f"S{i}" for i, c in enumerate(evidence, 1) if c.image and i <= max_images}
    for ref in parsed.sources:
        idx = int(ref.source_id[1:]) - 1
        if not 0 <= idx < len(evidence):
            problems.append(f"{ref.source_id} does not exist")
            continue
        c = evidence[idx]
        full_text = c.excerpt(10_000) + "\n" + lib.page_text(c.doc_id, c.page)
        if ref.evidence_type == "image":
            verified = ref.source_id in img_ids
            if not verified:
                problems.append(f"{ref.source_id} cited as image but no image was provided")
        else:
            verified = quote_in_text(ref.quote, full_text)
            if not verified:
                problems.append(f"quote for {ref.source_id} not found in that source")
        resolved.append({
            "source_id": ref.source_id, "doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page,
            "label": c.label, "image": c.image, "quote": ref.quote, "evidence_type": ref.evidence_type,
            "excerpt": c.excerpt(600), "verified": verified,
        })
    if parsed.found and not resolved:
        problems.append("answer claims to be found but cites no sources")
    return parsed, resolved, problems


def support_check(lib: Library, question: str, answer: str, sources: list[dict]) -> dict:
    """Ask the LLM (as a strict judge) whether the cited evidence supports the answer."""
    content: list[dict] = [{"type": "text", "text": f"Question: {question}\nProposed answer: {answer}\n\nCited evidence:"}]
    for s in sources[:4]:
        content.append({"type": "text", "text": f"[{s['source_id']}] {s['doc_name']} - {s['label']}\n{s['excerpt']}"})
        if s.get("image"):
            content.append(image_part(s["image"], 900_000))
    content.append({"type": "text", "text": "Does the cited evidence support every factual claim in the proposed answer? "
                    "Reply with JSON {\"verdict\": \"supported\"|\"partial\"|\"unsupported\", \"reason\": short reason}."})
    msgs = [{"role": "system", "content": "You are a strict fact-checker. Judge only against the evidence shown."},
            {"role": "user", "content": content}]
    raw = lib.c.llm.chat(msgs, json_schema=SUPPORT_SCHEMA, temperature=0.0, max_tokens=300)
    data = extract_json(raw)
    verdict = str(data.get("verdict", "unsupported")).lower()
    if verdict not in {"supported", "partial", "unsupported"}:
        verdict = "unsupported"
    return {"verdict": verdict, "reason": str(data.get("reason", ""))[:400]}


# --------------------------------------------------------------------------- main entry

def answer_question(lib: Library, question: str, doc_ids: list[str] | None = None,
                    use_reranker: bool | None = None, support: bool | None = None) -> dict:
    question = (question or "").strip()
    t_start = time.perf_counter()
    if not question:
        return {"answer": "Please type a question.", "found": False, "sources": [], "warnings": [], "timings": {}}
    if not lib.list_docs():
        return {"answer": "No course materials are loaded yet. Add files on the Materials tab first.",
                "found": False, "sources": [], "warnings": [], "timings": {}}

    ret: RetrievalResult = retrieve(lib, question, doc_ids, use_reranker)
    result = {"question": question, "found": False, "answer": NOT_FOUND, "sources": [], "support": None,
              "warnings": list(ret.warnings), "timings": dict(ret.timings), "reranked": ret.reranked,
              "retrieved": [{"doc_name": c.doc_name, "label": c.label, "doc_id": c.doc_id, "page": c.page,
                             "image": c.image, "ranks": c.ranks, "rrf": round(c.rrf, 4),
                             "rerank": None if c.rerank is None else round(c.rerank, 4)} for c in ret.candidates]}
    if not ret.evidence:
        result["timings"]["total"] = round(time.perf_counter() - t_start, 3)
        return result

    max_images = lib.s.max_evidence_images
    messages, _ = build_evidence_messages(question, ret.evidence, max_images)
    t0 = time.perf_counter()
    parsed = None
    resolved: list[dict] = []
    try:
        for attempt in range(2):
            raw = lib.c.llm.chat(messages, json_schema=ANSWER_SCHEMA)
            try:
                parsed, resolved, problems = validate_answer(raw, ret.evidence, lib, max_images)
            except ValueError as exc:
                problems = [str(exc)]
                parsed = None
            if parsed is not None and not problems:
                break
            if attempt == 0:  # one repair round with feedback
                messages = messages + [{"role": "assistant", "content": raw},
                                       {"role": "user", "content": "Your reply had problems: " + "; ".join(problems)
                                        + ". Fix them and return only the corrected JSON object."}]
        result["timings"]["generate"] = round(time.perf_counter() - t0, 3)
    except ServiceError as exc:
        result["timings"]["generate"] = round(time.perf_counter() - t0, 3)
        result["answer"] = ("The answer service is unavailable right now, so I can't generate an answer. "
                            "The most relevant course pages I found are shown below so you can read them directly.")
        result["warnings"].append(str(exc))
        result["sources"] = [{"source_id": f"S{i}", "doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page,
                              "label": c.label, "image": c.image, "quote": "", "evidence_type": "text",
                              "excerpt": c.excerpt(600), "verified": False}
                             for i, c in enumerate(ret.evidence, 1)]
        result["llm_error"] = True
        result["timings"]["total"] = round(time.perf_counter() - t_start, 3)
        return result

    if parsed is None:
        result["warnings"].append("The model did not return a valid structured answer.")
        result["timings"]["total"] = round(time.perf_counter() - t_start, 3)
        return result

    # keep only sources we could verify; never show invented citations
    good = [s for s in resolved if s["verified"]]
    dropped = len(resolved) - len(good)
    if dropped:
        result["warnings"].append(f"{dropped} citation(s) could not be verified against the source text and were removed.")
    # de-duplicate by page
    seen, uniq = set(), []
    for s in good:
        if (s["doc_id"], s["page"]) not in seen:
            seen.add((s["doc_id"], s["page"]))
            uniq.append(s)

    result["found"] = bool(parsed.found and uniq)
    if parsed.found and not uniq:
        result["answer"] = NOT_FOUND
        result["warnings"].append("The model's answer had no verifiable source, so it was withheld.")
    else:
        result["answer"] = parsed.answer if parsed.found else (parsed.answer or NOT_FOUND)
    result["sources"] = uniq if result["found"] else []

    do_support = lib.s.support_check if support is None else support
    if result["found"] and do_support:
        t0 = time.perf_counter()
        try:
            result["support"] = support_check(lib, question, result["answer"], result["sources"])
        except (ServiceError, ValueError) as exc:
            result["warnings"].append(f"support check skipped ({exc})")
        result["timings"]["support_check"] = round(time.perf_counter() - t0, 3)
    result["timings"]["total"] = round(time.perf_counter() - t_start, 3)
    return result
