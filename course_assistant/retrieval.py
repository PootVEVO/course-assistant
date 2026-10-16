"""Hybrid retrieval: keyword + text-vector + visual-vector search, fused with
reciprocal rank fusion (RRF) at the page/slide level, then multimodal reranking."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .clients import ServiceError
from .store import Library

log = logging.getLogger(__name__)
RRF_K = 60


@dataclass
class Candidate:
    doc_id: str
    doc_name: str
    page: int
    label: str
    image: str | None
    chunks: list[str] = field(default_factory=list)   # matching chunk texts (best first)
    ranks: dict[str, int] = field(default_factory=dict)  # retriever -> 1-based rank
    rrf: float = 0.0
    rerank: float | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_id, self.page)

    def excerpt(self, limit: int = 1500) -> str:
        seen, parts = set(), []
        for t in self.chunks:
            if t not in seen:
                seen.add(t)
                parts.append(t)
        return "\n".join(parts)[:limit]


@dataclass
class RetrievalResult:
    query: str
    evidence: list[Candidate]
    candidates: list[Candidate]
    warnings: list[str]
    timings: dict[str, float]
    reranked: bool


def _rrf(lists: dict[str, list[dict]], lib: Library) -> dict[tuple[str, int], Candidate]:
    cands: dict[tuple[str, int], Candidate] = {}
    for name, hits in lists.items():
        seen_pages: set[tuple[str, int]] = set()
        rank = 0
        for h in hits:
            key = (h["doc_id"], int(h["page"]))
            c = cands.get(key)
            if c is None:
                info = lib.page_info(h["doc_id"], int(h["page"])) or {}
                c = Candidate(doc_id=h["doc_id"], doc_name=h["doc_name"], page=int(h["page"]),
                              label=h["label"], image=info.get("image"))
                cands[key] = c
            if h.get("text"):
                c.chunks.append(h["text"])
            if key in seen_pages:  # only a page's best chunk counts for its rank in this list
                continue
            seen_pages.add(key)
            rank += 1
            c.ranks[name] = rank
            c.rrf += 1.0 / (RRF_K + rank)
    return cands


def retrieve(lib: Library, query: str, doc_ids: list[str] | None = None, use_reranker: bool | None = None,
             evidence_k: int | None = None) -> RetrievalResult:
    s = lib.s
    use_reranker = s.use_reranker if use_reranker is None else use_reranker
    evidence_k = evidence_k or s.evidence_k
    warnings: list[str] = []
    timings: dict[str, float] = {}
    if doc_ids is not None:
        doc_ids = [d for d in doc_ids if lib.get_doc(d)]
        if not doc_ids:
            return RetrievalResult(query, [], [], ["No documents selected."], timings, False)

    lists: dict[str, list[dict]] = {}
    for name, fn, k in (("keyword", lib.keyword_search, s.keyword_k),
                        ("text", lib.text_search, s.text_k),
                        ("visual", lib.visual_search, s.visual_k)):
        t0 = time.perf_counter()
        try:
            lists[name] = fn(query, k, doc_ids)
        except ServiceError as exc:  # degrade gracefully: keep the other retrievers
            warnings.append(f"{name} search unavailable ({exc})")
            lists[name] = []
        timings[f"{name}_search"] = round(time.perf_counter() - t0, 3)

    cands = sorted(_rrf(lists, lib).values(), key=lambda c: c.rrf, reverse=True)[: s.candidate_k]
    # make sure every candidate has some text for the reranker / LLM
    for c in cands:
        if not c.chunks:
            t = lib.page_text(c.doc_id, c.page)
            if t:
                c.chunks.append(t)

    reranked = False
    if use_reranker and cands:
        t0 = time.perf_counter()
        try:
            docs = [{"text": f"{c.doc_name} - {c.label}\n{c.excerpt(1500)}", "image": c.image} for c in cands]
            scores = lib.c.reranker.rerank(query, docs, batch=s.rerank_batch)
            for c, sc in zip(cands, scores):
                c.rerank = sc
            cands.sort(key=lambda c: (c.rerank if c.rerank is not None else -1e9), reverse=True)
            reranked = True
        except ServiceError as exc:
            warnings.append(f"reranker unavailable, using fused ranking ({exc})")
        timings["rerank"] = round(time.perf_counter() - t0, 3)

    return RetrievalResult(query, cands[:evidence_k], cands, warnings, timings, reranked)
