"""HTTP clients for the class model services (vLLM 0.29 servers).

Request formats follow the vLLM 0.29 pooling/scoring docs:
  * Text embeddings  -> Cohere-compatible  POST {base}/v2/embed
  * Visual embeddings-> OpenAI-compatible  POST {base}/embeddings with chat-style `messages`
                        (falls back to   POST {root}/v2/embed with `inputs`)
  * Reranker         -> Jina/Cohere-compatible POST {base}/rerank with `documents[].content`
  * Vision LLM / OCR -> OpenAI-compatible chat completions

Every client raises `ServiceError` with a sanitized message (no keys, no headers).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from PIL import Image

from .config import ServiceConfig, Settings

log = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    """A model service failed or is unreachable. Message is safe to show users."""

    def __init__(self, service: str, message: str, status: int | None = None):
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status = status


def redact(text: str, secrets: Iterable[str]) -> str:
    for s in secrets:
        if s and len(s) >= 3:
            text = text.replace(s, "***")
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1***", text)
    return text


# --------------------------------------------------------------------------- images

@lru_cache(maxsize=256)
def image_data_uri(path: str, max_pixels: int = 1_200_000, fmt: str = "JPEG", quality: int = 85) -> str:
    """Load an image file, downscale to <= max_pixels, return a base64 data URI."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def image_part(path: str | Path, max_pixels: int = 1_200_000) -> dict:
    return {"type": "image_url", "image_url": {"url": image_data_uri(str(path), max_pixels)}}


# --------------------------------------------------------------------------- base

class _HTTPService:
    def __init__(self, cfg: ServiceConfig, secrets: Sequence[str] = ()):  # noqa: D401
        self.cfg = cfg
        self._secrets = list(secrets) + ([cfg.api_key] if cfg.api_key else [])
        self.session = requests.Session()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def post(self, url: str, payload: dict, retries: int = 1) -> dict:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self.session.post(url, json=payload, headers=self._headers(), timeout=self.cfg.timeout)
            except requests.RequestException as exc:  # connection refused, timeout, DNS...
                last = ServiceError(self.cfg.name, f"service unreachable ({type(exc).__name__}). "
                                    "Check the endpoint URL and your network connection.")
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise last from None
            if r.status_code >= 500 and attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise ServiceError(self.cfg.name, "authentication failed (check COURSE_API_KEY in your .env).", r.status_code)
            if r.status_code >= 400:
                body = redact(r.text[:300], self._secrets)
                raise ServiceError(self.cfg.name, f"HTTP {r.status_code}: {body}", r.status_code)
            try:
                return r.json()
            except ValueError:
                raise ServiceError(self.cfg.name, "response was not valid JSON", r.status_code) from None
        raise last or ServiceError(self.cfg.name, "request failed")  # pragma: no cover

    @property
    def root(self) -> str:
        """Base URL without a trailing /v1 (for /v2/embed, /rerank)."""
        return re.sub(r"/v1/?$", "", self.cfg.base_url)


# --------------------------------------------------------------------------- embeddings

def _floats_from_embed_response(data: dict) -> list[list[float]]:
    if isinstance(data.get("embeddings"), dict) and "float" in data["embeddings"]:
        return data["embeddings"]["float"]
    if isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    if isinstance(data.get("data"), list):
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in items]
    raise ValueError("unrecognized embedding response")


class TextEmbedder(_HTTPService):
    """nvidia/Nemotron-3-Embed via the Cohere-compatible /v2/embed endpoint."""

    batch_size = 32

    def __init__(self, cfg: ServiceConfig, secrets: Sequence[str] = ()):
        super().__init__(cfg, secrets)
        self._input_type_ok: bool | None = None

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        url = f"{self.root}/v2/embed"
        payload: dict[str, Any] = {"model": self.cfg.model, "texts": texts, "embedding_types": ["float"], "truncate": "END"}
        if self._input_type_ok is not False:
            try:
                data = self.post(url, {**payload, "input_type": input_type})
                self._input_type_ok = True
                return _floats_from_embed_response(data)
            except ServiceError as exc:
                if exc.status not in (400, 422):
                    raise
                self._input_type_ok = False  # model has no prompt prefixes; add them ourselves
        prefix = "query: " if input_type == "query" else "passage: "
        try:
            data = self.post(url, {**payload, "texts": [prefix + t for t in texts]})
        except ServiceError as exc:
            if exc.status not in (404, 405):
                raise
            data = self.post(f"{self.root}/v1/embeddings", {"model": self.cfg.model, "input": [prefix + t for t in texts]})
        return _floats_from_embed_response(data)

    def embed(self, texts: Sequence[str], input_type: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(list(texts[i : i + self.batch_size]), input_type))
        return out


class VisualEmbedder(_HTTPService):
    """Qwen3-VL-Embedding: embeds page images and text queries into one space."""

    QUERY_INSTRUCTION = "Retrieve the course slide or document page image that best answers the user's question."
    DOC_INSTRUCTION = "Represent the user's input."

    def _messages(self, instruction: str, content: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _one(self, instruction: str, content: list[dict]) -> list[float]:
        try:
            data = self.post(f"{self.cfg.base_url}/embeddings",
                             {"model": self.cfg.model, "messages": self._messages(instruction, content),
                              "encoding_format": "float"})
        except ServiceError as exc:
            if exc.status not in (400, 404, 405, 422):
                raise
            data = self.post(f"{self.root}/v2/embed",
                             {"model": self.cfg.model, "inputs": [{"content": content}], "embedding_types": ["float"]})
        return _floats_from_embed_response(data)[0]

    def embed_images(self, paths: Sequence[str | Path]) -> list[list[float]]:
        return [self._one(self.DOC_INSTRUCTION, [image_part(p, 1_000_000)]) for p in paths]

    def embed_query(self, text: str) -> list[float]:
        return self._one(self.QUERY_INSTRUCTION, [{"type": "text", "text": text}])


# --------------------------------------------------------------------------- reranker

class Reranker(_HTTPService):
    """Qwen3-VL-Reranker via POST /rerank. Documents may mix text and one page image."""

    def rerank(self, query: str, docs: Sequence[dict], batch: int = 4) -> list[float]:
        """docs: [{"text": str, "image": path|None}] -> relevance score per doc (same order)."""
        scores = [0.0] * len(docs)
        for start in range(0, len(docs), max(1, batch)):
            part = docs[start : start + batch]
            documents = []
            for d in part:
                content: list[dict] = []
                if d.get("image"):
                    content.append(image_part(d["image"], 1_000_000))
                content.append({"type": "text", "text": (d.get("text") or "")[:2000] or "(no text)"})
                documents.append({"content": content})
            data = self.post(f"{self.root}/rerank", {"model": self.cfg.model, "query": query,
                                                     "documents": documents, "top_n": len(documents)})
            results = data.get("results") or data.get("data") or []
            for r in results:
                idx = r.get("index", 0)
                scores[start + idx] = float(r.get("relevance_score", r.get("score", 0.0)))
        return scores


# --------------------------------------------------------------------------- chat / OCR

_THINK = re.compile(r"<think>.*?</think>", re.S)


class ChatLLM(_HTTPService):
    """OpenAI-compatible chat completions (vision-capable)."""

    def __init__(self, cfg: ServiceConfig, secrets: Sequence[str] = (), disable_thinking: bool = True,
                 max_tokens: int = 1800):
        super().__init__(cfg, secrets)
        self.disable_thinking = disable_thinking
        self.max_tokens = max_tokens
        self._schema_ok: bool | None = None

    def chat(self, messages: list[dict], json_schema: dict | None = None, temperature: float = 0.1,
             max_tokens: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if json_schema is not None and self._schema_ok is not False:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "response", "schema": json_schema}}
        url = f"{self.cfg.base_url}/chat/completions"
        try:
            data = self.post(url, payload)
            if "response_format" in payload:
                self._schema_ok = True
        except ServiceError as exc:
            if exc.status not in (400, 422) or "response_format" not in payload:
                raise
            self._schema_ok = False  # server lacks structured outputs; rely on prompt + validation
            payload.pop("response_format")
            data = self.post(url, payload)
        try:
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise ServiceError(self.cfg.name, "unexpected response format") from None
        return _THINK.sub("", content).strip()


class DocParser(ChatLLM):
    """dots.mocr OCR for pages/images without a text layer."""

    PROMPT = "Extract all the text content from this image, in reading order. Output plain text only."

    def ocr(self, image_path: str | Path) -> str:
        msgs = [{"role": "user", "content": [image_part(image_path, 1_600_000), {"type": "text", "text": self.PROMPT}]}]
        return self.chat(msgs, temperature=0.0, max_tokens=4000)


# --------------------------------------------------------------------------- bundle

class Clients:
    """All services the app needs. Tests swap in fakes with the same interface."""

    def __init__(self, settings: Settings):
        sec = settings.secrets
        self.llm = ChatLLM(settings.llm, sec, settings.llm_disable_thinking, settings.llm_max_tokens)
        self.text_embedder = TextEmbedder(settings.text_embed, sec)
        self.visual_embedder = VisualEmbedder(settings.visual_embed, sec)
        self.reranker = Reranker(settings.reranker, sec)
        self.parser = DocParser(settings.parser, sec, disable_thinking=False)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may include code fences or prose."""
    text = _THINK.sub("", text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found in model output")
