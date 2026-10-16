"""Course library: documents on disk + three separate search indexes.

* keyword index  - BM25 (bm25s) over text chunks, rebuilt in memory from chunks.json files
* text index     - Chroma collection `text_chunks` (text embeddings of chunks)
* visual index   - Chroma collection `page_images` (visual embeddings of page images)
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import bm25s
import chromadb
from chromadb.config import Settings as ChromaSettings

from .clients import Clients, ServiceError
from .config import Settings
from .documents import chunk_pages, prepare

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class AddResult(dict):
    """{'status': added|duplicate|replaced|error, 'name', 'doc_id', 'message', 'warnings'}"""


class Library:
    def __init__(self, settings: Settings, clients: Clients):
        self.s = settings
        self.c = clients
        self.root = settings.data_dir
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "library.json"
        self._lock = threading.RLock()
        self.manifest: dict[str, dict] = self._load_manifest()
        self.chroma = chromadb.PersistentClient(path=str(self.root / "chroma"),
                                                settings=ChromaSettings(anonymized_telemetry=False))
        self.text_col = self.chroma.get_or_create_collection("text_chunks", metadata={"hnsw:space": "cosine"})
        self.image_col = self.chroma.get_or_create_collection("page_images", metadata={"hnsw:space": "cosine"})
        self.chunks: list[dict] = []
        self._bm25: bm25s.BM25 | None = None
        self._rebuild_keyword_index()

    # ------------------------------------------------------------------ manifest
    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2))
        tmp.replace(self.manifest_path)

    def list_docs(self) -> list[dict]:
        return sorted(self.manifest.values(), key=lambda d: d["name"].lower())

    def get_doc(self, doc_id: str) -> dict | None:
        return self.manifest.get(doc_id)

    def page_info(self, doc_id: str, page: int) -> dict | None:
        doc = self.manifest.get(doc_id)
        if not doc:
            return None
        for p in doc["pages"]:
            if p["number"] == page:
                return p
        return None

    def page_text(self, doc_id: str, page: int) -> str:
        return " ".join(c["text"] for c in self.chunks if c["doc_id"] == doc_id and c["page"] == page)

    # ------------------------------------------------------------------ keyword index
    def _rebuild_keyword_index(self) -> None:
        chunks: list[dict] = []
        for doc_id in self.manifest:
            path = self.docs_dir / doc_id / "chunks.json"
            if path.exists():
                chunks.extend(json.loads(path.read_text()))
        self.chunks = chunks
        if chunks:
            corpus = [f"{c['doc_name']} {c['label']} {c['text']}" for c in chunks]
            tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
            bm = bm25s.BM25()
            bm.index(tokens, show_progress=False)
            self._bm25 = bm
        else:
            self._bm25 = None

    # ------------------------------------------------------------------ add / remove
    def add_file(self, path: str | Path, display_name: str | None = None, progress: Progress | None = None) -> AddResult:
        src = Path(path)
        name = display_name or src.name
        prog = progress or (lambda f, d: None)
        with self._lock:
            digest = file_sha256(src)
            doc_id = digest[:12]
            if doc_id in self.manifest:
                return AddResult(status="duplicate", name=name, doc_id=doc_id,
                                 message=f"'{name}' is already in the library (identical content) - skipped.")
            same_name = [d for d in self.manifest.values() if d["name"] == name]
            doc_dir = self.docs_dir / doc_id
            warnings: list[str] = []
            try:
                doc_dir.mkdir(parents=True, exist_ok=True)
                stored = doc_dir / f"source{src.suffix.lower()}"
                shutil.copyfile(src, stored)
                prog(0.05, f"Preparing {name}")
                prepared = prepare(stored, doc_dir, self.s.render_zoom, self.s.soffice_path)
                pages = prepared.pages

                # OCR pages that have an image but no text layer (scans, photos, image files)
                if self.s.ocr_mode != "off":
                    need = [p for p in pages if p.image_path and len(p.text) < 15 and prepared.kind in ("pdf", "image")]
                    for k, p in enumerate(need):
                        prog(0.1 + 0.2 * k / max(len(need), 1), f"OCR {p.label}")
                        try:
                            p.text = self.c.parser.ocr(p.image_path).strip()
                            p.ocr = True
                        except ServiceError as exc:
                            warnings.append(f"OCR unavailable for {p.label}: {exc}")
                            break

                chunks = chunk_pages(pages, doc_id, name, self.s.chunk_size, self.s.chunk_overlap)
                prog(0.35, f"Text embeddings ({len(chunks)} chunks)")
                if chunks:
                    vecs = self.c.text_embedder.embed(
                        [f"{c['doc_name']} - {c['label']}\n{c['text']}" for c in chunks], input_type="document")
                    self.text_col.add(
                        ids=[c["chunk_id"] for c in chunks], embeddings=vecs, documents=[c["text"] for c in chunks],
                        metadatas=[{"doc_id": doc_id, "page": c["page"], "label": c["label"], "doc_name": name} for c in chunks])
                img_pages = [p for p in pages if p.image_path]
                for k, p in enumerate(img_pages):
                    prog(0.45 + 0.5 * k / max(len(img_pages), 1), f"Visual embedding {p.label}")
                    vec = self.c.visual_embedder.embed_images([p.image_path])[0]
                    self.image_col.add(ids=[f"{doc_id}:{p.number}"], embeddings=[vec],
                                       metadatas=[{"doc_id": doc_id, "page": p.number, "label": p.label, "doc_name": name}])
                (doc_dir / "chunks.json").write_text(json.dumps(chunks))
                self.manifest[doc_id] = {
                    "doc_id": doc_id, "name": name, "sha256": digest, "kind": prepared.kind,
                    "format": src.suffix.lower().lstrip("."), "added_at": time.strftime("%Y-%m-%d %H:%M"),
                    "n_pages": len(pages), "n_chunks": len(chunks),
                    "pages": [{"number": p.number, "label": p.label, "image": p.image_path,
                               "chars": len(p.text), "ocr": p.ocr} for p in pages],
                }
                # replacing an older version with the same file name
                for old in same_name:
                    self._remove_unlocked(old["doc_id"])
                self._save_manifest()
                self._rebuild_keyword_index()
            except Exception:
                self._purge_indexes(doc_id)
                shutil.rmtree(doc_dir, ignore_errors=True)
                self.manifest.pop(doc_id, None)
                raise
            prog(1.0, "Done")
            status = "replaced" if same_name else "added"
            msg = (f"Replaced older version of '{name}'." if same_name else f"Added '{name}'.") + \
                  f" {len(pages)} {'slides' if prepared.kind == 'slides' else 'pages/sections'}, {len(chunks)} text chunks."
            return AddResult(status=status, name=name, doc_id=doc_id, message=msg, warnings=warnings)

    def _purge_indexes(self, doc_id: str) -> None:
        for col in (self.text_col, self.image_col):
            try:
                col.delete(where={"doc_id": doc_id})
            except Exception as exc:  # noqa: BLE001
                log.warning("index purge failed for %s: %s", doc_id, exc)

    def _remove_unlocked(self, doc_id: str) -> bool:
        if doc_id not in self.manifest:
            return False
        self._purge_indexes(doc_id)
        shutil.rmtree(self.docs_dir / doc_id, ignore_errors=True)
        self.manifest.pop(doc_id, None)
        return True

    def remove(self, doc_id: str) -> bool:
        with self._lock:
            ok = self._remove_unlocked(doc_id)
            if ok:
                self._save_manifest()
                self._rebuild_keyword_index()
            return ok

    # ------------------------------------------------------------------ search
    def _where(self, doc_ids: Iterable[str] | None) -> dict | None:
        ids = [d for d in (doc_ids or []) if d in self.manifest]
        if not ids:
            return None
        return {"doc_id": ids[0]} if len(ids) == 1 else {"doc_id": {"$in": ids}}

    def keyword_search(self, query: str, k: int, doc_ids: Iterable[str] | None = None) -> list[dict]:
        if not self._bm25 or not self.chunks:
            return []
        allowed = set(doc_ids) if doc_ids else None
        qtok = bm25s.tokenize([query], stopwords="en", show_progress=False)
        n = min(len(self.chunks), max(k * 5, 50))
        try:
            idx, scores = self._bm25.retrieve(qtok, k=n, show_progress=False)
        except ValueError:  # query has no known tokens
            return []
        out = []
        for i, sc in zip(idx[0], scores[0]):
            if sc <= 0:
                continue
            c = self.chunks[int(i)]
            if allowed and c["doc_id"] not in allowed:
                continue
            out.append({**c, "score": float(sc)})
            if len(out) >= k:
                break
        return out

    def text_search(self, query: str, k: int, doc_ids: Iterable[str] | None = None) -> list[dict]:
        if self.text_col.count() == 0:
            return []
        vec = self.c.text_embedder.embed([query], input_type="query")[0]
        res = self.text_col.query(query_embeddings=[vec], n_results=min(k, self.text_col.count()),
                                  where=self._where(doc_ids))
        out = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"chunk_id": cid, "doc_id": meta["doc_id"], "doc_name": meta["doc_name"], "page": meta["page"],
                        "label": meta["label"], "text": doc, "score": 1 - float(dist)})
        return out

    def visual_search(self, query: str, k: int, doc_ids: Iterable[str] | None = None) -> list[dict]:
        if self.image_col.count() == 0:
            return []
        vec = self.c.visual_embedder.embed_query(query)
        res = self.image_col.query(query_embeddings=[vec], n_results=min(k, self.image_col.count()),
                                   where=self._where(doc_ids))
        return [{"doc_id": m["doc_id"], "doc_name": m["doc_name"], "page": m["page"], "label": m["label"],
                 "score": 1 - float(d)} for m, d in zip(res["metadatas"][0], res["distances"][0])]
