"""Runtime configuration.

All settings come from environment variables (optionally loaded from a local
`.env` file that is git-ignored). API keys are only ever read here and are
never written to logs, the UI, or result files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


@dataclass
class ServiceConfig:
    """One model service (base URL, model name, key)."""

    name: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)  # never printed
    timeout: float = 60.0


@dataclass
class Settings:
    data_dir: Path
    llm: ServiceConfig
    text_embed: ServiceConfig
    visual_embed: ServiceConfig
    reranker: ServiceConfig
    parser: ServiceConfig
    # Retrieval knobs
    use_reranker: bool = True
    keyword_k: int = 20
    text_k: int = 20
    visual_k: int = 10
    candidate_k: int = 12
    evidence_k: int = 4
    max_evidence_images: int = 4
    rerank_batch: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120
    # Generation knobs
    llm_disable_thinking: bool = True
    llm_max_tokens: int = 1800
    support_check: bool = True
    # Document parsing
    ocr_mode: str = "auto"  # auto | off  (auto = OCR pages that have no text layer)
    render_zoom: float = 2.0
    soffice_path: str = ""

    @property
    def secrets(self) -> list[str]:
        keys = {s.api_key for s in (self.llm, self.text_embed, self.visual_embed, self.reranker, self.parser)}
        return [k for k in keys if k]


def load_settings(env_file: str | os.PathLike | None = None, data_dir: str | os.PathLike | None = None) -> Settings:
    """Build Settings from the environment (and `.env` if present)."""
    if load_dotenv is not None:
        if env_file:
            load_dotenv(env_file, override=False)
        else:
            load_dotenv(override=False)

    host = os.getenv("CLASS_HOST", "http://dobolyi.com").rstrip("/")
    shared_key = os.getenv("COURSE_API_KEY", "")

    def svc(prefix: str, name: str, default_url: str, default_model: str, timeout: float) -> ServiceConfig:
        return ServiceConfig(
            name=name,
            base_url=os.getenv(f"{prefix}_BASE_URL", default_url).rstrip("/"),
            model=os.getenv(f"{prefix}_MODEL", default_model),
            api_key=os.getenv(f"{prefix}_API_KEY", shared_key),
            timeout=_float(f"{prefix}_TIMEOUT", timeout),
        )

    return Settings(
        data_dir=Path(data_dir or os.getenv("DATA_DIR", "data")).resolve(),
        llm=svc("LLM", "vision LLM", f"{host}:9001/v1", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", 240),
        text_embed=svc("TEXT_EMBED", "text embedding", f"{host}:9002", "nvidia/Nemotron-3-Embed-1B-BF16", 90),
        visual_embed=svc("VISUAL_EMBED", "visual embedding", f"{host}:9003/v1", "Qwen/Qwen3-VL-Embedding-2B", 120),
        reranker=svc("RERANK", "multimodal reranker", f"{host}:9004", "Qwen/Qwen3-VL-Reranker-2B", 120),
        parser=svc("PARSER", "document parser", f"{host}:9005/v1", "dots.mocr", 180),
        use_reranker=_bool("USE_RERANKER", True),
        keyword_k=_int("KEYWORD_K", 20),
        text_k=_int("TEXT_K", 20),
        visual_k=_int("VISUAL_K", 10),
        candidate_k=_int("CANDIDATE_K", 12),
        evidence_k=_int("EVIDENCE_K", 4),
        max_evidence_images=_int("MAX_EVIDENCE_IMAGES", 4),
        rerank_batch=_int("RERANK_BATCH", 4),
        chunk_size=_int("CHUNK_SIZE", 800),
        chunk_overlap=_int("CHUNK_OVERLAP", 120),
        llm_disable_thinking=_bool("LLM_DISABLE_THINKING", True),
        llm_max_tokens=_int("LLM_MAX_TOKENS", 1800),
        support_check=_bool("SUPPORT_CHECK", True),
        ocr_mode=os.getenv("OCR_MODE", "auto").strip().lower(),
        render_zoom=_float("RENDER_ZOOM", 2.0),
        soffice_path=os.getenv("SOFFICE_PATH", ""),
    )
