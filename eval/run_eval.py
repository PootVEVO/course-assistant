"""Answer-quality evaluation: compare reranking ON vs OFF on the same questions and files.

Usage (from the repo root, with .env configured):
    python -m eval.run_eval --materials /path/to/course/files
    python -m eval.run_eval --materials ./course_files --configs rerank_on rerank_off --repeat 1

Outputs (in results/):
    eval_<config>.json     full records (answers, sources, retrieval ranks, timings)
    eval_results.csv       one row per question x config, with blank columns for human grading
    eval_summary.md        summary table used in README.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from course_assistant.answering import answer_question  # noqa: E402
from course_assistant.clients import Clients  # noqa: E402
from course_assistant.config import load_settings  # noqa: E402
from course_assistant.documents import SUPPORTED_EXT  # noqa: E402
from course_assistant.store import Library  # noqa: E402

CONFIGS = {"rerank_on": {"use_reranker": True}, "rerank_off": {"use_reranker": False}}


def _match_doc(name: str, key: str) -> bool:
    return key.lower() in name.lower()


def _hit(pages: list[dict], expected: list[dict]) -> bool:
    for exp in expected:
        for p in pages:
            if _match_doc(p["doc_name"], exp["doc"]) and p["page"] in exp["pages"]:
                return True
    return False


def keyword_ok(answer: str, groups: list[list[str]]) -> bool:
    a = re.sub(r"\s+", " ", answer.lower().replace("-", " ").replace("’", "'"))
    return all(any(k.lower().replace("-", " ") in a for k in g) for g in groups)


def evaluate(lib: Library, questions: list[dict], config: str, k_evidence: int) -> list[dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        res = answer_question(lib, q["question"], use_reranker=CONFIGS[config]["use_reranker"])
        elapsed = time.perf_counter() - t0
        evidence = res.get("retrieved", [])[:k_evidence]
        cited = res.get("sources", [])
        if q["answerable"]:
            auto_correct = bool(res["found"]) and keyword_ok(res["answer"], q["keywords"])
            cited_ok = _hit(cited, q["expected_sources"])
            retrieval_hit = _hit(evidence, q["expected_sources"])
        else:
            auto_correct = not res["found"]
            cited_ok = not cited  # no invented citations
            retrieval_hit = None
        rows.append({
            "config": config, "id": q["id"], "type": q["type"], "question": q["question"],
            "answerable": q["answerable"], "found": res["found"], "answer": res["answer"],
            "auto_correct": auto_correct, "cited_expected_source": cited_ok, "retrieval_hit_at_k": retrieval_hit,
            "support_verdict": (res.get("support") or {}).get("verdict"),
            "sources": [f"{s['doc_name']} - {s['label']}" for s in cited],
            "evidence": [f"{c['doc_name']} - {c['label']} (ranks {c['ranks']}, rerank {c['rerank']})" for c in evidence],
            "seconds": round(elapsed, 2), "timings": res.get("timings", {}), "warnings": res.get("warnings", []),
        })
        print(f"[{config}] {q['id']} correct={auto_correct} cited={cited_ok} hit={retrieval_hit} {elapsed:.1f}s", flush=True)
    return rows


def summarize(all_rows: dict[str, list[dict]]) -> str:
    lines = ["| Metric | " + " | ".join(all_rows) + " |", "|---|" + "---|" * len(all_rows)]

    def pct(rows, key, only_answerable=False):
        vals = [r[key] for r in rows if r[key] is not None and (r["answerable"] or not only_answerable)]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}"

    def col(fn):
        return " | ".join(fn(rows) for rows in all_rows.values())

    lines.append("| Correct answers (auto-graded) | " + col(lambda r: pct(r, "auto_correct")) + " |")
    lines.append("| Cited the expected source | " + col(lambda r: pct(r, "cited_expected_source")) + " |")
    lines.append("| Expected page in evidence sent to LLM | " + col(lambda r: pct(r, "retrieval_hit_at_k", True)) + " |")
    lines.append("| Support check = supported | " + col(lambda r: f"{sum(x['support_verdict'] == 'supported' for x in r)}/{sum(x['found'] for x in r)}") + " |")
    lines.append("| Mean latency (s) | " + col(lambda r: f"{statistics.mean(x['seconds'] for x in r):.1f}") + " |")
    lines.append("| Median latency (s) | " + col(lambda r: f"{statistics.median(x['seconds'] for x in r):.1f}") + " |")
    lines.append("")
    lines.append("| Q | Type | " + " | ".join(f"{c}: correct / source / time" for c in all_rows) + " |")
    lines.append("|---|---|" + "---|" * len(all_rows))
    ids = [r["id"] for r in next(iter(all_rows.values()))]
    for i, qid in enumerate(ids):
        cells = []
        for rows in all_rows.values():
            r = rows[i]
            cells.append(f"{'yes' if r['auto_correct'] else 'NO'} / {'yes' if r['cited_expected_source'] else 'NO'} / {r['seconds']}s")
        lines.append(f"| {qid} | {rows[i]['type']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", required=True, help="folder with course files (PDF/PPTX/...)")
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--data-dir", default=str(ROOT / "eval" / ".data"), help="separate index for evaluation")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    settings = load_settings(data_dir=args.data_dir)
    lib = Library(settings, Clients(settings))
    files = sorted(p for p in Path(args.materials).iterdir() if p.suffix.lower() in SUPPORTED_EXT and not p.name.startswith("~$"))
    print(f"Loading {len(files)} files into {args.data_dir} ...")
    for f in files:
        t0 = time.perf_counter()
        r = lib.add_file(f)
        print(f"  {r['status']}: {f.name} ({time.perf_counter() - t0:.0f}s)")

    questions = json.loads(Path(args.questions).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = {}
    for cfg in args.configs:
        rows = evaluate(lib, questions, cfg, settings.evidence_k)
        all_rows[cfg] = rows
        (out / f"eval_{cfg}.json").write_text(json.dumps(rows, indent=2))
    with open(out / "eval_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "id", "type", "question", "answer", "sources", "auto_correct", "cited_expected_source",
                    "retrieval_hit_at_k", "support_verdict", "seconds", "human_correct", "human_sources_support", "notes"])
        for rows in all_rows.values():
            for r in rows:
                w.writerow([r["config"], r["id"], r["type"], r["question"], r["answer"], "; ".join(r["sources"]),
                            r["auto_correct"], r["cited_expected_source"], r["retrieval_hit_at_k"], r["support_verdict"],
                            r["seconds"], "", "", ""])
    summary = (f"Evaluation run {time.strftime('%Y-%m-%d %H:%M')} | files: {', '.join(f.name for f in files)} | "
               f"LLM: {settings.llm.model} | evidence pages per answer: {settings.evidence_k}\n\n" + summarize(all_rows))
    (out / "eval_summary.md").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
