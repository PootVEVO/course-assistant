"""Capture README screenshots from the REAL app (real class models).

Prereqs: .env configured, course files already loaded (e.g. by `python -m eval.run_eval`, which uses
eval/.data), and Playwright:  pip install playwright && python -m playwright install chromium

    python scripts/capture_screenshots.py --data-dir eval/.data

Writes docs/screenshots/answer_with_slide.png, quiz_feedback.png, answer_dark.png
"""
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from course_assistant.app import CSS, build_app  # noqa: E402
from course_assistant.config import load_settings  # noqa: E402

QUESTION = 'Find the meme about Vibe Coding on "Prod" in the Week 2 course slides. What do its image and text show?'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "eval" / ".data"))
    ap.add_argument("--quiz-doc", default="Week 5", help="part of the document name to quiz on")
    ap.add_argument("--quiz-topic", default="hybrid RAG and reranking")
    args = ap.parse_args()
    out = ROOT / "docs" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings(data_dir=args.data_dir)
    demo, lib = build_app(settings)
    if not lib.list_docs():
        sys.exit("No documents in the library - load course files first.")
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    demo.queue().launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                        allowed_paths=[str(settings.data_dir)], theme=gr.themes.Soft(), css=CSS)
    url = f"http://127.0.0.1:{port}"
    with sync_playwright() as p:
        b = p.chromium.launch()
        for scheme, name in (("light", "answer_with_slide.png"), ("dark", "answer_dark.png")):
            page = b.new_page(viewport={"width": 1400, "height": 1000}, color_scheme=scheme)
            page.goto(url + ("/?__theme=dark" if scheme == "dark" else ""))
            page.get_by_label("Your question").fill(QUESTION)
            page.get_by_role("button", name="Ask", exact=True).click()
            page.locator("text=Sources").first.wait_for(timeout=300_000)
            page.wait_for_timeout(2000)
            page.screenshot(path=str(out / name), full_page=True)
            print("saved", name)

        page = b.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(url)
        page.get_by_role("tab", name="Quiz").click()
        page.get_by_label("Course material for the quiz").click()
        page.get_by_role("option", name=args.quiz_doc).first.click()
        page.keyboard.press("Escape")
        page.get_by_label("Topic filter (optional)").fill(args.quiz_topic)
        page.locator("input[type=number]").first.fill("3")
        page.get_by_role("button", name="Generate quiz").click()
        page.get_by_text("Q1.").wait_for(timeout=300_000)
        page.get_by_role("radio").first.check()
        page.get_by_role("button", name="Check").first.click()
        page.get_by_text("Source:").first.wait_for(timeout=30_000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(out / "quiz_feedback.png"), full_page=True)
        print("saved quiz_feedback.png")
        b.close()
    demo.close()


if __name__ == "__main__":
    main()
