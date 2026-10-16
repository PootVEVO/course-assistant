"""Browser end-to-end test of the Gradio interface (fake models, real app + indexes).

Requires:  pip install playwright && playwright install chromium
Skipped automatically if Playwright or its browser is unavailable.
Saves light/dark screenshots to results/ui/ for a readability check.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

from course_assistant.app import CSS, build_app  # noqa: E402
from tests.conftest import CAPTIONS  # noqa: E402
from tests.fakes import FakeClients  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ui"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def app_url(settings):
    import gradio as gr
    demo, lib = build_app(settings, FakeClients(CAPTIONS))
    port = _free_port()
    demo.queue().launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True,
                        allowed_paths=[str(settings.data_dir)], theme=gr.themes.Soft(), css=CSS)
    yield f"http://127.0.0.1:{port}"
    demo.close()


@pytest.fixture
def browser():
    try:
        with pw.sync_playwright() as p:
            b = p.chromium.launch()
            yield b
            b.close()
    except Exception as exc:  # browser binary missing
        pytest.skip(f"Chromium not available for Playwright: {exc}")


def test_full_workflow_in_browser(app_url, browser, deck):
    OUT.mkdir(parents=True, exist_ok=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(app_url)
    page.get_by_role("tab", name="Materials").click()
    page.locator("input[type=file]").first.set_input_files(str(deck))
    page.get_by_role("button", name="Add to library").click()
    page.get_by_text("Added 'Week 2 Sample Slides.pdf'").wait_for(timeout=30_000)
    # re-adding the same file is skipped
    page.locator("input[type=file]").first.set_input_files(str(deck))
    page.get_by_role("button", name="Add to library").click()
    page.get_by_text("already in the library").wait_for(timeout=30_000)

    # Ask a visual question (after checking the Surprise me button fills the box)
    page.get_by_role("tab", name="Ask").click()
    page.get_by_role("button", name="Surprise me").click()
    page.wait_for_function("document.querySelector('textarea').value.length > 10", timeout=10_000)
    page.get_by_label("Your question").fill("Find the meme about Vibe Coding on Prod. What does it say?")
    page.get_by_role("button", name="Ask", exact=True).click()
    page.get_by_text("one does not simply", exact=False).first.wait_for(timeout=30_000)
    page.locator("text=Week 2 Sample Slides.pdf - Slide 2").first.wait_for(timeout=10_000)
    assert page.locator("img").count() >= 1
    page.screenshot(path=str(OUT / "ask_light.png"), full_page=True)

    # Quiz: answer the first question, check score + source
    page.get_by_role("tab", name="Quiz").click()
    page.get_by_label("Course material for the quiz").click()
    page.get_by_role("option").first.click()
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Generate quiz").click()
    page.get_by_text("Q1.").wait_for(timeout=30_000)
    assert page.get_by_text("Correct answer").count() == 0  # solutions hidden
    page.get_by_role("button", name="Show answer").first.click()
    page.get_by_text("Correct answer").first.wait_for(timeout=10_000)
    page.get_by_text("Score: 0 / 5").wait_for(timeout=10_000)
    page.get_by_text("1 of 5 done").wait_for(timeout=10_000)
    page.screenshot(path=str(OUT / "quiz_light.png"), full_page=True)

    # Dark mode readability snapshot
    dark = browser.new_page(viewport={"width": 1280, "height": 900}, color_scheme="dark")
    dark.goto(app_url + "/?__theme=dark")
    dark.get_by_role("tab", name="Materials").click()
    dark.get_by_role("button", name="Add to library").wait_for(timeout=10_000)
    dark.wait_for_timeout(1500)
    dark.screenshot(path=str(OUT / "materials_dark.png"), full_page=True)

    # Remove the document
    page.get_by_role("tab", name="Materials").click()
    page.get_by_label("Remove a document").click()
    page.get_by_role("option").first.click()
    page.get_by_role("button", name="Remove").click()
    page.get_by_text("and all of its searchable content").wait_for(timeout=10_000)
