"""Gradio interface: Ask / Quiz / Materials tabs.

Run:  python -m course_assistant.app      (then open http://127.0.0.1:7860)
"""
from __future__ import annotations

import argparse
import random
import html
import logging
from pathlib import Path

import gradio as gr

from .answering import answer_question
from .clients import Clients, ServiceError, redact
from .config import Settings, load_settings
from .documents import SUPPORTED_EXT, SUPPORTED_FORMATS_HELP, ConversionError, UnsupportedFormat
from .quiz import LETTERS, QuizSession, generate_quiz
from .store import Library

log = logging.getLogger("course_assistant")
MAX_Q = 10

SURPRISE_QUESTIONS = [
    'Find the meme about Vibe Coding on "Prod" in the Week 2 slides. What does it say?',
    "What does the Stack Overflow chart in Week 2 say about how LLMs changed things?",
    "Explain the hybrid RAG pipeline diagram like I'm a busy MBA student.",
    "What is speculative decoding, in one sentence?",
    "What happens if I turn an assignment in two days late?",
    "What is the 'lost in the middle' effect?",
    "What is the difference between vibe coding and agentic coding?",
    "How much of my grade is the final project?",
]
CHEERS = ["Nailed it!", "Boom.", "Textbook.", "Correct, and the slides back you up.", "You've clearly been to class."]
NUDGES = ["Not quite.", "Close, but no.", "Good guess, wrong slide.", "Plot twist."]


def final_message(correct: int, total: int, best_streak: int) -> str:
    ratio = correct / total if total else 0
    if ratio == 1:
        msg = "\U0001F389 **Perfect score!** You could teach this section."
    elif ratio >= 0.8:
        msg = "\U0001F31F **Great work!** Just a little polishing left."
    elif ratio >= 0.5:
        msg = "\U0001F4AA **Solid start.** Review the slides on the ones you missed, then try another round."
    else:
        msg = "\U0001F4DA **Good practice run.** Open the source slides above and try again. Same topic, new questions."
    if best_streak >= 3:
        msg += f" Best streak: \U0001F525 {best_streak}."
    return msg

CSS = """
.ca-badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.85em;font-weight:600;margin:4px 0;}
.ca-ok {background:#d1fae5;color:#065f46;} .ca-warn {background:#fef3c7;color:#92400e;} .ca-bad {background:#fee2e2;color:#991b1b;}
.ca-src {border-left:3px solid var(--color-accent);padding:4px 10px;margin:6px 0;}
.ca-muted {opacity:0.75;font-size:0.9em;}
.ca-progress {height:8px;border-radius:999px;background:var(--neutral-200);overflow:hidden;margin:4px 0 2px;}
.dark .ca-progress {background:var(--neutral-700);}
.ca-progress-fill {height:100%;background:linear-gradient(90deg,#6366f1,#22c55e);transition:width .4s ease;}
.ca-streak {font-size:0.8em;padding:2px 10px;border-radius:999px;background:#ffedd5;color:#9a3412;margin-left:8px;}
"""


def _esc(t: str) -> str:
    return html.escape(t or "")


def _doc_choices(lib: Library) -> list[tuple[str, str]]:
    return [(f"{d['name']} ({d['n_pages']} {'slides' if d['kind'] == 'slides' else 'pages'})", d["doc_id"])
            for d in lib.list_docs()]


def _library_table(lib: Library) -> list[list]:
    return [[d["name"], d["kind"], d["n_pages"], d["n_chunks"], d["added_at"], d["doc_id"]] for d in lib.list_docs()]


def format_answer(res: dict) -> tuple[str, str, list, str]:
    """-> (answer markdown, sources markdown, gallery items, details markdown)"""
    parts = [f"### Answer\n\n{res['answer']}"]
    sup = res.get("support")
    if sup:
        cls = {"supported": "ca-ok", "partial": "ca-warn", "unsupported": "ca-bad"}[sup["verdict"]]
        label = {"supported": "Sources support this answer", "partial": "Partially supported by sources",
                 "unsupported": "Sources may not support this answer"}[sup["verdict"]]
        parts.append(f"<span class='ca-badge {cls}'>{label}</span> <span class='ca-muted'>{_esc(sup['reason'])}</span>")
    elif res.get("found") is False and not res.get("llm_error"):
        parts.append("<span class='ca-badge ca-warn'>Not found in the loaded materials</span>")
    for w in res.get("warnings", []):
        parts.append(f"<span class='ca-muted'>Note: {_esc(w)}</span>")
    answer_md = "\n\n".join(parts)

    srcs = res.get("sources", [])
    if srcs:
        heading = "### Closest material (not verified)" if res.get("llm_error") else "### Sources"
        lines = [heading]
        for s in srcs:
            lines.append(f"**[{s['source_id']}] {_esc(s['doc_name'])} - {_esc(s['label'])}**")
            if s.get("evidence_type") == "image" and s.get("quote"):
                lines.append(f"*Visual evidence:* {_esc(s['quote'])}")
            elif s.get("quote"):
                lines.append(f"*Quoted:* “{_esc(s['quote'])}”")
            if s.get("excerpt"):
                lines.append("> " + _esc(s["excerpt"][:500]).replace("\n", "\n> "))
            lines.append("")
        sources_md = "\n\n".join(lines)
    else:
        sources_md = ""
    gallery = [(s["image"], f"[{s['source_id']}] {s['doc_name']} - {s['label']}") for s in srcs if s.get("image")]

    det = ["| # | Document | Page/slide | keyword | text | visual | RRF | rerank |", "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(res.get("retrieved", [])[:12], 1):
        r = c["ranks"]
        det.append(f"| {i} | {_esc(c['doc_name'])} | {c['label']} | {r.get('keyword', '-')} | {r.get('text', '-')} | "
                   f"{r.get('visual', '-')} | {c['rrf']} | {c['rerank'] if c['rerank'] is not None else '-'} |")
    t = res.get("timings", {})
    det.append("\n**Timings (s):** " + ", ".join(f"{k} {v}" for k, v in t.items()))
    det.append(f"\n**Reranker used:** {res.get('reranked', False)}")
    return answer_md, sources_md, gallery, "\n".join(det)


def build_app(settings: Settings | None = None, clients: Clients | None = None) -> tuple[gr.Blocks, Library]:
    settings = settings or load_settings()
    clients = clients or Clients(settings)
    lib = Library(settings, clients)
    secrets = settings.secrets

    def safe(msg: str) -> str:
        return redact(msg, secrets)

    # ------------------------------------------------------------------ handlers: materials
    def refresh_lists():
        ch = _doc_choices(lib)
        return (gr.update(choices=ch, value=[]), gr.update(choices=ch, value=[]),
                gr.update(choices=ch, value=None), gr.update(choices=ch, value=None), _library_table(lib))

    def add_files(files, progress=gr.Progress()):
        if not files:
            return ("Choose one or more files first.", None) + refresh_lists()
        msgs = []
        for f in files:
            path = Path(f if isinstance(f, str) else f.name)
            name = path.name
            if path.suffix.lower() not in SUPPORTED_EXT:
                msgs.append(f"- **{_esc(name)}**: not added - unsupported file type (see the list above).")
                continue
            try:
                res = lib.add_file(path, name, progress=lambda frac, d: progress(frac, desc=d))
                icon = {"added": "Added", "replaced": "Updated", "duplicate": "Skipped"}[res["status"]]
                msgs.append(f"- **{icon}:** {_esc(res['message'])}")
                for w in res.get("warnings", []):
                    msgs.append(f"  - Note: {_esc(safe(w))}")
            except (UnsupportedFormat, ConversionError, ServiceError) as exc:
                msgs.append(f"- **{_esc(name)}**: not added - {_esc(safe(str(exc)))}")
            except Exception as exc:  # noqa: BLE001
                log.exception("add failed")
                msgs.append(f"- **{_esc(name)}**: not added - unexpected error ({type(exc).__name__}).")
        return ("\n".join(msgs), None) + refresh_lists()  # clear the upload box after adding

    def remove_doc(doc_id):
        if not doc_id:
            return ("Pick a document to remove.",) + refresh_lists()
        doc = lib.get_doc(doc_id)
        ok = lib.remove(doc_id)
        msg = f"Removed **{_esc(doc['name'])}** and all of its searchable content." if ok and doc else "Document not found."
        return (msg,) + refresh_lists()

    def preview(doc_id, page):
        if not doc_id:
            return None, ""
        doc = lib.get_doc(doc_id)
        page = int(page or 1)
        info = lib.page_info(doc_id, page)
        if not info:
            return None, f"{doc['name']} has {doc['n_pages']} pages/slides."
        text = lib.page_text(doc_id, page)
        return info.get("image"), f"**{_esc(doc['name'])} - {info['label']}** ({info['chars']} characters of text{', OCR' if info.get('ocr') else ''})\n\n> " + _esc(text[:800]).replace("\n", "\n> ")

    # ------------------------------------------------------------------ handlers: ask
    def ask(question, doc_ids, use_rerank):
        try:
            res = answer_question(lib, question, doc_ids or None, use_reranker=use_rerank)
        except ServiceError as exc:
            return f"### Answer\n\nA model service is unavailable: {_esc(safe(str(exc)))}", "", [], ""
        except Exception as exc:  # noqa: BLE001
            log.exception("ask failed")
            return f"### Answer\n\nSomething went wrong ({type(exc).__name__}). Please try again.", "", [], ""
        return format_answer(res)

    # ------------------------------------------------------------------ handlers: quiz
    def slot_updates(session: QuizSession | None):
        outs = []
        for i in range(MAX_Q):
            if session and i < len(session.questions):
                q = session.public_question(i)
                outs += [gr.update(visible=True), f"**Q{i + 1}. {_esc(q['question'])}**",
                         gr.update(choices=[f"{LETTERS[k]}. {o}" for k, o in enumerate(q["options"])], value=None, interactive=True),
                         "", gr.update(value=None, visible=False), gr.update(interactive=True), gr.update(interactive=True)]
            else:
                outs += [gr.update(visible=False), "", gr.update(choices=[], value=None), "", gr.update(value=None, visible=False),
                         gr.update(interactive=True), gr.update(interactive=True)]
        return outs

    def score_md(session: QuizSession | None) -> str:
        if not session or not session.questions:
            return ""
        s = session.score()
        extra = f" - {s['revealed_without_answer']} revealed without answering" if s["revealed_without_answer"] else ""
        done = len(session.revealed)
        pct = int(100 * done / s["total"])
        bar = (f"<div class='ca-progress'><div class='ca-progress-fill' style='width:{pct}%'></div></div>"
               f"<span class='ca-muted'>{done} of {s['total']} done</span>")
        streak = f" <span class='ca-streak'>\U0001F525 {session.streak} in a row</span>" if session.streak >= 2 else ""
        md = f"### Score: {s['correct']} / {s['total']}{streak}  <span class='ca-muted'>({s['answered']} answered{extra})</span>\n\n{bar}"
        if session.finished:
            md += "\n\n" + final_message(s["correct"], s["total"], session.best_streak)
        return md

    def make_quiz(doc_ids, topic, n, progress=gr.Progress()):
        progress(0.1, desc="Selecting material and writing questions...")
        try:
            session = generate_quiz(lib, doc_ids or [], topic or "", int(n))
        except (ValueError, ServiceError) as exc:
            return [None, f"Could not create a quiz: {_esc(safe(str(exc)))}", ""] + slot_updates(None)
        got = len(session.questions)
        note = f"Generated {got} question{'s' if got != 1 else ''}"
        if got < int(n):
            note += f" (asked for {int(n)}; {int(n) - got} were dropped because they failed validation)"
        note += ". Pick an answer and press **Check**, or **Show answer**. Your first answer is final."
        return [session, note, score_md(session)] + slot_updates(session)

    def render_feedback(session: QuizSession, i: int):
        fb = session.feedback(i)
        q = session.questions[i]
        right = f"{LETTERS[fb['correct_option']]}. {q['options'][fb['correct_option']]}"
        if not fb["answered"]:
            head = f"<span class='ca-badge ca-warn'>Answer revealed</span> Correct answer: **{_esc(right)}**"
        elif fb["is_correct"]:
            head = f"<span class='ca-badge ca-ok'>Correct</span> {random.choice(CHEERS)} **{_esc(right)}**"
        else:
            head = (f"<span class='ca-badge ca-bad'>Incorrect</span> {random.choice(NUDGES)} You chose {LETTERS[fb['chosen']]}. "
                    f"Correct answer: **{_esc(right)}**")
        src = fb["source"]
        ev = (f"*Visual evidence:* {_esc(src['quote'])}" if src.get("evidence_type") == "image"
              else f"*Quoted:* “{_esc(src['quote'])}”")
        md = (f"{head}\n\n{_esc(fb['explanation'])}\n\n**Source: {_esc(src['doc_name'])} - {_esc(src['label'])}**  \n{ev}\n\n> "
              + _esc(src.get("excerpt", "")[:400]).replace("\n", "\n> "))
        img = gr.update(value=src.get("image"), visible=bool(src.get("image")), label=f"{src['doc_name']} - {src['label']}")
        return md, img

    def make_check(i):
        def _check(session: QuizSession | None, choice):
            if not session or i >= len(session.questions):
                return session, "No quiz loaded.", gr.update(), gr.update(), gr.update(), gr.update(), ""
            if not choice and i not in session.revealed:
                return session, "Pick an option first.", gr.update(), gr.update(), gr.update(), gr.update(), score_md(session)
            if choice and i not in session.revealed:
                session.submit(i, LETTERS.index(choice[0]))
            md, img = render_feedback(session, i)
            return (session, md, img, gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), score_md(session))
        return _check

    def make_reveal(i):
        def _reveal(session: QuizSession | None):
            if not session or i >= len(session.questions):
                return session, "No quiz loaded.", gr.update(), gr.update(), gr.update(), gr.update(), ""
            session.reveal(i)
            md, img = render_feedback(session, i)
            return (session, md, img, gr.update(interactive=False), gr.update(interactive=False),
                    gr.update(interactive=False), score_md(session))
        return _reveal

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title="Course Assistant") as demo:
        gr.Markdown("# \U0001F393 Course Assistant\nAsk anything about your course materials, or test yourself with a quiz. "
                    "Every answer comes from the files you load, with the receipts (the slides themselves) attached.")
        quiz_state = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question = gr.Textbox(label="Your question", lines=2,
                                              placeholder="e.g. Find the meme about Vibe Coding on \"Prod\" in the Week 2 slides")
                    with gr.Column(scale=2):
                        ask_docs = gr.Dropdown(label="Limit to documents (optional)", multiselect=True, choices=[])
                        use_rerank = gr.Checkbox(label="Use multimodal reranker", value=settings.use_reranker)
                with gr.Row():
                    ask_btn = gr.Button("Ask", variant="primary", scale=4)
                    surprise_btn = gr.Button("\U0001F3B2 Surprise me", scale=1)
                answer_md = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=3):
                        gallery = gr.Gallery(label="Source slides/pages", columns=2, height=560, object_fit="contain",
                                             preview=True, show_label=True)
                    with gr.Column(scale=2):
                        sources_md = gr.Markdown()
                with gr.Accordion("Retrieval details", open=False):
                    details_md = gr.Markdown()
                ask_btn.click(ask, [question, ask_docs, use_rerank], [answer_md, sources_md, gallery, details_md])
                question.submit(ask, [question, ask_docs, use_rerank], [answer_md, sources_md, gallery, details_md])
                surprise_btn.click(lambda: random.choice(SURPRISE_QUESTIONS), None, question)

            with gr.Tab("Quiz"):
                with gr.Row():
                    quiz_docs = gr.Dropdown(label="Course material for the quiz", multiselect=True, choices=[], scale=3)
                    quiz_topic = gr.Textbox(label="Topic filter (optional)", placeholder="e.g. RAG chunking", scale=2)
                    quiz_n = gr.Slider(1, MAX_Q, value=5, step=1, label="Questions", scale=1)
                quiz_btn = gr.Button("Generate quiz", variant="primary")
                quiz_status = gr.Markdown()
                quiz_score = gr.Markdown()
                slots = []
                for i in range(MAX_Q):
                    with gr.Group(visible=False) as grp:
                        q_md = gr.Markdown()
                        q_radio = gr.Radio(choices=[], label="Choose one", interactive=True)
                        with gr.Row():
                            q_check = gr.Button("Check", size="sm", variant="primary")
                            q_reveal = gr.Button("Show answer", size="sm")
                        q_fb = gr.Markdown()
                        q_img = gr.Image(visible=False, interactive=False, show_label=True, height=320)
                    slots.append((grp, q_md, q_radio, q_fb, q_img, q_check, q_reveal))
                slot_outputs = [comp for s in slots for comp in s]
                quiz_btn.click(make_quiz, [quiz_docs, quiz_topic, quiz_n], [quiz_state, quiz_status, quiz_score] + slot_outputs)
                for i, (grp, q_md, q_radio, q_fb, q_img, q_check, q_reveal) in enumerate(slots):
                    q_check.click(make_check(i), [quiz_state, q_radio], [quiz_state, q_fb, q_img, q_radio, q_check, q_reveal, quiz_score])
                    q_reveal.click(make_reveal(i), [quiz_state], [quiz_state, q_fb, q_img, q_radio, q_check, q_reveal, quiz_score])

            with gr.Tab("Materials"):
                gr.Markdown(SUPPORTED_FORMATS_HELP + "\n\nUploading a file that is already loaded is skipped; uploading a "
                            "changed file with the same name replaces the old version.")
                with gr.Row():
                    uploads = gr.File(label="Add course files", file_count="multiple",
                                      file_types=sorted(SUPPORTED_EXT), type="filepath")
                add_btn = gr.Button("Add to library", variant="primary")
                mat_status = gr.Markdown()
                table = gr.Dataframe(headers=["Document", "Type", "Pages/slides", "Chunks", "Added", "ID"],
                                     value=_library_table(lib), interactive=False, wrap=True)
                with gr.Row():
                    rm_doc = gr.Dropdown(label="Remove a document", choices=[], scale=3)
                    rm_btn = gr.Button("Remove", variant="stop", scale=1)
                gr.Markdown("#### Check a converted page/slide")
                with gr.Row():
                    pv_doc = gr.Dropdown(label="Document", choices=[], scale=3)
                    pv_page = gr.Number(label="Page / slide", value=1, precision=0, minimum=1, scale=1)
                    pv_btn = gr.Button("Show", scale=1)
                with gr.Row():
                    pv_img = gr.Image(label="Rendered page", interactive=False, height=360)
                    pv_text = gr.Markdown()

                list_outputs = [ask_docs, quiz_docs, rm_doc, pv_doc, table]
                add_btn.click(add_files, [uploads], [mat_status, uploads] + list_outputs)
                rm_btn.click(remove_doc, [rm_doc], [mat_status] + list_outputs)
                pv_btn.click(preview, [pv_doc, pv_page], [pv_img, pv_text])

        demo.load(refresh_lists, None, list_outputs)
    return demo, lib


def main() -> None:
    ap = argparse.ArgumentParser(description="Course Assistant")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = load_settings(data_dir=args.data_dir)
    if not settings.secrets:
        log.warning("No API key configured: copy .env.example to .env and set COURSE_API_KEY.")
    demo, _ = build_app(settings)
    demo.queue().launch(server_name=args.host, server_port=args.port, allowed_paths=[str(settings.data_dir)],
                        theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
