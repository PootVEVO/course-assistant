"""Generates docs/architecture.svg (run: python docs/make_architecture_svg.py)."""
from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1280, 860
out: list[str] = []
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"

SERVICE_COLORS = {
    "9001": ("#ede9fe", "#7c3aed"),
    "9002": ("#dbeafe", "#2563eb"),
    "9003": ("#dcfce7", "#16a34a"),
    "9004": ("#fee2e2", "#dc2626"),
    "9005": ("#fef9c3", "#ca8a04"),
}


def box(x, y, w, h, lines, fill="#ffffff", stroke="#334155", bold_first=True, port=None, size=13):
    out.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>")
    n = len(lines)
    lh = size + 3
    y0 = y + h / 2 - (n - 1) * lh / 2 + size / 3
    for i, t in enumerate(lines):
        weight = "700" if (i == 0 and bold_first) else "400"
        sz = size if i == 0 else size - 1
        out.append(f"<text x='{x + w / 2}' y='{y0 + i * lh}' text-anchor='middle' font-size='{sz}' font-weight='{weight}' "
                   f"fill='#0f172a' {FONT}>{escape(t)}</text>")
    if port:
        f, s = SERVICE_COLORS[port]
        bx, by = x + w - 46, y - 10
        out.append(f"<rect x='{bx}' y='{by}' width='50' height='20' rx='10' fill='{f}' stroke='{s}' stroke-width='1.5'/>")
        out.append(f"<text x='{bx + 25}' y='{by + 14}' text-anchor='middle' font-size='11' font-weight='700' fill='{s}' {FONT}>:{port}</text>")


def arrow(x1, y1, x2, y2, color="#475569", dashed=False):
    dash = " stroke-dasharray='6 4'" if dashed else ""
    out.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='1.8'{dash} marker-end='url(#arrow)'/>")


def poly(points, color="#475569"):
    pts = " ".join(f"{x},{y}" for x, y in points)
    out.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.8' marker-end='url(#arrow)'/>")


def label(x, y, text, size=15, weight="700", color="#0f172a", anchor="start"):
    out.append(f"<text x='{x}' y='{y}' font-size='{size}' font-weight='{weight}' fill='{color}' text-anchor='{anchor}' {FONT}>{escape(text)}</text>")


out.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
out.append("<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
           "<path d='M0,0 L10,5 L0,10 z' fill='#475569'/></marker></defs>")
out.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='#ffffff'/>")
label(20, 34, "Course Assistant - architecture (hybrid multimodal RAG)", 20)

# regions
out.append("<rect x='16' y='52' width='914' height='792' rx='12' fill='#f8fafc' stroke='#94a3b8' stroke-width='1.5'/>")
label(32, 76, "Your computer - Python app (Gradio UI at localhost:7860, data/ folder)", 14, "700", "#334155")
out.append("<rect x='946' y='52' width='318' height='792' rx='12' fill='#fff7ed' stroke='#fb923c' stroke-width='1.5'/>")
label(962, 76, "Class server - dobolyi.com (vLLM 0.29)", 14, "700", "#9a3412")

# services
services = [("9001", ["Vision LLM", "Qwen3.6-35B-A3B (AWQ 4-bit)", "answers, support check, quizzes"]),
            ("9002", ["Text embeddings", "nvidia/Nemotron-3-Embed-1B", "chunks + queries (/v2/embed)"]),
            ("9003", ["Visual embeddings", "Qwen3-VL-Embedding-2B", "page images + queries"]),
            ("9004", ["Multimodal reranker", "Qwen3-VL-Reranker-2B", "query x (text + slide image)"]),
            ("9005", ["Document parser (OCR)", "dots.mocr", "pages with no text layer"])]
for i, (port, lines) in enumerate(services):
    f, s = SERVICE_COLORS[port]
    box(966, 100 + i * 140, 278, 96, lines, fill=f, stroke=s, port=port)
label(966, 812, "Badges like :9003 on the left show", 12, "400", "#9a3412")
label(966, 828, "which service a step calls. Key in .env only.", 12, "400", "#9a3412")

# 1. document preparation
label(32, 108, "1  Document preparation (Materials tab)", 15, "700", "#1e293b")
box(36, 126, 160, 70, ["Add / remove files", "PDF, PPTX, DOCX, MD,", "images; SHA-256 dedupe"])
box(216, 126, 150, 70, ["LibreOffice", "PPTX/DOCX -> PDF", "(headless soffice)"])
box(386, 126, 170, 70, ["PyMuPDF", "page/slide PNG + text", "OCR if no text"], port="9005")
box(576, 126, 170, 70, ["Chunker", "800 chars, 120 overlap", "keeps doc + slide no."])
arrow(196, 161, 214, 161)
arrow(366, 161, 384, 161)
arrow(556, 161, 574, 161)
box(386, 226, 170, 62, ["page_images", "Chroma, visual vectors"], fill="#dcfce7", stroke="#16a34a", port="9003")
box(576, 226, 170, 62, ["text_chunks", "Chroma, text vectors"], fill="#dbeafe", stroke="#2563eb", port="9002")
box(766, 226, 150, 62, ["keyword index", "BM25 (bm25s)"], fill="#f1f5f9", stroke="#475569")
arrow(471, 196, 471, 224)
arrow(661, 196, 661, 224)
poly([(746, 176), (841, 176), (841, 224)])
label(40, 268, "Removing a file deletes its pages,", 12, "400", "#475569")
label(40, 284, "chunks and vectors from all 3 indexes.", 12, "400", "#475569")

# 2. answering
label(32, 330, "2  Answering a question (Ask tab)", 15, "700", "#1e293b")
box(36, 400, 140, 66, ["Question", "+ optional", "document filter"])
box(206, 346, 170, 50, ["Keyword search", "BM25 top 20 chunks"], fill="#f1f5f9", stroke="#475569")
box(206, 408, 170, 50, ["Text vector search", "top 20 chunks"], fill="#dbeafe", stroke="#2563eb", port="9002")
box(206, 470, 170, 50, ["Visual vector search", "top 10 page images"], fill="#dcfce7", stroke="#16a34a", port="9003")
for yy in (371, 433, 495):
    arrow(176, 433, 204, yy)
box(406, 404, 150, 58, ["RRF fusion", "page level, top 12"])
for yy in (371, 433, 495):
    arrow(376, yy, 404, 433)
box(586, 404, 150, 58, ["Reranker", "on/off toggle"], fill="#fee2e2", stroke="#dc2626", port="9004")
arrow(556, 433, 584, 433)
box(766, 404, 150, 58, ["Best evidence", "top 4 pages: text + image"])
arrow(736, 433, 764, 433)
box(766, 548, 150, 66, ["Vision LLM", "returns JSON with", "answer + sources"], fill="#ede9fe", stroke="#7c3aed", port="9001")
arrow(841, 462, 841, 546)
box(586, 548, 150, 66, ["Validate", "schema, source ids,", "quotes; 1 repair retry"])
arrow(766, 581, 738, 581)
box(406, 548, 150, 66, ["Support check", "LLM judge: supported /", "partial / unsupported"], fill="#ede9fe", stroke="#7c3aed", port="9001")
arrow(586, 581, 558, 581)
box(206, 548, 170, 66, ["Gradio answer view", "answer, cited excerpts,", "original slide images"])
arrow(406, 581, 378, 581)
label(40, 640, "No evidence / invalid citations -> \"not in the course materials\". A service that is down is skipped with a warning;", 12, "400", "#475569")
label(40, 656, "if the LLM is down, the closest pages are still shown.", 12, "400", "#475569")

# 3. quiz
label(32, 694, "3  Practice quiz (Quiz tab)", 15, "700", "#1e293b")
box(36, 712, 150, 70, ["Choose material", "+ optional topic", "+ no. of questions"])
box(206, 712, 170, 70, ["Pick evidence", "topic: hybrid search", "else: spread pages"])
box(396, 712, 160, 70, ["Vision LLM", "4-option MCQs, JSON,", "quote from source"], fill="#ede9fe", stroke="#7c3aed", port="9001")
box(576, 712, 160, 70, ["Validate + shuffle", "drop unsupported Qs,", "answer key fixed"])
box(756, 712, 160, 70, ["Quiz session (server)", "key hidden until answer", "or Show answer; scoring"])
arrow(186, 747, 204, 747)
arrow(376, 747, 394, 747)
arrow(556, 747, 574, 747)
arrow(736, 747, 754, 747)
label(40, 808, "Feedback shows correct option, explanation, document + slide/page, the quoted excerpt, and the slide image.", 12, "400", "#475569")
label(40, 826, "Stored locally: data/library.json, data/docs/<id>/pages/*.png, chunks.json, data/chroma/  (all git-ignored).", 12, "400", "#475569")

out.append("</svg>")
Path(__file__).with_name("architecture.svg").write_text("\n".join(out))
print("wrote docs/architecture.svg")
