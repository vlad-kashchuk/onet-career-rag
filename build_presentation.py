"""
Generates presentation.pptx for the Job & Career RAG Chatbot final project.
Run: python build_presentation.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

DARK_BG    = RGBColor(0x1E, 0x1E, 0x2E)
ACCENT     = RGBColor(0x74, 0xC7, 0xEC)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xCC, 0xCC, 0xCC)
ACCENT2    = RGBColor(0xA6, 0xE3, 0xA1)
RED_SOFT   = RGBColor(0xF3, 0x8B, 0xA8)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def bg(slide, color=DARK_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def txbox(slide, text, left, top, width, height,
          size=24, bold=False, color=WHITE, align=PP_ALIGN.LEFT, italic=False):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return tb


def accent_bar(slide, top=1.05):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(top), Inches(13.33), Inches(0.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


def bullet_box(slide, items, left, top, width, height, size=20, color=WHITE):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = "• " + item
        run.font.size = Pt(size)
        run.font.color.rgb = color


def divider(slide, left, top, width):
    line = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(0.03))
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()


# ── Slide 1: Title ─────────────────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=5.6)
txbox(s, "Job & Career Assistant", 0.6, 1.8, 12, 1.2, size=52, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
txbox(s, "A Domain-Specific RAG Chatbot over O*NET Occupational Data", 0.6, 3.1, 12, 0.8, size=24, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)
txbox(s, "CS3390R  ·  NLP Final Project", 0.6, 5.7, 12, 0.5, size=18, color=LIGHT_GRAY, align=PP_ALIGN.CENTER, italic=True)

# ── Slide 2: Problem Statement ──────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "Problem Statement", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)
bullet_box(s, [
    "Job-seekers lack accessible, accurate guidance on careers, required skills, and education paths.",
    "General-purpose LLMs hallucinate career facts — skills, education, and job duties need grounded data.",
    "O*NET (U.S. Dept. of Labor) covers 923 occupations with structured, authoritative data — but it is not conversational.",
    "Goal: build a chatbot that answers career questions accurately using retrieved, cited evidence from O*NET.",
], 0.6, 1.3, 12, 5.5, size=22)

# ── Slide 3: System Architecture ────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "System Architecture", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)

steps = [
    ("1  User\nQuery",       0.3),
    ("2  Embed\nQuery",      2.1),
    ("3  MMR\nRetrieval\n20 docs", 3.9),
    ("4  Cross-Encoder\nReranker\nTop 5", 5.9),
    ("5  Claude Haiku\nGenerate\n(streaming)", 8.0),
    ("6  Answer +\nO*NET\nSource Links", 10.3),
]
for label, x in steps:
    box = s.shapes.add_shape(1, Inches(x), Inches(2.0), Inches(1.9), Inches(1.8))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0x31, 0x32, 0x44)
    box.line.color.rgb = ACCENT
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = label
    r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = WHITE

for x in [2.2, 4.0, 5.95, 8.05, 10.35]:
    arr = s.shapes.add_shape(1, Inches(x), Inches(2.85), Inches(0.22), Inches(0.05))
    arr.fill.solid(); arr.fill.fore_color.rgb = ACCENT; arr.line.fill.background()

txbox(s, "sentence-transformers\nall-MiniLM-L6-v2", 0.3, 4.0, 2.2, 0.75, size=13, color=ACCENT2)
txbox(s, "ChromaDB  ·  LangChain", 3.8, 4.0, 2.3, 0.75, size=13, color=ACCENT2)
txbox(s, "ms-marco-MiniLM-L-6-v2", 5.7, 4.0, 2.5, 0.75, size=13, color=ACCENT2)
txbox(s, "Anthropic API  ·  LangChain", 7.8, 4.0, 2.7, 0.75, size=13, color=ACCENT2)

txbox(s, "ConversationBufferMemory keeps chat history across turns", 0.5, 5.0, 12, 0.5, size=16, color=LIGHT_GRAY, align=PP_ALIGN.CENTER, italic=True)

# ── Slide 4: Data ────────────────────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "Data — O*NET 30.2 Database", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)

txbox(s, "Dataset", 0.6, 1.25, 5.5, 0.4, size=20, bold=True, color=ACCENT2)
bullet_box(s, [
    "Source: onetcenter.org (public domain, CC license)",
    "923 occupations across all U.S. industries",
    "Tab-delimited .txt files, downloaded as a ZIP",
], 0.6, 1.7, 5.8, 2.0, size=20)

txbox(s, "Files Used", 7.0, 1.25, 5.5, 0.4, size=20, bold=True, color=ACCENT2)
bullet_box(s, [
    "Occupation Data.txt — titles & descriptions",
    "Skills.txt  ·  Knowledge.txt  ·  Abilities.txt",
    "Work Activities.txt",
    "Education, Training, and Experience.txt",
], 7.0, 1.7, 5.8, 2.2, size=20)

divider(s, 0.5, 3.9, 12.3)
txbox(s, "Education fix: the education file stores multiple rows per occupation with % of workers per level.\nIngestion now picks the level with the highest percentage — not just the first row.", 0.6, 4.05, 12, 1.1, size=19, color=LIGHT_GRAY)
txbox(s, "Each occupation → one document (title, description, top-5 skills, knowledge, abilities, work activities, education)", 0.6, 5.2, 12, 0.7, size=18, color=LIGHT_GRAY, italic=True)

# ── Slide 5: NLP Pipeline ────────────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "NLP Pipeline", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)

sections = [
    ("Embedding", [
        "Model: all-MiniLM-L6-v2  (384-dim dense vectors)",
        "Fast, free, no API required — runs fully locally at ingest time",
    ]),
    ("MMR Retrieval", [
        "Maximal Marginal Relevance: fetches 20 candidates, returns 5",
        "Balances relevance AND diversity — avoids returning 5 near-identical occupations",
        "ChromaDB local vector store, persistent across sessions",
    ]),
    ("Cross-Encoder Reranking", [
        "Model: cross-encoder/ms-marco-MiniLM-L-6-v2",
        "Attends to query and document jointly — more accurate than bi-encoder cosine similarity",
        "Reranks the 20 MMR candidates and returns the top 5 for the LLM",
    ]),
    ("Generation", [
        "Claude Haiku (Anthropic API)  ·  temperature=0.3  ·  max_tokens=1024",
        "Streaming LLM for answers  +  non-streaming LLM for question condensing",
        "Custom PromptTemplate injects context + chat history + question",
    ]),
]

y = 1.25
for title, bullets in sections:
    txbox(s, title, 0.6, y, 12, 0.38, size=20, bold=True, color=ACCENT2)
    bullet_box(s, bullets, 0.9, y + 0.38, 11.8, 0.85, size=17)
    y += 1.3

# ── Slide 6: Demo ─────────────────────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "Live Demo", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)
txbox(s, "streamlit run app.py  →  http://localhost:8501", 0.6, 1.2, 12, 0.55, size=22, color=ACCENT2, bold=True)

txbox(s, "Features to show:", 0.6, 2.0, 12, 0.4, size=20, bold=True, color=WHITE)
bullet_box(s, [
    "Streaming responses (tokens appear in real time)",
    "Source cards with clickable O*NET links per answer",
    "Multi-turn follow-up: ask about a career, then ask a follow-up question",
    "Clear Chat button resets both the display and the chain's conversational memory",
], 0.8, 2.45, 11.5, 2.0, size=20, color=LIGHT_GRAY)

txbox(s, "Sample questions:", 0.6, 4.55, 12, 0.4, size=20, bold=True, color=WHITE)
bullet_box(s, [
    "What skills do I need to become a software developer?",
    "What education do I need to be a mechanical engineer?",
    "Compare a UX designer and a graphic designer.",
], 0.8, 5.0, 11.5, 1.8, size=19, color=LIGHT_GRAY)

# ── Slide 7: Results & Evaluation ────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "Results & Evaluation", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)

txbox(s, "eval.py  — 10 questions, 3 metrics", 0.6, 1.15, 12, 0.45, size=20, bold=True, color=ACCENT2)

# Metric boxes
metrics = [
    ("90%", "Source Hit Rate", "9/10 expected\noccupations retrieved"),
    ("0.521", "Avg Cosine Relevance", "Query ↔ source title\nsimilarity"),
    ("216", "Avg Answer Length", "words per response"),
]
for i, (val, label, sub) in enumerate(metrics):
    x = 0.5 + i * 4.2
    box = s.shapes.add_shape(1, Inches(x), Inches(1.75), Inches(3.9), Inches(1.5))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0x31, 0x32, 0x44)
    box.line.color.rgb = ACCENT
    txbox(s, val,   x + 0.1, 1.85, 3.7, 0.65, size=36, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
    txbox(s, label, x + 0.1, 2.5,  3.7, 0.4,  size=16, bold=True, color=WHITE,  align=PP_ALIGN.CENTER)
    txbox(s, sub,   x + 0.1, 2.9,  3.7, 0.5,  size=13, color=LIGHT_GRAY, align=PP_ALIGN.CENTER)

divider(s, 0.5, 3.45, 12.3)

col_a = [
    "Correct source retrieved for 9/10 questions",
    "Responses grounded in O*NET — no hallucinated facts",
    "Multi-turn context maintained across follow-up questions",
    "Honest about missing data (e.g., no salary info in O*NET)",
]
col_b = [
    "Vague queries (e.g., 'teacher') miss the exact occupation",
    "O*NET knowledge/education fields sparse for some roles",
    "No salary data — O*NET does not include wages",
    "Memory unbounded in very long sessions",
]
txbox(s, "What Worked", 0.6, 3.55, 5.8, 0.4, size=20, bold=True, color=ACCENT2)
bullet_box(s, col_a, 0.6, 3.98, 5.8, 3.0, size=17)
txbox(s, "Limitations", 7.0, 3.55, 5.8, 0.4, size=20, bold=True, color=RED_SOFT)
bullet_box(s, col_b, 7.0, 3.98, 5.8, 3.0, size=17, color=LIGHT_GRAY)

# ── Slide 8: Tech Stack ───────────────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "Tech Stack", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)

rows = [
    ("Component",          "Tool",                                           True),
    ("LLM",                "Claude Haiku (Anthropic API) — streaming",       False),
    ("Embeddings",         "all-MiniLM-L6-v2  (sentence-transformers)",      False),
    ("Vector Store",       "ChromaDB  (local, persistent)",                  False),
    ("Retrieval",          "MMR  (fetch 20 → return 5)",                     False),
    ("Reranker",           "cross-encoder/ms-marco-MiniLM-L-6-v2",          False),
    ("RAG Framework",      "LangChain  (ConversationalRetrievalChain)",      False),
    ("UI",                 "Streamlit",                                      False),
    ("Data",               "O*NET 30.2 Database (public, CC license)",       False),
]

row_h = 0.58
for i, (comp, tool, header) in enumerate(rows):
    y = 1.25 + i * row_h
    box_l = s.shapes.add_shape(1, Inches(0.5),  Inches(y), Inches(5.5), Inches(row_h - 0.04))
    box_r = s.shapes.add_shape(1, Inches(6.15), Inches(y), Inches(6.7), Inches(row_h - 0.04))
    for box, clr in [(box_l, RGBColor(0x31,0x32,0x44)), (box_r, RGBColor(0x28,0x29,0x3A))]:
        box.fill.solid(); box.fill.fore_color.rgb = clr; box.line.fill.background()
    c1 = ACCENT if header else WHITE
    c2 = ACCENT2 if header else LIGHT_GRAY
    sz = 20 if header else 18
    txbox(s, comp, 0.6,  y + 0.07, 5.3, row_h, size=sz, bold=header, color=c1)
    txbox(s, tool, 6.25, y + 0.07, 6.4, row_h, size=sz, bold=header, color=c2)

# ── Slide 9: Conclusion ───────────────────────────────────────────────────────
s = add_slide(); bg(s)
accent_bar(s, top=1.05)
txbox(s, "Conclusion", 0.5, 0.2, 12, 0.75, size=36, bold=True, color=ACCENT)
bullet_box(s, [
    "Built a fully functional domain-specific RAG chatbot grounded in O*NET occupational data.",
    "Two-stage retrieval (MMR + cross-encoder reranking) achieves 90% source hit rate across test questions.",
    "Demonstrated core NLP techniques: embedding, semantic retrieval, reranking, and LLM generation.",
    "Conversational memory enables coherent multi-turn career advising sessions with streaming responses.",
    "System is fully reproducible: one-time ingest, one command to run, eval script for verification.",
    "Future work: add BLS salary data, sliding-window memory, and larger reranking candidate pool.",
], 0.6, 1.3, 12, 4.8, size=22)
txbox(s, "Thank you", 0.5, 6.5, 12, 0.7, size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────────────────
prs.save("presentation.pptx")
print("Saved presentation.pptx")
