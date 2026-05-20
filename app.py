"""
app.py
Streamlit UI for the Job/Career RAG Chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from langchain_core.callbacks.base import BaseCallbackHandler
from rag import load_vectorstore, build_chain, ask

st.set_page_config(
    page_title="Career Assistant",
    page_icon="💼",
    layout="centered",
)

st.title("💼 Job & Career Assistant")
st.caption("Ask me about careers, required skills, salaries, job outlooks, and more.")


# ── Stream handler ────────────────────────────────────────────────────────────
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


# ── Check that vector store exists ───────────────────────────────────────────
if not Path("data/chroma_db").exists() or not any(Path("data/chroma_db").iterdir()):
    st.error(
        "Vector store not found. Run `python ingest.py` first to index the O*NET data."
    )
    st.stop()


# ── Load chain once per session ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading career database...")
def get_chain():
    vs = load_vectorstore()
    return build_chain(vs)

chain = get_chain()


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I can help you explore careers, required skills, education paths, and job outlooks. What would you like to know?",
        }
    ]


# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                for src in msg["sources"]:
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"**{src['title']}**")
                    col2.markdown(f"[{src['code']}](https://www.onetonline.org/link/summary/{src['code']})")


# ── Helper: process a question through the chain ──────────────────────────────
def process_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        handler = StreamHandler(placeholder)
        result = ask(chain, question, callbacks=[handler])

        if result["sources"]:
            with st.expander(f"Sources ({len(result['sources'])})", expanded=False):
                for src in result["sources"]:
                    col1, col2 = st.columns([4, 1])
                    col1.write(f"**{src['title']}**")
                    col2.markdown(f"[{src['code']}](https://www.onetonline.org/link/summary/{src['code']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


# ── Chat input ────────────────────────────────────────────────────────────────
# Sidebar buttons store a pending question; handle it here in the main context
if "pending_question" in st.session_state:
    pending = st.session_state.pop("pending_question")
    process_question(pending)

if user_input := st.chat_input("Ask about a career..."):
    process_question(user_input)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Example Questions")
    examples = [
        "What skills do I need to become a software developer?",
        "What is the job outlook for data scientists?",
        "What does a registered nurse do day to day?",
        "What education do I need to be a mechanical engineer?",
        "Which careers are growing the fastest?",
        "Compare a UX designer and a graphic designer.",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        chain.memory.clear()
        st.rerun()

    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.85em; padding-top: 2em;'>
            Built by <b>Vlad Kashchuk</b><br>
            <a href='https://www.linkedin.com/in/vlad-kash/' target='_blank' style='color: inherit;'>LinkedIn</a>
            &nbsp;•&nbsp;
            <a href='https://github.com/vlad-kashchuk/onet-career-rag' target='_blank' style='color: inherit;'>GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
