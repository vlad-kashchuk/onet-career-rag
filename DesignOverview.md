# Design Overview

This project is a **Retrieval-Augmented Generation (RAG) career chatbot** built on O*NET occupational data. Users ask natural-language questions about careers and receive grounded, cited answers from a local vector store — with no hallucinated statistics.

---

## Architecture

The system has four layers:

**1. Data Ingestion (`ingest.py`)**
O*NET tab-separated files (Occupations, Skills, Knowledge, Abilities, Work Activities, Education) are loaded with Pandas, merged into one rich text document per occupation, embedded with `all-MiniLM-L6-v2` (HuggingFace), and stored in a local ChromaDB vector store in batches of 500.

**2. RAG Core (`rag.py`)**
At query time, the user's question is first condensed (with a non-streaming Claude Haiku call) to resolve pronouns and follow-ups against the conversation history. The condensed query is then sent to an MMR retriever (fetches 20, returns 5 diverse docs), optionally re-ranked by a cross-encoder (`ms-marco-MiniLM-L-6-v2`). The final context plus chat history is passed to a streaming Claude Haiku instance that generates the answer. Conversation memory persists across turns.

**3. Streamlit UI (`app.py`)**
A chat interface with live token streaming, collapsible source citations (linked to O*NET online), and sidebar example-question buttons. State is managed via `st.session_state`.

**4. Evaluation (`eval.py`)**
A fixed 10-question benchmark measures source hit rate (did the expected occupation appear in retrieved docs?), mean cosine relevance, average answer length, and honest-refusal rate.

---

## System Diagram

```mermaid
flowchart TD
    subgraph Data["Data Layer"]
        RAW["data/raw/\nO*NET .txt files\n(Occupations, Skills,\nKnowledge, Abilities,\nWork Activities, Education)"]
        CHROMA["data/chroma_db/\nChromaDB Vector Store"]
    end

    subgraph Ingestion["Ingestion Pipeline (ingest.py)"]
        LOAD["load_onet_file()\nPandas TSV loader"]
        BUILD["build_occupation_docs()\nCombine tables → Documents"]
        EMBED["HuggingFace Embeddings\nall-MiniLM-L6-v2"]
        STORE["Chroma.from_documents()\nbatch_size=500"]
    end

    subgraph RAGCore["RAG Core (rag.py)"]
        VS["load_vectorstore()\nChroma + HF Embeddings"]
        RETR["MMR Retriever\nfetch_k=20, k=5, λ=0.7"]
        RERANK["CrossEncoder Reranker\nms-marco-MiniLM-L-6-v2\n(optional)"]
        MEM["ConversationBufferMemory\nchat_history"]
        LLM_COND["Claude Haiku\ncondense_question_llm\ntemp=0, streaming=False"]
        LLM_ANS["Claude Haiku\nanswer LLM\ntemp=0.3, streaming=True"]
        CHAIN["ConversationalRetrievalChain"]
        ASK["ask(chain, question)\n→ {answer, sources}"]
    end

    subgraph UI["Streamlit UI (app.py)"]
        INPUT["Chat Input /\nSidebar Example Buttons"]
        STREAM["StreamHandler\n(live token streaming)"]
        MSGS["Session State\nmessages"]
        SOURCES["Source Expander\nwith O*NET links"]
    end

    subgraph Eval["Evaluation (eval.py)"]
        QUESTIONS["10 Fixed Eval Questions"]
        METRICS["Metrics:\n• Source Hit Rate\n• Cosine Relevance\n• Answer Length\n• Honest Refusal"]
    end

    RAW --> LOAD --> BUILD --> EMBED --> STORE --> CHROMA
    CHROMA --> VS --> RETR --> RERANK --> CHAIN
    MEM --> CHAIN
    LLM_COND --> CHAIN
    LLM_ANS --> CHAIN
    CHAIN --> ASK

    INPUT --> ASK
    ASK --> STREAM --> MSGS
    ASK --> SOURCES

    CHAIN --> Eval
    QUESTIONS --> Eval
    Eval --> METRICS
```