# Job & Career Assistant — RAG Chatbot

An NLP final project built for CS3390R. A domain-specific chatbot that answers career and job-related questions using Retrieval-Augmented Generation (RAG) over the O*NET occupational database.

## Overview

The system ingests O*NET occupation data, generates embeddings, stores them in a local vector database, and uses Claude Haiku (Anthropic) as the LLM to generate grounded, accurate responses to career questions. Retrieval uses MMR (Maximal Marginal Relevance) to reduce redundant results, followed by cross-encoder reranking for precision. Responses stream token-by-token, and every answer includes clickable O*NET source citations.

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Claude Haiku (Anthropic API) — streaming |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (local, persistent) |
| Retrieval | MMR (fetch 20 → return 5) + cross-encoder reranking |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAG Framework | LangChain (ConversationalRetrievalChain) |
| UI | Streamlit |
| Data | O*NET 30.2 Database |

## Project Structure

```
.
├── app.py              # Streamlit chat UI
├── rag.py              # RAG chain (retriever + reranker + LLM)
├── ingest.py           # Data ingestion and embedding pipeline
├── eval.py             # Evaluation script (source hit rate, relevance)
├── requirements.txt
├── .env.example        # API key template
├── data/
│   ├── raw/            # O*NET .txt files (not committed)
│   └── chroma_db/      # Vector store (not committed)
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### 4. Download O*NET data

1. Go to [onetcenter.org/database.html](https://www.onetcenter.org/database.html)
2. Download the latest **O*NET Database** zip (tab-delimited text format)
3. Extract and copy these files into `data/raw/`:

```
data/raw/
├── Occupation Data.txt
├── Skills.txt
├── Knowledge.txt
├── Abilities.txt
├── Work Activities.txt
└── Education, Training, and Experience.txt
```

### 5. Run ingestion (one time)

```bash
python ingest.py
```

This loads the O*NET data, builds one document per occupation, generates embeddings, and saves the vector store locally. Takes a few minutes on first run.

### 6. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Example Questions

- What skills do I need to become a software developer?
- What does a registered nurse do day to day?
- What education do I need to be a mechanical engineer?
- What abilities are important for a graphic designer?
- Which careers are growing the fastest?
- Compare a UX designer and a graphic designer.

## Architecture

```
User Query
    │
    ▼
Embed query (all-MiniLM-L6-v2)
    │
    ▼
ChromaDB MMR search → Top 20 candidate occupation docs
    │
    ▼
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) → Top 5 docs
    │
    ▼
Prompt = system instructions + retrieved context + chat history + query
    │
    ▼
Claude Haiku — streamed token by token
    │
    ▼
Grounded answer + clickable O*NET source citations
```

## Evaluation

Run the evaluation script to measure source hit rate and retrieval relevance across 10 test questions:

```bash
python eval.py
```

Latest results: **90% source hit rate**, 0.521 avg cosine relevance, 216 avg answer words.

## Dependencies

```
anthropic
langchain
langchain-anthropic
langchain-community
langchain-huggingface
langchain-chroma
chromadb
sentence-transformers
streamlit
pandas
python-dotenv
```
