---
title: Career Assistant
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Job and Career Assistant — RAG Chatbot

A domain-specific conversational assistant that answers career and job-related questions using Retrieval-Augmented Generation over the U.S. Department of Labor's [O\*NET](https://www.onetcenter.org/) occupational database (900+ occupations).

Built as the final project for CS3390R (Natural Language Processing).

![App screenshot](docs/screenshot.png)

---

## Overview

The system ingests structured O\*NET occupation data (skills, knowledge, abilities, work activities, education requirements), generates sentence embeddings, and stores them in a local vector database. At query time, it retrieves the most relevant occupations and passes them as grounded context to Claude Haiku, which streams a response token-by-token. Every answer ships with clickable O\*NET citations, and the model is instructed to refuse rather than fabricate when the retrieved context is insufficient.

## Key Features

- **Grounded answers.** Every response is constrained by retrieved O\*NET documents and is required to admit when data is missing rather than hallucinate.
- **Two-stage retrieval.** MMR retrieval (fetch 20, return 5) for diversity, followed by cross-encoder reranking for precision.
- **Conversational memory.** Pronouns and follow-ups are resolved against chat history via a dedicated question-condensing LLM.
- **Streaming UI.** Tokens render live in the Streamlit chat as the model generates them.
- **Source transparency.** Each answer expands to show the exact O\*NET occupations consulted, with direct links to onetonline.org.
- **Evaluated.** A 10-question benchmark measures source hit rate, retrieval relevance, and honest-refusal behavior — see [Evaluation](#evaluation).

## Architecture

```
User Query
    │
    ▼
Question condensing (Claude Haiku, non-streaming)
    │  resolves pronouns and follow-ups against chat history
    ▼
Embed query (all-MiniLM-L6-v2)
    │
    ▼
ChromaDB MMR search  →  top 20 candidate occupation documents
    │
    ▼
Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)  →  top 5 documents
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

A more detailed component-level diagram (Mermaid) is available in [DesignOverview.md](DesignOverview.md).

## Tech Stack

| Layer            | Tool                                                            |
| ---------------- | --------------------------------------------------------------- |
| LLM              | Claude Haiku 4.5 (Anthropic API), streaming                     |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2`                        |
| Vector store     | ChromaDB (local, persistent)                                    |
| Retrieval        | MMR (fetch 20 → return 5) + cross-encoder reranking             |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2`                          |
| RAG orchestration| LangChain (`ConversationalRetrievalChain`)                      |
| UI               | Streamlit                                                       |
| Data             | O\*NET 30.2 Database (U.S. Department of Labor)                 |

## Evaluation

Running `python eval.py` benchmarks the system against a fixed set of 10 career questions. Each question has a hand-labeled expected occupation; the script measures whether it was retrieved, the mean cosine relevance of the retrieved sources, and the model's tendency to refuse honestly when data is missing.

| Metric                  | Result      |
| ----------------------- | ----------- |
| Source hit rate         | 90% (9/10)  |
| Avg. source relevance   | 0.521 cosine|
| Avg. answer length      | 216 words   |
| Honest refusals         | Tracked     |

## Run Locally

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/vlad-kashchuk/onet-career-rag.git
cd onet-career-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Anthropic API key

```bash
cp .env.example .env
# then edit .env and paste your ANTHROPIC_API_KEY
```

Get a key at [console.anthropic.com](https://console.anthropic.com/).

### 4. Launch the app

The vector store is committed to the repo, so you can skip ingestion and run the app immediately:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### (Optional) Rebuild the vector store from scratch

If you want to re-ingest the O\*NET data yourself (e.g. to upgrade to a newer release):

1. Download the latest O\*NET database (tab-delimited text) from [onetcenter.org/database.html](https://www.onetcenter.org/database.html).
2. Extract these files into `data/raw/`:
   - `Occupation Data.txt`
   - `Skills.txt`
   - `Knowledge.txt`
   - `Abilities.txt`
   - `Work Activities.txt`
   - `Education, Training, and Experience.txt`
3. Re-run ingestion:
   ```bash
   python ingest.py
   ```

This takes a few minutes on first run while sentence-transformers downloads its model weights.

## Deployment (Hugging Face Spaces)

This repo is set up to deploy directly to [Hugging Face Spaces](https://huggingface.co/spaces) via the Docker SDK. The included `Dockerfile` and the YAML header at the top of this README configure the Space.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space). Choose **Docker** as the SDK (Blank template).
2. In the Space's **Settings → Variables and secrets**, add a secret named `ANTHROPIC_API_KEY` with your Anthropic key.
3. Push the repo to the Space's git remote (`git push hf main`).
4. The Space builds the Docker image automatically. The committed `data/chroma_db/` directory means the app comes up without any ingestion step.

The free-tier hardware (CPU basic, 16 GB RAM) is sufficient for this app, though the first build pulls dependencies and downloads sentence-transformer model weights (~160 MB), which can take a few minutes.

## Example Questions

- What skills do I need to become a software developer?
- What does a registered nurse do day to day?
- What education do I need to be a mechanical engineer?
- What abilities are important for a graphic designer?
- Which careers are growing the fastest?
- Compare a UX designer and a graphic designer.

## Project Structure

```
.
├── app.py              # Streamlit chat UI
├── rag.py              # RAG chain: retriever + reranker + LLM
├── ingest.py           # Data ingestion and embedding pipeline
├── eval.py             # Evaluation script (source hit rate, relevance, refusal)
├── Dockerfile          # Container image for Hugging Face Spaces deployment
├── requirements.txt    # Pinned Python dependencies
├── .env.example        # API key template
├── .streamlit/         # Streamlit toolbar config
├── DesignOverview.md   # Detailed architecture writeup (Mermaid diagram)
├── report.md           # Full project report
└── data/
    ├── raw/            # O*NET source .txt files (gitignored)
    └── chroma_db/      # Persistent vector store (committed)
```

## Data Source

This project uses the O\*NET 30.2 database, maintained by the U.S. Department of Labor / Employment and Training Administration and developed by the National Center for O\*NET Development. O\*NET data is used under the [O\*NET Data License](https://www.onetcenter.org/license_db.html). See [onetcenter.org](https://www.onetcenter.org/) for details.

## License

Code released under the MIT License. See [LICENSE](LICENSE) for details.

---

_Built as the CS3390R Natural Language Processing final project. See [report.md](report.md) for the full writeup and [DesignOverview.md](DesignOverview.md) for the detailed architecture._
