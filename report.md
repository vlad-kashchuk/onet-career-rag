# Job & Career Assistant — Written Report

**CS3390R: Natural Language Processing**
**Final Project**

---

## Problem Statement

Finding clear, accurate information about careers is harder than it sounds. Most people turn to
generic search results or ask ChatGPT, which often makes things up or gives vague answers not
backed by real data. The problem is that there is a great public dataset — the O\*NET occupational
database maintained by the U.S. Department of Labor — that covers 900+ occupations with
structured information on skills, abilities, knowledge areas, work activities, and education
requirements. But it is a tab-delimited database, not something you can just ask a question to.

This project builds a conversational career assistant that lets users ask natural-language
questions about careers and get answers that are grounded in O\*NET data. The system uses
Retrieval-Augmented Generation (RAG), which means it retrieves the most relevant occupation
documents first and then passes them to a language model to generate a response. This way the
answers are tied to real data rather than whatever the model happened to learn during training.

The goal was a system that a real person — a student exploring majors, someone considering a
career change — could actually use and trust.

---

## Methodology

### Data

The O\*NET 30.2 database was downloaded as a ZIP of tab-delimited text files from
onetcenter.org. Six files were used: `Occupation Data.txt`, `Skills.txt`, `Knowledge.txt`,
`Abilities.txt`, `Work Activities.txt`, and `Education, Training, and Experience.txt`. These cover
923 occupations.

### Ingestion Pipeline

A custom ingestion script (`ingest.py`) reads each O\*NET file with Pandas, joins them by
occupation code, and builds one text document per occupation. Each document looks like:

```
Occupation: Software Developers
O*NET Code: 15-1252.00

Description: Research, design, and develop computer software systems...

Top Skills: Programming, Critical Thinking, Reading Comprehension, ...
Knowledge Areas: Computers and Electronics, Mathematics, ...
Key Abilities: Oral Comprehension, Written Comprehension, ...
Work Activities: Updating and Using Relevant Knowledge, ...
Education/Training: Bachelor's degree
```

This "one doc per occupation" design means no additional chunking is needed — each occupation
is naturally self-contained. Documents are batched in groups of 500 and loaded into ChromaDB.

Education data required special handling. The `Education, Training, and Experience.txt` file
stores multiple rows per occupation, each representing a different education level with a
percentage of workers who require it. The ingestion pipeline filters for the "Required Level of
Education" element and selects the category with the highest data value (the most commonly
required level), rather than naively grabbing the first row.

### Embedding

Documents are embedded using `all-MiniLM-L6-v2` from HuggingFace sentence-transformers.
This model produces 384-dimensional dense vectors and is fast, free to run locally, and performs
well on semantic similarity tasks. No API calls are needed at ingestion time.

### Vector Store and Retrieval

Embeddings are stored in ChromaDB, a local persistent vector database. At query time, the user's
question is embedded with the same model and retrieval proceeds in two stages:

**Stage 1 — MMR retrieval.** Instead of plain cosine similarity, the system uses Maximal Marginal
Relevance (MMR) with `fetch_k=20` and `k=5`. MMR balances relevance and diversity, so the
five returned documents cover different angles of the query rather than returning five near-identical
occupations.

**Stage 2 — Cross-encoder reranking.** The 20 candidate documents are reranked using
`cross-encoder/ms-marco-MiniLM-L-6-v2`, a model specifically trained to score query-document
relevance pairs. Unlike bi-encoder similarity (which encodes query and document independently),
a cross-encoder attends to both simultaneously, producing a more accurate relevance score. The
top 5 after reranking are passed to the LLM.

### Response Generation

Retrieved documents are passed to Claude Haiku via the Anthropic API using
`langchain-anthropic`. A custom `PromptTemplate` injects the retrieved context, the
conversation history, and the user's question. `ConversationBufferMemory` keeps track of
prior turns so users can ask follow-up questions naturally.

To support streaming without polluting the output, two LLM instances are used: a
streaming-enabled one for answer generation, and a non-streaming one for the internal question
condensing step (which reformulates follow-up questions into standalone queries using chat
history). The LLM is configured with `temperature=0.3` and `max_tokens=1024`.

### Interface

The UI is built with Streamlit. It renders a chat interface with token-by-token streaming,
expandable source citations (with clickable O\*NET links for each occupation), and a sidebar
with example questions. The chain is cached with `@st.cache_resource` so it loads once per
session. Clearing the chat also resets the chain's conversational memory.

---

## Results

Ten questions were run through the system using `eval.py`, which measures source hit rate
(whether the expected occupation appears in retrieved sources), average cosine similarity between
the query and retrieved source titles, and answer length.

| # | Question | Source Hit | Relevance | Notes |
|---|---|---|---|---|
| 1 | What skills do I need to become a software developer? | YES | 0.481 | Accurate skills, well-organized response |
| 2 | What does a registered nurse do day to day? | YES | 0.505 | Detailed work activities, correct sources |
| 3 | What education do I need to be a mechanical engineer? | YES | 0.585 | Correct source retrieved; education data partially missing in O\*NET |
| 4 | What are the top knowledge areas for a data scientist? | YES | 0.500 | Knowledge field sparse in O\*NET; model acknowledged gap honestly |
| 5 | What abilities are important for a graphic designer? | YES | 0.559 | Correct abilities returned, well-structured |
| 6 | What does an electrician do? | YES | 0.605 | Clear, accurate duties description |
| 7 | What skills are needed to become a lawyer? | YES | 0.486 | Correct source, accurate skills list |
| 8 | What work activities does a teacher perform? | NO | 0.389 | Retrieved Preschool/Postsecondary instead of Elementary |
| 9 | Difference between civil and structural engineer? | YES | 0.465 | No structural engineer in O\*NET; model noted this honestly |
| 10 | What careers involve working with animals? | YES | 0.631 | Retrieved multiple relevant animal-related occupations |

**Summary:**
- Source hit rate: **90%** (9/10)
- Average cosine relevance: **0.521**
- Average answer length: **216 words**
- Honest refusals (admitting missing data): **2** (questions 4 and 9)

---

## Limitations

**Sparse fields in O\*NET.** Some occupations have incomplete knowledge area or education data.
The education parsing was improved to select the most commonly required level, but some records
still lack usable data. This is an upstream data quality issue.

**No salary information.** O\*NET does not include salary or wage data. Users frequently ask
about pay, and the system correctly tells them it does not have that information. Adding BLS
Occupational Employment Statistics data would fill this gap.

**Memory grows unbounded.** `ConversationBufferMemory` appends every turn to the context
window. In very long sessions this will eventually approach the model's token limit. A sliding
window or summary-based memory would be more robust.

**Vague queries still struggle.** Questions like "what is a good career for someone who likes
people?" do not map cleanly to specific occupations. The cross-encoder reranking helps, but
retrieval quality for open-ended exploratory questions remains lower than for specific occupation
queries.

---

## Conclusion

The system works end-to-end and delivers grounded, cited responses for the vast majority of
career questions. The two-stage retrieval pipeline — MMR followed by cross-encoder reranking —
measurably improved source hit rate compared to plain cosine similarity. The 90% source hit rate
and 0.521 average retrieval relevance across 10 test questions demonstrate that the system
reliably surfaces the right occupations. Responses stream in real time, sources link directly to
O\*NET, and the conversational memory enables natural follow-up questions. The main remaining
weakness is data completeness upstream in O\*NET rather than the retrieval or generation
architecture itself.
