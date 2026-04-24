# Presentation Q&A Prep

---

## General / Concept

**Q: What is RAG and why did you use it instead of just prompting an LLM?**
A: RAG stands for Retrieval-Augmented Generation. Instead of relying on whatever the LLM learned during training, you first retrieve relevant documents from a knowledge base and inject them into the prompt. I used it because O*NET has structured, authoritative career data that a general-purpose LLM either doesn't know or would hallucinate. RAG keeps answers grounded — every response is backed by actual O*NET records, and the sources are shown to the user.

**Q: Why not just fine-tune a model on O*NET instead?**
A: Fine-tuning bakes knowledge into the model's weights, which is expensive, slow to update, and hard to audit. With RAG, the knowledge lives in the vector store — if O*NET releases a new version, I just re-run `ingest.py`. Also fine-tuning doesn't prevent hallucination, it just shifts what the model hallucinates about. RAG with source citations gives the user a way to verify the answer.

**Q: Why did you choose O*NET as your data source?**
A: It's maintained by the U.S. Department of Labor, covers 923 occupations, is publicly available under a Creative Commons license, and has structured fields — skills, knowledge, abilities, work activities, and education — which map naturally to the kinds of questions users ask about careers.

---

## Technical — Embeddings & Retrieval

**Q: What embedding model did you use and why?**
A: `all-MiniLM-L6-v2` from HuggingFace sentence-transformers. It produces 384-dimensional dense vectors, runs fully locally with no API key, and has good benchmark performance on semantic similarity tasks. It's fast enough to embed all 923 occupation documents in a couple of minutes on a laptop.

**Q: What is MMR and why is it better than plain cosine similarity?**
A: MMR stands for Maximal Marginal Relevance. Plain cosine similarity just returns the top-k most similar documents, which often means you get 5 near-identical results for specific queries. MMR trades off between relevance and diversity — it penalizes documents that are too similar to ones already selected. In practice this means the 5 retrieved occupations cover different angles of the query rather than all being minor variants of the same job.

**Q: What is a cross-encoder reranker and how is it different from the bi-encoder you use for retrieval?**
A: The bi-encoder (sentence-transformers) encodes the query and each document independently into vectors, then compares them with cosine similarity. It's fast but approximate. A cross-encoder takes the query and a document together as a single input and produces a relevance score — it can attend to interactions between the two. That makes it more accurate but too slow to run over the entire corpus. So I use the bi-encoder to cheaply narrow to 20 candidates, then the cross-encoder to precisely rerank those 20 and return the top 5.

**Q: Why fetch 20 documents but only return 5?**
A: The bi-encoder retrieval is fast but imprecise — fetching more candidates gives the reranker more to work with. 20 is a common sweet spot. Returning only 5 to the LLM keeps the prompt size manageable and avoids flooding it with marginally relevant context.

**Q: Why did you use ChromaDB instead of FAISS or Pinecone?**
A: ChromaDB runs locally, persists to disk automatically, and has a clean LangChain integration. FAISS is slightly faster but doesn't persist natively and requires more setup. Pinecone is a managed cloud service — overkill for a local project and costs money. For 923 documents ChromaDB is more than fast enough.

---

## Technical — Generation & LangChain

**Q: Why Claude Haiku specifically?**
A: It's the fastest and cheapest model in the Claude family, which matters when every user message triggers an API call. It's also capable enough for factual Q&A over short retrieved contexts. For a production system I might use Sonnet, but for a class project Haiku keeps costs low.

**Q: Why two separate LLM instances — one streaming and one not?**
A: LangChain's `ConversationalRetrievalChain` makes two LLM calls internally: one to condense the user's follow-up question into a standalone query (using chat history), and one to generate the final answer. If both stream, the condensed question text leaks into the UI before the real answer starts. Using a non-streaming LLM for the condensing step means only the actual answer streams to the user.

**Q: What does `ConversationBufferMemory` do?**
A: It stores the full conversation history — every user message and assistant response — and injects it into each new prompt. This lets the user ask follow-up questions like "what about the salary?" without repeating the context. The trade-off is that the memory grows with every turn and will eventually hit the model's token limit in very long sessions.

**Q: What happens when you click "Clear Chat"?**
A: It clears `st.session_state.messages` (the display) AND calls `chain.memory.clear()` (the LangChain memory). Both need to be reset — otherwise the chain would still remember the old conversation even though the screen looks blank.

---

## Technical — Data & Ingestion

**Q: Why one document per occupation instead of chunking?**
A: Each occupation in O*NET is naturally self-contained — it has a defined set of fields that together describe one job. Chunking would split that into fragments (e.g., skills separate from work activities), which makes retrieval less coherent. A single document per occupation means the LLM always gets the full picture of a job when that occupation is retrieved.

**Q: How did you fix the education field?**
A: The `Education, Training, and Experience.txt` file stores multiple rows per occupation — one per education level — with a `Data Value` column representing the percentage of workers who require that level. The original code just grabbed the first row, which was often not the most relevant level. The fix filters for the "Required Level of Education" element and selects the category with the highest data value — the most commonly required level for that occupation.

**Q: What if a user asks about an occupation that isn't in O*NET?**
A: The system will retrieve the most semantically similar occupations it can find and answer from those. It may not find an exact match. In the evaluation, "structural engineer" is a good example — O*NET doesn't have that specific entry, so the system retrieved "Civil Engineers" and honestly told the user it couldn't make a complete comparison.

---

## Results & Evaluation

**Q: How did you evaluate the system?**
A: I wrote `eval.py` which runs 10 representative questions through the chain and measures three things: source hit rate (whether the expected occupation appears in the retrieved sources), average cosine similarity between the query and retrieved source titles, and average answer length as a proxy for completeness.

**Q: What was your source hit rate and what does it mean?**
A: 90% — 9 out of 10 test questions retrieved the expected occupation in the top 5 sources. The one miss was "teacher," which retrieved Preschool and Postsecondary teachers instead of Elementary School Teachers. It's still a relevant answer, just not the exact occupation I was testing for.

**Q: Why not use RAGAS or another automated evaluation framework?**
A: RAGAS is the standard tool for this, and it would measure faithfulness and answer relevance more rigorously. The main reason I didn't use it is that RAGAS needs a reference answer for each question, and O*NET doesn't come with a pre-built Q&A benchmark. Building one manually for 10 questions is doable but I wanted to focus on metrics I could compute objectively from the data itself.

**Q: Your system says it doesn't know about salaries — is that a failure?**
A: No, it's the correct behavior. O*NET doesn't include wage data. A hallucinated salary would be worse than an honest "I don't have that data." The system telling the user to check BLS is more useful than making up a number.

---

## Architecture / Design Decisions

**Q: Why Streamlit instead of a proper web framework like Flask or React?**
A: For a class project, Streamlit lets you build a functional, demo-ready UI in a single Python file without frontend knowledge. It has built-in chat components, session state, and caching that are exactly what a chatbot needs. For a production product I'd use something more flexible, but for a demo it's the right tool.

**Q: How does the sidebar question injection work?**
A: Sidebar buttons set a `pending_question` key in Streamlit's session state and trigger a rerun. On the next render cycle, the main chat area reads and pops that key, then calls `process_question()` in the main context. This avoids a bug where calling the question handler inside the sidebar block renders the response inside the sidebar before re-rendering it in the main chat.

**Q: How would you scale this to a larger corpus?**
A: A few changes: switch ChromaDB for a server-mode or cloud vector database (Qdrant, Weaviate, Pinecone), switch `ConversationBufferMemory` for a sliding-window or summary memory, and add async handling so multiple users don't block each other. The retrieval pipeline (MMR + reranking) scales well — only the top 20 candidates go through the cross-encoder regardless of corpus size.

---

## Reflection

**Q: What would you do differently if you started over?**
A: A few things. I would add BLS salary data from the start since it's the most common gap users hit. I would also build a small labeled evaluation set of 50 questions with reference answers so I could measure faithfulness quantitatively. And I would use LCEL (LangChain Expression Language) instead of the legacy `ConversationalRetrievalChain` — it's more transparent and streaming works more cleanly.

**Q: What was the hardest part?**
A: Getting the education field to parse correctly. The O*NET education file has a non-obvious schema — it's not just one row per occupation, it's a distribution of education levels per occupation. It took looking at the raw file to understand that I needed to aggregate by data value rather than just grab the first row.

**Q: What NLP concepts from the course does this project use?**
A: Tokenization and embedding (sentence-transformers encodes text into dense vectors), transformers (both the embedding model and Claude are transformer-based), semantic similarity (cosine similarity for retrieval), and generation (Claude Haiku generates the final answer). The reranker also uses a transformer cross-encoder, which is a classic NLP model architecture for relevance scoring.
