"""
rag.py
RAG chain: ChromaDB retriever + Claude API as LLM via LangChain.

Improvements over v1:
- MMR retrieval (fetch_k=20, return k=5) reduces redundant results
- Optional cross-encoder reranking for higher precision
- Streaming enabled on the answer LLM; condense step uses a separate non-streaming LLM
- ask() returns richer source metadata (title + onet_code)
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

load_dotenv()

CHROMA_DIR   = "data/chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Set to False to skip reranking (faster cold start, slightly lower precision)
USE_RERANKING = True

SYSTEM_PROMPT = """You are a helpful job and career advisor. Use the provided career information
to answer questions accurately. If the answer is not in the context, say so honestly rather
than making something up.

Context from career database:
{context}

Chat history:
{chat_history}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT,
)


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def _build_retriever(vectorstore: Chroma) -> BaseRetriever:
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )

    if not USE_RERANKING:
        return base_retriever

    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        print("  Loading cross-encoder reranker (first run downloads ~80 MB)...")
        cross_encoder = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
    except Exception as e:
        print(f"  [warn] Reranking unavailable ({e}), falling back to MMR only.")
        return base_retriever


def build_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    # Streaming LLM for answer generation
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.3,
        max_tokens=1024,
        streaming=True,
    )

    # Non-streaming LLM for question condensing (avoids polluting stream output)
    llm_condense = ChatAnthropic(
        model=CLAUDE_MODEL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
        max_tokens=256,
        streaming=False,
    )

    retriever = _build_retriever(vectorstore)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=llm_condense,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )

    return chain


def ask(
    chain: ConversationalRetrievalChain,
    question: str,
    callbacks: Optional[List] = None,
) -> Dict[str, Any]:
    config = {"callbacks": callbacks} if callbacks else {}
    result = chain.invoke({"question": question}, config=config)
    sources = [
        {"title": doc.metadata.get("title", "Unknown"), "code": doc.metadata.get("onet_code", "")}
        for doc in result["source_documents"]
    ]
    return {"answer": result["answer"], "sources": sources}


if __name__ == "__main__":
    print("Loading vector store...")
    vs = load_vectorstore()
    chain = build_chain(vs)

    print("Career Assistant ready. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        result = ask(chain, q)
        print(f"\nAssistant: {result['answer']}")
        titles = ", ".join(s["title"] for s in result["sources"][:3])
        print(f"Sources: {titles}\n")
