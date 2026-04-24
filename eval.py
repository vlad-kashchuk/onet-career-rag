"""
eval.py
Evaluates the RAG chain on a fixed question set.

Metrics:
  - Source Relevance: cosine similarity between query embedding and each retrieved doc
  - Answer Length:    proxy for response completeness
  - Has N/A:         whether the answer admitted missing data (honest refusal)

Usage:
    python eval.py
"""

import textwrap
from sentence_transformers import SentenceTransformer, util
from rag import load_vectorstore, build_chain, ask

EVAL_QUESTIONS = [
    ("What skills do I need to become a software developer?",       "Software Developers"),
    ("What does a registered nurse do day to day?",                 "Registered Nurses"),
    ("What education do I need to be a mechanical engineer?",       "Mechanical Engineers"),
    ("What are the top knowledge areas for a data scientist?",      "Data Scientists"),
    ("What abilities are important for a graphic designer?",        "Graphic Designers"),
    ("What does an electrician do?",                                "Electricians"),
    ("What skills are needed to become a lawyer?",                  "Lawyers"),
    ("What work activities does a teacher perform?",                "Elementary School Teachers"),
    ("What is the difference between a civil and structural engineer?", "Civil Engineers"),
    ("What careers involve working with animals?",                  "Animal Control Workers"),
]

SEP = "-" * 70


def score_source_relevance(query: str, sources: list, model: SentenceTransformer) -> float:
    """Mean cosine similarity between query and source titles."""
    if not sources:
        return 0.0
    q_emb = model.encode(query, convert_to_tensor=True)
    s_embs = model.encode([s["title"] for s in sources], convert_to_tensor=True)
    scores = util.cos_sim(q_emb, s_embs)[0]
    return float(scores.mean())


def expected_source_hit(sources: list, expected_title: str) -> bool:
    """Check whether the expected occupation appears in retrieved sources."""
    expected_lower = expected_title.lower()
    return any(expected_lower in s["title"].lower() for s in sources)


def main():
    print("Loading vector store and chain...")
    vs = load_vectorstore()
    chain = build_chain(vs)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\n{'='*70}")
    print("RAG Evaluation — 10 Questions")
    print(f"{'='*70}\n")

    results = []
    for question, expected in EVAL_QUESTIONS:
        result = ask(chain, question)
        answer = result["answer"]
        sources = result["sources"]

        relevance = score_source_relevance(question, sources, embed_model)
        hit = expected_source_hit(sources, expected)
        has_na = "n/a" in answer.lower() or "not in the" in answer.lower()
        answer_words = len(answer.split())

        results.append({
            "question": question,
            "expected": expected,
            "hit": hit,
            "relevance": relevance,
            "words": answer_words,
            "honest_refusal": has_na,
            "answer": answer,
            "sources": sources,
        })

        print(f"Q: {question}")
        print(f"   Expected source : {expected}")
        print(f"   Source hit      : {'YES' if hit else 'NO'}")
        print(f"   Avg relevance   : {relevance:.3f}")
        print(f"   Answer length   : {answer_words} words")
        print(f"   Honest refusal  : {'yes' if has_na else 'no'}")
        print(f"   Retrieved       : {', '.join(s['title'] for s in sources[:3])}")
        print(f"   Answer preview  : {textwrap.shorten(answer, width=120)}")
        print(SEP)

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(results)
    hit_rate = sum(r["hit"] for r in results) / n
    avg_relevance = sum(r["relevance"] for r in results) / n
    avg_words = sum(r["words"] for r in results) / n
    honest_count = sum(r["honest_refusal"] for r in results)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Questions evaluated : {n}")
    print(f"  Source hit rate     : {hit_rate:.0%}  ({sum(r['hit'] for r in results)}/{n} expected occupations retrieved)")
    print(f"  Avg source relevance: {avg_relevance:.3f}  (cosine sim, higher = better)")
    print(f"  Avg answer length   : {avg_words:.0f} words")
    print(f"  Honest refusals     : {honest_count}  (answers that admitted missing data)")
    print()


if __name__ == "__main__":
    main()
