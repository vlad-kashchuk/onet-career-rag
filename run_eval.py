"""Quick evaluation script — runs 10 questions through the RAG chain and prints results."""

from rag import load_vectorstore, build_chain, ask

questions = [
    "What skills do I need to become a software developer?",
    "What does a registered nurse do day to day?",
    "What education do I need to be a mechanical engineer?",
    "What are the top knowledge areas for a data scientist?",
    "What abilities are important for a graphic designer?",
    "What does an electrician do?",
    "What skills are needed to become a lawyer?",
    "What work activities does a teacher perform?",
    "What is the difference between a civil engineer and a structural engineer?",
    "What careers involve working with animals?",
]

vs = load_vectorstore()
chain = build_chain(vs)

for q in questions:
    result = ask(chain, q)
    sources = result["sources"][:3]
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:300]}...")
    print(f"Sources: {', '.join(sources)}")
    print("-" * 60)