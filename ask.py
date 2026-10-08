"""Chat with the RAG app yourself (to see what it does before evaluating)."""
from rag_app import RAGApp
app = RAGApp()
print("Ask me about machine learning / LLMs. Type 'quit' to stop.")
while True:
    q = input("\nYou: ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    ans, ctx = app.answer(q)
    print("Bot:", ans)
    print("   [retrieved:", " | ".join(c.split(":")[0] for c in ctx), "]")
