"""The LLM application we are going to evaluate: a small RAG question-answering bot.
Retrieval = TF-IDF (no big downloads).  Generation = free Groq LLM."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import llm

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context given. "
    "If the context does not contain the answer, reply exactly: "
    "\"I don't know based on the provided documents.\" Keep the answer short (1-3 sentences)."
)


def load_chunks(path="data/knowledge_base.txt"):
    """Each '# Title' section of the file is one chunk."""
    text = open(path, encoding="utf-8").read()
    parts = [p.strip() for p in re.split(r"(?m)^# ", text) if p.strip()]
    return [{"title": p.split("\n", 1)[0], "text": p.split("\n", 1)[1].strip()} for p in parts]


class RAGApp:
    def __init__(self, kb_path="data/knowledge_base.txt", top_k=3):
        self.chunks = load_chunks(kb_path)
        self.top_k = top_k
        docs = [c["title"] + ". " + c["text"] for c in self.chunks]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(docs)

    def retrieve(self, question):
        sims = cosine_similarity(self.vec.transform([question]), self.matrix)[0]
        best = sims.argsort()[::-1][: self.top_k]
        return [self.chunks[i]["title"] + ": " + self.chunks[i]["text"] for i in best]

    def answer(self, question):
        contexts = self.retrieve(question)
        ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        reply = llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}],
            model=llm.GEN_MODEL, temperature=0.0, max_tokens=250)
        return reply.strip(), contexts
