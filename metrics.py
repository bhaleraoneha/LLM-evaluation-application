"""4 evaluation metrics, all computed automatically with an LLM-as-judge.
Every metric returns (score between 0 and 1, short reason).

1. faithfulness        - Is every claim in the answer supported by the retrieved context? (hallucination check)
2. answer_relevancy    - Does the answer actually address the question that was asked?
3. context_precision   - Did the retriever put the useful chunks at the top? (retrieval quality)
4. answer_correctness  - Does the answer match the reference (ground-truth) answer?
"""
import llm

NOT_IN_DOCS = "NOT_IN_DOCS"


def _judge(prompt):
    out = llm.chat([{"role": "system", "content": "You are a strict, fair evaluator. Reply with JSON only."},
                    {"role": "user", "content": prompt}],
                   model=llm.JUDGE_MODEL, temperature=0.0, max_tokens=700, json_mode=True)
    return llm.parse_json(out)


def faithfulness(question, answer, contexts):
    """Judge splits the answer into claims and checks each one against the context.
    score = supported claims / total claims.  (Same idea as RAGAS faithfulness.)"""
    ctx = "\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(contexts))
    d = _judge(f"""Break the ANSWER into short factual claims. For each claim, say whether it is
directly supported by the CONTEXT (true) or not (false). Ignore phrases like "I don't know".
CONTEXT:
{ctx}

ANSWER: {answer}

Reply as JSON: {{"claims":[{{"claim":"...","supported":true}}], "reason":"one short sentence"}}""")
    claims = d.get("claims", [])
    if not claims:                       # e.g. "I don't know" -> nothing unsupported was said
        return 1.0, d.get("reason", "no factual claims made")
    ok = sum(1 for c in claims if c.get("supported") is True)
    bad = [c.get("claim") for c in claims if c.get("supported") is not True]
    return ok / len(claims), (f"unsupported: {bad}" if bad else "all claims supported")


def answer_relevancy(question, answer):
    """Judge scores 0-1 how directly the answer addresses the question (correctness is NOT judged here)."""
    d = _judge(f"""Rate how well the ANSWER addresses the QUESTION, from 0.0 to 1.0.
1.0 = directly and completely on-topic. 0.5 = partly on-topic or missing parts. 0.0 = off-topic.
An honest "I don't know / not in the documents" counts as relevant (1.0) only if it clearly refers to the question.
Do NOT judge whether the facts are true.
QUESTION: {question}
ANSWER: {answer}
Reply as JSON: {{"score":0.0,"reason":"one short sentence"}}""")
    return _clip(d.get("score")), d.get("reason", "")


def context_precision(question, reference, contexts):
    """Judge marks each retrieved chunk useful/not useful. Score = average precision, so useful chunks
    ranked FIRST score higher. Returns None for unanswerable questions (nothing useful exists)."""
    if reference.startswith(NOT_IN_DOCS):
        return None, "skipped (question is not answerable from the documents)"
    ctx = "\n".join(f"[Chunk {i+1}] {c}" for i, c in enumerate(contexts))
    d = _judge(f"""For each retrieved chunk, decide if it contains information that helps produce the REFERENCE ANSWER
to the QUESTION. Output one true/false per chunk, in order ({len(contexts)} chunks).
QUESTION: {question}
REFERENCE ANSWER: {reference}
{ctx}
Reply as JSON: {{"relevant":[true,false,...], "reason":"one short sentence"}}""")
    rel = [bool(x) for x in d.get("relevant", [])][: len(contexts)]
    rel += [False] * (len(contexts) - len(rel))
    if not any(rel):
        return 0.0, "no retrieved chunk was useful. " + d.get("reason", "")
    hits, total = 0, 0.0
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            total += hits / k
    return total / hits, f"relevant chunks by rank: {rel}. " + d.get("reason", "")


def answer_correctness(question, answer, reference):
    """Judge compares answer to the reference. 1.0 = same meaning, 0.5 = partly right, 0.0 = wrong."""
    extra = ""
    if reference.startswith(NOT_IN_DOCS):
        extra = "The documents do NOT contain the answer, so the ONLY correct behaviour is to say it doesn't know. Making up an answer = 0.0."
    d = _judge(f"""Compare the ANSWER with the REFERENCE for the QUESTION. Score 0.0 to 1.0:
1.0 = same meaning / all key points, 0.5 = partly correct or missing key points, 0.0 = wrong or contradicts.
{extra}
QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}
Reply as JSON: {{"score":0.0,"reason":"one short sentence"}}""")
    return _clip(d.get("score")), d.get("reason", "")


def _clip(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0
