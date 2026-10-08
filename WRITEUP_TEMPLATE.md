# Evaluation Write-up

## 1. App evaluated
A RAG Q&A bot over a 15-section knowledge base about machine learning and LLMs. Retrieval: TF-IDF (top 3 chunks). Generation: Groq `openai/gpt-oss-20b`.

## 2. Test set
20 questions: 8 easy, 6 medium, 4 hard, 2 unanswerable (answer not in the documents, to test if the bot admits it does not know). Each has a reference answer.

## 3. Metrics and how each is computed (LLM-as-judge, `openai/gpt-oss-120b`, temperature 0)
- Faithfulness: ...
- Answer relevancy: ...
- Context precision: ...
- Answer correctness: ...
(Copy the table from README.md)

## 4. Results
Paste the average table from `results/report.md` and insert `results/scores.png`.
Average faithfulness = __, relevancy = __, context precision = __, correctness = __.
Which question type scored lowest? ____

## 5. Failure analysis (pick 2-3 from the "Worst 3" in report.md)
**Failure 1 - Q__:** question / what the bot said / why it failed (retrieval? hallucination? bad prompt?).
**Failure 2 - Q__:** ...
**Failure 3 - Q__:** ...

## 6. Limitations and improvements
- The judge is also an LLM and can make mistakes (I manually checked a few scores).
- TF-IDF only matches keywords, not meaning -> try embeddings.
- Ideas tried / next: increase top_k, better prompt, re-ranking.
