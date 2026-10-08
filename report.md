# Evaluation Report

- App: RAG Q&A bot (TF-IDF retrieval, top_k=3) | Generator: `openai/gpt-oss-20b` | Judge: `openai/gpt-oss-120b`
- Test questions: 20

## Average scores

| Metric | Average |
|---|---|
| faithfulness | 0.90 |
| answer_relevancy | 0.99 |
| context_precision | 0.95 |
| answer_correctness | 1.00 |

## Scores by question type

| category     |   faithfulness |   answer_relevancy |   context_precision |   answer_correctness |
|:-------------|---------------:|-------------------:|--------------------:|---------------------:|
| easy         |              1 |               1    |                0.94 |                    1 |
| hard         |              1 |               1    |                0.9  |                    1 |
| medium       |              1 |               0.97 |                1    |                    1 |
| unanswerable |              0 |               1    |              nan    |                    1 |

## Worst 3 failure cases

### Q19 (unanswerable): How many parameters does GPT-4 have?
- **Reference:** NOT_IN_DOCS - the documents do not state this, so the app should say it does not know.
- **App answer:** I don't know based on the provided documents.
- **Retrieved:** Large Language Models: A large language model (LLM) is a tra... || Evaluating Classifiers: Accuracy is the share of predictions... || Fine-tuning: Fine-tuning continues the training of an existi...
- **Scores:** faithfulness=0.00, relevancy=1.00, context_precision=n/a, correctness=1.00
- **Judge said:** answer correctly states lack of knowledge as reference indicates | unsupported: ["I don't know based on the provided documents."] | skipped (question is not answerable from the documents)
- **Auto-diagnosis:** HALLUCINATION: the answer contains claims that are not in the retrieved context.
- **YOUR ANALYSIS (write in your own words):** ...

### Q18 (unanswerable): Who is the CEO of OpenAI?
- **Reference:** NOT_IN_DOCS - the documents do not contain this information, so the app should say it does not know.
- **App answer:** I don't know based on the provided documents.
- **Retrieved:** Fine-tuning: Fine-tuning continues the training of an existi... || Evaluating Classifiers: Accuracy is the share of predictions... || Chunking: Before documents are stored for RAG they are split...
- **Scores:** faithfulness=0.00, relevancy=1.00, context_precision=n/a, correctness=1.00
- **Judge said:** The answer correctly states it does not know, matching the reference. | unsupported: ["I don't know based on the provided documents."] | skipped (question is not answerable from the documents)
- **Auto-diagnosis:** HALLUCINATION: the answer contains claims that are not in the retrieved context.
- **YOUR ANALYSIS (write in your own words):** ...

### Q8 (easy): What is a token in the context of LLMs?
- **Reference:** A token is a piece of text such as a whole word, part of a word or a single character into which text is split before an LLM reads it.
- **App answer:** A token is a unit of text that an LLM processes—this can be a whole word, part of a word, or even a single character.
- **Retrieved:** Large Language Models: A large language model (LLM) is a tra... || Tokens: LLMs do not read raw words. Text is split into token... || Chunking: Before documents are stored for RAG they are split...
- **Scores:** faithfulness=1.00, relevancy=1.00, context_precision=0.5, correctness=1.00
- **Judge said:** Answer matches reference meaning and includes all key points | all claims supported | relevant chunks by rank: [False, True, False]. Only chunk 2 defines tokens as text pieces, the others do not
- **Auto-diagnosis:** No clear single cause - read the judge reasons.
- **YOUR ANALYSIS (write in your own words):** ...
