# LLM Evaluation Framework (RAG Q&A bot) - Valentius Kryptix Task

**What this project does (in simple words)**
1. `rag_app.py` = a small Q&A bot. It looks up the best paragraphs from `data/knowledge_base.txt` and asks a free LLM to answer using only those paragraphs (this is called RAG).
2. `data/test_set.json` = 20 test questions (easy, medium, hard, and 2 "trick" questions whose answer is NOT in the documents), each with a correct reference answer.
3. `evaluate.py` = runs all 20 questions through the bot and **automatically scores** each answer with 4 metrics (using a second LLM as "judge").
4. It saves a CSV, a chart and a `report.md` that lists the 3 worst cases with reasons.

## The 4 metrics (score 0 to 1, higher = better)
| Metric | Question it answers | How it is computed |
|---|---|---|
| Faithfulness | Did the bot make things up? | Judge splits the answer into claims and checks each one against the retrieved text. Score = supported claims / all claims |
| Answer relevancy | Does the answer address the question? | Judge gives 0-1 for how on-topic the answer is |
| Context precision | Did retrieval find the right paragraphs, ranked first? | Judge marks each retrieved chunk useful / not useful; score = average precision (useful chunks at top = higher) |
| Answer correctness | Does it match the reference answer? | Judge compares answer to reference (1 = same meaning, 0.5 = partly, 0 = wrong) |

## Setup - step by step (Windows)
1. Install **Python 3.10+** from python.org (tick "Add Python to PATH").
2. Unzip this folder. Open **Command Prompt** inside the folder (type `cmd` in the folder's address bar and press Enter).
3. Create a virtual environment and install packages:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   (Mac/Linux: `source venv/bin/activate`)
4. Get your **free** API key: go to https://console.groq.com/keys -> sign up -> "Create API Key" -> copy it. No credit card needed.
5. Copy `.env.example` to `.env` (`copy .env.example .env`), open `.env` in Notepad, and paste your key after `GROQ_API_KEY=`.
6. (Optional) Try the bot yourself: `python ask.py`
7. Test quickly with 3 questions: `python evaluate.py --limit 3`
8. Run the full evaluation: `python evaluate.py` (takes about 8-12 minutes because of free-tier speed limits).
9. Open the `results` folder: `report.md`, `results.csv`, `scores.png`.

## Problems?
- **"model decommissioned / not found"** -> Groq sometimes retires models. Open https://console.groq.com/docs/models, pick a current model name, and put it in `.env` as `GEN_MODEL` / `JUDGE_MODEL`.
- **Rate limit (429) messages** -> normal on the free plan, the script waits and retries. If it is too slow, set `JUDGE_MODEL=openai/gpt-oss-20b` in `.env`.
- **Test without internet/key** -> `set MOCK=1` (Windows) or `export MOCK=1` (Mac/Linux) then run `python evaluate.py`. Scores are random in this mode - only for checking the code runs.

## How to improve it (good for your write-up)
Change `top_k` in `rag_app.py` (3 -> 5), edit the prompt, or add your own questions to `test_set.json`, run again and compare the scores before/after.

## Files
```
rag_app.py     the app being evaluated (retrieve + generate)
llm.py         free Groq API helper (retry + rate-limit handling)
metrics.py     the 4 metrics (LLM-as-judge)
evaluate.py    runs everything, aggregates, writes report
ask.py         chat with the bot manually
data/          knowledge_base.txt, test_set.json
results/       created after you run evaluate.py
WRITEUP_TEMPLATE.md   fill this and submit on the Interns Hub
```
