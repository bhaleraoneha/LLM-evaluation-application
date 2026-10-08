"""Runs the whole test set through the RAG app, scores every answer automatically,
then saves results + a report with the worst failure cases.

Run:  python evaluate.py            (all questions)
      python evaluate.py --limit 5  (quick test with 5 questions)"""
import json, argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rag_app import RAGApp
import metrics as M
import llm

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]


def diagnose(row):
    """Simple rule-based hint about WHY a question scored badly (use it to write your own analysis)."""
    hints = []
    cp, f, c = row["context_precision"], row["faithfulness"], row["answer_correctness"]
    if cp is not None and cp == cp and cp < 0.5:
        hints.append("RETRIEVAL problem: the retriever (TF-IDF keyword matching) fetched mostly useless chunks.")
    if f < 0.7:
        hints.append("HALLUCINATION: the answer contains claims that are not in the retrieved context.")
    if c < 0.6 and (cp is None or cp != cp or cp >= 0.5) and f >= 0.7:
        hints.append("GENERATION problem: good context was retrieved but the answer is incomplete/wrong.")
    if row["category"] == "unanswerable" and c < 0.6:
        hints.append("The app failed to say 'I don't know' for a question outside its documents.")
    return " ".join(hints) or "No clear single cause - read the judge reasons."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tests = json.load(open("data/test_set.json", encoding="utf-8"))
    if args.limit:
        tests = tests[: args.limit]
    app = RAGApp()
    os.makedirs("results", exist_ok=True)
    rows = []

    for t in tests:
        print(f"[{t['id']}/{len(tests)}] {t['question']}")
        answer, contexts = app.answer(t["question"])
        f, f_why = M.faithfulness(t["question"], answer, contexts)
        r, r_why = M.answer_relevancy(t["question"], answer)
        p, p_why = M.context_precision(t["question"], t["reference"], contexts)
        c, c_why = M.answer_correctness(t["question"], answer, t["reference"])
        print(f"    faith={f:.2f} relevancy={r:.2f} ctx_prec={'n/a' if p is None else f'{p:.2f}'} correct={c:.2f}")
        rows.append({"id": t["id"], "category": t["category"], "question": t["question"],
                     "reference": t["reference"], "answer": answer,
                     "retrieved_chunks": " || ".join(x[:60] + "..." for x in contexts),
                     "faithfulness": f, "answer_relevancy": r, "context_precision": p, "answer_correctness": c,
                     "faithfulness_why": f_why, "relevancy_why": r_why,
                     "context_precision_why": p_why, "correctness_why": c_why})

    df = pd.DataFrame(rows)
    df.to_csv("results/results.csv", index=False)
    df.to_json("results/results.json", orient="records", indent=2, force_ascii=False)

    # ---- aggregate ----
    overall = df[METRICS].mean(numeric_only=True)
    by_cat = df.groupby("category")[METRICS].mean().round(2)
    df["overall"] = df[METRICS].mean(axis=1, skipna=True)
    worst = df.sort_values("overall").head(3)

    # ---- chart ----
    ax = overall.plot(kind="bar", ylim=(0, 1), color=["#4c78a8", "#f58518", "#54a24b", "#e45756"], figsize=(7, 4))
    ax.set_title(f"Average scores over {len(df)} test questions")
    ax.set_ylabel("score (0-1)")
    plt.xticks(rotation=20)
    for i, v in enumerate(overall):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig("results/scores.png", dpi=120)

    # ---- report ----
    L = ["# Evaluation Report\n",
         f"- App: RAG Q&A bot (TF-IDF retrieval, top_k={app.top_k}) | Generator: `{llm.GEN_MODEL}` | Judge: `{llm.JUDGE_MODEL}`",
         f"- Test questions: {len(df)}\n", "## Average scores\n",
         "| Metric | Average |", "|---|---|"]
    L += [f"| {m} | {overall[m]:.2f} |" for m in METRICS]
    L += ["\n## Scores by question type\n", by_cat.to_markdown(), "\n## Worst 3 failure cases\n"]
    for _, w in worst.iterrows():
        L += [f"### Q{w['id']} ({w['category']}): {w['question']}",
              f"- **Reference:** {w['reference']}", f"- **App answer:** {w['answer']}",
              f"- **Retrieved:** {w['retrieved_chunks']}",
              f"- **Scores:** faithfulness={w['faithfulness']:.2f}, relevancy={w['answer_relevancy']:.2f}, "
              f"context_precision={'n/a' if pd.isna(w['context_precision']) else round(w['context_precision'], 2)}, "
              f"correctness={w['answer_correctness']:.2f}",
              f"- **Judge said:** {w['correctness_why']} | {w['faithfulness_why']} | {w['context_precision_why']}",
              f"- **Auto-diagnosis:** {diagnose(w)}",
              "- **YOUR ANALYSIS (write in your own words):** ...\n"]
    open("results/report.md", "w", encoding="utf-8").write("\n".join(L))

    print("\n=== AVERAGES ===")
    print(overall.round(2).to_string())
    print("\nSaved: results/results.csv, results/results.json, results/scores.png, results/report.md")


if __name__ == "__main__":
    main()
