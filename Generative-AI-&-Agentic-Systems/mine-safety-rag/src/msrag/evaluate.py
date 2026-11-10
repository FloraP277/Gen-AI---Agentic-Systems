"""Evaluate retrieval and answers on a question set, across configurations.

Question file (JSON Lines), one question per line:
    {"id": "q01", "question": "...", "answer": "reference answer",
     "sections": ["75.400"], "type": "single" | "multi" | "unanswerable"}

Retrieval metrics (free: no LLM needed):
    hit@k     at least one correct section retrieved
    recall@k  share of the correct sections retrieved (matters for "multi")
    mrr       1 / rank of the first correct section

Answer metrics (need an LLM; --with-answers):
    correct          an LLM judge compares the answer with the reference
    faithful         an LLM judge checks every claim is supported by the sources
    citation_ok      every cited section was actually retrieved
    abstain_ok       "unanswerable" questions answered NOT_IN_SOURCES
    false_abstain    answerable questions wrongly answered NOT_IN_SOURCES
    bertscore_f1     meaning overlap with the reference (if bert-score is installed)

    python -m msrag.evaluate --corpus local:../regs-to-vectors/vectors --questions eval/questions.jsonl \
        --configs dense bm25 hybrid hybrid_rerank --strategies fixed section --with-answers
"""
import argparse
import json
import os
import re

import pandas as pd

from . import retrieve
from .answer import NOT_IN_SOURCES, answer, format_sources
from .corpus import load
from .embed import get_embedder

JUDGE_CORRECT = """You grade answers about mine safety regulations.
Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}
Is the candidate answer correct and consistent with the reference? Minor wording differences are fine;
missing or wrong facts are not. Reply with JSON only: {{"correct": true}} or {{"correct": false}}"""

JUDGE_FAITHFUL = """You check whether an answer is supported by its sources.
Sources:
{sources}

Answer: {candidate}
Is every factual claim in the answer supported by the sources above? Reply with JSON only:
{{"faithful": true}} or {{"faithful": false}}"""


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_metrics(retrieved: list[str], gold: list[str]) -> dict:
    if not gold:
        return {"hit": None, "recall": None, "mrr": None}
    ranks = [i for i, s in enumerate(retrieved, 1) if s in gold]
    found = {s for s in retrieved if s in gold}
    return {"hit": float(bool(ranks)), "recall": len(found) / len(set(gold)),
            "mrr": 1.0 / ranks[0] if ranks else 0.0}


def _judge(llm, prompt: str, key: str) -> float | None:
    text = llm.generate(prompt).text
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        return float(bool(json.loads(m.group(0))[key])) if m else None
    except (json.JSONDecodeError, KeyError):
        return None


def evaluate_config(questions, corpus, embedder, kind: str, strategy: str, k: int,
                    llm=None, judge=None, reranker=None) -> pd.DataFrame:
    sub = corpus.subset(strategy) if kind != "no_rag" else None
    retriever = retrieve.build(kind, sub, embedder, reranker) if sub is not None else None
    rows = []
    for q in questions:
        gold = q.get("sections", [])
        row = {"config": kind, "strategy": strategy if sub is not None else "-", "id": q["id"], "type": q["type"]}
        hits = retriever.search(q["question"], k) if retriever else []
        retrieved = []
        for h in hits:   # unique sections in rank order
            if h.section_id not in retrieved:
                retrieved.append(h.section_id)
        row.update(retrieval_metrics(retrieved, gold) if retriever else {"hit": None, "recall": None, "mrr": None})
        if llm is not None:
            a = answer(q["question"], llm, retriever, corpus.sections, k)
            unanswerable = q["type"] == "unanswerable"
            row.update({
                "answer": a.text,
                "abstain_ok": float(a.abstained) if unanswerable else None,
                "false_abstain": None if unanswerable else float(a.abstained),
                "citation_ok": float(not a.invalid_citations) if retriever else None,
                "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "seconds": a.seconds,
            })
            if judge is not None and not unanswerable and not a.abstained:
                row["correct"] = _judge(judge, JUDGE_CORRECT.format(
                    question=q["question"], reference=q["answer"], candidate=a.text), "correct")
                if retriever:
                    row["faithful"] = _judge(judge, JUDGE_FAITHFUL.format(
                        sources=format_sources(hits, corpus.sections), candidate=a.text), "faithful")
            elif not unanswerable:
                row["correct"] = 0.0 if a.abstained else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if llm is not None:
        df = add_bertscore(df, questions)
    return df


def add_bertscore(df: pd.DataFrame, questions) -> pd.DataFrame:
    try:
        from bert_score import score
    except ImportError:
        return df
    refs = {q["id"]: q["answer"] for q in questions}
    mask = df["answer"].notna() & ~df["answer"].str.startswith(NOT_IN_SOURCES) & df["id"].map(refs).astype(bool)
    if mask.any():
        _, _, f1 = score(df.loc[mask, "answer"].tolist(), df.loc[mask, "id"].map(refs).tolist(), lang="en")
        df.loc[mask, "bertscore_f1"] = f1.numpy()
    return df


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in ["hit", "recall", "mrr", "correct", "faithful", "citation_ok", "abstain_ok",
                           "false_abstain", "bertscore_f1", "seconds"] if c in results]
    out = results.groupby(["config", "strategy"])[metrics].mean().round(3)
    order = {"no_rag": 0, "dense": 1, "bm25": 2, "hybrid": 3, "hybrid_rerank": 4}
    return out.sort_index(key=lambda idx: idx.map(lambda v: order.get(v, 9)) if idx.name == "config" else idx)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--configs", nargs="+", default=["dense", "bm25", "hybrid", "hybrid_rerank"])
    p.add_argument("--strategies", nargs="+", default=["fixed", "section"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--embedder", default="minilm")
    p.add_argument("--with-answers", action="store_true", help="Also generate and judge answers (uses Gemini)")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    corpus, embedder = load(args.corpus), get_embedder(args.embedder)
    questions = load_questions(args.questions)
    reranker = retrieve.CrossEncoderReranker() if "hybrid_rerank" in args.configs else None
    llm = judge = None
    configs = list(args.configs)
    if args.with_answers:
        from .llm import GeminiLLM
        llm, judge = GeminiLLM(args.model), GeminiLLM(args.model)
        configs = ["no_rag"] + configs

    frames = []
    for kind in configs:
        for strategy in (["-"] if kind == "no_rag" else args.strategies):
            frames.append(evaluate_config(questions, corpus, embedder, kind, strategy, args.k, llm, judge, reranker))
    results = pd.concat(frames, ignore_index=True)
    os.makedirs(args.out, exist_ok=True)
    results.to_csv(os.path.join(args.out, "per_question.csv"), index=False)
    table = summarize(results)
    table.to_csv(os.path.join(args.out, "summary.csv"))
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table.to_markdown())
    print(table.to_string())


if __name__ == "__main__":
    main()
