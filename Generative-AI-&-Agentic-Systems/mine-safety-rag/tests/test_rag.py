import json
import os

import pandas as pd

from msrag import retrieve
from msrag.answer import NOT_IN_SOURCES, answer
from msrag.evaluate import evaluate_config, load_questions, main, retrieval_metrics, summarize
from msrag.llm import FakeLLM

ROOT = os.path.join(os.path.dirname(__file__), "..")
QUESTIONS = os.path.join(ROOT, "eval", "questions.sample.jsonl")


def test_tokenizer_keeps_section_numbers_whole():
    assert "75.400" in retrieve.tokenize("See § 75.400 for coal dust")
    assert "56.14107(a)" in retrieve.tokenize("under 56.14107(a), guards")


def test_bm25_finds_a_section_by_its_number(corpus):
    hits = retrieve.BM25Retriever(corpus.subset("section")).search("what does 75.403 require", 3)
    assert hits[0].section_id == "75.403"


def test_dense_retriever_and_part_filter(corpus, embedder):
    sub = corpus.subset("section", part="56")
    hits = retrieve.DenseRetriever(sub, embedder).search("rock dust incombustible", 3)
    assert all(corpus.sections[h.section_id]["part"] == "56" for h in hits)


def test_hybrid_rrf_rewards_agreement():
    class Fixed:
        def __init__(self, ids):
            self.ids = ids

        def search(self, q, k):
            return [retrieve.Hit(i, i.split(":")[1], i, 1.0) for i in self.ids[:k]]

    a = Fixed(["c:A", "c:B", "c:C"])
    b = Fixed(["c:C", "c:D", "c:B"])
    ranked = [h.chunk_id for h in retrieve.HybridRetriever([a, b]).search("q", 4)]
    assert ranked[:2] == ["c:C", "c:B"]   # found by both beats found by one


def test_reranker_reorders(corpus, embedder):
    class Prefers75512:
        def score(self, q, texts):
            return [1.0 if "electric" in t else 0.0 for t in texts]

    base = retrieve.build("hybrid", corpus.subset("section"), embedder)
    hits = retrieve.Reranked(base, Prefers75512(), candidates=7).search("equipment examination", 2)
    assert hits[0].section_id == "75.512"


def test_answer_parses_citations_and_flags_invented_ones(corpus, embedder):
    retriever = retrieve.build("bm25", corpus.subset("section"), embedder)
    llm = FakeLLM(lambda p, s: "At least 80 percent [§ 75.403], see also [§ 99.999].")
    a = answer("rock dust incombustible content 75.403", llm, retriever, corpus.sections, k=3)
    assert a.cited_sections == ["75.403", "99.999"]
    assert a.invalid_citations == ["99.999"]
    assert "[1] § " in llm.calls[0][1]          # numbered sources were sent


def test_abstention_and_no_rag_baseline(corpus, embedder):
    retriever = retrieve.build("dense", corpus.subset("section"), embedder)
    a = answer("speed limit for haul trucks", FakeLLM(lambda p, s: NOT_IN_SOURCES), retriever, corpus.sections)
    assert a.abstained
    base = answer("speed limit", FakeLLM(lambda p, s: "25 mph [§ 56.9101]"))
    assert base.retrieved_sections == [] and base.cited_sections == ["56.9101"]


def test_retrieval_metrics():
    m = retrieval_metrics(["A", "B", "C"], ["B", "D"])
    assert m == {"hit": 1.0, "recall": 0.5, "mrr": 0.5}
    assert retrieval_metrics(["A"], [])["hit"] is None


def test_evaluate_with_fake_llm_and_judge(corpus, embedder):
    questions = load_questions(QUESTIONS)

    def respond(prompt, system):
        if "speed limit" in prompt and "Sources" in prompt:
            return NOT_IN_SOURCES
        return "Answer from the rule [§ 75.403]."

    judge = FakeLLM(lambda p, s: '{"correct": true}' if "Reference answer" in p else '{"faithful": true}')
    df = evaluate_config(questions, corpus, embedder, "hybrid", "section", 5, FakeLLM(respond), judge)
    assert len(df) == len(questions)
    unanswerable = df[df["type"] == "unanswerable"].iloc[0]
    assert unanswerable["abstain_ok"] == 1.0
    assert df["correct"].dropna().eq(1.0).all()
    assert set(summarize(df).columns) >= {"hit", "recall", "mrr", "correct", "faithful", "abstain_ok"}


def test_cli_retrieval_only(corpus_dir, tmp_path):
    out = tmp_path / "results"
    main(["--corpus", f"local:{corpus_dir}", "--questions", QUESTIONS, "--configs", "dense", "bm25", "hybrid",
          "--embedder", "hashing", "--out", str(out)])
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["config"]) == {"dense", "bm25", "hybrid"}
    assert set(summary["strategy"]) == {"fixed", "section"}
    assert (summary["hit"] > 0).all()
    assert (out / "summary.md").exists()
