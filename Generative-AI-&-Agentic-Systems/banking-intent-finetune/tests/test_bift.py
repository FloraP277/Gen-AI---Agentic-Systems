"""Tests for everything except the GPU training itself (which runs on Colab)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bift.data import split_train, stratified_sample, to_prompt_completion  # noqa: E402
from bift.labels import parse_prediction                                    # noqa: E402
from bift.metrics import comparison_table, cost_per_1000, prompted_baseline, score  # noqa: E402
from bift.pairs import build_pairs                                          # noqa: E402

INTENTS = ["card_arrival", "lost_or_stolen_card", "failed_transfer", "exchange_rate", "request_refund"]


@pytest.fixture
def train():
    rows = [(f"customer message {k}-{i}", intent) for k, intent in enumerate(INTENTS) for i in range(40)]
    return pd.DataFrame({"id": [f"train-{i}" for i in range(len(rows))],
                         "text": [r[0] for r in rows], "intent": [r[1] for r in rows]})


def test_split_train_is_disjoint_and_stratified(train):
    parts = split_train(train)
    ids = [set(p["id"]) for p in parts.values()]
    assert sum(len(s) for s in ids) == len(train)
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for p in parts.values():
        assert set(p["intent"]) == set(INTENTS)       # every intent in every part
    assert len(parts["val"]) == 10 and len(parts["pref"]) == 30


def test_stratified_sample_is_deterministic(train):
    # Same data in the same order (as both projects load it) -> the same test set.
    a, b = stratified_sample(train, 25), stratified_sample(train.copy(), 25)
    assert a["intent"].value_counts().tolist() == [5] * 5
    assert a["id"].tolist() == b["id"].tolist()


def test_prompt_completion_puts_intent_only_in_completion(train):
    rec = to_prompt_completion(train.head(1))[0]
    assert rec["completion"] == [{"role": "assistant", "content": "card_arrival"}]
    assert all("card_arrival" not in m["content"] for m in rec["prompt"])


@pytest.mark.parametrize("text,expected", [
    ("card_arrival", "card_arrival"),
    ("Card Arrival\n", "card_arrival"),
    ("  lost or stolen card.", "lost_or_stolen_card"),
    ("failed_transfr", "failed_transfer"),            # small typo
    ("I think it's about money", None),
    ("", None),
])
def test_parse_prediction(text, expected):
    assert parse_prediction(text, INTENTS) == expected


def test_build_pairs_from_mistakes_with_cap():
    preds = pd.DataFrame({
        "text": [f"t{i}" for i in range(6)],
        "true_intent": ["card_arrival"] * 5 + ["exchange_rate"],
        "pred_intent": ["lost_or_stolen_card"] * 4 + ["card_arrival", None],
    })
    pairs = build_pairs(preds, max_per_confusion=3)
    assert len(pairs) == 4                             # 3 capped + 1 invalid; the correct one skipped
    assert pairs[0]["chosen"][0]["content"] == "card_arrival"
    assert pairs[0]["rejected"][0]["content"] == "lost_or_stolen_card"
    assert pairs[-1]["rejected"][0]["content"] == "unknown"


def test_score_and_comparison(tmp_path):
    df = pd.DataFrame({"true_intent": ["a", "a", "b", "b"], "pred_intent": ["a", None, "b", "a"]})
    s = score(df)
    assert s["accuracy"] == 0.5 and s["invalid_rate"] == 0.25
    assert cost_per_1000(0.036, 1.0) == 0.01

    harness = pd.DataFrame({
        "run": ["few_shot@v1 t=0.0 p=1.0"] * 4 + ["zero_shot@v1 t=0.0 p=1.0"] * 4,
        "repeat": [0] * 8,
        "true_intent": ["a", "a", "b", "b"] * 2,
        "pred_intent": ["a", "a", "b", "a", "a", None, "a", "a"],
        "seconds": [0.5] * 8,
    })
    path = tmp_path / "predictions.csv"
    harness.to_csv(path, index=False)
    base = prompted_baseline(str(path))
    assert base["run"].startswith("few_shot") and base["accuracy"] == 0.75
    table = comparison_table([{"model": "ft", **s, "seconds_per_message": 0.01, "cost_per_1000": 0.003},
                              {"model": "prompted", **{k: base[k] for k in ("accuracy", "macro_f1", "invalid_rate")},
                               "seconds_per_message": 0.5, "cost_per_1000": None}])
    assert list(table.index) == ["ft", "prompted"]
