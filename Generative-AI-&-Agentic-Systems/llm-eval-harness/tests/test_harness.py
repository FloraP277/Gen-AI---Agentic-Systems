import json
import os
import re
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from evalharness import metrics                         # noqa: E402
from evalharness.cli import main                        # noqa: E402
from evalharness.data import few_shot_examples, load_csv, stratified_sample  # noqa: E402
from evalharness.llm import FakeLLM                     # noqa: E402
from evalharness.parse import parse_reply               # noqa: E402
from evalharness.prompts import load_prompts            # noqa: E402
from evalharness.queues import all_queues, needs_human, queue_for  # noqa: E402
from evalharness.runner import Experiment, run          # noqa: E402

FIX = os.path.join(ROOT, "tests", "fixtures")
PROMPTS = os.path.join(ROOT, "prompts")


@pytest.fixture
def ds():
    return load_csv(os.path.join(FIX, "train.csv"), os.path.join(FIX, "test.csv"))


def oracle(wrong_for=(), invalid_for=()):
    """A fake model that answers correctly except for chosen messages."""
    truth = pd.concat([pd.read_csv(os.path.join(FIX, f)) for f in ("train.csv", "test.csv")])
    answer = dict(zip(truth["text"], truth["label"]))

    def respond(system, user, s):
        msg = re.search(r'Message: "(.*)"', user).group(1)
        if msg in invalid_for:
            return "Sorry, I think this is about a card."
        true = answer[msg]
        intent = (("change_pin" if true != "change_pin" else "exchange_rate")
                  if msg in wrong_for else true)
        return json.dumps({"intent": intent, "queue": queue_for(intent), "needs_human": needs_human(intent)})
    return respond


def test_queue_rules_route_security_and_disputes_to_humans():
    assert queue_for("lost_or_stolen_card") == "security" and needs_human("lost_or_stolen_card")
    assert queue_for("request_refund") == "disputes" and needs_human("request_refund")
    assert queue_for("card_arrival") == "cards" and not needs_human("card_arrival")
    assert queue_for("something_else") == "account"


def test_parse_handles_messy_replies(ds):
    q = all_queues()
    p = parse_reply('Sure! {"intent": "Card Arrival", "queue": "cards", "needs_human": false}', ds.intents, q)
    assert p.valid_json and p.intent == "card_arrival" and p.queue == "cards"
    assert not parse_reply("no json here", ds.intents, q).valid_json
    bad = parse_reply('{"intent": "made_up_intent", "queue": "x"}', ds.intents, q)
    assert bad.valid_json and not bad.valid_intent and bad.queue is None


def test_sampling_is_stratified_and_few_shot_uses_train_only(ds):
    s = stratified_sample(ds.test, 16)
    assert s["intent"].value_counts().max() == 2
    ex = few_shot_examples(ds.train, 1)
    assert set(ex["text"]).issubset(set(ds.train["text"]))
    assert len(ex) == len(ds.train["intent"].unique())


def test_prompts_render_with_all_fields(ds):
    prompts = load_prompts(PROMPTS)
    assert set(prompts) == {"zero_shot", "few_shot", "chain_of_thought", "role"}
    system, user = prompts["few_shot"].render("Where is my card?", ds.intents, all_queues(),
                                              few_shot_examples(ds.train, 1))
    assert "card_arrival" in system and "Example" in system and "Where is my card?" in user
    assert '{"intent"' in system


def test_run_metrics_and_cache(ds, tmp_path):
    test_msgs = list(ds.test["text"])
    llm = FakeLLM(oracle(wrong_for=test_msgs[:4], invalid_for=test_msgs[4:6]))
    exp = Experiment(name="t", model="fake", prompts=["zero_shot", "few_shot"], test_size=24)
    cache = str(tmp_path / "cache.jsonl")
    df = run(exp, ds, load_prompts(PROMPTS), llm, cache)
    assert llm.calls == 48
    table = metrics.summary(df)
    row = table.loc["zero_shot@v1 t=0.0 p=1.0"]
    assert row["intent_accuracy"] == pytest.approx(18 / 24, abs=1e-4)
    assert row["valid_json"] == pytest.approx(22 / 24, abs=1e-4)

    llm2 = FakeLLM(oracle())
    run(exp, ds, load_prompts(PROMPTS), llm2, cache)
    assert llm2.calls == 0                       # everything came from the cache

    pairs = metrics.confused_pairs(df, "zero_shot@v1 t=0.0 p=1.0")
    assert pairs["count"].sum() == 6


def test_consistency_detects_unstable_answers(ds, tmp_path):
    flip = {"n": 0}

    def respond(system, user, s):
        flip["n"] += 1
        intent = "card_arrival" if flip["n"] % 2 else "change_pin"
        return json.dumps({"intent": intent, "queue": "cards", "needs_human": False})

    exp = Experiment(name="c", model="fake", prompts=["zero_shot"], temperatures=[1.2], repeats=2, test_size=7)
    df = run(exp, ds, load_prompts(PROMPTS), FakeLLM(respond), str(tmp_path / "c.jsonl"))
    assert metrics.summary(df)["consistency"].iloc[0] == 0.0


def test_cli_writes_report(ds, tmp_path):
    cfg = tmp_path / "exp.yaml"
    cfg.write_text("name: demo\nmodel: fake\nprompts: [zero_shot, chain_of_thought]\ntest_size: 24\n"
                   "price_per_million_input: 0.3\nprice_per_million_output: 2.5\n")
    main(["--config", str(cfg), "--out", str(tmp_path / "out")], llm=FakeLLM(oracle()), dataset=ds)
    report = (tmp_path / "out" / "demo" / "report.md").read_text()
    assert "Most confused intent pairs" in report and "cost_per_1000_messages" in report
