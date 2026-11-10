"""Run every prompt x setting combination over the test set, with a response cache.

The cache (JSON Lines) is keyed by prompt fingerprint + settings + message +
repeat number, so re-running an experiment, or adding one new setting, only
calls the API for combinations that haven't been run yet.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field

import pandas as pd

from .data import few_shot_examples, stratified_sample
from .llm import Settings
from .parse import parse_reply
from .queues import all_queues, needs_human, queue_for


@dataclass
class Experiment:
    name: str
    model: str
    prompts: list
    temperatures: list = field(default_factory=lambda: [0.0])
    top_ps: list = field(default_factory=lambda: [1.0])
    repeats: int = 1
    test_size: int = 300
    max_output_tokens: int = 512
    stop_sequences: list = field(default_factory=list)
    price_per_million_input: float | None = None
    price_per_million_output: float | None = None
    seed: int = 13


class Cache:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def get(self, key):
        return self.data.get(key)

    def put(self, rec):
        self.data[rec["key"]] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


def run(exp: Experiment, dataset, prompts: dict, llm, cache_path: str) -> pd.DataFrame:
    test = stratified_sample(dataset.test, exp.test_size, exp.seed)
    queues = all_queues()
    cache = Cache(cache_path)
    rows = []
    for pname in exp.prompts:
        tmpl = prompts[pname]
        examples = few_shot_examples(dataset.train, tmpl.few_shot_per_intent) if tmpl.few_shot_per_intent else None
        for temp in exp.temperatures:
            for top_p in exp.top_ps:
                s = Settings(exp.model, temp, top_p, exp.max_output_tokens, tuple(exp.stop_sequences))
                run_id = f"{tmpl.name}@v{tmpl.version} t={temp} p={top_p}"
                for rep in range(exp.repeats):
                    for msg in test.itertuples():
                        key = hashlib.sha256(f"{tmpl.fingerprint}|{s.key()}|{msg.text}|{rep}".encode()).hexdigest()
                        rec = cache.get(key)
                        if rec is None:
                            system, user = tmpl.render(msg.text, dataset.intents, queues, examples)
                            r = llm.generate(system, user, s)
                            rec = {"key": key, "text": r.text, "input_tokens": r.input_tokens,
                                   "output_tokens": r.output_tokens, "seconds": r.seconds}
                            cache.put(rec)
                        p = parse_reply(rec["text"], dataset.intents, queues)
                        rows.append({
                            "run": run_id, "prompt": tmpl.name, "technique": tmpl.technique,
                            "temperature": temp, "top_p": top_p, "repeat": rep, "message_id": msg.id,
                            "text": msg.text, "true_intent": msg.intent, "true_queue": queue_for(msg.intent),
                            "true_needs_human": needs_human(msg.intent),
                            "pred_intent": p.intent, "pred_queue": p.queue, "pred_needs_human": p.needs_human,
                            "valid_json": p.valid_json, "valid_intent": p.valid_intent,
                            "input_tokens": rec["input_tokens"], "output_tokens": rec["output_tokens"],
                            "seconds": rec["seconds"], "raw": rec["text"],
                        })
    return pd.DataFrame(rows)
