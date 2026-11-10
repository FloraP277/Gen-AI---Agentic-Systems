"""Preference pairs for DPO.

DPO (direct preference optimization) learns from pairs of a better and a worse
answer to the same prompt. Here the "worse" answers are real mistakes the SFT
model made on held-out training messages, plus the intents it confuses most,
so preference tuning targets exactly where the model is weak.
"""
from collections import Counter

import pandas as pd

from .data import SYSTEM


def build_pairs(pref_preds: pd.DataFrame, max_per_confusion: int = 50) -> list[dict]:
    """pref_preds: text, true_intent, pred_intent from the SFT model on the `pref` split."""
    wrong = pref_preds[pref_preds["pred_intent"] != pref_preds["true_intent"]]
    confusions = Counter()
    pairs = []
    for r in wrong.itertuples():
        rejected = r.pred_intent if isinstance(r.pred_intent, str) and r.pred_intent else "unknown"
        if confusions[(r.true_intent, rejected)] >= max_per_confusion:
            continue
        confusions[(r.true_intent, rejected)] += 1
        pairs.append({
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r.text}],
            "chosen": [{"role": "assistant", "content": r.true_intent}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return pairs
