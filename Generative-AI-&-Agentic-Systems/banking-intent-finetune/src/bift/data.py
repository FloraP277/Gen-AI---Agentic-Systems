"""Banking77 data, formatted for fine-tuning.

The test set is chosen with the same stratified sampling (same seed) as the
LLM Evaluation Harness project, so the fine-tuned model is scored on exactly
the same 300 messages as the prompted models.
"""
import random

import pandas as pd

SYSTEM = "You route bank customer messages. Reply with the single intent name only, exactly as written in training."


def load_banking77():
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names

    def frame(split):
        df = ds[split].to_pandas()
        return pd.DataFrame({"id": [f"{split}-{i}" for i in range(len(df))],
                             "text": df["text"], "intent": df["label"].map(lambda i: names[i])})

    return frame("train"), frame("test"), list(names)


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 13) -> pd.DataFrame:
    """Identical to evalharness.data.stratified_sample: do not change one without the other."""
    rng = random.Random(seed)
    groups = {k: list(g.index) for k, g in df.groupby("intent")}
    for idx in groups.values():
        rng.shuffle(idx)
    picked, i = [], 0
    while len(picked) < min(n, len(df)):
        for intent in sorted(groups):
            if i < len(groups[intent]) and len(picked) < n:
                picked.append(groups[intent][i])
        i += 1
    return df.loc[picked].reset_index(drop=True)


def split_train(train: pd.DataFrame, val_frac: float = 0.05, pref_frac: float = 0.15, seed: int = 7):
    """Split the TRAIN split three ways, stratified by intent:
    sft (fit the model), val (watch for overfitting), pref (find the SFT model's
    mistakes to build DPO preference pairs). The test split is never touched."""
    parts = {"sft": [], "val": [], "pref": []}
    for _, g in train.groupby("intent"):
        g = g.sample(frac=1, random_state=seed)
        n_val, n_pref = max(1, round(len(g) * val_frac)), max(1, round(len(g) * pref_frac))
        parts["val"].append(g.iloc[:n_val])
        parts["pref"].append(g.iloc[n_val:n_val + n_pref])
        parts["sft"].append(g.iloc[n_val + n_pref:])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}


def to_prompt_completion(df: pd.DataFrame) -> list[dict]:
    """Conversational prompt/completion records: the loss is computed on the
    completion (the intent) only, not on the instructions."""
    return [{"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
             "completion": [{"role": "assistant", "content": i}]}
            for t, i in zip(df["text"], df["intent"])]


def prompt_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
