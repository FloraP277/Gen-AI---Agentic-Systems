"""Banking77: 13,083 banking customer queries labeled with 77 intents.

Loaded from Hugging Face (PolyAI/banking77, CC BY 4.0) or from local CSVs with
columns `text,label` where label is the intent name.
"""
import random
from dataclasses import dataclass

import pandas as pd


@dataclass
class Dataset:
    train: pd.DataFrame      # columns: id, text, intent
    test: pd.DataFrame
    intents: list


def load_banking77() -> Dataset:
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names

    def frame(split):
        df = ds[split].to_pandas()
        return pd.DataFrame({"id": [f"{split}-{i}" for i in range(len(df))],
                             "text": df["text"], "intent": df["label"].map(lambda i: names[i])})

    return Dataset(frame("train"), frame("test"), list(names))


def load_csv(train_path: str, test_path: str) -> Dataset:
    def frame(path, split):
        df = pd.read_csv(path)
        return pd.DataFrame({"id": [f"{split}-{i}" for i in range(len(df))],
                             "text": df["text"], "intent": df["label"]})

    train, test = frame(train_path, "train"), frame(test_path, "test")
    return Dataset(train, test, sorted(set(train["intent"]) | set(test["intent"])))


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 13) -> pd.DataFrame:
    """Take n rows spread as evenly as possible across intents, reproducibly."""
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


def few_shot_examples(train: pd.DataFrame, per_intent: int = 1, intents=None, seed: int = 7) -> pd.DataFrame:
    """Fixed examples from the TRAIN split only, so the test set never leaks into prompts."""
    pool = train if intents is None else train[train["intent"].isin(intents)]
    return (pool.sample(frac=1, random_state=seed).groupby("intent").head(per_intent)
            .sort_values("intent").reset_index(drop=True))
