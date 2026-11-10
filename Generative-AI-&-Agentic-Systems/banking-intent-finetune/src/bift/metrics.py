"""Scoring and the fine-tuned vs prompted comparison."""
import pandas as pd
from sklearn.metrics import f1_score


def score(df: pd.DataFrame) -> dict:
    """df needs true_intent and pred_intent (None = unparseable, counted wrong)."""
    pred = df["pred_intent"].fillna("<invalid>")
    return {
        "accuracy": round(float((pred == df["true_intent"]).mean()), 4),
        "macro_f1": round(float(f1_score(df["true_intent"], pred, average="macro", zero_division=0)), 4),
        "invalid_rate": round(float(df["pred_intent"].isna().mean()), 4),
        "messages": int(len(df)),
    }


def cost_per_1000(seconds_per_message: float, gpu_dollars_per_hour: float) -> float:
    """Self-hosted cost: GPU time to classify 1,000 messages."""
    return round(seconds_per_message * 1000 / 3600 * gpu_dollars_per_hour, 4)


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """rows: {"model": ..., "accuracy": ..., "macro_f1": ..., "invalid_rate": ...,
    "seconds_per_message": ..., "cost_per_1000": ...}"""
    cols = ["model", "accuracy", "macro_f1", "invalid_rate", "seconds_per_message", "cost_per_1000"]
    return pd.DataFrame(rows).reindex(columns=cols).set_index("model")


def prompted_baseline(predictions_csv: str, run: str | None = None) -> dict:
    """Read the best run from the LLM Evaluation Harness predictions.csv."""
    df = pd.read_csv(predictions_csv)
    if run is None:
        acc = df.assign(ok=df["pred_intent"] == df["true_intent"]).groupby("run")["ok"].mean()
        run = acc.idxmax()
    g = df[(df["run"] == run) & (df["repeat"] == 0)]
    g = g.assign(pred_intent=g["pred_intent"].where(g["pred_intent"].notna(), None))
    return {"run": run, **score(g), "seconds_per_message": round(float(g["seconds"].mean()), 3)}
