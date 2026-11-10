"""Metrics that show failures, not just averages (Week 4: averages hide the failures that matter)."""
import pandas as pd
from sklearn.metrics import f1_score


def summary(df: pd.DataFrame, price_in: float | None = None, price_out: float | None = None) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run", sort=False):
        pred = g["pred_intent"].fillna("<invalid>")
        consistent = (g.groupby("message_id")["pred_intent"]
                      .agg(lambda s: s.fillna("<invalid>").nunique() == 1).mean())
        row = {
            "run": run,
            "intent_accuracy": (pred == g["true_intent"]).mean(),
            "macro_f1": f1_score(g["true_intent"], pred, average="macro", zero_division=0),
            "queue_accuracy": (g["pred_queue"] == g["true_queue"]).mean(),
            "needs_human_accuracy": (g["pred_needs_human"] == g["true_needs_human"]).mean(),
            "valid_json": g["valid_json"].mean(),
            "valid_intent": g["valid_intent"].mean(),
            "consistency": consistent,
            "p50_seconds": g["seconds"].median(),
            "p95_seconds": g["seconds"].quantile(0.95),
            "tokens_per_message": (g["input_tokens"] + g["output_tokens"]).mean(),
        }
        if price_in is not None and price_out is not None:
            per_msg = (g["input_tokens"] * price_in + g["output_tokens"] * price_out).mean() / 1e6
            row["cost_per_1000_messages"] = per_msg * 1000
        rows.append(row)
    return pd.DataFrame(rows).set_index("run").round(4)


def per_group(df: pd.DataFrame, run: str, column: str = "true_queue") -> pd.DataFrame:
    g = df[df["run"] == run]
    return (g.assign(correct=g["pred_intent"] == g["true_intent"])
            .groupby(column)["correct"].agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "messages"}).sort_values("accuracy").round(3))


def confused_pairs(df: pd.DataFrame, run: str, top: int = 10) -> pd.DataFrame:
    g = df[(df["run"] == run) & (df["pred_intent"] != df["true_intent"])]
    return (g.assign(predicted=g["pred_intent"].fillna("<invalid>"))
            .groupby(["true_intent", "predicted"]).size().rename("count")
            .sort_values(ascending=False).head(top).reset_index())
