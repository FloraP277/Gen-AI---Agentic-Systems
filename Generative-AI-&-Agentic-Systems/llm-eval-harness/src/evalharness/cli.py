"""Run an experiment and write a report.

    export GEMINI_API_KEY=...
    python -m evalharness.cli --config experiments/main.yaml
"""
import argparse
import os

import yaml

from . import metrics
from .data import load_banking77, load_csv
from .prompts import load_prompts
from .runner import Experiment, run

HERE = os.path.dirname(__file__)
DEFAULT_PROMPTS = os.path.join(HERE, "..", "..", "prompts")


def write_report(df, exp: Experiment, out_dir: str) -> str:
    table = metrics.summary(df, exp.price_per_million_input, exp.price_per_million_output)
    best = table["intent_accuracy"].idxmax()
    lines = [
        f"# Results: {exp.name}",
        "",
        f"Model `{exp.model}` · {df['message_id'].nunique()} test messages · {exp.repeats} repeat(s) per setting",
        "",
        "## All runs",
        "",
        table.to_markdown(),
        "",
        f"## Best run: {best}",
        "",
        "### Accuracy by queue",
        "",
        metrics.per_group(df, best).to_markdown(),
        "",
        "### Weakest intents",
        "",
        metrics.per_group(df, best, "true_intent").head(10).to_markdown(),
        "",
        "### Most confused intent pairs",
        "",
        metrics.confused_pairs(df, best).to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        "_Write your recommendation here: which prompt and settings to use, and why._",
    ]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "report.md")
    open(path, "w").write("\n".join(lines))
    df.drop(columns=["raw"]).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    table.to_csv(os.path.join(out_dir, "summary.csv"))
    return path


def main(argv=None, llm=None, dataset=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--train-csv")
    p.add_argument("--test-csv")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    exp = Experiment(**yaml.safe_load(open(args.config)))
    if dataset is None:
        dataset = load_csv(args.train_csv, args.test_csv) if args.train_csv else load_banking77()
    if llm is None:
        from .llm import GeminiLLM
        llm = GeminiLLM()
    out = os.path.join(args.out, exp.name)
    df = run(exp, dataset, load_prompts(args.prompts), llm, os.path.join(out, "cache.jsonl"))
    path = write_report(df, exp, out)
    print(open(path).read())
    return df


if __name__ == "__main__":
    main()
