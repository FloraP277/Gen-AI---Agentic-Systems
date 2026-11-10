"""Score fine-tuned models on the shared 300-message test set and compare them
with the best prompted model from the LLM Evaluation Harness.

    python -m bift.evaluate --runs runs/sft runs/dpo \
        --prompted ../llm-eval-harness/results/techniques/predictions.csv \
        --prompted-cost-per-1000 <from that project's report> --gpu-dollars-per-hour <your GPU price>
"""
import argparse
import json
import os

from .data import load_banking77, stratified_sample
from .metrics import comparison_table, cost_per_1000, prompted_baseline, score
from .predict import load_model, predict


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--prompted", help="predictions.csv from the LLM Evaluation Harness")
    p.add_argument("--prompted-cost-per-1000", type=float)
    p.add_argument("--gpu-dollars-per-hour", type=float, help="What your GPU costs per hour")
    p.add_argument("--test-size", type=int, default=300)
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    _, test, _ = load_banking77()
    test = stratified_sample(test, args.test_size).rename(columns={"intent": "true_intent"})
    rows = []
    os.makedirs(args.out, exist_ok=True)
    for run in args.runs:
        meta = json.load(open(os.path.join(run, "training.json")))
        intents = json.load(open(os.path.join(run, "intents.json")))
        tok, model = load_model(meta["base_model"], os.path.join(run, "adapter"))
        preds = predict(test, tok, model, intents)
        preds.to_csv(os.path.join(args.out, f"{os.path.basename(run)}_predictions.csv"), index=False)
        spm = float(preds["seconds_per_message"].iloc[0])
        rows.append({"model": f"fine-tuned ({os.path.basename(run)})", **score(preds),
                     "seconds_per_message": round(spm, 4),
                     "cost_per_1000": cost_per_1000(spm, args.gpu_dollars_per_hour) if args.gpu_dollars_per_hour else None})
    if args.prompted:
        base = prompted_baseline(args.prompted)
        rows.append({"model": f"prompted ({base['run']})", **{k: base[k] for k in ("accuracy", "macro_f1", "invalid_rate")},
                     "seconds_per_message": base["seconds_per_message"], "cost_per_1000": args.prompted_cost_per_1000})
    table = comparison_table(rows)
    table.to_csv(os.path.join(args.out, "comparison.csv"))
    open(os.path.join(args.out, "comparison.md"), "w").write(table.to_markdown())
    print(table.to_string())


if __name__ == "__main__":
    main()
