"""Does adding synthetic images help the classifier on rare classes?

For each ratio r, the training set is all real TRAIN images plus, for each rare
class, r synthetic images per real image of that class (r = 0 is the baseline).
Validation and test sets are always real images only, and the same in every run.

    python -m sld.experiment --work work --ratios 0 0.5 1 2 --epochs 8
"""
import argparse
import glob
import json
import os

import pandas as pd

from .metrics import per_class_report


def training_set(splits: pd.DataFrame, synthetic_dir: str, rare: list[str], ratio: float) -> pd.DataFrame:
    train = splits[splits["split"] == "train"][["path", "dx"]].assign(synthetic=False)
    extra = []
    for dx in rare:
        files = sorted(glob.glob(os.path.join(synthetic_dir, dx, "*.jpg")))
        want = int(round(ratio * (train["dx"] == dx).sum()))
        if want > len(files):
            raise ValueError(f"ratio {ratio} needs {want} synthetic {dx} images, only {len(files)} generated")
        extra += [{"path": f, "dx": dx, "synthetic": True} for f in files[:want]]
    return pd.concat([train, pd.DataFrame(extra, columns=["path", "dx", "synthetic"])], ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="work")
    p.add_argument("--ratios", type=float, nargs="+", default=[0, 0.5, 1, 2])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args(argv)

    from .classifier import train_and_evaluate

    splits = pd.read_csv(os.path.join(args.work, "splits.csv"))
    rare = json.load(open(os.path.join(args.work, "rare.json")))["rare_classes"]
    classes = sorted(splits["dx"].unique())
    val, test = splits[splits["split"] == "val"], splits[splits["split"] == "test"]
    rows = []
    for ratio in args.ratios:
        train = training_set(splits, os.path.join(args.work, "synthetic"), rare, ratio)
        for seed in args.seeds:
            preds = train_and_evaluate(train, val, test, classes, epochs=args.epochs, seed=seed)
            report = per_class_report(test["dx"].tolist(), preds, classes, rare)
            rows.append({"ratio": ratio, "seed": seed, "train_images": len(train),
                         "synthetic_images": int(train["synthetic"].sum()), **report})
            print(rows[-1])
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(args.work, "results"), exist_ok=True)
    df.to_csv(os.path.join(args.work, "results", "experiment.csv"), index=False)
    summary = df.drop(columns="seed").groupby("ratio").mean().round(3)
    open(os.path.join(args.work, "results", "experiment.md"), "w").write(summary.to_markdown())
    print(summary.to_string())


if __name__ == "__main__":
    main()
