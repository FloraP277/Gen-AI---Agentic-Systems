"""Build the splits and the diffusion fine-tuning set.

    python -m sld.prepare --root data/ham10000 --out work

Writes:
    work/splits.csv                       every image with its lesion-level split
    work/diffusion_train/<dx>/            TRAIN images of each rare class + metadata.jsonl
                                          (image + caption), the format the Hugging Face
                                          diffusers LoRA training script reads
"""
import argparse
import json
import os
import shutil

from .data import load_metadata, prompt_for, rare_classes, split_by_lesion


def build_diffusion_sets(splits, out_dir: str, classes: list[str]) -> dict:
    """Only TRAIN images are used to teach the diffusion model, so synthetic
    images can never be copies of test images."""
    counts = {}
    for dx in classes:
        d = os.path.join(out_dir, dx)
        os.makedirs(d, exist_ok=True)
        rows = splits[(splits["split"] == "train") & (splits["dx"] == dx)]
        with open(os.path.join(d, "metadata.jsonl"), "w") as f:
            for r in rows.itertuples():
                name = os.path.basename(r.path)
                shutil.copy(r.path, os.path.join(d, name))
                f.write(json.dumps({"file_name": name, "text": prompt_for(dx)}) + "\n")
        counts[dx] = len(rows)
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="work")
    p.add_argument("--rare", type=int, default=3)
    args = p.parse_args(argv)
    splits = split_by_lesion(load_metadata(args.root))
    os.makedirs(args.out, exist_ok=True)
    splits.to_csv(os.path.join(args.out, "splits.csv"), index=False)
    rare = rare_classes(splits, args.rare)
    counts = build_diffusion_sets(splits, os.path.join(args.out, "diffusion_train"), rare)
    print(splits.groupby(["dx", "split"]).size().unstack(fill_value=0))
    print(f"rare classes: {rare}; diffusion training images: {counts}")
    json.dump({"rare_classes": rare}, open(os.path.join(args.out, "rare.json"), "w"))


if __name__ == "__main__":
    main()
