"""Tests for the data handling and scoring (GPU training and generation run on Colab)."""
import json
import os
import random
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sld.data import load_metadata, rare_classes, split_by_lesion  # noqa: E402
from sld.experiment import training_set                              # noqa: E402
from sld.metrics import per_class_report                             # noqa: E402
from sld.prepare import build_diffusion_sets                          # noqa: E402

COUNTS = {"nv": 60, "mel": 20, "bkl": 20, "bcc": 14, "akiec": 10, "vasc": 8, "df": 7}   # lesions per class


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """A fake HAM10000 folder: tiny JPEGs, some lesions photographed more than once."""
    from PIL import Image

    root = tmp_path_factory.mktemp("ham")
    rng = random.Random(0)
    rows, k = [], 0
    for dx, n in COUNTS.items():
        for lesion in range(n):
            for _ in range(rng.choice([1, 1, 2, 3])):
                image_id = f"ISIC_{k:07d}"
                sub = root / ("part_1" if k % 2 else "part_2")
                sub.mkdir(exist_ok=True)
                Image.new("RGB", (8, 8), (k % 255, 80, 120)).save(sub / f"{image_id}.jpg")
                rows.append({"lesion_id": f"HAM_{dx}_{lesion}", "image_id": image_id, "dx": dx})
                k += 1
    rows.append({"lesion_id": "HAM_missing", "image_id": "ISIC_9999999", "dx": "nv"})   # listed, no file
    pd.DataFrame(rows).to_csv(root / "HAM10000_metadata.csv", index=False)
    return root


def test_load_finds_images_in_subfolders_and_drops_missing(root):
    meta = load_metadata(str(root))
    assert meta["path"].notna().all()
    assert "ISIC_9999999" not in set(meta["image_id"])


def test_split_by_lesion_never_leaks_a_lesion(root):
    s = split_by_lesion(load_metadata(str(root)))
    per_lesion = s.groupby("lesion_id")["split"].nunique()
    assert (per_lesion == 1).all()                         # every photo of a lesion in one split
    for part in ("train", "val", "test"):
        assert set(s[s["split"] == part]["dx"]) == set(COUNTS)   # every class in every split
    again = split_by_lesion(load_metadata(str(root)))
    assert s["split"].tolist() == again["split"].tolist()       # reproducible


def test_rare_classes_and_diffusion_set_use_train_only(root, tmp_path):
    s = split_by_lesion(load_metadata(str(root)))
    rare = rare_classes(s, 3)
    assert set(rare) == {"akiec", "vasc", "df"}
    counts = build_diffusion_sets(s, str(tmp_path / "dt"), rare)
    for dx in rare:
        lines = [json.loads(l) for l in open(tmp_path / "dt" / dx / "metadata.jsonl")]
        names = {l["file_name"] for l in lines}
        train_names = {os.path.basename(p) for p in s[(s["split"] == "train") & (s["dx"] == dx)]["path"]}
        assert names == train_names and counts[dx] == len(names)
        assert all("dermatoscopic" in l["text"] for l in lines)


def test_training_set_adds_the_right_amount_of_synthetic_images(root, tmp_path):
    s = split_by_lesion(load_metadata(str(root)))
    syn = tmp_path / "synthetic" / "df"
    syn.mkdir(parents=True)
    for i in range(30):
        (syn / f"synthetic_df_{i:05d}.jpg").write_bytes(b"x")
    real_df = ((s["split"] == "train") & (s["dx"] == "df")).sum()
    base = training_set(s, str(tmp_path / "synthetic"), ["df"], 0)
    assert base["synthetic"].sum() == 0
    double = training_set(s, str(tmp_path / "synthetic"), ["df"], 2)
    assert double["synthetic"].sum() == 2 * real_df
    assert set(double[double["synthetic"]]["dx"]) == {"df"}
    with pytest.raises(ValueError, match="only 30 generated"):
        training_set(s, str(tmp_path / "synthetic"), ["df"], 20)


def test_per_class_report_exposes_a_majority_class_model():
    y_true = ["nv"] * 8 + ["df", "vasc"]
    always_nv = ["nv"] * 10
    r = per_class_report(y_true, always_nv, ["df", "nv", "vasc"], ["df", "vasc"])
    assert r["accuracy"] == 0.8
    assert r["rare_class_recall"] == 0.0 and r["balanced_accuracy"] == pytest.approx(0.333, abs=1e-3)
