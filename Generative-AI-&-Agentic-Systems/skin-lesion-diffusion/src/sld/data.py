"""HAM10000: 10,015 dermatoscopic images of 7 lesion types.

Expected layout after downloading (Harvard Dataverse or Kaggle):
    <root>/HAM10000_metadata.csv        columns include lesion_id, image_id, dx
    <root>/**/<image_id>.jpg            images, in any subfolders

Some lesions were photographed several times. Splitting by IMAGE would put
photos of the same lesion in both train and test and inflate the scores, so
every split here is by LESION.
"""
import glob
import os
import random

import pandas as pd

CLASSES = {
    "nv": "melanocytic nevus", "mel": "melanoma", "bkl": "benign keratosis-like lesion",
    "bcc": "basal cell carcinoma", "akiec": "actinic keratosis", "vasc": "vascular lesion",
    "df": "dermatofibroma",
}


def load_metadata(root: str) -> pd.DataFrame:
    meta = pd.read_csv(os.path.join(root, "HAM10000_metadata.csv"))
    missing = {"lesion_id", "image_id", "dx"} - set(meta.columns)
    if missing:
        raise ValueError(f"metadata is missing columns {missing}")
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)}
    meta = meta.assign(path=meta["image_id"].map(paths))
    lost = meta["path"].isna().sum()
    if lost:
        print(f"warning: {lost} images listed in the metadata were not found and are dropped")
    return meta.dropna(subset=["path"]).reset_index(drop=True)


def split_by_lesion(meta: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Assign each LESION to train/val/test, stratified by diagnosis."""
    rng = random.Random(seed)
    lesions = meta.groupby("lesion_id")["dx"].first()
    split = {}
    for dx, ids in lesions.groupby(lesions):
        ids = sorted(ids.index)
        rng.shuffle(ids)
        n_test, n_val = max(1, round(len(ids) * test)), max(1, round(len(ids) * val))
        for i, lid in enumerate(ids):
            split[lid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return meta.assign(split=meta["lesion_id"].map(split))


def rare_classes(meta: pd.DataFrame, n: int = 3) -> list[str]:
    """The n diagnoses with the fewest training lesions."""
    train = meta[meta["split"] == "train"]
    return train.groupby("dx")["lesion_id"].nunique().sort_values().index[:n].tolist()


def prompt_for(dx: str) -> str:
    return f"a dermatoscopic image of a {CLASSES[dx]}, dermatology photograph, close-up skin lesion"
