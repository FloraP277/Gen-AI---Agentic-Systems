import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from msrag.corpus import Corpus          # noqa: E402
from msrag.embed import HashingEmbedder  # noqa: E402


def _chunks():
    data = json.load(open(os.path.join(ROOT, "tests", "fixtures", "sample_sections.json")))["sections"]
    rows, sections = [], {}
    for sid, s in data.items():
        sections[sid] = {"part": s["part"], "title": s["title"], "url": f"https://www.ecfr.gov/current/title-30/section-{sid}"}
        text = "\n".join(s["paragraphs"])
        rows.append({"chunk_id": f"section:{sid}:0", "section_id": sid, "part": s["part"], "strategy": "section",
                     "position": 0, "text": f"§ {sid} {s['title']}.\n{text}", "n_words": len(text.split())})
        words = text.split()
        for i, start in enumerate(range(0, len(words), 30)):
            piece = " ".join(words[start:start + 40])
            rows.append({"chunk_id": f"fixed:{sid}:{i}", "section_id": sid, "part": s["part"], "strategy": "fixed",
                         "position": i, "text": piece, "n_words": len(piece.split())})
    return pd.DataFrame(rows), sections


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder()


@pytest.fixture(scope="session")
def corpus(embedder):
    chunks, sections = _chunks()
    return Corpus(chunks, embedder.embed(chunks["text"].tolist()), sections)


@pytest.fixture
def corpus_dir(tmp_path, corpus):
    """The same corpus saved in the Regulations-to-Vectors local-store format."""
    corpus.chunks.to_parquet(tmp_path / "chunks.parquet", index=False)
    np.save(tmp_path / "embeddings.npy", corpus.embeddings)
    json.dump(corpus.sections, open(tmp_path / "sections.json", "w"))
    return tmp_path
