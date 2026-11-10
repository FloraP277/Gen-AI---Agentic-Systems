"""Embedding models.

- MiniLMEmbedder: sentence-transformers/all-MiniLM-L6-v2, a free open-source
  model that runs on a laptop CPU. 384 dimensions. Use this for real runs.
- HashingEmbedder: a deterministic, dependency-free stand-in used by the tests.
  It maps words to vector positions with a hash, so texts that share words get
  similar vectors. It does not understand meaning; never use it for real search.
"""
import hashlib
import re

import numpy as np


class HashingEmbedder:
    name = "hashing-test-only"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class MiniLMEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"
    dim = 384

    def __init__(self, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)


def get_embedder(name: str):
    if name == "minilm":
        return MiniLMEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {name!r}; use 'minilm' or 'hashing'")
