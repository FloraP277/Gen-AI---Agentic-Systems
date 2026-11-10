"""Query embedding. Must be the same model used to embed the chunks
(all-MiniLM-L6-v2 in the Regulations-to-Vectors project)."""
import hashlib
import re

import numpy as np


class HashingEmbedder:
    """Test-only stand-in: hashes words into a vector. Understands no meaning."""
    name = "hashing-test-only"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class MiniLMEmbedder:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.name)

    def embed(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def get_embedder(name: str):
    return {"minilm": MiniLMEmbedder, "hashing": HashingEmbedder}[name]()
