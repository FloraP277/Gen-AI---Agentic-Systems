"""Load the chunks and embeddings produced by the Regulations-to-Vectors project.

Two sources, matching that project's stores:
- a local folder (chunks.parquet, embeddings.npy, sections.json)
- PostgreSQL + pgvector (tables reg_chunks and reg_sections)
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Corpus:
    chunks: pd.DataFrame          # chunk_id, section_id, part, strategy, text, ...
    embeddings: np.ndarray        # one row per chunk, L2-normalized
    sections: dict                # section_id -> {title, url, part, ...}

    def subset(self, strategy: str, part: str | None = None) -> "Corpus":
        mask = (self.chunks["strategy"] == strategy).to_numpy(copy=True)
        if part is not None:
            mask = mask & (self.chunks["part"].astype(str) == str(part)).to_numpy()
        return Corpus(self.chunks[mask].reset_index(drop=True), self.embeddings[mask], self.sections)

    def __len__(self):
        return len(self.chunks)


def load_local(path: str) -> Corpus:
    chunks = pd.read_parquet(os.path.join(path, "chunks.parquet"))
    emb = np.load(os.path.join(path, "embeddings.npy"))
    sections = json.load(open(os.path.join(path, "sections.json")))
    return Corpus(chunks, emb, sections)


def load_pg(dsn: str) -> Corpus:
    import psycopg
    from pgvector.psycopg import register_vector

    with psycopg.connect(dsn) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, section_id, part, strategy, position, text, n_words, embedding"
                        " FROM reg_chunks ORDER BY chunk_id")
            rows = cur.fetchall()
            cur.execute("SELECT section_id, part, subpart, title, url FROM reg_sections")
            sections = {r[0]: {"part": r[1], "subpart": r[2], "title": r[3], "url": r[4]} for r in cur.fetchall()}
    cols = ["chunk_id", "section_id", "part", "strategy", "position", "text", "n_words"]
    chunks = pd.DataFrame([r[:-1] for r in rows], columns=cols)
    emb = np.array([r[-1] for r in rows], dtype=np.float32)
    return Corpus(chunks, emb, sections)


def load(spec: str) -> Corpus:
    """'local:PATH' or 'pg:DSN'."""
    kind, _, target = spec.partition(":")
    if kind == "local":
        return load_local(target)
    if kind == "pg":
        return load_pg(target)
    raise ValueError("corpus must look like local:PATH or pg:DSN")
