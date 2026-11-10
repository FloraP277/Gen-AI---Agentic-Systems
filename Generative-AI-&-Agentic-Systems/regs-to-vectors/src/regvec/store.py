"""Where sections, chunks and embeddings are stored.

PgVectorStore: PostgreSQL with the pgvector extension (the real target).
  - reg_sections: one row per section with its content hash
  - reg_chunks: chunk text, metadata, a `vector` embedding with an HNSW index
    for similarity search, and a generated `tsvector` column with a GIN index
    for keyword search
LocalStore: files on disk (Parquet + NumPy), for development and tests.

Both support the same operations, so the pipeline doesn't care which is used.
"""
import json
import os

import numpy as np
import pandas as pd

CHUNK_COLS = ["chunk_id", "section_id", "part", "strategy", "position", "text", "n_words"]


class LocalStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._sections_file = os.path.join(path, "sections.json")
        self._chunks_file = os.path.join(path, "chunks.parquet")
        self._emb_file = os.path.join(path, "embeddings.npy")
        self.sections = json.load(open(self._sections_file)) if os.path.exists(self._sections_file) else {}
        if os.path.exists(self._chunks_file):
            self.chunks = pd.read_parquet(self._chunks_file)
            self.emb = np.load(self._emb_file)
        else:
            self.chunks = pd.DataFrame(columns=CHUNK_COLS)
            self.emb = np.zeros((0, 0), dtype=np.float32)

    def section_hashes(self) -> dict:
        return {k: v["content_hash"] for k, v in self.sections.items()}

    def replace_section(self, section, chunks, embeddings: np.ndarray, snapshot_date: str, model: str):
        self.delete_sections([section.section_id])
        self.sections[section.section_id] = {
            "part": section.part, "subpart": section.subpart, "title": section.title,
            "content_hash": section.content_hash, "snapshot_date": snapshot_date,
            "url": section.url, "embedding_model": model,
        }
        rows = pd.DataFrame([{**{c: getattr(ch, c) for c in CHUNK_COLS[:-1]}, "n_words": ch.n_words}
                             for ch in chunks], columns=CHUNK_COLS)
        self.chunks = pd.concat([self.chunks, rows], ignore_index=True) if len(self.chunks) else rows
        self.emb = embeddings if self.emb.size == 0 else np.vstack([self.emb, embeddings])

    def delete_sections(self, section_ids):
        ids = set(section_ids)
        for s in ids:
            self.sections.pop(s, None)
        if len(self.chunks):
            keep = ~self.chunks["section_id"].isin(ids)
            self.chunks = self.chunks[keep].reset_index(drop=True)
            self.emb = self.emb[keep.to_numpy()]

    def save(self):
        json.dump(self.sections, open(self._sections_file, "w"), indent=1)
        self.chunks.to_parquet(self._chunks_file, index=False)
        np.save(self._emb_file, self.emb)

    def all_chunks(self) -> pd.DataFrame:
        return self.chunks.copy()

    def embeddings(self) -> np.ndarray:
        return self.emb

    def dense_search(self, vec: np.ndarray, k: int = 5, strategy: str = "section", part: str | None = None):
        mask = (self.chunks["strategy"] == strategy).to_numpy(copy=True)
        if part is not None:
            mask = mask & (self.chunks["part"] == str(part)).to_numpy()
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        scores = self.emb[idx] @ vec
        order = idx[np.argsort(-scores)[:k]]
        return [(self.chunks.iloc[i].to_dict(), float(self.emb[i] @ vec)) for i in order]


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS reg_sections (
    section_id     text PRIMARY KEY,
    part           text NOT NULL,
    subpart        text,
    title          text NOT NULL,
    content_hash   text NOT NULL,
    snapshot_date  date NOT NULL,
    url            text,
    embedding_model text NOT NULL
);
CREATE TABLE IF NOT EXISTS reg_chunks (
    chunk_id    text PRIMARY KEY,
    section_id  text NOT NULL REFERENCES reg_sections(section_id) ON DELETE CASCADE,
    part        text NOT NULL,
    strategy    text NOT NULL,
    position    int  NOT NULL,
    text        text NOT NULL,
    n_words     int  NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS reg_chunks_embedding_idx ON reg_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS reg_chunks_tsv_idx ON reg_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS reg_chunks_filter_idx ON reg_chunks (strategy, part);
"""


class PgVectorStore:
    """PostgreSQL + pgvector. Needs `pip install "psycopg[binary]" pgvector`."""

    def __init__(self, dsn: str, dim: int = 384):
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(dsn, autocommit=False)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()
        register_vector(self.conn)
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA.format(dim=dim))
        self.conn.commit()

    def section_hashes(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute("SELECT section_id, content_hash FROM reg_sections")
            return dict(cur.fetchall())

    def replace_section(self, section, chunks, embeddings, snapshot_date: str, model: str):
        # One transaction per section: readers never see a half-updated rule.
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("DELETE FROM reg_sections WHERE section_id = %s", (section.section_id,))
            cur.execute(
                "INSERT INTO reg_sections VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (section.section_id, section.part, section.subpart, section.title,
                 section.content_hash, snapshot_date, section.url, model))
            cur.executemany(
                "INSERT INTO reg_chunks (chunk_id, section_id, part, strategy, position, text, n_words, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [(c.chunk_id, c.section_id, c.part, c.strategy, c.position, c.text, c.n_words, e)
                 for c, e in zip(chunks, embeddings)])

    def delete_sections(self, section_ids):
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("DELETE FROM reg_sections WHERE section_id = ANY(%s)", (list(section_ids),))

    def save(self):
        self.conn.commit()

    def all_chunks(self) -> pd.DataFrame:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT {', '.join(CHUNK_COLS)} FROM reg_chunks ORDER BY chunk_id")
            return pd.DataFrame(cur.fetchall(), columns=CHUNK_COLS)

    def embeddings(self) -> np.ndarray:
        with self.conn.cursor() as cur:
            cur.execute("SELECT embedding FROM reg_chunks ORDER BY chunk_id")
            return np.array([r[0] for r in cur.fetchall()], dtype=np.float32)

    def dense_search(self, vec, k: int = 5, strategy: str = "section", part: str | None = None):
        sql = ("SELECT chunk_id, section_id, part, strategy, position, text, n_words,"
               " 1 - (embedding <=> %s) AS score FROM reg_chunks WHERE strategy = %s")
        params = [vec, strategy]
        if part is not None:
            sql += " AND part = %s"
            params.append(str(part))
        sql += " ORDER BY embedding <=> %s LIMIT %s"
        params += [vec, k]
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return [(dict(zip(CHUNK_COLS, r[:-1])), float(r[-1])) for r in cur.fetchall()]


def open_store(spec: str, dim: int = 384):
    """'local:./vectors' or 'pg:postgresql://user:pass@host/db'."""
    kind, _, target = spec.partition(":")
    if kind == "local":
        return LocalStore(target)
    if kind == "pg":
        return PgVectorStore(target, dim)
    raise ValueError("store must look like local:PATH or pg:DSN")
