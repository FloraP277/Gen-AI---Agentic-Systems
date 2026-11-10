"""Integration test against a real PostgreSQL + pgvector database.

Skipped unless REGVEC_PG_DSN is set. CI sets it using the pgvector Docker
image; locally you can run:

    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
    REGVEC_PG_DSN=postgresql://postgres:postgres@localhost/postgres pytest tests/test_pgvector.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DSN = os.environ.get("REGVEC_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="REGVEC_PG_DSN not set")

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "title-30-sample.xml")


def test_pgvector_store_sync_search_and_delete():
    import psycopg

    from regvec.checks import run_checks
    from regvec.embed import HashingEmbedder
    from regvec.parse import parse_sections
    from regvec.pipeline import sync
    from regvec.store import PgVectorStore

    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS reg_chunks; DROP TABLE IF EXISTS reg_sections;")

    sections = parse_sections(open(SAMPLE, "rb").read())
    emb = HashingEmbedder()
    store = PgVectorStore(DSN)
    first = sync(sections, store, emb, "2026-11-01")
    assert first["new"] == 8
    assert sync(sections, store, emb, "2026-11-08")["unchanged"] == 8

    top = store.dense_search(emb.embed(["rock dust incombustible 80 percent"])[0], k=1)
    assert top[0][0]["section_id"] == "75.403"

    sync(sections[:-1], store, emb, "2026-11-15")
    assert "75.512" not in set(store.all_chunks()["section_id"])
    assert run_checks(store, embedder_dim=384)["issues"] == []
