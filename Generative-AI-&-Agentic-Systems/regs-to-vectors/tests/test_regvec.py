import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from regvec.checks import run_checks
from regvec.chunk import fixed_chunks, section_chunks
from regvec.embed import HashingEmbedder
from regvec.parse import Section, parse_sections
from regvec.pipeline import sync
from regvec.store import LocalStore

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "sample_data", "title-30-sample.xml")


@pytest.fixture
def sections():
    return parse_sections(open(SAMPLE, "rb").read())


def long_section(n_paragraphs=12, words=40):
    paras = [" ".join(f"word{p}_{w}" for w in range(words)) for p in range(n_paragraphs)]
    return Section("99.1", "99", "Subpart A", "Long test section", paras)


def test_parse_reads_parts_subparts_and_skips_reserved(sections):
    ids = [s.section_id for s in sections]
    assert ids == ["56.14100", "56.14101", "56.14107", "56.14130", "56.18002", "75.400", "75.403", "75.512"]
    s = sections[0]
    assert s.part == "56" and s.subpart == "Subpart M"
    assert s.title == "Safety defects; examination, correction and records"
    assert len(s.paragraphs) == 3
    assert s.url.endswith("section-56.14100")


def test_fixed_chunks_overlap_and_cover_all_words():
    s = long_section()
    chunks = fixed_chunks(s, size=100, overlap=20)
    assert all(c.n_words <= 100 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]                       # 20-word overlap
    covered = set(w for c in chunks for w in c.text.split())
    assert covered == set(s.text.split())                   # nothing dropped


def test_section_chunks_keep_paragraphs_whole_and_add_header():
    s = long_section()
    chunks = section_chunks(s, max_words=150)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith("§ 99.1 Long test section.")
        assert c.n_words <= 150
        body = c.text.split("\n")[1:]
        assert all(p in s.paragraphs for p in body)         # no paragraph split


def test_sync_is_incremental(tmp_path, sections):
    store = LocalStore(str(tmp_path / "v"))
    emb = HashingEmbedder()

    first = sync(sections, store, emb, "2026-11-01")
    assert (first["new"], first["changed"], first["removed"]) == (8, 0, 0)

    again = sync(sections, LocalStore(str(tmp_path / "v")), emb, "2026-11-08")
    assert (again["new"], again["changed"], again["unchanged"], again["chunks_written"]) == (0, 0, 8, 0)

    edited = [Section(s.section_id, s.part, s.subpart, s.title, list(s.paragraphs)) for s in sections]
    edited[0].paragraphs[0] += " Amended."
    edited = edited[:-1]                                     # last section repealed
    store = LocalStore(str(tmp_path / "v"))
    third = sync(edited, store, emb, "2026-11-15")
    assert (third["new"], third["changed"], third["removed"], third["unchanged"]) == (0, 1, 1, 6)
    assert "75.512" not in set(store.all_chunks()["section_id"])
    assert run_checks(store, embedder_dim=384)["issues"] == []


def test_checks_catch_problems(tmp_path, sections):
    store = LocalStore(str(tmp_path / "v"))
    sync(sections, store, HashingEmbedder(), "2026-11-01")
    store.emb[0, 0] = np.nan
    store.chunks.loc[1, "text"] = "   "
    issues = run_checks(store, embedder_dim=384)["issues"]
    assert any("NaN" in i for i in issues)
    assert any("empty" in i for i in issues)


def test_dense_search_finds_the_right_rule_and_filters_by_part(tmp_path, sections):
    store = LocalStore(str(tmp_path / "v"))
    emb = HashingEmbedder()
    sync(sections, store, emb, "2026-11-01")
    q = emb.embed(["rock dust incombustible content 80 percent"])[0]
    top = store.dense_search(q, k=1, strategy="section")
    assert top[0][0]["section_id"] == "75.403"
    only_56 = store.dense_search(q, k=3, strategy="section", part="56")
    assert all(r[0]["part"] == "56" for r in only_56)
