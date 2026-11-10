"""Regulations -> sections -> chunks -> embeddings -> vector store.

Incremental: each section's content hash is compared with what is stored.
Only new or changed sections are re-chunked and re-embedded, and sections that
disappeared from the regulations are deleted.

    # Real run: download parts of 30 CFR from eCFR, embed with MiniLM, load into Postgres
    python -m regvec.pipeline --parts 46 48 56 57 75 77 --embedder minilm \
        --store pg:postgresql://localhost/regs

    # Offline: a saved XML file into a local store
    python -m regvec.pipeline --xml sample_data/title-30-sample.xml --embedder hashing --store local:./vectors
"""
import argparse
import datetime as dt
import json
import time

from .checks import run_checks
from .chunk import STRATEGIES
from .embed import get_embedder
from .parse import download_part, latest_date, parse_sections
from .store import open_store


def sync(sections, store, embedder, snapshot_date: str, strategies=("fixed", "section")) -> dict:
    start = time.perf_counter()
    stored = store.section_hashes()
    incoming = {s.section_id: s for s in sections}
    if len(incoming) != len(sections):
        raise ValueError("duplicate section ids in the input")

    new = [s for sid, s in incoming.items() if sid not in stored]
    changed = [s for sid, s in incoming.items() if sid in stored and stored[sid] != s.content_hash]
    removed = [sid for sid in stored if sid not in incoming]

    chunks_written = 0
    for section in new + changed:
        chunks = [c for name in strategies for c in STRATEGIES[name](section)]
        vectors = embedder.embed([c.text for c in chunks])
        store.replace_section(section, chunks, vectors, snapshot_date, embedder.name)
        chunks_written += len(chunks)
    if removed:
        store.delete_sections(removed)
    store.save()

    return {
        "snapshot_date": snapshot_date,
        "sections_in_source": len(incoming),
        "new": len(new), "changed": len(changed), "removed": len(removed),
        "unchanged": len(incoming) - len(new) - len(changed),
        "chunks_written": chunks_written,
        "seconds": round(time.perf_counter() - start, 2),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--parts", nargs="+", help="Parts of 30 CFR to download from eCFR")
    src.add_argument("--xml", nargs="+", help="Local eCFR XML files")
    p.add_argument("--date", default="latest", help="eCFR snapshot date (YYYY-MM-DD) or 'latest'")
    p.add_argument("--embedder", default="minilm", choices=["minilm", "hashing"])
    p.add_argument("--store", default="local:./vectors")
    args = p.parse_args(argv)

    if args.parts:
        date = latest_date() if args.date == "latest" else args.date
        sections = [s for part in args.parts for s in parse_sections(download_part(part, date))]
    else:
        date = dt.date.today().isoformat() if args.date == "latest" else args.date
        sections = [s for path in args.xml for s in parse_sections(open(path, "rb").read())]

    embedder = get_embedder(args.embedder)
    store = open_store(args.store, dim=getattr(embedder, "dim", 384))
    report = sync(sections, store, embedder, date)
    report["checks"] = run_checks(store, embedder_dim=getattr(embedder, "dim", 384))
    print(json.dumps(report, indent=2))
    if report["checks"]["issues"]:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main()
