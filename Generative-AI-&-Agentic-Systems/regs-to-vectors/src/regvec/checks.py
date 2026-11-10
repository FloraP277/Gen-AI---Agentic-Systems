"""Quality checks on what's in the store after a run."""
import numpy as np


def run_checks(store, embedder_dim: int, max_words: int = 300) -> dict:
    chunks = store.all_chunks()
    emb = store.embeddings()
    sections = set(store.section_hashes())
    issues = []

    empty = chunks[chunks["text"].str.strip() == ""]
    if len(empty):
        issues.append(f"{len(empty)} empty chunks")
    too_long = chunks[chunks["n_words"] > max_words]
    if len(too_long):
        issues.append(f"{len(too_long)} chunks longer than {max_words} words")
    orphans = chunks[~chunks["section_id"].isin(sections)]
    if len(orphans):
        issues.append(f"{len(orphans)} chunks whose section is missing")
    no_chunks = sections - set(chunks["section_id"])
    if no_chunks:
        issues.append(f"{len(no_chunks)} sections with no chunks")
    if len(chunks) != len(emb):
        issues.append(f"{len(chunks)} chunks but {len(emb)} embeddings")
    elif len(emb):
        if emb.shape[1] != embedder_dim:
            issues.append(f"embedding size {emb.shape[1]}, expected {embedder_dim}")
        if not np.isfinite(emb).all():
            issues.append("embeddings contain NaN or infinite values")
    dup = chunks["chunk_id"].duplicated().sum()
    if dup:
        issues.append(f"{dup} duplicate chunk ids")

    by_strategy = chunks.groupby("strategy")["n_words"].agg(["count", "mean", "max"]).round(1)
    return {
        "sections": len(sections),
        "chunks": len(chunks),
        "by_strategy": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_strategy.to_dict("index").items()},
        "issues": issues,
    }
