# Regulations-to-Vectors

An incremental pipeline that turns U.S. mine safety regulations (30 CFR, from eCFR) into clean, searchable chunks with embeddings in PostgreSQL + pgvector. It is the data layer for the [Mine Safety Regulations RAG Assistant](../mine-safety-rag).

**Stack:** Python · lxml · sentence-transformers (all-MiniLM-L6-v2) · PostgreSQL · pgvector · pytest · GitHub Actions

## Results

> **Real-data results:** _run on the real regulations and fill in: sections loaded, chunks per strategy, average chunk size, full run time, and incremental run time._

| | Sections | Chunks (fixed) | Chunks (section-aware) | Full run | Re-run, nothing changed |
| --- | --- | --- | --- | --- | --- |
| Your run | _[N]_ | _[N]_ | _[N]_ | _[N] s_ | _[N] s_ |

On the bundled sample file (8 sections), the tests confirm the pipeline:
- writes only new and changed sections on a re-run, and nothing when nothing changed;
- deletes the chunks of sections that were removed from the regulations;
- passes every post-run check (no empty chunks, no oversized chunks, no orphaned chunks, one embedding per chunk, no NaN values).

## Architecture

```mermaid
flowchart LR
    A[eCFR API<br/>Title 30 XML by part] --> B[Parse<br/>part / subpart / section<br/>paragraphs + metadata]
    B --> C{Content hash<br/>changed?}
    C -->|no| D[Skip]
    C -->|yes| E[Chunk two ways<br/>fixed-size with overlap<br/>section-aware]
    E --> F[Embed<br/>all-MiniLM-L6-v2]
    F --> G[(PostgreSQL + pgvector<br/>HNSW vector index<br/>GIN full-text index)]
    G --> H[Checks]
```

## Two chunking strategies

Both are stored side by side so the RAG project can measure which retrieves better.

| Strategy | How it works | Trade-off |
| --- | --- | --- |
| `fixed` | Windows of 200 words with a 40-word overlap | Simple and predictable, but can cut a rule in half |
| `section` | Packs whole paragraphs up to 250 words and starts every chunk with the section number and title | Each chunk keeps its context ("§ 75.400 Accumulation of combustible materials") even when retrieved alone |

## Database schema

- `reg_sections`: one row per section with part, subpart, title, URL, snapshot date, embedding model and a **content hash**.
- `reg_chunks`: chunk text and metadata, a `vector(384)` embedding with an **HNSW** index for similarity search, and a generated `tsvector` column with a **GIN** index for keyword search. Deleting a section cascades to its chunks.

## Run it

```bash
pip install -r requirements.txt
docker compose up -d            # PostgreSQL 16 with pgvector

# Load the regulations for metal/non-metal and coal mines from eCFR
PYTHONPATH=src python -m regvec.pipeline --parts 56 57 75 77 \
    --store pg:postgresql://postgres:postgres@localhost:5432/regs

# Run it again: only changed sections are re-embedded
PYTHONPATH=src python -m regvec.pipeline --parts 56 57 75 77 \
    --store pg:postgresql://postgres:postgres@localhost:5432/regs
```

Offline, with the sample file and a file-based store:

```bash
PYTHONPATH=src python -m regvec.pipeline --xml sample_data/title-30-sample.xml \
    --embedder hashing --store local:./vectors
pytest
```

`sample_data/title-30-sample.xml` copies eCFR's XML structure, but its wording was written for testing and is **not** official regulation text. `--embedder hashing` is a word-hashing stand-in used only so tests run without downloading a model; it does not understand meaning.

## Design decisions

- **Content hashing for incremental loads.** Regulations change rarely. Hashing each section and re-embedding only changed ones turns a full re-embed into a few seconds of work, and keeps embedding costs near zero if you switch to a paid embedding API.
- **One transaction per section.** A section's old chunks are deleted and new ones inserted together, so a search never sees half of an updated rule.
- **Keyword index next to the vectors.** The generated `tsvector` column lets the RAG project run hybrid search (keywords + meaning) from one table. Regulations are full of exact terms like section numbers and "float coal dust" that pure vector search can miss.
- **Section-level metadata on every chunk.** Part and section IDs let the RAG assistant filter ("only Part 75") and cite the exact rule.
- **A free local embedding model.** all-MiniLM-L6-v2 runs on a laptop CPU, so the project costs nothing to run or rerun.

## Limitations and next steps

- Tables inside regulations are flattened into text; a table-aware parser would keep rows together.
- The embedding model is general-purpose. Fine-tuning it on regulatory language, or comparing a larger model, is a natural experiment.
- Footnotes and cross-references ("as required by § 75.360") are kept as text, not linked.

## Data source

[eCFR](https://www.ecfr.gov/current/title-30/chapter-I) (Electronic Code of Federal Regulations), Title 30, Chapter I: Mine Safety and Health Administration. U.S. government work, public domain.
