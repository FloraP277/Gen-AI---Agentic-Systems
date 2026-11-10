"""Answer a question from retrieved regulation chunks, with citations."""
import re
from dataclasses import dataclass, field

NOT_IN_SOURCES = "NOT_IN_SOURCES"

RAG_SYSTEM = f"""You answer questions about U.S. mine safety regulations (30 CFR).
Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- After every claim, cite the section it came from in square brackets, like [§ 75.400].
- If the sources do not contain the answer, reply with exactly: {NOT_IN_SOURCES}
- Be concise: two to five sentences."""

NO_RAG_SYSTEM = """You answer questions about U.S. mine safety regulations (30 CFR).
Cite section numbers in square brackets, like [§ 75.400], when you refer to a rule.
Be concise: two to five sentences."""

CITATION = re.compile(r"§\s*(\d+\.\d+)")


@dataclass
class Answer:
    question: str
    text: str
    retrieved_sections: list = field(default_factory=list)
    cited_sections: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)   # cited but not retrieved
    abstained: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def format_sources(hits, sections: dict) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        title = sections.get(h.section_id, {}).get("title", "")
        lines.append(f"[{i}] § {h.section_id} {title}\n{h.text}")
    return "\n\n".join(lines)


def cited(text: str) -> list[str]:
    return sorted(set(CITATION.findall(text)))


def answer(question: str, llm, retriever=None, sections: dict | None = None, k: int = 5) -> Answer:
    """With a retriever: RAG. Without one: the LLM alone (the Lab 2 'no-RAG' baseline)."""
    if retriever is None:
        res = llm.generate(f"Question: {question}", system=NO_RAG_SYSTEM)
        return Answer(question, res.text.strip(), [], cited(res.text), [],
                      NOT_IN_SOURCES in res.text, res.input_tokens, res.output_tokens, res.seconds)

    hits = retriever.search(question, k)
    retrieved = [h.section_id for h in hits]
    prompt = f"Sources:\n\n{format_sources(hits, sections or {})}\n\nQuestion: {question}"
    res = llm.generate(prompt, system=RAG_SYSTEM)
    text = res.text.strip()
    c = cited(text)
    return Answer(question, text, retrieved, c, [s for s in c if s not in retrieved],
                  text.startswith(NOT_IN_SOURCES), res.input_tokens, res.output_tokens, res.seconds)
