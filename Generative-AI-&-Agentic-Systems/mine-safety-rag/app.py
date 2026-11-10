"""Streamlit app: ask a question, see the answer next to the passages it cites.

    GEMINI_API_KEY=... CORPUS=local:../regs-to-vectors/vectors streamlit run app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from msrag import retrieve                    # noqa: E402
from msrag.answer import NOT_IN_SOURCES, answer   # noqa: E402
from msrag.corpus import load                  # noqa: E402
from msrag.embed import get_embedder           # noqa: E402
from msrag.llm import GeminiLLM                # noqa: E402

st.set_page_config(page_title="Mine Safety Regulations Assistant", layout="wide")


@st.cache_resource
def setup():
    corpus = load(os.environ.get("CORPUS", "local:../regs-to-vectors/vectors"))
    return corpus, get_embedder(os.environ.get("EMBEDDER", "minilm")), retrieve.CrossEncoderReranker()


corpus, embedder, reranker = setup()
st.title("Mine Safety Regulations Assistant")
st.caption("Answers come only from 30 CFR text retrieved for your question, with the section cited. "
           "This is a learning project, not legal advice.")

parts = ["All"] + sorted(corpus.chunks["part"].astype(str).unique().tolist())
col1, col2 = st.columns([3, 1])
question = col1.text_input("Question", placeholder="How often must working places be examined?")
part = col2.selectbox("Limit to part", parts)

if question:
    sub = corpus.subset("section", None if part == "All" else part)
    retriever = retrieve.build("hybrid_rerank", sub, embedder, reranker)
    hits = retriever.search(question, 5)
    a = answer(question, GeminiLLM(), retriever, corpus.sections)
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Answer")
        if a.abstained:
            st.warning("The retrieved regulations don't answer this question.")
        else:
            st.write(a.text)
        if a.invalid_citations:
            st.error(f"Cited sections that were not retrieved: {', '.join(a.invalid_citations)}")
    with right:
        st.subheader("Sources")
        for h in hits:
            meta = corpus.sections.get(h.section_id, {})
            marker = "✅ cited" if h.section_id in a.cited_sections else ""
            with st.expander(f"§ {h.section_id} {meta.get('title', '')} {marker}"):
                st.write(h.text)
                if meta.get("url"):
                    st.markdown(f"[Read on eCFR]({meta['url']})")
