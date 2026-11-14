# Generative AI and Agentic Systems

LLM projects covering retrieval-augmented generation, agents with tool calling, fine-tuning and evaluation. Several use U.S. mine safety data, which connects to my [Mine Safety Lakehouse](https://github.com/FloraP277/Data-Engineering/tree/main/data-engineering/mine-safety-lakehouse). Each project has its own README with an architecture diagram, setup steps and tests.

| Project | What it does | Stack | Status |
| --- | --- | --- | --- |
| [Mine Safety Regulations RAG Assistant](Generative-AI-%26-Agentic-Systems/mine-safety-rag) | Answers questions about U.S. mine safety regulations (30 CFR) with a citation for every answer, comparing no retrieval, dense, BM25, hybrid and reranked search | sentence-transformers, pgvector, Gemini 2.5 Flash, Streamlit | Built and tested; evaluation run pending |
| [Regulations-to-Vectors](Generative-AI-%26-Agentic-Systems/regs-to-vectors) | Incremental pipeline that turns eCFR regulations into chunks and embeddings in PostgreSQL + pgvector; the data layer for the RAG assistant | Python, lxml, sentence-transformers, pgvector | Tested on a sample file; full run pending |
| [Mine Operations Agent with a Custom MCP Server](Generative-AI-%26-Agentic-Systems/mine-ops-agent) | AI analyst that writes mine safety reports by calling tools from my own Model Context Protocol server, compared with a fixed workflow | MCP Python SDK, Gemini tool calling, pydantic | Built and tested; comparison run pending |
| [LLM Evaluation Harness](Generative-AI-%26-Agentic-Systems/llm-eval-harness) | Measures which prompting techniques and sampling settings work for routing bank customer messages to 77 intents with structured JSON output | Gemini 2.5 Flash, Hugging Face Datasets, scikit-learn | Built and tested; experiment run pending |
| [Fine-Tuning a Small LLM for Banking Support](Generative-AI-%26-Agentic-Systems/banking-intent-finetune) | Fine-tunes Qwen2.5-0.5B with LoRA (SFT, then DPO) and compares it with prompting a larger model on accuracy, speed and cost | Transformers, PEFT, TRL, Google Colab | Built and tested; training run pending |

## About me

I'm Flora Phiri, a data and AI engineer completing an M.S. in Information Systems Technology at George Washington University (December 2026), focused on cloud and generative AI.

[LinkedIn](https://www.linkedin.com/in/flora-phiri/) · [More projects](https://github.com/FloraP277)
