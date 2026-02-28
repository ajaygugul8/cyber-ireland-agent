# Cyber Ireland 2022 — Agentic Knowledge Backend

A production-grade autonomous intelligence system that transforms the Cyber Ireland 2022 PDF report into a queryable knowledge base with multi-step reasoning, verifiable citations, and reliable math.

---

## Architecture Overview

```
PDF
 │
 ▼
┌──────────────────────────────────┐
│  ETL Pipeline (etl/ingest.py)    │
│  ├── pdfplumber  (text + tables) │
│  ├── Chunking with page metadata │
│  └── ChromaDB vector store       │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│  FastAPI Backend (agent/api.py)  │
│  POST /query                     │
│                                  │
│  ┌────────────────────────────┐  │
│  │   LangChain Agent          │  │
│  │   (ReAct / Tool-calling)   │  │
│  │                            │  │
│  │  Tools:                    │  │
│  │  ├── vector_search         │  │
│  │  ├── keyword_search        │  │
│  │  ├── get_page_text         │  │
│  │  └── calculator            │  │
│  └────────────────────────────┘  │
└──────────────────────────────────┘
           │
           ▼
    JSON response with:
    - answer
    - citations (page, excerpt)
    - agent reasoning trace
```

### Why These Choices?

| Decision | Choice | Justification |
|----------|--------|---------------|
| PDF parsing | `pdfplumber` | Best-in-class table extraction; preserves layout better than PyMuPDF for structured reports |
| Chunking | Page-level + paragraph-level hybrid | Page-level preserves citation integrity; paragraph-level improves semantic retrieval |
| Vector DB | ChromaDB (local) | Zero-infrastructure, persists to disk, easy to swap for Pinecone/Weaviate in prod |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, local, no API cost; swap for OpenAI `text-embedding-3-small` for production |
| Agent framework | LangChain ReAct agent | Transparent reasoning trace; tool-use with self-correction; structured output |
| LLM | Claude claude-sonnet-4-6 via Anthropic API (configurable) | Strong instruction-following; reliable tool selection |
| Math | Dedicated `calculator` tool | LLMs must not do CAGR/arithmetic themselves; Python executes it deterministically |

---

## Project Structure

```
cyber-ireland-agent/
├── README.md
├── requirements.txt
├── .env.example
├── etl/
│   ├── ingest.py          # ETL pipeline: PDF → ChromaDB
│   └── table_parser.py    # Dedicated table extraction + serialization
├── agent/
│   ├── api.py             # FastAPI app with /query endpoint
│   ├── agent_runner.py    # LangChain agent + tool definitions
│   └── tools.py           # All tool implementations
├── data/
│   └── chroma_db/         # Persisted vector store (auto-created)
├── logs/
│   └── traces/            # JSON execution traces (auto-created)
└── tests/
    └── run_eval.py        # Runs all 3 test queries and saves traces
```

---

## Setup & Execution

### 1. Prerequisites

```bash
python 3.10+
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

### 4. Download the PDF

Place the Cyber Ireland 2022 report PDF at:
```
data/cyber_ireland_2022.pdf
```

### 5. Run the ETL pipeline

```bash
python etl/ingest.py
```

This will:
- Extract all text and tables from the PDF with page-level metadata
- Serialize tables as structured markdown
- Embed and store all chunks in ChromaDB
- Print a summary of what was indexed

Expected output:
```
[ETL] Extracting text from 40 pages...
[ETL] Found 12 tables across document
[ETL] Indexed 187 chunks into ChromaDB
[ETL] Done. Vector store saved to data/chroma_db/
```

### 6. Start the API server

```bash
uvicorn agent.api:app --reload --port 8000
```

### 7. Run the 3 evaluation queries

```bash
python tests/run_eval.py
```

This executes all three test queries and saves detailed JSON traces to `logs/traces/`.

### 8. Manual query via curl

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total number of jobs reported?"}'
```

---

## Limitations & Production Scaling

### Current Limitations

1. **OCR dependency**: If the PDF is scanned (image-based), pdfplumber will miss text. Solution: add Tesseract OCR fallback via `pdf2image` + `pytesseract`.

2. **Table spanning pages**: Tables that span multiple pages are split into separate extractions. Solution: implement a page-merge heuristic based on column header repetition.

3. **Local embeddings**: `all-MiniLM-L6-v2` is fast but less accurate than OpenAI/Cohere embeddings for domain-specific text.

4. **Single PDF**: The pipeline is document-scoped, not a general knowledge base.

5. **No re-ranking**: Pure vector similarity without cross-encoder re-ranking may surface less-relevant chunks.

### Production Scaling Path

| Concern | Solution |
|---------|----------|
| Multiple documents | Add document ID metadata; partition ChromaDB collections |
| Better retrieval | Add BM25 hybrid search + cross-encoder re-ranker (FlashRank) |
| Observability | Integrate LangSmith or Langfuse for trace storage |
| Vector DB | Swap ChromaDB → Pinecone or Weaviate for horizontal scale |
| LLM costs | Cache frequent queries with Redis; use `claude-haiku` for cheap retrieval steps |
| Table accuracy | Use Unstructured.io or Camelot for complex multi-header tables |
| Auth | Add API key middleware to FastAPI |
| Async load | Use background tasks for ETL; expose job status endpoint |
