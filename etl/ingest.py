"""
ingest.py
---------
ETL pipeline: PDF → ChromaDB vector store

Strategy:
1. Extract page-level text (for citation accuracy)
2. Extract and serialize all tables (markdown + prose)
3. Chunk text into overlapping paragraphs
4. Embed everything with sentence-transformers
5. Store in ChromaDB with rich metadata

Run with:  python etl/ingest.py
"""

import os
import sys
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from table_parser import extract_text_by_page, extract_all_tables

import chromadb
from chromadb.utils import embedding_functions

PDF_PATH = os.getenv("PDF_PATH", "data/cyber_ireland_2022.pdf")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 150     # overlap between chunks


def chunk_text(text: str, page_num: int, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split page text into overlapping chunks.
    Tries to split on paragraph boundaries first, falls back to character splits.
    """
    chunks = []

    # Split on double newlines (paragraphs)
    paragraphs = re.split(r"\n\n+", text)

    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += (" " if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + " " + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If no paragraphs found, do character-level chunking
    if not chunks and text:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])

    return [
        {
            "text": chunk,
            "page_num": page_num,
            "chunk_type": "text",
        }
        for chunk in chunks
        if len(chunk.strip()) > 50  # skip tiny fragments
    ]


# def generate_doc_id(content: str, metadata: dict) -> str:
#     """Generate a stable, unique ID for each chunk."""
#     key = f"{metadata.get('page_num', 0)}_{metadata.get('chunk_type', '')}_{content[:100]}"
#     return hashlib.md5(key.encode()).hexdigest()
_id_counter = 0

def generate_doc_id(content: str, metadata: dict) -> str:
    """Generate a stable, unique ID for each chunk."""
    global _id_counter
    _id_counter += 1
    key = f"{_id_counter}_{metadata.get('page_num', 0)}_{metadata.get('chunk_type', '')}_{content[:80]}"
    return hashlib.md5(key.encode()).hexdigest()


def build_chroma_client() -> chromadb.ClientAPI:
    """Initialize ChromaDB persistent client."""
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client


def run_etl():
    print(f"\n{'='*60}")
    print("  Cyber Ireland 2022 — ETL Ingestion Pipeline")
    print(f"{'='*60}\n")

    # Validate PDF exists
    if not Path(PDF_PATH).exists():
        print(f"[ERROR] PDF not found at: {PDF_PATH}")
        print("  Please place the Cyber Ireland 2022 PDF at data/cyber_ireland_2022.pdf")
        sys.exit(1)

    # ── Step 1: Extract text by page ──────────────────────────────────
    print("[Step 1] Extracting page text...")
    pages = extract_text_by_page(PDF_PATH)
    print(f"         {len(pages)} pages extracted\n")

    # ── Step 2: Extract tables ────────────────────────────────────────
    print("[Step 2] Extracting tables...")
    tables = extract_all_tables(PDF_PATH)
    print(f"         {len(tables)} tables found\n")

    # ── Step 3: Build all documents to index ─────────────────────────
    print("[Step 3] Building chunk corpus...")
    all_docs = []  # {id, text, metadata}

    # 3a. Text chunks (paragraph level, with page metadata)
    for page in pages:
        if not page["text"]:
            continue
        chunks = chunk_text(page["text"], page["page_num"])
        for i, chunk in enumerate(chunks):
            doc_id = generate_doc_id(chunk["text"], {**chunk, "chunk_idx": i})
            all_docs.append({
                "id": doc_id,
                "text": chunk["text"],
                "metadata": {
                    "page_num": chunk["page_num"],
                    "chunk_type": "text",
                    "chunk_idx": i,
                    "source": "page_text",
                }
            })

        # Also store full page text as a single chunk for exact citation lookup
        full_page_id = generate_doc_id(page["text"], {"page_num": page["page_num"], "type": "full_page"})
        all_docs.append({
            "id": full_page_id,
            "text": f"[Full text of page {page['page_num']}]\n{page['text']}",
            "metadata": {
                "page_num": page["page_num"],
                "chunk_type": "full_page",
                "source": "page_full",
            }
        })

    # 3b. Table chunks (markdown representation for structure-aware retrieval)
    for tbl in tables:
        # Index markdown version
        md_id = generate_doc_id(tbl["markdown"], {"page_num": tbl["page_num"], "type": "table_md"})
        all_docs.append({
            "id": md_id,
            "text": tbl["markdown"],
            "metadata": {
                "page_num": tbl["page_num"],
                "chunk_type": "table_markdown",
                "table_idx": tbl["table_idx"],
                "source": "table",
            }
        })

        # Index prose version (better semantic retrieval)
        if tbl["prose"]:
            prose_id = generate_doc_id(tbl["prose"], {"page_num": tbl["page_num"], "type": "table_prose"})
            all_docs.append({
                "id": prose_id,
                "text": tbl["prose"],
                "metadata": {
                    "page_num": tbl["page_num"],
                    "chunk_type": "table_prose",
                    "table_idx": tbl["table_idx"],
                    "source": "table",
                }
            })

    print(f"         {len(all_docs)} total chunks prepared\n")

    # ── Step 4: Initialize ChromaDB + Embedding function ─────────────
    print(f"[Step 4] Initializing ChromaDB at {CHROMA_DB_PATH}...")
    client = build_chroma_client()

    # Delete existing collection to start fresh
    try:
        client.delete_collection("cyber_ireland_2022")
        print("         Deleted existing collection")
    except Exception:
        pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.create_collection(
        name="cyber_ireland_2022",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"         Collection created with embedding model: {EMBEDDING_MODEL}\n")

    # ── Step 5: Batch upsert into ChromaDB ───────────────────────────
    print("[Step 5] Embedding and indexing chunks...")
    BATCH_SIZE = 50

    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        print(f"         Indexed {min(i + BATCH_SIZE, len(all_docs))}/{len(all_docs)} chunks...", end="\r")

    print(f"\n         ✓ All {len(all_docs)} chunks indexed\n")

    # ── Step 6: Save table index as JSON (for direct lookup by agent) ─
    print("[Step 6] Saving table index...")
    Path("data").mkdir(exist_ok=True)
    table_index = []
    for tbl in tables:
        table_index.append({
            "page_num": tbl["page_num"],
            "table_idx": tbl["table_idx"],
            "markdown": tbl["markdown"],
            "prose": tbl["prose"],
        })

    with open("data/table_index.json", "w") as f:
        json.dump(table_index, f, indent=2)
    print(f"         Saved {len(table_index)} tables to data/table_index.json\n")

    # ── Step 7: Save page text index (for exact citation) ────────────
    print("[Step 7] Saving page text index...")
    page_index = {str(p["page_num"]): p["text"] for p in pages}
    with open("data/page_index.json", "w") as f:
        json.dump(page_index, f, indent=2)
    print(f"         Saved {len(page_index)} pages to data/page_index.json\n")

    print(f"{'='*60}")
    print("  ETL COMPLETE")
    print(f"  - {len(pages)} pages indexed")
    print(f"  - {len(tables)} tables indexed")
    print(f"  - {len(all_docs)} total vector chunks")
    print(f"  - Vector store: {CHROMA_DB_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_etl()
