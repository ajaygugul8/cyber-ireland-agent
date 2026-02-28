"""
tools.py
--------
All tools available to the LangChain ReAct agent.

Design principles:
- Each tool has a clear, narrow responsibility
- Tools return structured data (not just strings) where possible
- The calculator tool uses Python eval() with a safe math namespace —
  LLMs must NEVER compute CAGR or percentages themselves
- Citation tool verifies against raw page text for hallucination prevention
"""

import os
import json
import math
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain.tools import tool

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Shared state (loaded once) ──────────────────────────────────────────────

_collection = None
_page_index = None
_table_index = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = client.get_collection(
            name="cyber_ireland_2022",
            embedding_function=embedding_fn,
        )
    return _collection


def _get_page_index() -> dict:
    global _page_index
    if _page_index is None:
        with open("data/page_index.json") as f:
            _page_index = json.load(f)
    return _page_index


def _get_table_index() -> list:
    global _table_index
    if _table_index is None:
        with open("data/table_index.json") as f:
            _table_index = json.load(f)
    return _table_index


# ── Tools ───────────────────────────────────────────────────────────────────

@tool
def vector_search(query: str, n_results: int = 5) -> str:
    """
    Search the Cyber Ireland 2022 document using semantic similarity.
    Use this as your primary retrieval tool for natural language queries.
    
    Args:
        query: Natural language query string
        n_results: Number of results to return (default 5, max 10)
    
    Returns:
        JSON string with matching chunks, their page numbers, and relevance scores.
    """
    n_results = min(int(n_results), 10)
    collection = _get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "page_num": meta.get("page_num"),
            "chunk_type": meta.get("chunk_type"),
            "relevance_score": round(1 - dist, 3),  # cosine distance → similarity
            "excerpt": doc[:800],  # truncate for context window
        })

    return json.dumps(output, indent=2)


@tool
def keyword_search(keyword: str) -> str:
    """
    Search for an exact keyword or phrase across all pages of the document.
    Use this when you need precise string matching, e.g., finding an exact number
    or a specific term like '6,500 jobs' or 'Pure-Play'.
    
    Args:
        keyword: Exact string to search for (case-insensitive)
    
    Returns:
        JSON with all pages containing the keyword and surrounding context.
    """
    page_index = _get_page_index()
    keyword_lower = keyword.lower()

    matches = []
    for page_num_str, page_text in page_index.items():
        if keyword_lower in page_text.lower():
            # Find the position and extract context
            idx = page_text.lower().find(keyword_lower)
            start = max(0, idx - 200)
            end = min(len(page_text), idx + 300)
            context = page_text[start:end]

            matches.append({
                "page_num": int(page_num_str),
                "context": context,
                "keyword_found": keyword,
            })

    if not matches:
        return json.dumps({"result": f"Keyword '{keyword}' not found in document."})

    return json.dumps({"total_matches": len(matches), "matches": matches}, indent=2)


@tool
def get_page_text(page_num: int) -> str:
    """
    Retrieve the complete raw text of a specific page from the document.
    Use this for citation verification — when you need to confirm exactly what
    appears on a specific page before quoting it.
    
    Args:
        page_num: Page number (1-indexed)
    
    Returns:
        Full text content of the page.
    """
    page_index = _get_page_index()
    text = page_index.get(str(page_num))

    if text is None:
        return f"Page {page_num} not found. Document has {len(page_index)} pages."

    return f"=== PAGE {page_num} ===\n\n{text}"


@tool
def get_all_tables(page_num: Optional[int] = None) -> str:
    """
    Retrieve tables extracted from the document, optionally filtered by page.
    Use this to access regional data tables, comparison tables, or statistical tables.
    
    Args:
        page_num: Optional page number to filter by. If None, returns all tables.
    
    Returns:
        JSON list of tables with their markdown representation and page numbers.
    """
    table_index = _get_table_index()

    if page_num is not None:
        tables = [t for t in table_index if t["page_num"] == int(page_num)]
    else:
        tables = table_index

    if not tables:
        return f"No tables found{' on page ' + str(page_num) if page_num else ''}."

    # Return markdown + page num for each table
    result = []
    for tbl in tables:
        result.append({
            "page_num": tbl["page_num"],
            "table_idx": tbl["table_idx"],
            "markdown": tbl["markdown"],
        })

    return json.dumps(result, indent=2)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely using Python.
    ALWAYS use this tool for any arithmetic, percentages, or CAGR calculations.
    Never attempt math in your own reasoning — use this tool instead.
    
    Supported functions: sqrt, log, log10, exp, pow, abs, round, ceil, floor
    Supported operators: +, -, *, /, **, %
    
    For CAGR calculation use:
        (end_value / start_value) ** (1 / years) - 1
    
    Args:
        expression: Python math expression as a string
    
    Returns:
        Result of the calculation with the expression echoed back.
    
    Examples:
        "2030 - 2022"  → 8
        "(17000 / 6500) ** (1/8) - 1"  → CAGR as decimal
        "round(((17000 / 6500) ** (1/8) - 1) * 100, 2)"  → CAGR as percentage
    """
    # Safe math namespace — no builtins, only math functions
    safe_namespace = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "pow": math.pow,
        "abs": abs,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Sanitize: only allow math chars
        if re.search(r"[a-zA-Z_]", expression):
            # Allow known function names only
            allowed_words = set(safe_namespace.keys())
            words_in_expr = set(re.findall(r"[a-zA-Z_]+", expression))
            disallowed = words_in_expr - allowed_words
            if disallowed:
                return json.dumps({
                    "error": f"Disallowed identifiers: {disallowed}. Only math functions allowed."
                })

        result = eval(expression, safe_namespace)  # noqa: S307
        return json.dumps({
            "expression": expression,
            "result": result,
            "result_formatted": f"{result:.6f}" if isinstance(result, float) else str(result),
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


@tool
def search_tables_for_keyword(keyword: str) -> str:
    """
    Search all extracted tables for a specific keyword.
    Best for finding regional data (e.g., 'South-West', 'National Average', 'Pure-Play').
    
    Args:
        keyword: Text to search for in table content
    
    Returns:
        All tables containing the keyword with full markdown.
    """
    table_index = _get_table_index()
    keyword_lower = keyword.lower()

    matches = []
    for tbl in table_index:
        if (keyword_lower in tbl["markdown"].lower() or
                keyword_lower in tbl.get("prose", "").lower()):
            matches.append({
                "page_num": tbl["page_num"],
                "table_idx": tbl["table_idx"],
                "markdown": tbl["markdown"],
            })

    if not matches:
        return json.dumps({"result": f"No tables found containing '{keyword}'."})

    return json.dumps({
        "keyword": keyword,
        "tables_found": len(matches),
        "tables": matches,
    }, indent=2)


# Export all tools as a list for the agent
ALL_TOOLS = [
    vector_search,
    keyword_search,
    get_page_text,
    get_all_tables,
    calculator,
    search_tables_for_keyword,
]
