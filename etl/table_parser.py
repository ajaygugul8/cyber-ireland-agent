"""
table_parser.py
---------------
Dedicated table extraction and serialization from PDF pages.
Uses pdfplumber's table detection with fallback strategies.
"""

import pdfplumber
import pandas as pd
from typing import List, Dict, Any, Optional
import re


def clean_cell(value: Optional[str]) -> str:
    """Normalize a table cell value."""
    if value is None:
        return ""
    # Remove excessive whitespace and newlines
    return re.sub(r"\s+", " ", str(value)).strip()


def table_to_markdown(table: List[List[Optional[str]]], page_num: int, table_idx: int) -> str:
    """
    Convert a raw pdfplumber table (list of rows) into a markdown table string.
    This is what gets embedded into the vector store — markdown preserves structure.
    """
    if not table or len(table) < 2:
        return ""

    # Clean all cells
    cleaned = [[clean_cell(cell) for cell in row] for row in table]

    # Use first row as headers if it looks like a header
    headers = cleaned[0]
    rows = cleaned[1:]

    # Build markdown
    lines = []
    lines.append(f"<!-- Table {table_idx + 1} on Page {page_num} -->")

    # Header row
    lines.append("| " + " | ".join(headers) + " |")
    # Separator
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    # Data rows
    for row in rows:
        # Pad row if needed
        while len(row) < len(headers):
            row.append("")
        lines.append("| " + " | ".join(row[:len(headers)]) + " |")

    return "\n".join(lines)


def table_to_prose(table: List[List[Optional[str]]], page_num: int, table_idx: int) -> str:
    """
    Also generate a prose description of the table for better semantic retrieval.
    For regional data tables, this is critical so the agent can find them via
    natural language queries like 'South-West pure-play firms'.
    """
    if not table or len(table) < 2:
        return ""

    cleaned = [[clean_cell(cell) for cell in row] for row in table]
    headers = cleaned[0]
    rows = cleaned[1:]

    lines = [f"Table on page {page_num} contains the following data:"]
    for row in rows:
        if any(row):  # Skip empty rows
            row_parts = []
            for h, v in zip(headers, row):
                if h and v:
                    row_parts.append(f"{h}: {v}")
            if row_parts:
                lines.append(", ".join(row_parts) + ".")

    return " ".join(lines)


def extract_all_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all tables from the PDF with rich metadata.

    Returns a list of dicts:
    {
        "page_num": int,           # 1-indexed page number
        "table_idx": int,          # table index on that page
        "markdown": str,           # markdown representation
        "prose": str,              # natural language description
        "raw": List[List[str]],    # raw cell data
        "dataframe": pd.DataFrame  # pandas DF for numeric operations
    }
    """
    tables_data = []

    table_settings = {
        # pdfplumber settings tuned for typical report PDFs
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 5,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Try strict line-based detection first
            tables = page.extract_tables(table_settings)

            # Fallback: looser detection if nothing found
            if not tables:
                tables = page.extract_tables()

            for table_idx, table in enumerate(tables):
                if not table:
                    continue

                # Filter out single-row or empty tables
                non_empty_rows = [r for r in table if any(clean_cell(c) for c in r)]
                if len(non_empty_rows) < 2:
                    continue

                markdown = table_to_markdown(table, page_num, table_idx)
                prose = table_to_prose(table, page_num, table_idx)

                # Build DataFrame for numeric queries
                cleaned = [[clean_cell(cell) for cell in row] for row in table]
                try:
                    df = pd.DataFrame(cleaned[1:], columns=cleaned[0])
                except Exception:
                    df = pd.DataFrame(cleaned)

                tables_data.append({
                    "page_num": page_num,
                    "table_idx": table_idx,
                    "markdown": markdown,
                    "prose": prose,
                    "raw": table,
                    "dataframe": df,
                })

    print(f"[TableParser] Extracted {len(tables_data)} tables from {pdf_path}")
    return tables_data


def extract_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract full text from every page with page number metadata.
    Critically important for citation verification (Test 1).

    Returns list of:
    {
        "page_num": int,
        "text": str,
        "word_count": int
    }
    """
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Normalize whitespace
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            pages_data.append({
                "page_num": page_num,
                "text": text,
                "word_count": len(text.split()),
            })

    print(f"[TableParser] Extracted text from {len(pages_data)} pages")
    return pages_data
