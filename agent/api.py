"""
api.py
------
FastAPI backend exposing the /query endpoint.

Startup: uvicorn agent.api:app --reload --port 8000

Endpoint: POST /query
  Body: {"query": "your question here"}
  Returns: Full agent response with reasoning trace
"""

import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Cyber Ireland 2022 — Agentic Query API",
    description="""
    Autonomous intelligence backend for the Cyber Ireland 2022 report.
    
    Powered by a LangChain ReAct agent with tools for:
    - Semantic vector search
    - Exact keyword search with citations
    - Table extraction and comparison
    - Safe mathematical computation (CAGR, percentages)
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question about the Cyber Ireland 2022 report")
    save_trace: bool = Field(default=True, description="Whether to save the reasoning trace to disk")


class ToolCallSummary(BaseModel):
    tool: str
    input_preview: str


class ReasoningStep(BaseModel):
    tool: str
    tool_input: str
    observation: str
    thought: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    reasoning_trace: list
    tool_calls: list
    execution_time_seconds: float
    timestamp: str
    llm_provider: str
    llm_model: str
    trace_saved_to: Optional[str] = None


# ── Lazy agent loading ───────────────────────────────────────────────────────

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from agent.agent_runner import get_agent_runner
        _agent = get_agent_runner()
    return _agent


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Cyber Ireland 2022 Agentic Query Backend",
        "status": "running",
        "endpoints": {
            "POST /query": "Submit a natural language query",
            "GET /health": "Health check",
            "GET /docs": "Swagger UI",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Submit a natural language query about the Cyber Ireland 2022 report.
    
    The agent will autonomously:
    1. Break down the query into retrieval steps
    2. Search the document using relevant tools
    3. Verify facts and compute any required math
    4. Return a cited, accurate answer with its full reasoning trace
    
    **Example queries:**
    - "What is the total number of jobs reported?"
    - "Compare Pure-Play cybersecurity firms in the South-West vs national average"
    - "What CAGR is needed to hit the 2030 job target from the 2022 baseline?"
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    agent = get_agent()

    try:
        result = agent.run(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Save trace in background if requested
    trace_path = None
    if request.save_trace:
        label = request.query[:50]
        trace_path = agent.save_trace(result, label=label)
        result["trace_saved_to"] = trace_path

    return QueryResponse(**{k: v for k, v in result.items() if k in QueryResponse.model_fields})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
