"""
agent_runner.py - Updated for LangChain 0.3.x
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")


def _build_llm():
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=LLM_MODEL, temperature=0, max_tokens=4096,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=LLM_MODEL, temperature=0, max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=LLM_MODEL, temperature=0, max_tokens=4096,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


SYSTEM_PROMPT = """You are an expert data analyst querying the Cyber Ireland 2022 Cybersecurity Industry Report.

Answer questions with absolute factual accuracy, including exact page citations.

## CRITICAL RULES:
1. NEVER perform arithmetic yourself. Always use the `calculator` tool for any math including CAGR.
2. Always cite your sources with exact page numbers.
3. Verify numbers using `get_page_text` or `keyword_search` before stating them.
4. For tables use `search_tables_for_keyword` or `get_all_tables`.
5. If retrieval fails, try alternative keywords — do not guess.

## TOOL SELECTION GUIDE:
- Number/fact queries: keyword_search → get_page_text to verify → answer with citation
- Comparative queries: search_tables_for_keyword → extract both values → compare
- CAGR/math queries: find both numbers → calculator → present result
"""


class AgentRunner:
    def __init__(self):
        from agent.tools import ALL_TOOLS
        self.tools = ALL_TOOLS
        self.llm = _build_llm()
        self._build_agent()

    def _build_agent(self):
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def run(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        print(f"\n{'─'*60}")
        print(f"QUERY: {query}")
        print(f"{'─'*60}")

        try:
            result = self.executor.invoke({"input": query})
        except Exception as e:
            return {
                "query": query,
                "answer": f"Agent encountered an error: {str(e)}",
                "error": str(e),
                "reasoning_trace": [],
                "tool_calls": [],
                "execution_time_seconds": time.time() - start_time,
                "timestamp": datetime.utcnow().isoformat(),
                "llm_provider": LLM_PROVIDER,
                "llm_model": LLM_MODEL,
            }

        elapsed = time.time() - start_time
        reasoning_trace = []
        tool_calls_summary = []

        for step in result.get("intermediate_steps", []):
            action, observation = step
            step_dict = {
                "tool": getattr(action, "tool", str(action)),
                "tool_input": str(getattr(action, "tool_input", "")),
                "observation": str(observation)[:2000],
                "thought": getattr(action, "log", "").strip(),
            }
            reasoning_trace.append(step_dict)
            tool_calls_summary.append({
                "tool": step_dict["tool"],
                "input_preview": step_dict["tool_input"][:200],
            })

        return {
            "query": query,
            "answer": result.get("output", ""),
            "reasoning_trace": reasoning_trace,
            "tool_calls": tool_calls_summary,
            "execution_time_seconds": round(elapsed, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL,
        }

    def save_trace(self, response: Dict[str, Any], label: str = "") -> str:
        logs_path = Path(os.getenv("LOGS_PATH", "logs/traces"))
        logs_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_").replace("/", "-")[:40]
        filepath = logs_path / f"{timestamp}_{safe_label}.json"
        with open(filepath, "w") as f:
            json.dump(response, f, indent=2, default=str)
        print(f"\n[Trace saved] {filepath}")
        return str(filepath)


_runner_instance = None

def get_agent_runner() -> AgentRunner:
    global _runner_instance
    if _runner_instance is None:
        _runner_instance = AgentRunner()
    return _runner_instance