"""
run_eval.py
-----------
Runs all 3 evaluation test queries and saves execution traces.

Usage: python tests/run_eval.py

This script directly invokes the agent (no HTTP server needed).
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agent.agent_runner import AgentRunner

# ── Test Queries ─────────────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "id": "test_1_verification",
        "label": "Test 1 – Verification Challenge",
        "query": "What is the total number of jobs or employees reported in the cyber security sector in Ireland, and on which page is this stated? Search for '7,351' and 'employees'.",
        "expected_behavior": "Must return 7,351, exact page number, and verifiable citation.",
    },
    {
        "id": "test_2_synthesis",
        "label": "Test 2 – Data Synthesis Challenge",
        "query": "Compare the concentration of dedicated (pure-play) cybersecurity firms in Cork versus Dublin, using the regional office data table. Extract the number of dedicated offices and total offices for each region and calculate the percentage.",
        "expected_behavior": "Must use page 15 table: Cork 37 dedicated / 129 total, Dublin 100 dedicated / 397 total. Use calculator for percentages.",
    },
    {
        "id": "test_3_forecasting",
        "label": "Test 3 – Forecasting Challenge",
        "query": "Based on the 2022 baseline employment figure of 7,351 and the National Cyber Security Strategy job target for 2030, what is the required CAGR? Search for the 2030 target number first, then use the calculator tool.",
        "expected_behavior": "Must find 2030 target, use calculator for CAGR formula.",
    },
]

# ── Main ─────────────────────────────────────────────────────────────────────

def run_evaluations():
    print("\n" + "=" * 70)
    print("  CYBER IRELAND 2022 — AGENT EVALUATION SUITE")
    print("=" * 70)

    # Initialize agent once
    print("\n[Init] Loading agent...")
    agent = AgentRunner()
    print("[Init] Agent ready.\n")

    logs_dir = Path(os.getenv("LOGS_PATH", "logs/traces"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for i, test in enumerate(TEST_QUERIES, 1):
        print(f"\n{'='*70}")
        print(f"  {test['label']}")
        print(f"  Query: {test['query']}")
        print(f"  Expected: {test['expected_behavior']}")
        print(f"{'='*70}")

        result = agent.run(test["query"])
        result["test_id"] = test["id"]
        result["test_label"] = test["label"]
        result["expected_behavior"] = test["expected_behavior"]

        # Save individual trace
        trace_path = agent.save_trace(result, label=test["id"])

        # Print summary
        print(f"\n✅ ANSWER:\n{result['answer']}\n")
        print(f"📊 Tools used: {[t['tool'] for t in result['tool_calls']]}")
        print(f"⏱  Time: {result['execution_time_seconds']}s")
        print(f"💾 Trace: {trace_path}")

        all_results.append(result)

    # Save combined results
    combined_path = logs_dir / f"eval_all_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  EVALUATION COMPLETE")
    print(f"  Combined trace: {combined_path}")
    print(f"{'='*70}\n")

    # Print final summary table
    print("\n📋 RESULTS SUMMARY:")
    print(f"{'─'*70}")
    for r in all_results:
        status = "✅" if r.get("answer") and not r.get("error") else "❌"
        print(f"{status} {r['test_label']}")
        print(f"   Tools: {[t['tool'] for t in r['tool_calls']]}")
        print(f"   Time:  {r['execution_time_seconds']}s")
        print(f"   Answer preview: {r['answer'][:150]}...")
        print()


if __name__ == "__main__":
    run_evaluations()
