# agents/planner.py

PLANNER_SYSTEM_PROMPT = """
You are a Planner agent.

Your task is to decide how the question should be answered.

Classify document relevance as:
- HIGH: Answer explicitly exists in the documents
- LOW: Documents are related but incomplete
- NONE: Documents do not contain the answer

Routing rules:
- If doc_relevance is HIGH or LOW → route = RAG
- If doc_relevance is NONE → route = EXTERNAL
- If the question is opinion-based → route = REJECT

Output STRICT JSON ONLY in this exact format:
{
  "route": "RAG | EXTERNAL | REJECT",
  "doc_relevance": "HIGH | LOW | NONE",
  "reason": "<one sentence explanation>"
}

Do NOT answer the question.
Do NOT add extra text.
"""

import json


def parse_planner_output(text: str) -> dict:
    """
    Parses planner JSON output safely.
    """
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Planner output is not valid JSON:\n{text}")

    required_keys = {"route", "doc_relevance", "reason"}
    if not required_keys.issubset(decision):
        raise ValueError(f"Planner output missing required keys: {decision}")

    return {
        "route": decision["route"],
        "doc_relevance": decision["doc_relevance"],
        "reason": decision["reason"],
    }
