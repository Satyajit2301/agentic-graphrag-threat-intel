"""Replay of a recorded run: the tool calls the agent made during the evaluation are executed again on the graph.

The tools are deterministic, so this rebuilds the full tool results and the evidence graph without any LLM call.
Only the language-model answer text itself is taken from the recording (data/recorded_runs.json, made by the packaging block).
"""
from __future__ import annotations

from .agent import parse_value
from .graph import build_context
from .tools import Sess, entity_sets, run_tool


def replay_agent(question: str, rec: dict) -> dict:
    ses = Sess()
    entity_sets(ses, question)
    trace = []
    for i, t in enumerate(rec.get("trace", []), 1):
        res = run_tool(ses, t["tool"], t["args"]) if t["tool"] != "LLM error" else t.get("result", "")
        trace.append({"step": i, "tool": t["tool"], "args": t["args"], "result": res})
    return {"text": rec.get("raw", ""), "value": parse_value(rec.get("raw", "")), "handle": rec.get("handle"), "pt": rec.get("pt", 0), "ct": rec.get("ct", 0),
            "calls": rec.get("calls", 0), "lat": rec.get("lat", 0.0), "trace": trace, "ses": ses, "forced": bool(rec.get("forced")), "tool_errors": rec.get("tool_errors", 0), "recorded": True}


def replay_graphrag(question: str, rec: dict) -> dict:
    ctx, ent = build_context(question)
    return {"text": rec.get("raw", ""), "value": parse_value(rec.get("raw", "")), "pt": rec.get("pt", 0), "ct": rec.get("ct", 0), "calls": 1,
            "lat": rec.get("lat", 0.0), "context": ctx, "entities": ent, "recorded": True}
