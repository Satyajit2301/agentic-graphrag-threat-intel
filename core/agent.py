"""Agentic GraphRAG loop and the single-call GraphRAG answer (port of Block 12, part 2, and Block 11).

The instructions below are the FIRST version of the agent instructions: the version that was frozen before the test questions were run.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from . import graph
from .graph import G
from .llm import LLM, LLMError
from .tools import Sess, entity_sets, run_tool, tools_schema, label

MAX_STEPS = 8                     # at most this many LLM calls with tools per question; then the agent must answer
AG_MAX_TOKENS = 700
GENERIC_SPEC = ('a JSON object {"answer": ...}: a list of IDs (CVE, technique, tactic, group, mitigation), an integer for a count, '
                'or a short text if no ID answer fits')

SYSTEM_AGENT = ("You are a cybersecurity analyst who answers questions by querying a knowledge graph with tools (data: NVD CVEs, CISA KEV, EPSS, MITRE ATT&CK, expert CVE-to-ATT&CK mappings). "
                "Every tool result is a numbered set (S1, S2, ...) with its size and a preview; later tool calls take a set name as `source`. Entities found in the question are already loaded as sets. "
                "Plan the shortest chain of tool calls, use ONLY what the tools return, use exact IDs, and follow every rule stated in the question exactly. "
                "Call neighbors() with several source items to see how many of them each result is linked to (sorted by that number); min_links=\"all\" keeps only results linked to every source item. "
                'When you are done, reply with ONLY one JSON object {"answer": ...}. For a list answer you can submit a whole set by name, e.g. {"answer": "S4"} (for a ranked answer the first three items of the set are used). For a count give the integer.')


def kind_of(question: str, kind: str | None = None) -> str:
    """how a submitted set is turned into an answer: 'number' (its size), 'ranked' (first three items) or 'list' (all items).
    For benchmark questions the answer kind is known; for free questions it is guessed from the wording."""
    if kind:
        return "number" if kind == "number" else "ranked" if kind.endswith("_ranked") else "list"
    if re.match(r"\s*how many", question, re.I):
        return "number"
    if re.search(r"priority order|\b(top|first)\s+(three|3)\b|\bthree\b.*\b(first|best|highest)\b", question, re.I):
        return "ranked"
    return "list"


def resolve_answer(ses: Sess, text: str, kind: str):
    """{"answer": "S4"}: submit a whole set by name. Returns (text, handle)."""
    m = re.search(r'"answer"\s*:\s*\[?\s*"(S\d+)"\s*\]?', text or "") or re.fullmatch(r"\s*(S\d+)\s*", text or "")
    if not m or m.group(1) not in ses.H:
        return text, None
    k, items = m.group(1), ses.H[m.group(1)]["items"]
    val = len(items) if kind == "number" else items[:3] if kind == "ranked" else items
    return json.dumps({"answer": val}), k


def parse_value(text: str):
    """the value of the {"answer": ...} object, or None if the reply is not JSON"""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    for cand in (t, *[t[t.find(a):t.rfind(b) + 1] for a, b in (("{", "}"), ("[", "]")) if a in t and b in t]):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj.get("answer", obj.get("Answer", next(iter(obj.values()), None)))
        return obj
    return None


def _agent_chat(llm: LLM, messages, tools, max_tokens=AG_MAX_TOKENS):
    """like llm.chat, but a call the server could not process (for example an unreadable tool call) is retried at once with a slightly higher temperature"""
    r = {}
    for temp in (0.0, 0.3, 0.7):
        r = llm.chat(messages, tools, max_tokens, temperature=temp)
        if r["finish"] != "error":
            return r
    return r


def run_agent(question: str, llm: LLM, answer_spec: str = GENERIC_SPEC, on_event: Callable[[dict], None] | None = None, max_steps: int = MAX_STEPS,
              answer_kind: str | None = None) -> dict:
    """Answer a question by letting the LLM call the graph tools. Returns the answer, the tool trace and the sets that were built."""
    tools = tools_schema()
    ses = Sess()
    ents = entity_sets(ses, question)
    n_ent = ses.n
    msgs = [{"role": "system", "content": SYSTEM_AGENT},
            {"role": "user", "content": f"Question: {question}\n\nAnswer format: {answer_spec}.\n\nEntities found in the question:\n{ents}"}]
    pt = ct = calls = errors = 0
    lat = 0.0
    trace, final, forced = [], None, False

    def emit(ev):
        if on_event:
            on_event(ev)

    for step in range(max_steps + 1):
        if step == max_steps:
            forced = True
            msgs.append({"role": "user", "content": 'You have used all your tool steps. Give your final answer now as {"answer": ...}.'})
        r = _agent_chat(llm, msgs, None if forced else tools)
        calls += 1; pt += r["pt"]; ct += r["ct"]; lat += r["lat"]
        if r["finish"] == "error":
            errors += 1
            trace.append({"step": len(trace) + 1, "tool": "LLM error", "args": "", "result": r.get("error", "")[:200]})
            emit({"type": "step", **trace[-1]})
            msgs.append({"role": "user", "content": "Your last message could not be processed. Call one tool again with simple arguments, or give the final JSON answer."})
            continue
        if r["tool_calls"] and not forced:
            msgs.append({"role": "assistant", "content": r["content"] or "", "tool_calls": r["tool_calls"]})
            for tc in r["tool_calls"]:
                fn_, args_ = tc["function"]["name"], tc["function"].get("arguments")
                res = run_tool(ses, fn_, args_)
                errors += res.startswith("ERROR")
                msgs.append({"role": "tool", "tool_call_id": tc.get("id") or fn_, "content": res})
                trace.append({"step": len(trace) + 1, "tool": fn_, "args": args_ if isinstance(args_, str) else json.dumps(args_), "result": res})
                emit({"type": "step", **trace[-1]})
            continue
        final = r["content"]
        break
    text, handle = resolve_answer(ses, final or "", kind_of(question, answer_kind))
    return {"text": text, "value": parse_value(text), "handle": handle, "pt": pt, "ct": ct, "calls": calls, "lat": lat,
            "trace": trace, "ses": ses, "forced": forced, "tool_errors": int(errors), "entity_sets": n_ent}


def run_graphrag(question: str, llm: LLM, answer_spec: str = GENERIC_SPEC) -> dict:
    """GraphRAG: one fixed neighbourhood context and ONE LLM call (no iteration)."""
    msgs, ctx, ent = graph.graph_messages(question, answer_spec)
    r = llm.chat(msgs, None, 900)
    if r["finish"] == "error":
        raise LLMError(r.get("error", "the LLM call failed"))
    return {"text": r["content"], "value": parse_value(r["content"]), "pt": r["pt"], "ct": r["ct"], "calls": 1, "lat": r["lat"], "context": ctx, "entities": ent}


def items_table(value, ses: Sess | None = None):
    """turn an answer value into rows (id, kind, details) for display; plain numbers and text are returned as they are"""
    if isinstance(value, list):
        rows = []
        for x in value:
            x = str(x).strip()
            typ_ = _type_of(x)
            rows.append({"id": x, "type": typ_ or "", "details": label(x, typ_) if typ_ else ""})
        return rows
    return value


def _type_of(x: str):
    from .tools import T, id_type
    t = id_type(x)
    return t if t and x in T.VALID_ID.get(t, ()) else None
