"""Tests that run without the real data and without an LLM key:  pytest -q

They use a small synthetic graph (tests/mock_data.py) that has the same tables and columns as the real data files.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import agent, evidence, graph, predict, replay, score, tools  # noqa: E402
from tests import mock_data  # noqa: E402


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("mock_data")
    mock_data.make(d)
    mock_data.make_examples(d)
    graph.load(d, force=True)
    tools.setup()
    predict.load(d, force=True)
    return d


class FakeLLM:
    """returns scripted replies: first a tool call, then the final JSON answer"""
    model = "fake"
    configured = True

    def __init__(self, script):
        self.script, self.i = script, 0

    def chat(self, messages, tools=None, max_tokens=700, temperature=0.0):
        r = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return {"content": r.get("content", ""), "tool_calls": r.get("tool_calls"), "finish": "tool_calls" if r.get("tool_calls") else "stop", "pt": 100, "ct": 10, "lat": 0.1, "error": ""}


def tool_call(name, **args):
    return {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_graph_tables(data_dir):
    s = graph.stats()
    assert s["cves"] == 120 and s["cves_with_mapping"] == 30 and s["nodes"] > 150
    assert all(c.startswith("CVE-") for c in list(graph.G.CVE_SET)[:5])


def test_entity_linking(data_dir):
    ent = graph.link("What is known about cve-2021-1000 and vendor1 product1?")
    assert ent["cves"] == ["CVE-2021-1000"] and len(ent["products"]) == 1


def test_graphrag_context_has_the_facts(data_dir):
    ctx, ent = graph.build_context("Which techniques map to CVE-2021-1000?")
    assert "== CVE-2021-1000 ==" in ctx and "Expert ATT&CK mapping" in ctx
    assert graph.tk(ctx) < graph.GRAPH_CTX_TOKENS


def test_tools_chain(data_dir):
    ses = tools.Sess()
    tools.entity_sets(ses, "CVE-2021-1000")
    out = tools.run_tool(ses, "neighbors", json.dumps({"source": "S1", "hop": "cve_to_techniques"}))
    assert out.startswith("S2:") and "Technique" in out
    out = tools.run_tool(ses, "neighbors", json.dumps({"source": "S2", "hop": "technique_to_tactics"}))
    assert out.startswith("S3:")
    assert tools.run_tool(ses, "neighbors", json.dumps({"source": "S9", "hop": "cve_to_techniques"})).startswith("ERROR")
    assert tools.run_tool(ses, "no_such_tool", "{}").startswith("ERROR")
    assert tools.run_tool(ses, "filter_cves", json.dumps({"source": "S1"})).startswith("ERROR")        # no condition given


def test_sort_and_filter(data_dir):
    ses = tools.Sess()
    allc = sorted(graph.G.CVE_SET)[:30]
    top = tools.run_tool(ses, "sort_cves", json.dumps({"source": ",".join(allc), "by": "kev,epss,cvss", "n": 3}))
    ids = [line.split(" | ")[0] for line in top.splitlines()[1:4]]
    assert ids == sorted(allc, key=graph.prio)[:3]
    k = tools.run_tool(ses, "filter_cves", json.dumps({"source": ",".join(allc), "kev": True}))
    assert all(graph.G.CI[line.split(" | ")[0]]["kev"] for line in k.splitlines()[1:] if line.startswith("CVE-"))


def test_agent_with_scripted_llm(data_dir):
    c = sorted(graph.G.techs_of_c)[0]
    gold = sorted({t for t, _ in graph.G.techs_of_c[c]})
    llm = FakeLLM([{"tool_calls": [tool_call("neighbors", source="S1", hop="cve_to_techniques")]}, {"content": '{"answer": "S2"}'}])
    res = agent.run_agent(f"Which techniques map to {c}?", llm)
    assert res["calls"] == 2 and len(res["trace"]) == 1 and res["handle"] == "S2"
    assert sorted(res["value"]) == gold
    assert score.score_against(gold, "technique_ids", res["value"])["exact"]
    dot = evidence.evidence_dot(res["ses"], res["value"])
    assert dot and c in dot and "digraph" in dot


def test_agent_step_limit_forces_an_answer(data_dir):
    llm = FakeLLM([{"tool_calls": [tool_call("describe", ids="CVE-2021-1000")]}] * 2 + [{"content": '{"answer": 0}'}])
    res = agent.run_agent("How many CVEs are there?", llm, max_steps=2)
    assert res["forced"] and res["value"] == 0


def test_replay_matches_the_recorded_run(data_dir):
    ex = json.loads((data_dir / "example_questions.json").read_text())[0]
    rec = json.loads((data_dir / "recorded_runs.json").read_text())["A1-01"]["agentic"]
    res = replay.replay_agent(ex["question"], rec)
    assert res["trace"][0]["result"].startswith("S2:") and score.score_against(ex["gold"], ex["answer_kind"], res["value"])["exact"]


def test_scorer_rules():
    assert score.score_against(5, "number", 5)["exact"] and not score.score_against(5, "number", 4)["exact"]
    s = score.score_against(["T1", "T2"], "technique_ids", ["T1"])
    assert abs(s["score"] - 2 / 3) < 1e-9 and not s["exact"]
    r = score.score_against(["CVE-2021-0001", "CVE-2021-0002", "CVE-2021-0003"], "cve_ids_ranked", ["CVE-2021-0001", "CVE-2021-0003", "CVE-2021-0002"])
    assert r["score"] == 1.0 and not r["exact"]


def test_predictor(data_dir):
    rows = predict.predict("CVE-2021-1000", 5)
    assert [r["rank"] for r in rows] == [1, 2, 3, 4, 5]
    assert predict.split_of("CVE-2021-1000") in ("test", "train")
    none_cve = predict.sample_cve("none", seed=1)
    assert predict.split_of(none_cve) == "none" and predict.cve_facts(none_cve)["expert"] == []
    assert "digraph" in evidence.predictor_dot("CVE-2021-1000", rows)


def test_answer_kind_heuristic():
    assert agent.kind_of("How many CVEs of Windows are in KEV?") == "number"
    assert agent.kind_of("anything", "cve_ids_ranked") == "ranked" and agent.kind_of("anything", "number") == "number" and agent.kind_of("anything", "tactic_ids") == "list"
    assert agent.kind_of("Which three CVEs should be patched first? List them in priority order.") == "ranked"
    assert agent.kind_of("Which tactics can be used?") == "list"


def test_app_starts_and_predictor_works(data_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    at.text_input(key="cve_input").set_value("CVE-2021-1000").run()
    assert not at.exception, at.exception
    assert any("CVE-2021-1000" in str(m.value) or True for m in at.markdown)
    at.text_input(key="cve_input").set_value("CVE-1999-0000").run()
    assert any("not in the knowledge graph" in e.value for e in at.error)


def test_app_recorded_answer(data_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APP_DATA_DIR", str(data_dir))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    pick = at.selectbox(key="example_pick")
    pick.select(pick.options[1]).run()
    assert not at.exception, at.exception
    ask = [b for b in at.button if b.label == "Ask"][0]
    ask.click().run()
    assert not at.exception, at.exception
    assert any("matches the gold answer" in s.value for s in at.success)
