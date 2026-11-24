"""Check on the REAL data (runs only if data/benchmark.json exists):  pytest -q tests/test_real_data.py -s

Reproduces the 'scripted tool check' of the notebook (Block 12): for every one of the 108 benchmark questions, the ideal sequence of tool calls
must return exactly the gold answer. If this passes on the data of the deployed app, the graph and the tools in the app are the same as in the evaluation.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "benchmark.json").exists(), reason="real data not present in data/")

from core import agent, graph, score, tools  # noqa: E402


@pytest.fixture(scope="module")
def bench():
    graph.load(DATA, force=True)
    tools.setup()
    return json.loads((DATA / "benchmark.json").read_text())


def scripted(q):
    ses = tools.Sess()
    tools.entity_sets(ses, q["question"])
    t_ = q["template"]

    def go(name, **a):
        res = tools.run_tool(ses, name, json.dumps(a))
        if res.startswith("ERROR"):
            raise RuntimeError(res)
        return f"S{ses.n}"
    s = "S1"
    if t_ == "A1":
        s = go("neighbors", source=s, hop="cve_to_techniques")
    elif t_ == "A2":
        s = go("filter_cves", source=go("neighbors", source=s, hop="product_to_cves"), kev=True)
    elif t_ in ("B1", "B2", "B3"):
        hop = {"B1": "technique_to_tactics", "B2": "technique_to_mitigations", "B3": "technique_to_groups"}[t_]
        s = go("neighbors", source=go("neighbors", source=s, hop="cve_to_techniques"), hop=hop, **({"min_links": "all"} if t_ == "B3" else {}))
    elif t_ in ("C1", "C2"):
        s = go("neighbors", source=go("neighbors", source=go("neighbors", source=s, hop="product_to_cves"), hop="cve_to_techniques"), hop="technique_to_tactics" if t_ == "C1" else "technique_to_groups")
    elif t_ == "C3":
        s = go("filter_cves", source=go("neighbors", source=s, hop="product_to_cves"), kev=True, via_tactic=q["entities"]["tactic"])
    elif t_ == "D1":
        s = go("sort_cves", source=go("neighbors", source=s, hop="product_to_cves"), by="kev,epss,cvss", n=3)
    text, _ = agent.resolve_answer(ses, json.dumps({"answer": s}), agent.kind_of(q["question"], q["answer_kind"]))
    return score.score_against(q["gold"], q["answer_kind"], agent.parse_value(text))["exact"]


def test_all_benchmark_questions_are_solvable_with_the_tools(bench):
    bad = [q["id"] for q in bench if not scripted(q)]
    print(f"\n{len(bench) - len(bad)} of {len(bench)} benchmark questions solved exactly by the scripted tool sequences; failures: {bad or 'none'}")
    assert not bad


def test_graphrag_context_fits_the_budget(bench):
    over = [q["id"] for q in bench if graph.tk(graph.build_context(q["question"])[0]) > graph.GRAPH_CTX_TOKENS + 400]
    assert not over, over
