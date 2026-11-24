"""Builds a SMALL synthetic knowledge graph in exactly the format of the real data files, so the tests run without the real data.

The numbers in it mean nothing; it only has the same tables, columns and relations as data/kg_nodes.parquet and data/kg_edges.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make(dest: str | Path, n_cves: int = 120, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    nodes, edges = [], []

    def node(nid, typ, ext, name, text="", attrs=None):
        nodes.append({"node_id": nid, "node_type": typ, "ext_id": ext, "name": name, "text": text, "attrs": json.dumps(attrs or {})})

    def edge(s, rel, d, st, dt, attr=""):
        edges.append({"src": s, "rel": rel, "dst": d, "src_type": st, "dst_type": dt, "source": "mock", "attr": attr})

    tactics = {f"TA{i:04d}": n for i, n in zip(range(1, 8), ["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access", "Impact"])}
    for t, n in tactics.items():
        node(f"tactic:{t}", "Tactic", t, n, attrs={"shortname": n.lower().replace(" ", "-")})
    techs = [f"T{1000 + i}" for i in range(24)]
    for i, t in enumerate(techs):
        node(f"tech:{t}", "Technique", t, f"Technique {t}", text=f"Description of technique {t}.", attrs={"is_sub": False, "tactics": "", "platforms": ""})
        for ta in rng.choice(list(tactics), size=int(rng.integers(1, 3)), replace=False):
            edge(f"tech:{t}", "in_tactic", f"tactic:{ta}", "Technique", "Tactic")
    groups = [f"G{i:04d}" for i in range(1, 9)]
    for g in groups:
        node(f"group:{g}", "Group", g, f"Group {g}", attrs={"aliases": ""})
        for t in rng.choice(techs, size=8, replace=False):
            edge(f"group:{g}", "uses", f"tech:{t}", "Group", "Technique")
    mits = [f"M{1000 + i}" for i in range(6)]
    for m in mits:
        node(f"mit:{m}", "Mitigation", m, f"Mitigation {m}")
        for t in rng.choice(techs, size=6, replace=False):
            edge(f"mit:{m}", "mitigates", f"tech:{t}", "Mitigation", "Technique")
    cwes = [f"CWE-{n}" for n in (79, 89, 22, 787, 20)]
    for c in cwes:
        node(f"cwe:{c}", "Weakness", c, f"Weakness {c}", attrs={})
    prods = [f"a:vendor{i % 5}:product{i}" for i in range(14)]
    for p in prods:
        v, pr = p.split(":")[1:]
        node(f"prod:{p}", "Product", p, f"{v} {pr}", attrs={"part": "a", "vendor": v, "product": pr, "n_cves": 0})
    for i in range(n_cves):
        cid = f"CVE-2021-{1000 + i}"
        kev = bool(rng.random() < 0.25)
        attrs = {"published": "2021-06-01T00:00:00", "base_score": float(np.round(rng.uniform(3, 10), 1)), "severity": "HIGH", "epss": float(np.round(rng.random() ** 2, 4)),
                 "in_kev": kev, "kev_date_added": "2022-01-01" if kev else None, "ransomware_use": "Known" if kev and rng.random() < 0.3 else "Unknown", "ssvc_exploitation": "active" if kev else "none"}
        node(f"cve:{cid}", "CVE", cid, cid, text=f"Mock vulnerability {cid} allows remote attackers to do something bad.", attrs=attrs)
        for p in rng.choice(prods, size=int(rng.integers(1, 4)), replace=False):
            edge(f"cve:{cid}", "affects", f"prod:{p}", "CVE", "Product")
        edge(f"cve:{cid}", "has_weakness", f"cwe:{rng.choice(cwes)}", "CVE", "Weakness")
        if i % 4 == 0:                                            # about a quarter of the CVEs have an expert mapping
            for t in rng.choice(techs, size=int(rng.integers(1, 4)), replace=False):
                edge(f"cve:{cid}", "maps_to", f"tech:{t}", "CVE", "Technique", "primary_impact")
    pd.DataFrame(nodes).to_parquet(dest / "kg_nodes.parquet")
    pd.DataFrame(edges).to_parquet(dest / "kg_edges.parquet")

    # link-prediction files in the format of Block 15 / the packaging block
    rows, info = [], []
    mapped = {e["src"][4:] for e in edges if e["rel"] == "maps_to"}
    for i in range(n_cves):
        cid = f"CVE-2021-{1000 + i}"
        split = ("test" if i % 8 == 0 else "train") if cid in mapped else "none"
        info.append({"cve_id": cid, "split": split})
        order = rng.permutation(techs)[:20]
        for r, t in enumerate(order, 1):
            rows.append({"cve_id": cid, "rank": r, "technique_id": t, "score": float(20 - r), "technique_name": f"Technique {t}", "has_expert_labels": cid in mapped, "model": "MLP"})
    pd.DataFrame(rows).to_parquet(dest / "lp_predictions.parquet")
    pd.DataFrame(info).to_parquet(dest / "lp_cve_info.parquet")
    (dest / "lp_test_results.json").write_text(json.dumps({"selected_model": "MLP", "test_metrics": {"MLP": {"hits@1": 0.5, "hits@3": 0.7, "hits@10": 0.9, "mrr": 0.6, "ndcg@10": 0.6}}}))
    return dest


def make_examples(dest: str | Path) -> None:
    """example_questions.json and recorded_runs.json for the mock graph (same format as the packaging block writes)"""
    from core import graph, tools
    dest = Path(dest)
    graph.load(dest)
    tools.setup()
    c = sorted(graph.G.techs_of_c)[0]
    gold = sorted({t for t, _ in graph.G.techs_of_c[c]})
    ex = [{"id": "A1-01", "template": "A1", "split": "dev", "question": f"Which ATT&CK techniques does the expert mapping link to {c}?", "answer_kind": "technique_ids",
           "answer_spec": 'a JSON object {"answer": [technique ids]}', "gold": gold, "entities": {"cves": [c]}, "hops": 1}]
    (dest / "example_questions.json").write_text(json.dumps(ex))
    rec = {"A1-01": {"agentic": {"raw": json.dumps({"answer": gold}), "trace": [{"tool": "neighbors", "args": json.dumps({"source": "S1", "hop": "cve_to_techniques"}), "result": ""}],
                                "pt": 1000, "ct": 100, "calls": 2, "lat": 5.0},
                     "graphrag": {"raw": json.dumps({"answer": gold}), "pt": 800, "ct": 50, "lat": 4.0}}}
    (dest / "recorded_runs.json").write_text(json.dumps(rec))
