"""Knowledge graph: lookup tables, entity linking and the GraphRAG context.

This module is a port of Block 11 (GraphRAG) and the first part of Block 12 (tables) of the Colab notebook.
The logic is the same; only the notebook globals have become the namespace ``G``.

    from core.graph import G, load
    load("data")            # reads data/kg_nodes.parquet and data/kg_edges.parquet once
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

G = SimpleNamespace(ready=False)          # everything that load() builds lives here

GRAPH_CTX_TOKENS = 3600                   # context budget of GraphRAG (estimated tokens), same as in the evaluation
TOK_PER_CHAR = 0.55                       # token estimate for lines full of identifiers
PRIORITY_ROWS = 15                        # rows of the "highest exploitation priority" view


def tk(s: str) -> int:
    return int(len(s) * TOK_PER_CHAR)


def _r4(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


def load(data_dir: str | Path = "data", force: bool = False) -> SimpleNamespace:
    """Read the graph tables and build every lookup table. Safe to call many times (does the work once)."""
    if G.ready and not force:
        return G
    d = Path(data_dir)
    NODES = pd.read_parquet(d / "kg_nodes.parquet")
    EDGES = pd.read_parquet(d / "kg_edges.parquet")
    G.NODES, G.EDGES = NODES, EDGES
    G.ATTR = {n: json.loads(a) for n, a in zip(NODES.node_id, NODES["attrs"])}
    G.EXT = dict(zip(NODES.node_id, NODES.ext_id))
    G.NM = dict(zip(NODES.node_id, NODES["name"]))
    G.TEXT = dict(zip(NODES.node_id, NODES["text"].fillna("")))
    EXT, NM = G.EXT, G.NM

    def erel(rel, st=None, dt=None):
        e = EDGES[EDGES.rel == rel]
        return e[(e.src_type == st) & (e.dst_type == dt)] if st else e

    cves_of_p, n_prod_of_c, techs_of_c, cwe_of_c = defaultdict(set), Counter(), defaultdict(list), defaultdict(list)
    tactics_of_t, mits_of_t, groups_of_t = defaultdict(list), defaultdict(list), defaultdict(list)
    for s, t in zip(*[erel("affects")[c] for c in ("src", "dst")]):
        cves_of_p[t].add(EXT[s]); n_prod_of_c[EXT[s]] += 1
    for s, t, a in zip(*[erel("maps_to")[c] for c in ("src", "dst", "attr")]):
        techs_of_c[EXT[s]].append((EXT[t], a or ""))
    for s, t in zip(*[erel("has_weakness")[c] for c in ("src", "dst")]):
        cwe_of_c[EXT[s]].append(EXT[t])
    for s, t in zip(*[erel("in_tactic")[c] for c in ("src", "dst")]):
        tactics_of_t[EXT[s]].append(EXT[t])
    for s, t in zip(*[erel("mitigates")[c] for c in ("src", "dst")]):
        mits_of_t[EXT[t]].append(EXT[s])
    for s, t in zip(*[erel("uses", "Group", "Technique")[c] for c in ("src", "dst")]):
        groups_of_t[EXT[t]].append(EXT[s])
    G.cves_of_p, G.n_prod_of_c, G.techs_of_c, G.cwe_of_c = cves_of_p, n_prod_of_c, techs_of_c, cwe_of_c
    G.tactics_of_t, G.mits_of_t, G.groups_of_t = tactics_of_t, mits_of_t, groups_of_t
    G.TNAME = {EXT[n]: NM[n] for n in NODES.node_id if n.startswith("tech:")}
    G.TACNAME = {EXT[n]: NM[n] for n in NODES.node_id if n.startswith("tactic:")}
    G.CVE_SET = {EXT[n] for n in NODES.node_id if n.startswith("cve:")}

    def cinfo(c):
        a = G.ATTR["cve:" + c]
        return {"epss": _r4(a.get("epss")), "cvss": _r4(a.get("base_score")), "sev": a.get("severity"), "kev": bool(a.get("in_kev")), "a": a}
    G.CI = {c: cinfo(c) for c in G.CVE_SET}

    by_name = defaultdict(list)
    for n in NODES.node_id[NODES.node_type == "Product"]:
        by_name[NM[n].lower()].append(n)
    G.by_name = by_name
    G.NAMES_LONGEST_FIRST = sorted((n for n in by_name if len(n) >= 4), key=len, reverse=True)

    def names_of(prefix):
        return {EXT[n]: str(NM[n] or "") for n in NODES.node_id if n.startswith(prefix)}
    G.GNAME, G.MNAME, G.WNAME = names_of("group:"), names_of("mit:"), names_of("cwe:")
    G.PNAME = {n: str(NM[n]) for n in NODES.node_id if n.startswith("prod:")}
    G.LEGEND = "Tactic IDs: " + "; ".join(f"{k} {v}" for k, v in sorted(G.TACNAME.items())) + "."
    G.ready = True
    return G


def prio(c):
    """patch priority key: KEV first, then higher EPSS, then higher CVSS, then CVE id"""
    ci = G.CI[c]
    return (not ci["kev"], -(ci["epss"] if ci["epss"] is not None else -1), -(ci["cvss"] if ci["cvss"] is not None else -1), c)


def stats() -> dict:
    """numbers for the About tab, computed from the loaded graph"""
    nt = G.NODES.node_type.value_counts().to_dict()
    et = G.EDGES.rel.value_counts().to_dict()
    return {"nodes": int(len(G.NODES)), "edges": int(len(G.EDGES)), "nodes_by_type": nt, "edges_by_relation": et,
            "cves": len(G.CVE_SET), "cves_with_mapping": len(G.techs_of_c), "kev_cves": sum(ci["kev"] for ci in G.CI.values())}


# ------------------------------------------------------------------------------------------------------------
# entity linking
# ------------------------------------------------------------------------------------------------------------
def link_products(text: str):
    text, out = " " + text.lower() + " ", []
    for n in G.NAMES_LONGEST_FIRST:
        if n not in text:
            continue
        m = re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", text)
        if m:
            out.append(max(G.by_name[n], key=lambda p: len(G.cves_of_p.get(p, ()))))
            text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]     # longest names first; matched text is blanked
    return out


def link(text: str) -> dict:
    cves = [c for c in dict.fromkeys(re.findall(r"CVE-\d{4}-\d{4,}", text.upper())) if c in G.CVE_SET]
    tacs = [t for t in dict.fromkeys(re.findall(r"\bTA\d{4}\b", text.upper())) if t in G.TACNAME]
    return {"cves": cves, "products": link_products(text), "tactics": tacs}


# ------------------------------------------------------------------------------------------------------------
# GraphRAG context: the neighbourhood of the entities, written as compact facts
# ------------------------------------------------------------------------------------------------------------
def role(a: str) -> str:
    return a.replace("_", " ").replace("|", " / ")


def cve_card(c, group_budget=900):
    a, ci = G.ATTR["cve:" + c], G.CI[c]
    kev = f"yes (added {a.get('kev_date_added')}, ransomware use: {a.get('ransomware_use')})" if ci["kev"] else "no"
    L = [f"== {c} ==", f"CVSS {ci['cvss']} ({ci['sev']}); EPSS {ci['epss']}; CISA KEV: {kev}; SSVC exploitation: {a.get('ssvc_exploitation') or 'n/a'}; "
                       f"weaknesses: {', '.join(sorted(G.cwe_of_c.get(c, []))) or 'n/a'}; affects {G.n_prod_of_c.get(c, 0)} products."]
    ts = sorted(G.techs_of_c.get(c, []))
    if not ts:
        return L + ["Expert ATT&CK mapping: none recorded."]
    L.append("Expert ATT&CK mapping (technique; role of the CVE; tactics; mitigations):")
    for t, r_ in ts:
        L.append(f"  {t} {G.TNAME.get(t, '')} ({role(r_)}); tactics {', '.join(sorted(G.tactics_of_t.get(t, []))) or 'n/a'}; mitigations {', '.join(sorted(G.mits_of_t.get(t, []))) or 'none'}")
    per = max(8, group_budget // max(len(ts), 1) // 4)
    L.append("Groups recorded as using each technique (IDs):")
    for t, _ in ts:
        g = sorted(G.groups_of_t.get(t, []))
        L.append(f"  {t} ({len(g)} groups): {', '.join(g[:per])}" + (f" ... (showing {per} of {len(g)})" if len(g) > per else ""))
    return L


def stack_block(prods, budget):
    names = [G.NM[p] for p in prods]
    sets_ = [G.cves_of_p.get(p, set()) for p in prods]
    allc = set().union(*sets_)
    kev, lab = {c for c in allc if G.CI[c]["kev"]}, {c for c in allc if c in G.techs_of_c}
    L = ["== Products =="]
    for p, s_ in zip(prods, sets_):
        L.append(f"{G.NM[p]}: {len(s_)} CVEs list it as vulnerable; {len(s_ & kev)} of them are in the CISA KEV catalogue; {len(s_ & lab)} have an expert ATT&CK mapping.")
    if len(prods) > 1:
        L.append(f"All these products together: {len(allc)} distinct CVEs; {len(kev)} in the KEV catalogue; {len(lab)} with an expert ATT&CK mapping.")
    top = sorted(allc, key=prio)[:PRIORITY_ROWS]
    L += [f"== View 1: highest exploitation priority among these CVEs (KEV first, then EPSS, then CVSS); showing {len(top)} of {len(allc)} ==", "CVE | KEV | EPSS | CVSS | ATT&CK techniques"]
    L += [f"{c} | {'KEV' if G.CI[c]['kev'] else '-'} | {G.CI[c]['epss']} | {G.CI[c]['cvss']} | {','.join(sorted({t for t, _ in G.techs_of_c.get(c, [])})) or '-'}" for c in top]
    used = tk("\n".join(L))

    def gl(t):
        return f"{t} {G.TNAME.get(t, '')} -> {', '.join(sorted(G.tactics_of_t.get(t, []))) or 'n/a'}"
    techs = {t for c in top for t, _ in G.techs_of_c.get(c, [])}
    left = budget - used - tk(G.LEGEND) - sum(tk(gl(t)) for t in techs) - 100
    labl = sorted(lab, key=lambda c: (not G.CI[c]["kev"], c))
    rows = []
    for c in labl:
        ts = {t for t, _ in G.techs_of_c[c]}
        cost = tk(f"{c} | KEV | " + ",".join(sorted(ts))) + sum(tk(gl(t)) for t in ts - techs)
        if cost > left:
            break
        rows.append(c); techs |= ts; left -= cost
    L += [f"== View 2: CVEs with an expert ATT&CK mapping (KEV first); showing {len(rows)} of {len(lab)} ==", "CVE | KEV | ATT&CK techniques"]
    L += [f"{c} | {'KEV' if G.CI[c]['kev'] else '-'} | {','.join(sorted({t for t, _ in G.techs_of_c[c]}))}" for c in rows]
    L += ["== Technique -> tactic (for the techniques above) =="] + [gl(t) for t in sorted(techs)]
    return L


def build_context(text: str):
    ent = link(text)
    parts = []
    if ent["cves"]:
        for c in ent["cves"][:3]:
            parts += cve_card(c, 900 // len(ent["cves"][:3]))
    if ent["products"]:
        parts += stack_block(ent["products"][:4], GRAPH_CTX_TOKENS - tk("\n".join(parts)))
    if not parts:
        parts = ["No entity of the question was found in the knowledge graph."]
    parts.append(G.LEGEND)
    return "\n".join(parts), ent


SYSTEM_GRAPH = ("You are a cybersecurity analyst. Answer the question using ONLY the knowledge-graph facts below (NVD, CISA KEV, EPSS, MITRE ATT&CK and expert CVE-to-ATT&CK mappings). Use exact identifiers as given. "
                'Lists marked "showing N of M" are shortened. Work through the relevant facts carefully before answering. '
                "If the facts do not contain enough information, give the best answer you can from them (an empty list, or 0 for a count, if there is nothing). "
                'Reply with ONLY one JSON object of the form {"answer": ...} and nothing else.')


def graph_messages(question: str, answer_spec: str):
    ctx, ent = build_context(question)
    msgs = [{"role": "system", "content": SYSTEM_GRAPH},
            {"role": "user", "content": f"Knowledge-graph facts:\n{ctx}\n\nQuestion: {question}\n\nAnswer format: {answer_spec}.\nReply with only the JSON object."}]
    return msgs, ctx, ent
