"""Draw the evidence behind an answer as a small graph (Graphviz DOT text, shown with st.graphviz_chart).

Idea: start from the items of the answer and walk the hops that the agent followed BACKWARDS; keep for every item a few of the
source items that really link to it. The result is a path graph "question entities -> ... -> answer" that is small enough to read.
"""
from __future__ import annotations

from .graph import G
from .tools import T, Sess, id_type, short_label

COLOR = {"CVE": "#eb6834", "Product": "#2a78d6", "Technique": "#1baf7a", "Tactic": "#e87ba4", "Group": "#eda100", "Mitigation": "#7c5cbf", "Weakness": "#5B6577"}
MAX_PER_LEVEL, PREDS_PER_NODE = 8, 2


def node_type(x: str):
    t = id_type(x)
    return t if t and x in T.VALID_ID.get(t, ()) else None


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', "'")


def _node_line(x: str, highlight: bool) -> str:
    typ_ = node_type(x) or "Weakness"
    lab = short_label(x, typ_)
    lab = lab if len(lab) <= 30 else lab[:28] + "…"
    extra = ' penwidth=3 color="#1B2330"' if highlight else ""
    return f'  "{_esc(x)}" [label="{_esc(lab)}", fillcolor="{COLOR.get(typ_, "#999999")}", fontcolor="white"{extra}];'


def evidence_edges(ses: Sess, answer_items=None):
    """list of (source id, target id, hop name) that support the answer"""
    hops = [e for e in ses.log if e["kind"] == "hop"]
    if not hops:
        return [], []
    if not answer_items:
        last = hops[-1]["dst"]
        answer_items = ses.H[last]["items"]
    needed = [x for x in answer_items if node_type(str(x))][:MAX_PER_LEVEL]
    answer = list(needed)
    edges = []
    for ev in reversed(hops):
        _, tt, table, _ = T.HOPS[ev["hop"]]
        targets = [b for b in needed if node_type(b) == tt]
        if not targets:
            continue
        nxt = []
        for b in targets:
            preds = [a for a in ev["src_items"] if b in table.get(a, ())][:PREDS_PER_NODE]
            for a in preds:
                edges.append((a, b, ev["hop"]))
                nxt.append(a)
        needed = list(dict.fromkeys(nxt + [b for b in needed if b not in targets]))[:MAX_PER_LEVEL]
    return list(dict.fromkeys(edges)), answer


def evidence_dot(ses: Sess, answer_items=None) -> str | None:
    edges, answer = evidence_edges(ses, answer_items)
    if not edges:
        return None
    nodes = list(dict.fromkeys([x for e in edges for x in e[:2]]))
    lines = ["digraph G {", '  rankdir=LR; bgcolor="transparent";', '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.12,0.06"];',
             '  edge [color="#8a93a3", fontname="Helvetica", fontsize=9, fontcolor="#5B6577"];']
    lines += [_node_line(n, n in answer) for n in nodes]
    for a, b, hop in edges:
        lines.append(f'  "{_esc(a)}" -> "{_esc(b)}" [label="{hop.split("_to_")[-1]}"];')
    lines.append("}")
    return "\n".join(lines)


def predictor_dot(cve: str, rows: list[dict]) -> str:
    """CVE -> predicted techniques -> their tactics. rows: dicts with technique_id, hit (bool)"""
    lines = ["digraph G {", '  rankdir=LR; bgcolor="transparent";', '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.12,0.06"];',
             '  edge [color="#8a93a3", fontsize=9, fontcolor="#5B6577", fontname="Helvetica"];']
    lines.append(f'  "{cve}" [label="{cve}", fillcolor="{COLOR["CVE"]}", fontcolor="white", penwidth=3, color="#1B2330"];')
    seen = set()
    for r in rows:
        t = r["technique_id"]
        lines.append(_node_line(t, bool(r.get("hit"))))
        lines.append(f'  "{cve}" -> "{t}" [label="{r.get("rank", "")}", style={"solid" if r.get("hit") else "dashed"}];')
        for ta in G.tactics_of_t.get(t, []):
            if ta not in seen:
                lines.append(_node_line(ta, False)); seen.add(ta)
            lines.append(f'  "{t}" -> "{ta}";')
    lines.append("}")
    return "\n".join(lines)
