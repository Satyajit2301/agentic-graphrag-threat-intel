"""The six graph tools of the Agentic GraphRAG system (port of Block 12, part 1).

Every tool result is a numbered SET (S1, S2, ...) that later calls can refer to, so large intermediate results never have to fit into the prompt.
Compared with the notebook, Sess also keeps a small log of the hops that were followed (``ses.log``); the app uses it to draw the evidence graph.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from types import SimpleNamespace

from .graph import G, link

PREVIEW = 25                      # items shown per tool result (the rest stays in the set and can be paged with `show`)
T = SimpleNamespace(ready=False)  # tables that depend on the loaded graph (built by setup())


def _rev(d):
    r = defaultdict(set)
    for k, vs in d.items():
        for v in vs:
            r[v].add(k)
    return r


def setup():
    """build the hop tables once the graph is loaded"""
    if T.ready:
        return T
    if not G.ready:
        raise RuntimeError("load the graph first: core.graph.load('data')")
    T.cve_techs = {c: {t for t, _ in v} for c, v in G.techs_of_c.items()}
    T.prods_of_c = _rev(G.cves_of_p)
    T.HOPS = {   # hop name: (type of the source items, type of the result items, lookup table, meaning)
        "product_to_cves": ("Product", "CVE", G.cves_of_p, "CVEs that list the product as vulnerable"),
        "cve_to_products": ("CVE", "Product", T.prods_of_c, "products listed as vulnerable in the CVE"),
        "cve_to_techniques": ("CVE", "Technique", T.cve_techs, "ATT&CK techniques the expert mapping links to the CVE"),
        "technique_to_cves": ("Technique", "CVE", _rev(T.cve_techs), "CVEs mapped to the technique"),
        "technique_to_tactics": ("Technique", "Tactic", G.tactics_of_t, "ATT&CK tactics the technique belongs to"),
        "tactic_to_techniques": ("Tactic", "Technique", _rev(G.tactics_of_t), "techniques of the tactic"),
        "technique_to_mitigations": ("Technique", "Mitigation", {t: set(m) for t, m in G.mits_of_t.items()}, "ATT&CK mitigations for the technique"),
        "technique_to_groups": ("Technique", "Group", {t: set(g) for t, g in G.groups_of_t.items()}, "ATT&CK groups recorded as using the technique"),
        "group_to_techniques": ("Group", "Technique", _rev({t: set(g) for t, g in G.groups_of_t.items()}), "techniques used by the group"),
        "cve_to_weaknesses": ("CVE", "Weakness", {c: set(w) for c, w in G.cwe_of_c.items()}, "CWE weaknesses of the CVE")}
    T.VALID_ID = {"CVE": G.CVE_SET, "Technique": set(G.TNAME), "Tactic": set(G.TACNAME), "Group": set(G.GNAME), "Mitigation": set(G.MNAME),
                  "Weakness": set(G.WNAME), "Product": set(G.PNAME)}
    T.ID_RE = [("CVE", r"CVE-\d{4}-\d{4,}"), ("Technique", r"T\d{4}(?:\.\d{3})?"), ("Tactic", r"TA\d{4}"), ("Group", r"G\d{4}"), ("Mitigation", r"M\d{4}"), ("Weakness", r"CWE-\d+")]
    T.NAMES = {"Technique": G.TNAME, "Tactic": G.TACNAME, "Group": G.GNAME, "Mitigation": G.MNAME, "Weakness": G.WNAME}
    T.ready = True
    return T


def id_type(x):
    if x.startswith("prod:"):
        return "Product"
    return next((t for t, p in T.ID_RE if re.fullmatch(p, x)), None)


def label(x, typ_):
    if typ_ == "CVE":
        ci = G.CI[x]
        return f"{x} | {'KEV' if ci['kev'] else 'not in KEV'} | EPSS {ci['epss']} | CVSS {ci['cvss']}"
    if typ_ == "Product":
        return f"{x} | {G.PNAME[x]} | {len(G.cves_of_p.get(x, ()))} CVEs"
    return f"{x} {T.NAMES[typ_].get(x, '')}".strip()


def short_label(x, typ_):
    """a short human label for tables and the graph view"""
    if typ_ == "Product":
        return G.PNAME.get(x, x)
    if typ_ == "CVE":
        return x
    return f"{x} {T.NAMES.get(typ_, {}).get(x, '')}".strip()


class Sess:
    """the sets of ONE question"""

    def __init__(self):
        setup()
        self.H, self.n, self.log = {}, 0, []

    def new(self, typ_, items, via="", cnt=None, src_n=1):
        self.n += 1
        self.H[f"S{self.n}"] = {"type": typ_, "items": list(items), "via": via, "cnt": cnt or {}, "src_n": src_n}
        return f"S{self.n}"

    def set_of(self, source):
        s = str(source).strip().upper()
        return s if re.fullmatch(r"S\d+", s) and s in self.H else None

    def items_of(self, source, expect=None):
        src = str(source).strip()
        if re.fullmatch(r"S\d+", src, re.I):
            k = src.upper()
            if k not in self.H:
                raise ValueError(f'there is no set {k}; existing sets: {", ".join(self.H) or "none"}')
            h = self.H[k]
            items, typ_ = h["items"], h["type"]
        else:
            items = [x if x.startswith("prod:") else x.upper() for x in re.split(r"[,;\s]+", src) if x]
            if not items:
                raise ValueError("source is empty: give a set name such as S1 or an ID")
            types = {id_type(x) for x in items}
            if len(types) != 1 or None in types:
                raise ValueError(f"could not read these as IDs of one kind: {src[:80]} (products need their prod:... id: use find_product)")
            typ_ = types.pop()
            for x in items:
                if x not in T.VALID_ID[typ_]:
                    raise ValueError(f"unknown {typ_} id {x}" + (" (use find_product to look a product up by name)" if typ_ == "Product" else ""))
        if expect and typ_ != expect:
            raise ValueError(f"this needs {expect} items but {src} holds {typ_} items")
        return items

    def show(self, k, offset=0, n=PREVIEW):
        h = self.H[k]
        it, typ_ = h["items"], h["type"]
        head = f"{k}: {len(it)} {typ_} item(s)" + (f" ({h['via']})" if h["via"] else "")
        if not it:
            return head + " - the set is empty."
        rows = [label(x, typ_) + (f"   [linked to {h['cnt'][x]} of {h['src_n']} source items]" if h["cnt"] else "") for x in it[offset:offset + n]]
        more = len(it) - offset - len(rows)
        return head + f"; showing {offset + 1}-{offset + len(rows)}:\n" + "\n".join(rows) + (f"\n... {more} more (use show with offset {offset + len(rows)})" if more > 0 else "")


# ------------------------------------------------------------------------------------------------------------
# the tools
# ------------------------------------------------------------------------------------------------------------
def t_find_product(ses, name):
    q_ = str(name).lower().strip()
    cand = sorted(G.by_name.get(q_, []), key=lambda p: -len(G.cves_of_p.get(p, ())))
    if not cand:
        toks = re.findall(r"[a-z0-9]+", q_)
        cand = sorted((p for n_, ps in G.by_name.items() if toks and all(t_ in n_ for t_ in toks) for p in ps), key=lambda p: (len(G.PNAME[p]), -len(G.cves_of_p.get(p, ()))))
    k = ses.new("Product", cand[:5], f'products matching "{name}"')
    return ses.show(k) if cand else f'{k}: no product matches "{name}".'


def t_neighbors(ses, source, hop, min_links=1):
    if hop not in T.HOPS:
        raise ValueError(f'unknown hop "{hop}". Hops: ' + ", ".join(T.HOPS))
    st, tt, table, _ = T.HOPS[hop]
    src = ses.items_of(source, st)
    cnt = Counter()
    for x in src:
        for y in set(table.get(x, ())):
            cnt[y] += 1
    need = len(src) if str(min_links).strip().lower() == "all" else int(float(min_links))
    items = sorted((y for y, c in cnt.items() if c >= need), key=lambda y: (-cnt[y], y))
    k = ses.new(tt, items, f"{hop} from {len(src)} source item(s)" + (f", linked to at least {need}" if need > 1 else ""), {y: cnt[y] for y in items} if len(src) > 1 else {}, len(src))
    ses.log.append({"kind": "hop", "hop": hop, "src": ses.set_of(source) or str(source), "src_items": list(src), "dst": k})
    return ses.show(k)


def t_filter_cves(ses, source, kev=None, min_epss=None, min_cvss=None, via_tactic=None):
    src = ses.items_of(source, "CVE")
    out, how = list(src), []
    if kev not in (None, ""):
        want = str(kev).strip().lower() in ("true", "1", "yes")
        out = [c for c in out if G.CI[c]["kev"] == want]
        how.append("in the CISA KEV catalogue" if want else "not in the KEV catalogue")
    if min_epss not in (None, ""):
        out = [c for c in out if (G.CI[c]["epss"] if G.CI[c]["epss"] is not None else -1) >= float(min_epss)]
        how.append(f"EPSS >= {min_epss}")
    if min_cvss not in (None, ""):
        out = [c for c in out if (G.CI[c]["cvss"] if G.CI[c]["cvss"] is not None else -1) >= float(min_cvss)]
        how.append(f"CVSS >= {min_cvss}")
    if via_tactic not in (None, ""):
        ta = str(via_tactic).strip().upper()
        if ta not in G.TACNAME:
            raise ValueError(f"unknown tactic id {ta}")
        out = [c for c in out if any(ta in G.tactics_of_t.get(t_, ()) for t_ in T.cve_techs.get(c, ()))]
        how.append(f"mapped to at least one technique of tactic {ta}")
    if not how:
        raise ValueError("give at least one condition: kev, min_epss, min_cvss or via_tactic")
    k = ses.new("CVE", sorted(out), "filter from " + str(source) + ": " + ", ".join(how))
    ses.log.append({"kind": "filter", "src": ses.set_of(source) or str(source), "dst": k})
    return ses.show(k)


def t_sort_cves(ses, source, by="kev,epss,cvss", n=3):
    src = ses.items_of(source, "CVE")
    keys = [k.strip().lower() for k in (by if isinstance(by, list) else str(by).split(",")) if k.strip()]
    if not keys or any(k not in ("kev", "epss", "cvss") for k in keys):
        raise ValueError("by must list some of: kev, epss, cvss (each sorted best first; ties by CVE id)")
    kf = {"kev": lambda c: not G.CI[c]["kev"], "epss": lambda c: -(G.CI[c]["epss"] if G.CI[c]["epss"] is not None else -1),
          "cvss": lambda c: -(G.CI[c]["cvss"] if G.CI[c]["cvss"] is not None else -1)}
    top = sorted(src, key=lambda c: tuple(kf[k](c) for k in keys) + (c,))[:max(1, int(n))]
    k = ses.new("CVE", top, f'top {len(top)} of {len(src)} CVEs sorted by {", ".join(keys)} (best first)')
    ses.log.append({"kind": "filter", "src": ses.set_of(source) or str(source), "dst": k})
    return ses.show(k)


def t_show(ses, source, offset=0, n=PREVIEW):
    k = str(source).strip().upper()
    if k not in ses.H:
        raise ValueError(f'there is no set {k}; existing sets: {", ".join(ses.H) or "none"}')
    return ses.show(k, max(0, int(offset)), min(60, max(1, int(n))))


def t_describe(ses, ids):
    out = []
    for x in [x if x.startswith("prod:") else x.upper() for x in re.split(r"[,;\s]+", str(ids)) if x][:5]:
        typ_ = id_type(x)
        if typ_ is None or x not in T.VALID_ID[typ_]:
            out.append(f"{x}: unknown id")
            continue
        if typ_ == "CVE":
            a, ci = G.ATTR["cve:" + x], G.CI[x]
            out.append(f"{x}: CVSS {ci['cvss']} ({ci['sev']}); EPSS {ci['epss']}; KEV {'yes, added ' + str(a.get('kev_date_added')) if ci['kev'] else 'no'}; SSVC exploitation {a.get('ssvc_exploitation') or 'n/a'}; "
                       f"weaknesses {', '.join(sorted(G.cwe_of_c.get(x, []))) or 'n/a'}; affects {G.n_prod_of_c.get(x, 0)} products; techniques {', '.join(sorted(T.cve_techs.get(x, []))) or 'none (no expert mapping)'}")
        elif typ_ == "Technique":
            out.append(f"{x} {G.TNAME[x]}: tactics {', '.join(sorted(G.tactics_of_t.get(x, []))) or 'n/a'}; {len(G.mits_of_t.get(x, []))} mitigations; used by {len(G.groups_of_t.get(x, []))} groups; "
                       f"mapped from {len(T.HOPS['technique_to_cves'][2].get(x, ()))} CVEs")
        elif typ_ == "Product":
            out.append(f"{x}: {G.PNAME[x]}; {len(G.cves_of_p.get(x, ()))} CVEs list it as vulnerable, {sum(G.CI[c]['kev'] for c in G.cves_of_p.get(x, ()))} of them in KEV")
        else:
            out.append(label(x, typ_))
    return "\n".join(out) or "no ids given"


TOOL_FUNCS = {"find_product": t_find_product, "neighbors": t_neighbors, "filter_cves": t_filter_cves, "sort_cves": t_sort_cves, "show": t_show, "describe": t_describe}
TOOL_USAGE = {"find_product": '{"name": "..."}', "neighbors": '{"source": "S1", "hop": "cve_to_techniques", "min_links": "all" (optional)}',
              "filter_cves": '{"source": "S2", "kev": true, "min_epss": 0.5, "min_cvss": 7, "via_tactic": "TA0001"} (any of the conditions)',
              "sort_cves": '{"source": "S2", "by": "kev,epss,cvss", "n": 3}', "show": '{"source": "S3", "offset": 25}', "describe": '{"ids": "CVE-2021-1234, T1190"}'}


def tool_def(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": req}}}


def tools_schema():
    """the tool descriptions sent to the LLM (identical wording to the evaluated version)"""
    setup()
    STR = lambda d: {"type": "string", "description": d}
    SRC = STR("a set name such as S1, or one or more IDs separated by commas")
    return [
        tool_def("find_product", "Look up product nodes by (part of) a product name when a product of the question was not found automatically. Returns a set of up to 5 candidates with their prod:... ids.", {"name": STR("product name or part of it")}, ["name"]),
        tool_def("neighbors", "Follow one hop of the knowledge graph from every item of a set and return the set of items reached. When the source has several items, every result shows how many source items are linked to it and the results are sorted by that number (highest first, ties by ID).",
                 {"source": SRC, "hop": {"type": "string", "enum": list(T.HOPS), "description": "; ".join(f"{k}: {v[3]}" for k, v in T.HOPS.items())},
                  "min_links": STR('optional: keep only results linked to at least this many source items; "all" keeps only results linked to every source item (an intersection). Default 1')}, ["source", "hop"]),
        tool_def("filter_cves", "Keep the CVEs of a set that satisfy all given conditions.", {"source": SRC, "kev": {"type": "boolean", "description": "true = only CVEs in the CISA KEV catalogue, false = only CVEs not in it"},
                 "min_epss": {"type": "number", "description": "minimum EPSS score"}, "min_cvss": {"type": "number", "description": "minimum CVSS base score"},
                 "via_tactic": STR("ATT&CK tactic ID such as TA0001: keep CVEs mapped to at least one technique of that tactic")}, ["source"]),
        tool_def("sort_cves", "Sort the CVEs of a set and keep the best n. Sort keys are applied in the order given (kev: KEV-listed first; epss: higher first; cvss: higher first); ties are broken by the alphabetically first CVE ID.",
                 {"source": SRC, "by": STR("comma-separated keys from: kev, epss, cvss"), "n": {"type": "integer", "description": "how many CVEs to keep"}}, ["source", "by", "n"]),
        tool_def("show", "Show more items of a set (paging).", {"source": STR("set name"), "offset": {"type": "integer", "description": "index of the first item to show (0-based)"}, "n": {"type": "integer", "description": "how many items (max 60)"}}, ["source"]),
        tool_def("describe", "Show the facts of up to 5 nodes (CVE scores, KEV status, techniques; technique tactics; product CVE counts).", {"ids": STR("IDs separated by commas")}, ["ids"])]


def run_tool(ses, name, args):
    if name not in TOOL_FUNCS:
        return f"ERROR: unknown tool '{name}'. Tools: {', '.join(TOOL_FUNCS)}."
    try:
        a = json.loads(args) if isinstance(args, str) and args.strip() else (args or {})
        if not isinstance(a, dict):
            raise ValueError("not an object")
    except Exception:
        return f"ERROR: the arguments are not a valid JSON object. Expected: {TOOL_USAGE[name]}"
    try:
        return TOOL_FUNCS[name](ses, **a)
    except TypeError as e:
        return f"ERROR: wrong arguments for {name} ({str(e)[:120]}). Expected: {TOOL_USAGE[name]}"
    except Exception as e:
        return f"ERROR: {str(e)[:300]}"


def entity_sets(ses, text):
    """the entities found in the question are loaded as the first sets"""
    ent = link(text)
    lines = []
    if ent["products"]:
        lines.append(ses.show(ses.new("Product", ent["products"][:4], "products found in the question")))
    if ent["cves"]:
        lines.append(ses.show(ses.new("CVE", ent["cves"], "CVEs found in the question")))
    return "\n".join(lines) or "no entity of the question was recognised: use find_product or give IDs yourself"
