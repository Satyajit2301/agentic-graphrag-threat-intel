"""Streamlit application: Agentic GraphRAG threat-intelligence analyst with CVE-to-ATT&CK link prediction.

Run locally:   streamlit run app.py
Secrets:       GROQ_API_KEY (and optionally GROQ_MODEL, LLM_BASE_URL) in .streamlit/secrets.toml or in the Streamlit Cloud settings.
Data:          data/*.parquet and data/*.json, created by the packaging block of the notebook (see README.md).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from core import graph, predict, tools
from core.agent import GENERIC_SPEC, items_table, kind_of, run_agent, run_graphrag
from core.evidence import evidence_dot, predictor_dot
from core.llm import LLM, LLMError, QuotaExhausted
from core.replay import replay_agent, replay_graphrag
from core.score import score_against

DATA = Path(os.environ.get("APP_DATA_DIR", Path(__file__).parent / "data"))
MAX_QUESTIONS_PER_SESSION = 25            # protects the free LLM quota during a demo

st.set_page_config(page_title="Threat-Intel Analyst: Agentic GraphRAG", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------------------------------------------------
# loading (once per server process)
# ------------------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the knowledge graph (first start only)...")
def load_everything():
    graph.load(DATA)
    tools.setup()
    has_pred = (DATA / "lp_predictions.parquet").exists()
    if has_pred:
        predict.load(DATA)
    return {"predictions": has_pred}


@st.cache_data
def read_json(name: str):
    p = DATA / name
    return json.loads(p.read_text()) if p.exists() else None


def secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name)
    except Exception:                      # no secrets file at all
        v = None
    return str(v or os.environ.get(name, default))


def get_llm() -> LLM:
    return LLM(api_key=secret("GROQ_API_KEY"), model=secret("GROQ_MODEL") or None, base_url=secret("LLM_BASE_URL") or None, effort="low")


if not (DATA / "kg_nodes.parquet").exists() or not (DATA / "kg_edges.parquet").exists():
    st.error(f"The graph files are missing: expected {DATA / 'kg_nodes.parquet'} and {DATA / 'kg_edges.parquet'}. "
             "Create them with the packaging block of the notebook (README, step 2) and put them in the data/ folder.")
    st.stop()
STATE = load_everything()
EXAMPLES = read_json("example_questions.json") or []
RECORDED = read_json("recorded_runs.json") or {}
LLM_CLIENT = get_llm()

st.session_state.setdefault("n_questions", 0)
st.session_state.setdefault("question", "")
st.session_state.setdefault("example_id", None)
st.session_state.setdefault("last", None)


# ------------------------------------------------------------------------------------------------------------
# sidebar
# ------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Threat-Intel Analyst")
    st.caption("Agentic GraphRAG over a knowledge graph of NVD, CISA KEV, EPSS, MITRE ATT&CK and expert CVE-to-ATT&CK mappings.")
    mode = st.radio("Question-answering system", ["Agentic GraphRAG (tools)", "GraphRAG (one LLM call)"], index=0,
                    help="The agent calls graph tools step by step. GraphRAG puts a fixed neighbourhood of the question's entities into one prompt.")
    max_steps = st.slider("Maximum tool steps (agent)", 3, 10, 8, help="The evaluation used at most 8 LLM calls per question.")
    st.divider()
    if LLM_CLIENT.configured:
        st.success(f"Live LLM: {LLM_CLIENT.model}")
    else:
        st.warning("No LLM key found. Recorded examples and the link predictor still work. Add GROQ_API_KEY to the app secrets for live questions.")
    st.caption(f"Questions asked in this session: {st.session_state['n_questions']} / {MAX_QUESTIONS_PER_SESSION}")
    st.divider()
    st.caption("A student project (CSE 4192, Siksha 'O' Anusandhan). Answers are generated from the data in the graph; check important results against the sources.")


# ------------------------------------------------------------------------------------------------------------
# helpers for the question tab
# ------------------------------------------------------------------------------------------------------------
def on_example_change():
    pick = st.session_state.get("example_pick")
    if pick and pick != "— write your own question —":
        ex = next(e for e in EXAMPLES if f"{e['id']}  ·  {e['question'][:90]}" == pick)
        st.session_state["question"] = ex["question"]
        st.session_state["example_id"] = ex["id"]
    else:
        st.session_state["example_id"] = None


def current_example():
    eid = st.session_state.get("example_id")
    ex = next((e for e in EXAMPLES if e["id"] == eid), None)
    if ex and ex["question"].strip() != st.session_state["question"].strip():
        return None                        # the text was edited: it is no longer the benchmark question
    return ex


def render_answer(res: dict, question: str, example: dict | None, system: str):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM calls", res.get("calls", 0))
    c2.metric("Prompt tokens", f"{res.get('pt', 0):,}")
    c3.metric("Seconds", f"{res.get('lat', 0):.1f}")
    c4.metric("Tool steps", len(res.get("trace", [])) if "trace" in res else "-")
    if res.get("recorded"):
        st.info("This answer was recorded during the evaluation with the local gpt-oss-20b model. The tool calls were executed again on the graph (no LLM call was made now).")
    st.subheader("Answer")
    value = res.get("value")
    rows = items_table(value, res.get("ses")) if value is not None else None
    if isinstance(rows, list) and rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    elif isinstance(rows, list):
        st.write("The answer is an empty list.")
    elif value is not None:
        st.metric("Answer", value) if isinstance(value, (int, float)) and not isinstance(value, bool) else st.write(value)
    else:
        st.warning("The reply could not be read as a JSON answer. Raw reply below.")
        st.code(res.get("text", "") or "(empty)")
    if res.get("forced"):
        st.caption("The step limit was reached, so the agent was asked to answer with what it had.")
    if example is not None:
        sc = score_against(example["gold"], example["answer_kind"], value)
        verdict = "matches the gold answer" if sc["exact"] else "differs from the gold answer"
        (st.success if sc["exact"] else st.warning)(f"Check against the gold answer computed from the graph: {verdict} ({sc['text']}).")
        with st.expander("Show the gold answer"):
            st.write(example["gold"])
    with st.expander("Raw reply of the model"):
        st.code(res.get("text", "") or "(empty)", language="json")

    if system == "agent":
        t1, t2 = st.tabs(["Tool trace", "Evidence graph"])
        with t1:
            trace = res.get("trace", [])
            if not trace:
                st.write("The agent answered without calling a tool.")
            for t in trace:
                with st.expander(f"Step {t['step']}: {t['tool']}   {t['args'][:110]}", expanded=t["step"] == 1):
                    st.code(t["result"], language="text")
        with t2:
            ses = res.get("ses")
            ans_ids = value if isinstance(value, list) else None
            dot = evidence_dot(ses, ans_ids) if ses is not None else None
            if dot:
                st.caption("Boxes with a thick border are items of the answer. Arrows show the graph hops the agent followed, read from the question's entities to the answer.")
                st.graphviz_chart(dot, width="stretch")
            else:
                st.write("No graph hops were followed (the answer did not need a hop), so there is nothing to draw.")
    else:
        with st.expander("Knowledge-graph facts given to the model (the whole context of this one call)"):
            st.code(res.get("context", ""), language="text")


# ------------------------------------------------------------------------------------------------------------
# tabs
# ------------------------------------------------------------------------------------------------------------
tab_q, tab_p, tab_a = st.tabs(["💬 Ask the analyst", "🔗 ATT&CK link predictor", "ℹ️ About and data"])

with tab_q:
    st.header("Ask a question about vulnerabilities, products and attack techniques")
    st.write("Mention CVE ids (for example `CVE-2021-44228`) or product names. Multi-hop questions work best: "
             "*which tactics could attackers use against this stack?*, *which three CVEs should be patched first?*")
    if EXAMPLES:
        opts = ["— write your own question —"] + [f"{e['id']}  ·  {e['question'][:90]}" for e in EXAMPLES]
        st.selectbox("Example questions from the benchmark (the gold answer is computed from the graph)", opts, key="example_pick", on_change=on_example_change)
    st.text_area("Your question", key="question", height=110)
    ex = current_example()
    rec_avail = bool(ex and ex["id"] in RECORDED)
    use_rec = False
    if rec_avail:
        use_rec = st.checkbox("Use the recorded answer of the evaluation run (no internet or LLM key needed)", value=not LLM_CLIENT.configured)
    go = st.button("Ask", type="primary")

    if go:
        q = st.session_state["question"].strip()
        system = "agent" if mode.startswith("Agentic") else "graphrag"
        if not q:
            st.warning("Please type a question.")
        elif use_rec:
            rec = RECORDED[ex["id"]].get("agentic" if system == "agent" else "graphrag")
            if rec is None:
                st.warning("No recorded run of this system for this question.")
            else:
                res = replay_agent(q, rec) if system == "agent" else replay_graphrag(q, rec)
                st.session_state["last"] = {"res": res, "q": q, "ex": ex, "system": system}
        elif not LLM_CLIENT.configured:
            st.error("No LLM key is configured. Tick the recorded answer (for example questions) or add GROQ_API_KEY to the secrets.")
        elif st.session_state["n_questions"] >= MAX_QUESTIONS_PER_SESSION:
            st.error("The question limit of this session is reached (it protects the free LLM quota). Reload the page to continue.")
        else:
            st.session_state["n_questions"] += 1
            spec = ex["answer_spec"] if ex else GENERIC_SPEC
            try:
                if system == "agent":
                    with st.status("The agent is querying the graph...", expanded=True) as status:
                        def on_event(ev):
                            status.write(f"**Step {ev['step']}: {ev['tool']}**  `{ev['args'][:100]}`  →  {ev['result'].splitlines()[0][:110] if ev['result'] else ''}")
                        res = run_agent(q, LLM_CLIENT, spec, on_event=on_event, max_steps=max_steps, answer_kind=ex["answer_kind"] if ex else None)
                        status.update(label=f"Done: {res['calls']} LLM calls, {len(res['trace'])} tool steps", state="complete", expanded=False)
                else:
                    with st.spinner("Building the graph context and asking the model..."):
                        res = run_graphrag(q, LLM_CLIENT, spec)
                st.session_state["last"] = {"res": res, "q": q, "ex": ex, "system": system}
            except QuotaExhausted as e:
                st.error(str(e))
            except LLMError as e:
                st.error(str(e))

    last = st.session_state.get("last")
    if last:
        st.divider()
        st.markdown(f"**Question:** {last['q']}")
        render_answer(last["res"], last["q"], last["ex"], last["system"])

with tab_p:
    st.header("Predict ATT&CK techniques for a CVE")
    if not STATE["predictions"]:
        st.warning("data/lp_predictions.parquet is missing, so the predictor is switched off.")
    else:
        info = read_json("lp_test_results.json") or {}
        sel = info.get("selected_model", predict.P.model)
        st.write("Only about 12 % of the CVEs have an expert ATT&CK mapping. A model trained on the mapped CVEs ranks techniques for every CVE from its description, "
                 "weakness types, CVSS vector and vendor. The table shows the ten highest-ranked techniques; for CVEs with an expert mapping, techniques that match it are marked.")
        st.session_state.setdefault("cve_input", "CVE-2021-44228" if "CVE-2021-44228" in graph.G.CVE_SET else "")
        def pick_random(kind):
            c = predict.sample_cve(kind)
            if c:
                st.session_state["cve_input"] = c
        cA, cB, cC = st.columns([3, 1.3, 1.6])
        cve_in = cA.text_input("CVE id", key="cve_input")
        cB.button("Random held-out test CVE", on_click=pick_random, args=("test",))
        cC.button("Random CVE without expert mapping", on_click=pick_random, args=("none",))
        k = st.slider("How many predicted techniques to show", 3, 20, 10)
        cve = cve_in.strip().upper()
        if cve:
            if cve not in graph.G.CVE_SET:
                st.error(f"{cve} is not in the knowledge graph (it contains {len(graph.G.CVE_SET):,} CVEs).")
            else:
                f = predict.cve_facts(cve)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("CVSS", f["cvss"]); m2.metric("Severity", f["severity"]); m3.metric("EPSS", f["epss"]); m4.metric("In CISA KEV", "yes" if f["kev"] else "no")
                st.write(f["description"])
                st.caption(f"Published {f['published']}; weaknesses: {', '.join(f['weaknesses']) or 'n/a'}; affects {f['n_products']} products.")
                sp = predict.split_of(cve)
                st.info(predict.SPLIT_TEXT.get(sp, "This CVE has an expert mapping."))
                rows = predict.predict(cve, k)
                if not rows:
                    st.warning("There is no prediction stored for this CVE.")
                else:
                    df = pd.DataFrame(rows)
                    df["match with expert mapping"] = df.hit.map({True: "✓", False: ""}) if f["expert"] else "(no expert mapping)"
                    st.dataframe(df[["rank", "technique_id", "technique", "tactics", "score", "match with expert mapping"]].rename(columns={"technique_id": "technique id"}),
                                 width="stretch", hide_index=True)
                    if f["expert"]:
                        st.write("**Expert mapping (the labels):** " + "; ".join(f"{e['technique_id']} {e['technique']}" for e in f["expert"]))
                        hits = sum(r["hit"] for r in rows)
                        st.caption(f"{hits} of the top {len(rows)} predicted techniques are in the expert mapping.")
                    st.graphviz_chart(predictor_dot(cve, rows[:min(k, 8)]), width="stretch")
        with st.expander("About the model"):
            st.write(f"Selected model: **{sel}**, chosen by its NDCG@10 on the validation CVEs. It was fitted on the 1,026 labelled CVEs of the training and validation splits and evaluated once on the 181 test CVEs.")
            tm = (info.get("test_metrics") or {}).get(sel)
            if tm:
                st.write(f"Test results: Hits@1 {tm['hits@1']:.3f}, Hits@3 {tm['hits@3']:.3f}, Hits@10 {tm['hits@10']:.3f}, MRR {tm['mrr']:.3f}, NDCG@10 {tm['ndcg@10']:.3f}. "
                         "All five compared models were statistically indistinguishable on these 181 test CVEs; the popularity baseline reached NDCG@10 0.446.")
            st.write("Limits: the expert labels are incomplete, many CVEs have near-copies in the training data, and a predicted technique is a suggestion for an analyst to check, not a verified fact.")

with tab_a:
    st.header("About the system and the data")
    s = graph.stats()
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Nodes", f"{s['nodes']:,}"); a2.metric("Edges", f"{s['edges']:,}"); a3.metric("CVEs", f"{s['cves']:,}"); a4.metric("CVEs with expert mapping", f"{s['cves_with_mapping']:,}")
    c1, c2 = st.columns(2)
    c1.write("**Nodes by type**"); c1.dataframe(pd.DataFrame({"type": list(s["nodes_by_type"]), "nodes": list(s["nodes_by_type"].values())}), hide_index=True, width="stretch")
    c2.write("**Edges by relation**"); c2.dataframe(pd.DataFrame({"relation": list(s["edges_by_relation"]), "edges": list(s["edges_by_relation"].values())}), hide_index=True, width="stretch")
    st.markdown("""
**How a question is answered**

1. *Entity linking*: CVE ids, tactic ids and product names in the question are found in the graph.
2. *GraphRAG*: the neighbourhood of these entities (products → CVEs → ATT&CK techniques → tactics, mitigations, groups) is written as compact facts and sent to the language model in one call.
3. *Agentic GraphRAG*: the model gets six tools (`find_product`, `neighbors`, `filter_cves`, `sort_cves`, `show`, `describe`) and calls them one after another. Every result is a numbered set (S1, S2, ...) that later calls can refer to, so large intermediate results never have to fit into the prompt. The answer is given as JSON, or as the name of a set.

**Results on the 72 test questions of the benchmark** (same local gpt-oss-20b model for all three systems; average score, bootstrap 95 % interval):
plain RAG 17.3 % (10.2–25.2), GraphRAG 83.1 % (74.0–91.0), Agentic GraphRAG 92.1 % (85.6–97.2).

**Data sources**: NVD (CVE records), CISA KEV, EPSS, MITRE ATT&CK, and the expert CVE-to-ATT&CK mappings of the Center for Threat-Informed Defense. The data is a snapshot; KEV and EPSS change over time.
""")
