"""CVE -> ATT&CK technique predictor: a lookup in the predictions of the selected link-prediction model.

Block 15 of the notebook fitted the selected model (a multilayer perceptron) on the 1,026 training and validation CVEs and saved its
20 highest-scoring techniques for EVERY CVE of the graph (data/lp_predictions.parquet). The app shows those lists; it does not train anything.
data/lp_cve_info.parquet (made by the packaging block) says which split a CVE belongs to, so the app can say whether the model has seen its labels.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .graph import G

P = SimpleNamespace(ready=False)

SPLIT_TEXT = {
    "test": "Held-out test CVE: the model never saw its expert labels, so the comparison below is a fair test.",
    "val": "Validation CVE: its labels were used to choose the model settings and to fit the final model.",
    "train": "Training CVE: its labels were used to fit the model, so a good match here is expected and proves little.",
    "none": "No expert mapping exists for this CVE: the predicted techniques are new, unverified suggestions.",
}


def load(data_dir: str | Path = "data", force: bool = False) -> SimpleNamespace:
    if P.ready and not force:
        return P
    d = Path(data_dir)
    P.pred = pd.read_parquet(d / "lp_predictions.parquet")
    info = d / "lp_cve_info.parquet"
    P.split = dict(zip(*[pd.read_parquet(info)[c] for c in ("cve_id", "split")])) if info.exists() else {}
    P.model = str(P.pred["model"].iloc[0]) if "model" in P.pred.columns and len(P.pred) else "MLP"
    P.by_cve = {c: g.sort_values("rank") for c, g in P.pred.groupby("cve_id")}
    P.ready = True
    return P


def split_of(cve: str) -> str:
    if cve in P.split:
        return str(P.split[cve])
    return "labelled" if cve in G.techs_of_c else "none"


def predict(cve: str, k: int = 10) -> list[dict]:
    """top-k techniques for a CVE with the tactics from the graph and a hit mark against the expert mapping (if there is one)"""
    cve = cve.strip().upper()
    g = P.by_cve.get(cve)
    if g is None:
        return []
    expert = {t for t, _ in G.techs_of_c.get(cve, [])}
    rows = []
    for r in g.head(k).itertuples():
        t = r.technique_id
        rows.append({"rank": int(r.rank), "technique_id": t, "technique": G.TNAME.get(t) or getattr(r, "technique_name", "") or "", "score": float(r.score),
                     "tactics": ", ".join(G.TACNAME.get(x, x) for x in sorted(G.tactics_of_t.get(t, []))) or "n/a", "hit": t in expert})
    return rows


def cve_facts(cve: str) -> dict:
    cve = cve.strip().upper()
    a, ci = G.ATTR["cve:" + cve], G.CI[cve]
    return {"cve": cve, "description": G.TEXT.get("cve:" + cve, ""), "cvss": ci["cvss"], "severity": ci["sev"], "epss": ci["epss"], "kev": ci["kev"],
            "published": str(a.get("published", ""))[:10], "weaknesses": sorted(G.cwe_of_c.get(cve, [])), "n_products": G.n_prod_of_c.get(cve, 0),
            "expert": [{"technique_id": t, "technique": G.TNAME.get(t, ""), "role": r} for t, r in sorted(G.techs_of_c.get(cve, []))]}


def sample_cve(kind: str, seed: int | None = None) -> str | None:
    """a random CVE for the demo: kind = 'test' (held-out, has expert labels) or 'none' (no expert mapping)"""
    cands = sorted(c for c in P.by_cve if split_of(c) == kind and c in G.CVE_SET)
    if not cands:
        return None
    import random
    return random.Random(seed).choice(cands) if seed is not None else random.choice(cands)
