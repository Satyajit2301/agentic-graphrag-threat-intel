"""Compare an answer with the gold answer of a benchmark question (same rules as the notebook scorer, Block 9).

set answers: F1 of the sets; ranked answers: overlap of the top three divided by three; counts: exact match.
"""
from __future__ import annotations

import re


def ids_of(value) -> list[str]:
    if isinstance(value, list):
        out = []
        for x in value:
            m = re.findall(r"\b(?:CVE-\d{4}-\d{4,}|CWE-\d+|TA\d{4}|[TGM]\d{4}(?:\.\d{3})?)\b", str(x), flags=re.I)
            out += [y.upper() for y in m] if m else [str(x).strip()]
        return list(dict.fromkeys(out))
    return []


def score_against(gold, kind: str, value) -> dict:
    """-> {'score': 0..1, 'exact': bool, 'text': short sentence for the screen}"""
    if kind == "number":
        v = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if v is None and isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
            v = int(value)
        ok = v is not None and int(v) == int(gold)
        return {"score": float(ok), "exact": ok, "text": f"expected {gold}, got {v if v is not None else 'no number'}"}
    pred = ids_of(value)
    g = list(gold)
    if kind.endswith("_ranked"):
        sc = len(set(pred[:3]) & set(g)) / 3
        return {"score": sc, "exact": pred[:3] == g, "text": f"top-3 overlap {sc:.2f}" + ("; same order" if pred[:3] == g else "; order differs or items missing")}
    tp = len(set(pred) & set(g))
    p, r = (tp / len(set(pred)) if pred else 0.0), (tp / len(g) if g else 0.0)
    f1 = 2 * p * r / (p + r) if tp else 0.0
    return {"score": f1, "exact": set(pred) == set(g), "text": f"F1 {f1:.2f} (precision {p:.2f}, recall {r:.2f})"}
