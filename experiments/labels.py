"""One source of truth for class labels and row selection.

Two scripts each carrying their own copy of a classification rule drifted twice in
one session, and both times the drift surfaced as a RESULT rather than an error:
once putting a rebranded abliteration in the benign class, once showing a false
positive in the figure that did not exist. Every consumer imports from here.
"""
import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

# fallback only -- repo names miss cases the model card states outright
ABL = re.compile(r"ablitera|abliter|heretic|orthogonali[sz]|refusal.?direction", re.I)
AMB = re.compile(r"uncensor|amoral|unalign|unrestrict|jailbr|nsfw|unfilter|"
                 r"decensor|josiefied|dolphin", re.I)

_CARDS = {}
try:
    _CARDS = json.loads((RESULTS / "labels.json").read_text())
except Exception:
    pass


def label_of(rid):
    """ablation | ambiguous | benign, from the model card where available."""
    lab = _CARDS.get(rid, {}).get("label")
    if lab:
        return lab
    return ("ablation" if ABL.search(rid)
            else "ambiguous" if AMB.search(rid) else "benign")


def load_rows(min_rel=1e-4, prefer_scan=True):
    """Every scored model, one row each, labelled.

    - near-mirrors (rel < min_rel) are dropped: the delta is numerical noise and
      every statistic is degenerate, so counting them as negatives is a free win
    - a rescored model appears in BOTH its original tag file and the rb_* file;
      keep the row carrying the scanned resid
    - resid is reported as the SCANNED value wherever it exists: a fixed k is a
      different and worse statistic, not a simplification
    """
    out = []
    for f in sorted(RESULTS.glob("d6_*.json")):
        blob = json.loads(f.read_text())
        for r in blob["rows"]:
            if "error" in r or r.get("resid") is None:
                continue
            if r.get("rel") is not None and r["rel"] < min_rel:
                continue
            if prefer_scan and r.get("resid_best") is not None:
                r["resid"] = r["resid_best"]
            r["base"] = blob.get("base")
            r["claim"] = label_of(r["id"])
            out.append(r)
    best = {}
    for r in out:
        prev = best.get(r["id"])
        if prev is None or (r.get("resid_best") is not None
                            and prev.get("resid_best") is None):
            best[r["id"]] = r
    return list(best.values())


if __name__ == "__main__":
    rs = load_rows()
    n = {}
    for r in rs:
        n[r["claim"]] = n.get(r["claim"], 0) + 1
    scan = sum(1 for r in rs if r.get("resid_best") is not None)
    print(f"{len(rs)} rows  {n}  scanned resid: {scan}/{len(rs)}")
    for c in ("ablation", "ambiguous", "benign"):
        v = sorted(r["resid"] for r in rs if r["claim"] == c)
        if v:
            print(f"  {c:<10} {v[0]:.3f} .. {v[len(v)//2]:.3f} .. {v[-1]:.3f}")
