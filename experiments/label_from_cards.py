"""Label models from their CARD TEXT, not their repo name.

Names miss cases: VIDraft/Gemma-3-R1984-1B gives no hint in its name, its card says
"delivers uncensored responses", and its scanned resid is 0.133 with the per-layer
family signature. Counting it as benign depresses the measured false-positive rate
and hides a true detection.

Writes results/labels.json: {repo_id: {"label":..., "evidence":...}}
  ablation   card names a projection method (abliterat/orthogonali/refusal direction)
  ambiguous  card claims uncensored/unfiltered behaviour, method unstated
  benign     neither
Card text is weak supervision, not ground truth; disagreements are for inspection.
"""
import json, glob, re
from pathlib import Path
from huggingface_hub import hf_hub_download

RESULTS = Path(__file__).resolve().parent / "results"
ABL = re.compile(r"ablitera|abliter|orthogonali[sz]|refusal[- ]?direction|"
                 r"refusal[- ]?ablat|heretic", re.I)
AMB = re.compile(r"uncensor|unfilter|unalign|unrestrict|amoral|jailbreak|nsfw|"
                 r"decensor|no refusal|without refusal", re.I)
# Safety models DISCUSS these words without having them done to them:
# Qwen3Guard-Gen-0.6B says "Jailbreak" because it DETECTS jailbreaks. Card text has
# its own failure mode, symmetric to the one in repo names.
GUARD = re.compile(r"\bguard\b|moderation|classif|detect(or|ion)|safety filter|"
                   r"content filter|harmful content|policy violation", re.I)


def card(rid):
    for fn in ("README.md",):
        try:
            return Path(hf_hub_download(rid, fn)).read_text(errors="ignore")
        except Exception:
            pass
    return ""


def main():
    ids = set()
    for f in glob.glob(str(RESULTS / "d6_*.json")):
        for r in json.loads(Path(f).read_text())["rows"]:
            if "error" not in r:
                ids.add(r["id"])
    out = {}
    p = RESULTS / "labels.json"
    if p.exists():
        out = json.loads(p.read_text())
    for i, rid in enumerate(sorted(ids - set(out)), 1):
        txt = rid + "\n" + card(rid)
        m = ABL.search(txt)
        if m:
            lab, ev = "ablation", m.group(0)
        else:
            m = AMB.search(txt)
            if m and GUARD.search(txt):
                lab, ev = "benign", f"guard model (said {m.group(0)!r})"
            else:
                lab, ev = ("ambiguous", m.group(0)) if m else ("benign", "")
        out[rid] = {"label": lab, "evidence": ev}
        print(f"[{i}] {lab:<10} {ev[:22]:<22} {rid[:46]}")
        p.write_text(json.dumps(out, indent=1))
    n = {k: sum(1 for v in out.values() if v["label"] == k)
         for k in ("ablation", "ambiguous", "benign")}
    print(f"\n{len(out)} labelled: {n}")
    print("->", p)


if __name__ == "__main__":
    main()
