"""Do INDEPENDENT authors recover the SAME direction from the same base model?

The evaluation establishes that resid detects a weight-space projection. It does not,
on its own, establish that a projection edit is the thing we care about -- that
inference currently rests on self-reported model cards.

This closes the gap from weights alone. If the recovered subspace really is the
refusal direction, then different people using different tools on the same base must
converge on it. If instead each toolchain recovers its own arbitrary direction,
agreement sits at chance and the self-reported labels are carrying all the weight.

Chance for |cos| between two random unit vectors in d dimensions is sqrt(2/(pi*d)),
but chance is the WRONG control on its own. The right control is a BENIGN finetune of
the same base: its delta also has a leading direction, and if that lands as close to
an abliteration's direction as two abliterations land to each other, the convergence
means nothing.

A mismatched-tensor control does NOT work when either model is global-family: its
direction is identical in every tensor, so shuffling tensors resamples the same
distribution. Preliminary run on Qwen3-1.7B hit exactly that -- matched 0.3312 vs
"control" 0.3299 -- and the control was uninformative rather than reassuring.

Predictions, both falsifiable:
  global-direction family   high agreement between authors, in every tensor
  per-layer family          high agreement WITHIN a layer, lower across layers,
                            because the direction is recomputed per layer
"""
import json, re, math, itertools
from pathlib import Path
import torch

R = Path(__file__).resolve().parent / "results"
DIRS = R / "dirs"


def load():
    out = {}
    for p in sorted(DIRS.glob("dir_*.pt")):
        d = torch.load(p)
        out[p.stem[4:]] = {k: v for k, v in zip(d["keys"], d["U"].float())}
    return out


def layer(k):
    m = re.search(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else -1


def main():
    if not DIRS.exists():
        print("no direction files yet -- they are written by the rescore pass")
        return
    labels = {}
    try:
        labels = json.loads((R / "labels.json").read_text())
    except Exception:
        pass
    mods = load()
    norm = {re.sub(r"[^A-Za-z0-9]+", "_", k): k for k in labels}
    abl = [m for m in mods
           if labels.get(norm.get(m, ""), {}).get("label") in ("ablation", "ambiguous")]
    if len(abl) < 2:
        print(f"{len(abl)} ablation direction sets saved; need 2+ on a shared base")
        return

    ben = [m for m in mods
           if labels.get(norm.get(m, ""), {}).get("label") == "benign"]

    def family(m):
        """global-family models share ONE direction across all tensors; per-layer
        models recompute it per layer. The comparison means different things in
        each case, so report them separately."""
        ks = sorted(mods[m])
        if len(ks) < 3:
            return "?"
        c = [abs(float(mods[m][a] @ mods[m][b]))
             for a, b in zip(ks[:-1], ks[1:]) if layer(a) != layer(b)]
        if not c:
            return "?"
        return "global" if sorted(c)[len(c) // 2] > 0.9 else "per-layer"

    def pair(a, b):
        shared = sorted(set(mods[a]) & set(mods[b]))
        if not shared:
            return None
        # direction files carry no base id; models from different bases live in
        # different spaces, so a shared tensor NAME is not enough -- require the
        # same dimension too.
        if mods[a][shared[0]].numel() != mods[b][shared[0]].numel():
            return None
        cos = sorted(abs(float(mods[a][k] @ mods[b][k])) for k in shared)
        d = mods[a][shared[0]].numel()
        return (len(shared), cos[len(cos) // 2], cos[-1],
                math.sqrt(2.0 / (math.pi * d)))

    fam = {m: family(m) for m in abl}
    print(f"{'pair':<56}{'n':>5}{'median':>9}{'max':>8}{'chance':>8}")
    print("-" * 86)
    print("GLOBAL x GLOBAL -- THE DECISIVE TEST. Two tools that each commit to a")
    print("single direction must agree UNIFORMLY if that direction is intrinsic to")
    print("the model, and sit at chance if it is an artefact of the toolchain.")
    gg = [m for m in abl if fam[m] == "global"]
    for a, b in itertools.combinations(sorted(gg), 2):
        r = pair(a, b)
        if r:
            print(f"  {a[:26]:<26} x {b[:26]:<26}{r[0]:>5}{r[1]:>9.4f}{r[2]:>8.4f}{r[3]:>8.4f}")
    print("\nGLOBAL x PER-LAYER  (expect a PEAK at one layer, decaying away)")
    for a in sorted(gg):
        for b in sorted(m for m in abl if fam[m] == "per-layer"):
            r = pair(a, b)
            if r:
                print(f"  {a[:26]:<26} x {b[:26]:<26}{r[0]:>5}{r[1]:>9.4f}{r[2]:>8.4f}{r[3]:>8.4f}")
    print("\nABLATION x BENIGN    (the control: must sit far lower)")
    for a in sorted(abl):
        for b in sorted(ben):
            r = pair(a, b)
            if r:
                print(f"  {a[:24]:<24} x {b[:24]:<24}{r[0]:>5}{r[1]:>9.4f}{r[2]:>8.4f}{r[3]:>8.4f}")


if __name__ == "__main__":
    main()
