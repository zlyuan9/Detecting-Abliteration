"""Figures and the appendix table for paper/abliteration.tex.
Reads results/d6_*.json only; no downloads.

  fig_scan.pdf    median r_k over tensors vs k, four public abliterations with
                  different minimising k, all other derivatives in grey
  fig_resid.pdf   scanned resid per public model, by declared class
  tab_models.tex  one row per scanned model (appendix longtable body)
and prints the numbers behind the hardening, fixed-k and agreement tables.
"""
import json
import random
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from labels import load_rows, RESULTS

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

SHOW = [
    ("mlabonne/gemma-3-1b-it-abliterated-v2", r"mlabonne, Gemma-3-1B ($\hat k=1$)"),
    ("failspy/Llama-3-8B-Instruct-abliterated", r"failspy v1, Llama-3-8B ($\hat k=2$)"),
    ("Goekdeniz-Guelmez/Qwen3-0.6B-gabliterated-Dev",
     r"Goekdeniz-Guelmez, Qwen3-0.6B ($\hat k=3$)"),
    ("failspy/Meta-Llama-3-8B-Instruct-abliterated-v3", r"failspy v3, Llama-3-8B ($\hat k=4$)"),
]
CLASSES = [("ablation", "Declared\nabliteration"),
           ("ambiguous", "Permissive,\nmethod unstated"),
           ("benign", "Other\nderivatives")]
BASES = {"Qwen/Qwen2.5-0.5B-Instruct": ("Qwen2.5-0.5B", "o"),
         "Qwen/Qwen3-0.6B": ("Qwen3-0.6B", "s"),
         "google/gemma-3-1b-it": ("Gemma-3-1B", "^"),
         "meta-llama/Llama-3.2-1B-Instruct": ("Llama-3.2-1B", "D"),
         "NousResearch/Meta-Llama-3-8B-Instruct": ("Llama-3-8B", "v")}


def curves():
    """resid_curve of the scanned row for every model load_rows keeps."""
    return {r["id"]: r["resid_curve"] for r in load_rows()
            if r.get("resid_curve") and r.get("resid_best") is not None}


def fig_scan(rows):
    cur = curves()
    cls = {r["id"]: r["claim"] for r in rows}
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    first = True
    for rid, c in cur.items():
        if cls.get(rid) == "benign":
            ax.plot(range(1, len(c) + 1), c, color="0.75", lw=0.8,
                    label="other derivatives" if first else None)
            first = False
    for rid, lab in SHOW:
        c = cur[rid]
        ax.plot(range(1, len(c) + 1), c, marker="o", ms=3, lw=1.2, label=lab)
    ax.set_xlabel("subspace dimension $k$")
    ax.set_ylabel(r"median $r_k$ over tensors")
    ax.set_xlim(0.5, 32.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([1, 2, 4, 8, 16, 24, 32])
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scan.pdf")
    plt.close(fig)


def fig_resid(rows, classes=CLASSES, fname="fig_resid.pdf", width=5.0):
    rows = [r for r in rows if r.get("resid_best") is not None]
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(width, 3.0))
    seen = set()
    for x, (c, _lab) in enumerate(classes):
        for r in rows:
            if r["claim"] != c:
                continue
            name, m = BASES[r["base"]]
            ax.scatter(x + rng.uniform(-0.18, 0.18), r["resid_best"], marker=m,
                       s=18, color=f"C{list(BASES).index(r['base'])}",
                       label=None if name in seen else name)
            seen.add(name)
    ax.set_xticks(range(len(classes)), [lab for _c, lab in classes])
    ax.set_ylabel(r"projection residual $r$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


def write_model_table(rows):
    """LaTeX rows: model, base, class, r, k_hat, ||D||/||W||, changed tensors."""
    tag = {"ablation": "A", "ambiguous": "P", "benign": "O"}
    order = {"ablation": 0, "ambiguous": 1, "benign": 2}
    rows = sorted((r for r in rows if r.get("resid_best") is not None),
                  key=lambda r: (order[r["claim"]], r["resid_best"], r["id"]))
    lines = []
    for r in rows:
        lines.append(
            rf"\nolinkurl{{{r['id']}}} & {BASES[r['base']][0]} & {tag[r['claim']]} & "
            rf"{r['resid_best']:.3f} & {r['k_best']} & {r['rel']:.3f} & "
            rf"{r['n_tensors']}/{r['n_candidates']} \\")
    (OUT.parent / "tab_models.tex").write_text("\n".join(lines) + "\n")


def separation(vals):
    """AUC (lower = abliteration), margin, and abliterations at/above the lowest other."""
    a = [v for c, v in vals if c == "ablation"]
    b = [v for c, v in vals if c == "benign"]
    auc = sum(1.0 if x < y else 0.5 if x == y else 0.0 for x in a for y in b) / (len(a) * len(b))
    return auc, min(b) - max(a), sum(x >= min(b) for x in a)


def print_tables(rows):
    scanned = {r["id"]: r for r in rows if r.get("resid_best") is not None}

    print("\n--- Llama-3-8B-Instruct derivatives (Table: hardening) ---")
    for r in sorted((r for r in scanned.values()
                     if r["base"] == "NousResearch/Meta-Llama-3-8B-Instruct"),
                    key=lambda r: (r["claim"] != "ablation", r["resid_best"])):
        print(f"  {r['id']:<72} {r['n_tensors']:>2}/{r['n_candidates']}  "
              f"rel {r['rel']:.3f}  r {r['resid_best']:.4f}  k {r['k_best']}")

    print("\n--- scan vs fixed k (Table: fixed) ---")
    fixed = {r["id"]: r["resid"] for r in load_rows(prefer_scan=False)}
    cur = curves()
    cls = [(scanned[i]["claim"], i) for i in scanned]
    for k in (1, 2, 4, 8, 16, 32):
        auc, gap, miss = separation([(c, cur[i][k - 1]) for c, i in cls])
        print(f"  fixed k={k:<3} AUC {auc:.3f}  margin {gap:+.3f}  missed {miss}")
    auc, gap, miss = separation([(c, fixed[i]) for c, i in cls])
    print(f"  99% energy  AUC {auc:.3f}  margin {gap:+.3f}  missed {miss}")
    auc, gap, miss = separation([(c, scanned[i]["resid_best"]) for c, i in cls])
    print(f"  scan        AUC {auc:.3f}  margin {gap:+.3f}  missed {miss}")

    print("\n--- direction agreement, Qwen3-0.6B (Table: agreement) ---")
    blob = json.loads((RESULTS / "xauthor_Qwen3-0_6B.json").read_text())
    ids = {re.sub(r"[^A-Za-z0-9]+", "_", i): i
           for i in json.loads((RESULTS / "labels.json").read_text())}
    owner = lambda s: ids.get(s, s).split("/")[0]
    groups = {}
    for p in blob["pairs"]:
        if p["cls_a"] == p["cls_b"] == "ablation" and owner(p["a"]) == owner(p["b"]):
            key = "same author"
        else:
            key = " x ".join(sorted((p["cls_a"], p["cls_b"])))
        groups.setdefault(key, []).append(p["par"])
    for key, v in sorted(groups.items()):
        print(f"  {key:<22} n={len(v):>3}  median {statistics.median(v):.3f}  "
              f"range {min(v):.3f}-{max(v):.3f}")
    print(f"  chance {blob['chance']:.3f}")



def _best_errors(vals):
    """Fewest misclassified models over all thresholds (lower = abliteration), in-sample."""
    xs = sorted({v for _c, v in vals})
    best = (len(vals), None)
    for lo, hi in zip(xs, xs[1:]):
        cut = (lo + hi) / 2
        err = sum((c == "ablation") != (v < cut) for c, v in vals)
        if err < best[0]:
            best = (err, cut)
    return best


def _band_top1(r, lo=0.33, hi=0.67):
    """Hurtado's E_1: top-1 energy fraction of Delta averaged over the mid-stack band."""
    lay = [(int(m.group(1)), t["top1"]) for t in r.get("per_tensor") or []
           if t.get("top1") is not None and (m := re.search(r"layers\.(\d+)\.", t["key"]))]
    if not lay:
        return None
    n_layers = max(l for l, _ in lay) + 1
    sel = [v for l, v in lay if int(lo * n_layers) <= l <= int(hi * n_layers)]
    return sum(sel) / len(sel) if sel else None


def print_baselines(rows):
    """Weights-only alternatives to the projection residual on the same models."""
    rows = [r for r in rows if r.get("resid_best") is not None
            and r["claim"] in ("ablation", "benign")]
    stats = [  # name, value per row, True if a HIGH value indicates abliteration
        ("projection residual (scan)", lambda r: r["resid_best"], False),
        ("top-1 energy fraction (=1/stable rank)", lambda r: r["top1"], True),
        ("effective rank of Delta", lambda r: r["eff_rank"], False),
        ("Hurtado E1 (mid-stack band top-1 energy)", _band_top1, True),
    ]
    print("\n--- baselines on the declared-abliteration vs other models ---")
    for name, f, high in stats:
        vals = [(r["claim"], f(r), r["id"]) for r in rows if f(r) is not None]
        sign = -1 if high else 1
        pairs = [(c, sign * v) for c, v, _i in vals]
        auc, gap, miss = separation(pairs)
        err, cut = _best_errors(pairs)
        na = sum(c == "ablation" for c, _v, _i in vals)
        print(f"  {name:<42} n={na}/{len(vals) - na}  AUC {auc:.3f}  "
              f"margin {gap:+.3f}  errors at best threshold {err}")
        if err:
            cut = sign * cut
            bad = [i for c, v, i in vals if (c == "ablation") != ((v > cut) if high else (v < cut))]
            print("      misclassified:", "; ".join(bad))
    multi = sorted((r["k_best"], r["id"]) for r in rows
                   if r["claim"] == "ablation" and r["k_best"] >= 2)
    print("  abliterations with k_hat >= 2:", "; ".join(f"{i} (k={k})" for k, i in multi))
    top = sorted(((r["top1"], r["id"]) for r in rows if r["claim"] == "benign"), reverse=True)[:2]
    print("  highest top-1 among other models:", "; ".join(f"{i} {v:.3f}" for v, i in top))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fig_scan(rows)
    fig_resid(rows)
    fig_resid(rows, [CLASSES[0], CLASSES[2]], "fig_resid_main.pdf", width=3.6)
    write_model_table(rows)
    print("wrote", OUT / "fig_scan.pdf", OUT / "fig_resid.pdf",
          OUT.parent / "tab_models.tex")
    print_tables(rows)
    print_baselines(rows)
