"""Synthetic edits on real weights, scored with the scanned projection residual.

Every edit removes ARBITRARY orthonormal directions (seeded Gaussian, QR), never a
behavioural refusal direction, so no safety-relevant artifact is produced. One R is
shared by all tensors, as in abliteration.

Base: Qwen/Qwen3-0.6B, all 56 residual-writing matrices (o_proj, down_proj).

  exact       W' = W - a R R^T W, k in {1,2,4,8,16,32}, a in {0.2,0.5,1.0}
  repair      exact k=4, a=1, plus lam * (||D_abl|| / ||D_ft||) * D_ft per tensor, where
              D_ft is the real delta of a public fine-tune of the same base
              (suayptalha/Qwen3-0.6B-Code-Expert); lam is the perturbation-to-edit
              norm ratio
  multi       exact k in {1,4,8,16}, a=1, plus D_ft at lam in {0, 0.5, 1, 2}: scan vs
              a fixed k=1
  projection  projection-shaped edits that are NOT removals of typical directions:
              an ADDED projection (W + 0.5 R R^T W), and truncated-SVD compression
              removing the 4, 32 or 512 lowest singular directions of each W
  controls    D_ft alone; random rank-8 product (LoRA-shaped); pure rescale W' = 0.5 W,
              the last at kmax=32 and kmax=d to show why kmax must be small

Reports, per condition, medians over tensors of resid_best, k_best, the fitted
strength alpha at k_best, the energy ratio, and r_1 (the residual at a fixed k=1).
Writes results/synth_scan.json. ~2.7 GB download on first run, ~10-15 min CPU.
"""
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from d1_deltastruct import RESID_WRITE, RESULTS
from mech import resid_scan

BASE = "Qwen/Qwen3-0.6B"
FT = "suayptalha/Qwen3-0.6B-Code-Expert"
KS = [1, 2, 4, 8, 16, 32]
ALPHAS = [0.2, 0.5, 1.0]
LAMS = [0.0, 0.3, 1.0, 3.0, 5.0, 10.0]
MULTI_KS = [1, 4, 8, 16]
MULTI_LAMS = [0.0, 0.5, 1.0, 2.0]
TRUNC = [4, 32, 512]


def resid_tensors(rid):
    A = load_file(hf_hub_download(rid, "model.safetensors"))
    return {k: v.float() for k, v in A.items()
            if any(q in k for q in RESID_WRITE) and v.dim() == 2}


def score(W, D, kmax=32):
    U, s, Vh = torch.linalg.svd(D, full_matrices=False)
    r, k, curve, er, a = resid_scan(D, W, U, s, Vh, kmax)
    return {"r": r, "k": float(k), "alpha": a, "er": er, "r1": float(curve[0])}


def summarise(rows):
    med = lambda f: float(torch.tensor([x[f] for x in rows]).median())
    return {"resid_median": med("r"),
            "resid_max": float(max(x["r"] for x in rows)),
            "k_best_median": med("k"),
            "alpha_median": med("alpha"),
            "energy_ratio_median": med("er"),
            "r1_median": med("r1")}


def with_ft(Wb, Wf, t, R, lam):
    Da = -R @ (R.T @ Wb[t])
    Df = Wf[t] - Wb[t]
    return Da + lam * (Da.norm() / Df.norm()) * Df


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    Wb = resid_tensors(BASE)
    Wf = resid_tensors(FT)
    keys = sorted(k for k in Wb if k in Wf and Wf[k].shape == Wb[k].shape)
    d = Wb[keys[0]].shape[0]
    g = torch.Generator().manual_seed(0)
    Rfull, _ = torch.linalg.qr(torch.randn(d, max(KS), generator=g))
    out = {"base": BASE, "ft": FT, "n_tensors": len(keys), "d": d,
           "exact": [], "repair": [], "multi": [], "projection": {}, "controls": {}}

    for k in KS:
        R = Rfull[:, :k]
        for a in ALPHAS:
            rows = [score(Wb[t], -a * R @ (R.T @ Wb[t])) for t in keys]
            out["exact"].append({"k": k, "alpha": a, **summarise(rows)})
            print("exact", out["exact"][-1])

    R4 = Rfull[:, :4]
    for lam in LAMS:
        rows = [score(Wb[t], with_ft(Wb, Wf, t, R4, lam)) for t in keys]
        out["repair"].append({"lambda": lam, **summarise(rows)})
        print("repair", out["repair"][-1])

    for k in MULTI_KS:
        R = Rfull[:, :k]
        for lam in MULTI_LAMS:
            rows = [score(Wb[t], with_ft(Wb, Wf, t, R, lam)) for t in keys]
            out["multi"].append({"k": k, "lambda": lam, **summarise(rows)})
            print("multi", out["multi"][-1])

    p = out["projection"]
    p["added_k4"] = summarise([score(Wb[t], 0.5 * R4 @ (R4.T @ Wb[t])) for t in keys])
    for m in TRUNC:
        rows = []
        for t in keys:
            UW = torch.linalg.svd(Wb[t], full_matrices=False)[0]
            B = UW[:, -m:]
            rows.append(score(Wb[t], -B @ (B.T @ Wb[t])))
        p[f"truncate_{m}"] = summarise(rows)
    print("projection", json.dumps(p, indent=1))

    c = out["controls"]
    c["finetune"] = summarise([score(Wb[t], Wf[t] - Wb[t]) for t in keys])
    rows = []
    for t in keys:
        W = Wb[t]
        B = torch.randn(W.shape[0], 8, generator=g)
        A = torch.randn(8, W.shape[1], generator=g)
        rows.append(score(W, B @ A))
    c["lowrank8"] = summarise(rows)
    c["rescale_kmax32"] = summarise([score(Wb[t], -0.5 * Wb[t], 32) for t in keys])
    c["rescale_kmaxd"] = summarise([score(Wb[t], -0.5 * Wb[t], d) for t in keys])
    print("controls", json.dumps(c, indent=1))

    (RESULTS / "synth_scan.json").write_text(json.dumps(out, indent=1))
    print("->", RESULTS / "synth_scan.json")


if __name__ == "__main__":
    main()
