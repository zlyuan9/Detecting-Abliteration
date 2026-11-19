"""Exact edits of Qwen3-0.6B, then rounded to the precision a release is stored in.

An abliteration computed as W' = W - a R R^T W is exactly a scaled projection in exact
arithmetic, so r = 0. A release stores W' in bf16 or fp16, which adds rounding error e
to the difference: Delta = -a R R^T W + e. If e is uncorrelated with the edit and of
norm ||e||, then with rho = ||e|| / ||a R R^T W||

    r = rho / sqrt(1 + rho^2)                      (same algebra as the paper's Eq. 5)

and rho grows as the edit shrinks, because the rounding error does not.

Base: Qwen/Qwen3-0.6B (stored in bf16), all 56 residual-writing matrices, the same
seeded random R as synth_scan.py. Edits are k in {1, 4} directions at a in
{0.2, 0.5, 1.0}, rounded to fp32 (no rounding), bf16 and fp16.

Reports, per condition, the median over matrices of the scanned residual, the relative
edit size ||Delta|| / ||W||, and rho (measured against the exact edit) with the residual
it predicts. Writes results/synth_bf16.json. Needs only the base model (~1.2 GB).
"""
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from d1_deltastruct import RESID_WRITE, RESULTS
from synth_scan import score

BASE = "Qwen/Qwen3-0.6B"
KS = [1, 4]
ALPHAS = [0.2, 0.5, 1.0]
STORE = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    A = load_file(hf_hub_download(BASE, "model.safetensors"))
    keys = sorted(k for k, v in A.items()
                  if any(q in k for q in RESID_WRITE) and v.dim() == 2)
    dtypes = sorted({str(A[k].dtype) for k in keys})
    Wb = {k: A[k].float() for k in keys}
    d = Wb[keys[0]].shape[0]
    g = torch.Generator().manual_seed(0)
    R, _ = torch.linalg.qr(torch.randn(d, max(KS), generator=g))
    med = lambda xs: float(torch.tensor(xs).median())
    out = {"base": BASE, "stored_dtype": dtypes, "n_tensors": len(keys), "rows": []}
    for k in KS:
        Rk = R[:, :k]
        for a in ALPHAS:
            for name, dt in STORE.items():
                r, kk, al, er, rel, rho, nz = [], [], [], [], [], [], []
                for t in keys:
                    W = Wb[t]
                    Dex = -a * Rk @ (Rk.T @ W)
                    Wp = W + Dex
                    if dt is not None:
                        Wp = Wp.to(dt).float()
                    D = Wp - W
                    sc = score(W, D)
                    r.append(sc["r"]); kk.append(sc["k"]); al.append(sc["alpha"])
                    er.append(sc["er"])
                    rel.append(float(D.norm() / W.norm()))
                    rho.append(float((D - Dex).norm() / Dex.norm()))
                    nz.append(float((D - Dex).norm() / W.norm()))
                row = {"k": k, "alpha": a, "storage": name, "resid_median": med(r),
                       "k_best_median": med(kk), "alpha_hat_median": med(al),
                       "energy_ratio_median": med(er),
                       "rel_median": med(rel), "rho_median": med(rho),
                       "noise_rel_median": med(nz),
                       "predicted": med(rho) / (1 + med(rho) ** 2) ** 0.5}
                out["rows"].append(row)
                print(row, flush=True)
    (RESULTS / "synth_bf16.json").write_text(json.dumps(out, indent=1))
    print("->", RESULTS / "synth_bf16.json")


if __name__ == "__main__":
    main()
