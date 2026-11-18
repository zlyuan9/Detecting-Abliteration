"""Cross-author agreement with the BASE's own dominant direction partialled out.

Raw cross-author |cos| is confounded: base models are anisotropic (Qwen bases show
cross-tensor agreement 0.49-0.76 against a 0.02 chance floor), so two unrelated
deltas can agree simply by both leaning on the base's leading direction. Benign
models with high base alignment (0.79-0.93) reach |cos| 0.28 against an ablation
purely this way.

The fix: for each tensor, remove the base's top left-singular direction from every
delta direction, renormalise, and compare what is left. If two ablations still agree
after that, they share something the base does not already supply.
"""
import json
import math
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from d1_deltastruct import shard_groups, drop, RESID_WRITE

RESULTS = Path(__file__).resolve().parent / "results"


def base_dirs(base, nvec=1):
    """Top-n left singular directions of each residual-writing tensor of the base."""
    out = {}
    for sb, _sd, ks in shard_groups(base, base):
        p = hf_hub_download(base, sb)
        A = load_file(p)
        for k in (ks or sorted(A)):
            if k in A and any(q in k for q in RESID_WRITE) \
               and A[k].dim() == 2 and min(A[k].shape) <= 4096:
                U, _, _ = torch.linalg.svd(A[k].float(), full_matrices=False)
                out[k] = U[:, :nvec].contiguous()
        del A
        drop(p)
    return out


def residualise(v, B):
    """Remove span(B) from v and renormalise. None if nothing survives."""
    w = v - B @ (B.T @ v)
    n = float(w.norm())
    return w / n if n > 1e-6 else None


def main(base, nvec=1):
    torch.set_grad_enabled(False)
    BD = base_dirs(base, nvec)
    mods = {}
    for p in sorted((RESULTS / "dirs").glob("dir_*.pt")):
        d = torch.load(p)
        v = {k: x.float() for k, x in zip(d["keys"], d["U"])}
        if any(k in BD and x.numel() == BD[k].shape[0] for k, x in v.items()):
            mods[p.stem[4:]] = v
    names = {re.sub(r"[^A-Za-z0-9]+", "_", i): i for i in
             [l.strip() for l in sys.stdin] if i} if not sys.stdin.isatty() else {}

    dm = next(iter(BD.values())).shape[0]
    ch = math.sqrt(2.0 / (math.pi * dm))
    print(f"base {base}  d={dm}  chance={ch:.4f}  partialling out top-{nvec} base direction\n")
    keys = sorted(mods)
    from labels import label_of
    norm = {re.sub(r"[^A-Za-z0-9]+", "_", i): i for i in
            json.loads((RESULTS / "labels.json").read_text())}
    cls = {k: label_of(norm.get(k, k)) for k in keys}
    dump = []
    print(f"{'pair':<58}{'raw':>8}{'partialled':>12}")
    print("-" * 80)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            sh = [k for k in mods[a] if k in mods[b] and k in BD
                  and mods[a][k].numel() == dm]
            if len(sh) < 10:
                continue
            raw, par = [], []
            for k in sh:
                raw.append(abs(float(mods[a][k] @ mods[b][k])))
                ra = residualise(mods[a][k], BD[k])
                rb = residualise(mods[b][k], BD[k])
                if ra is not None and rb is not None:
                    par.append(abs(float(ra @ rb)))
            if not par:
                continue
            raw.sort(); par.sort()
            print(f"  {a[:26]:<26} x {b[:26]:<26}"
                  f"{raw[len(raw)//2]:>8.4f}{par[len(par)//2]:>12.4f}")
            dump.append({"a": a, "b": b, "cls_a": cls[a], "cls_b": cls[b],
                         "n": len(sh), "raw": raw[len(raw) // 2],
                         "par": par[len(par) // 2]})
    out = RESULTS / ("xauthor_" + base.split("/")[-1].replace(".", "_") + ".json")
    out.write_text(json.dumps({"base": base, "d": dm, "chance": ch,
                               "nvec": nvec, "pairs": dump}, indent=1))
    print("->", out, f"({len(dump)} pairs)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct",
         int(sys.argv[2]) if len(sys.argv) > 2 else 1)
