"""Does the WEIGHT DELTA of a released derivative reveal HOW it was modified?

Lineage tools answer "which base is this". This asks the next question: given a
base/derivative pair that genuinely IS a finetune, what was done to it?

The hypothesis is mechanical. Refusal-direction ablation ("abliteration") computes a
single direction r in residual-stream space and, for every matrix that WRITES to the
residual stream, applies

    W <- W - r r^T W

so the delta is exactly rank 1, and its top left-singular vector is r -- the SAME r
for every such tensor in the checkpoint. Ordinary finetuning (task SFT, distillation)
produces a diffuse, roughly full-rank delta whose principal directions have no reason
to agree across tensors.

So two weights-only statistics should separate the techniques:

  top1      sigma_1^2 / sum sigma_i^2 for Delta, per tensor.  ~1.0 if rank-1 edit.
  align     median |cos(u_1^(s), u_1^(t))| over pairs of residual-writing tensors.
            ~1.0 if one shared direction was removed everywhere, ~0 otherwise.

align is the discriminative one: top1 alone can be high for a low-rank adapter merge,
but a SHARED top direction across unrelated tensors is specific to a global
projection. Chance level for |cos| between two random unit vectors in d dimensions is
sqrt(2/(pi*d)), reported alongside.

No forward passes, no GPU, no eval data. Shards stream one at a time and are deleted,
so peak disk is about two shards rather than two checkpoints. All linalg is forced to
CPU: torch svd wedges on MPS.

Usage:
  python d1_deltastruct.py --base google/gemma-3-1b-it \
                           --deriv DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated \
                           --tag heretic1b
"""

import argparse, json, re, shutil
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

RESULTS = Path(__file__).resolve().parent / "results"
# Matrices whose OUTPUT dimension is the residual stream: these are what a global
# projection would have to touch, so their left singular vectors are comparable.
RESID_WRITE = ("self_attn.o_proj", "mlp.down_proj")


def drop(path):
    blob = Path(path).resolve()
    shutil.rmtree(Path(path).parent, ignore_errors=True)
    blob.unlink(missing_ok=True)


def shard_map(rid):
    try:
        p = hf_hub_download(rid, "model.safetensors.index.json")
        idx = json.loads(Path(p).read_text())["weight_map"]
        drop(p)
    except Exception:
        return {"model.safetensors": None}
    out = defaultdict(list)
    for k, v in idx.items():
        out[v].append(k)
    return dict(out)


def bucket(key):
    for pat, name in (("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
                      ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_o"),
                      ("mlp.gate_proj", "mlp_gate"), ("mlp.up_proj", "mlp_up"),
                      ("mlp.down_proj", "mlp_down"), ("embed_tokens", "embed")):
        if pat in key:
            return name
    return "norm" if "norm" in key else "other"


def spectrum(D):
    """Singular values of Delta on CPU. Falls back to randomized SVD when huge."""
    D = D.float().cpu()
    m = min(D.shape)
    if m > 4096:
        q = min(64, m)
        _, s, _ = torch.svd_lowrank(D, q=q, niter=4)
        return s, None
    U, s, _ = torch.linalg.svd(D, full_matrices=False)
    return s, U[:, 0].contiguous()


def key_map(rid):
    """tensor key -> shard filename, independent of how the repo is sharded."""
    from huggingface_hub.errors import EntryNotFoundError
    try:
        p = hf_hub_download(rid, "model.safetensors.index.json")
        wm = json.loads(Path(p).read_text())["weight_map"]
        drop(p)
        return wm
    except EntryNotFoundError:
        # genuinely a single-file repo: one shard holding everything
        return None
    except Exception as e:
        # auth/gating/network must NOT masquerade as "single-file repo", which
        # silently yields zero comparable tensors and looks like a clean negative.
        raise RuntimeError(f"cannot read shard index for {rid}: "
                           f"{type(e).__name__}: {str(e)[:160]}") from e


def shard_groups(base, deriv, max_shards=0):
    """Ordered [(base_shard, deriv_shard, keys_or_None)] pairs that must be co-resident.

    Repos sharded differently (2-file base vs 1-file derivative) are matched on
    tensor KEY, not on filename, so layout differences are not a hard failure.
    """
    km_b, km_d = key_map(base), key_map(deriv)

    def shard_of(km, k):
        return km[k] if km else "model.safetensors"

    if km_b is not None and km_d is not None:
        keys_all = sorted(set(km_b) & set(km_d))
    elif km_b is not None:
        keys_all = sorted(km_b)
    elif km_d is not None:
        keys_all = sorted(km_d)
    else:
        keys_all = None

    groups = defaultdict(list)
    if keys_all is None:
        groups[("model.safetensors", "model.safetensors")] = None
    else:
        for k in keys_all:
            groups[(shard_of(km_b, k), shard_of(km_d, k))].append(k)
    order = sorted(groups)
    if max_shards:
        seen, keep = [], []
        for g in order:
            if g[0] not in seen:
                seen.append(g[0])
            if len(seen) <= max_shards:
                keep.append(g)
        order = keep
    if not order:
        raise RuntimeError(f"no comparable tensors: {base} vs {deriv}")
    return [(sb, sd, groups[(sb, sd)]) for (sb, sd) in order]


def analyse(base, deriv, max_shards=0, save_u1=True, tag=None):
    order = shard_groups(base, deriv, max_shards)

    per_tensor, u1 = [], {}
    for (sb, sd, ks) in order:
        pb = hf_hub_download(base, sb)
        pd = hf_hub_download(deriv, sd)
        A, B = load_file(pb), load_file(pd)
        if ks is None:
            ks = sorted(A.keys() & B.keys())
        for k in ks:
            if k not in A or k not in B:
                continue
            if "vision" in k or "multi_modal" in k:
                continue
            if A[k].shape != B[k].shape or A[k].dim() != 2:
                continue
            a, b = A[k].float(), B[k].float()
            D = b - a
            nb, nd = float(a.norm()), float(D.norm())
            rec = {"key": k, "bucket": bucket(k), "shape": list(a.shape),
                   "rel": nd / nb if nb else None}
            if nd > 1e-8 and nb > 0:
                sv, u = spectrum(D)
                e = (sv ** 2)
                tot = float(e.sum())
                rec["top1"] = float(e[0]) / tot if tot else None
                rec["top5"] = float(e[:5].sum()) / tot if tot else None
                # participation ratio of the squared spectrum
                rec["eff_rank"] = float((tot ** 2) / float((e ** 2).sum())) if tot else None
                if u is not None and any(p in k for p in RESID_WRITE):
                    u1[k] = u
            per_tensor.append(rec)
        del A, B
        drop(pb); drop(pd)

    # cross-tensor agreement of the leading delta direction
    keys = sorted(u1)
    cos, pairs = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = abs(float(torch.dot(u1[keys[i]], u1[keys[j]])))
            cos.append(c)
            pairs.append([keys[i], keys[j], c])
    cos.sort()
    d_model = u1[keys[0]].numel() if keys else None
    chance = (2.0 / (3.141592653589793 * d_model)) ** 0.5 if d_model else None

    # persist the leading directions so structural questions never need a redownload
    if save_u1 and keys and tag:
        RESULTS.mkdir(exist_ok=True)
        torch.save({"keys": keys, "U": torch.stack([u1[k] for k in keys])},
                   RESULTS / f"u1_{tag}.pt")

    return {"base": base, "deriv": deriv, "n_tensors": len(per_tensor),
            "n_resid_write": len(keys), "d_model": d_model,
            "align_median": cos[len(cos) // 2] if cos else None,
            "align_min": cos[0] if cos else None, "align_max": cos[-1] if cos else None,
            "align_chance": chance, "align_pairs": pairs, "per_tensor": per_tensor}


def summarise(r):
    pt = [t for t in r["per_tensor"] if t.get("top1") is not None]
    def med(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else None
    rw = [t for t in pt if any(p in t["key"] for p in RESID_WRITE)]
    return {"rel_median": med([t["rel"] for t in pt]),
            "top1_median_residwrite": med([t["top1"] for t in rw]),
            "eff_rank_median_residwrite": med([t["eff_rank"] for t in rw]),
            "align_median": r["align_median"], "align_chance": r["align_chance"],
            "n_resid_write": r["n_resid_write"], "n_tensors_moved": len(pt)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--deriv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-shards", type=int, default=0)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    res = analyse(a.base, a.deriv, a.max_shards, tag=a.tag)
    res["summary"] = summarise(res)
    out = RESULTS / f"d1_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    print("->", out)
