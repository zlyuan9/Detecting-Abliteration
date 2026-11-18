"""Score many derivatives of ONE base, holding the base resident.

d1/d4 re-fetch the base for every pair, which is fine for eight pairs and absurd for
eighty. This loads the base's residual-writing tensors once and streams derivatives
past them, so the marginal cost of another model is one derivative download.

Per derivative it records every statistic in play, so the comparison between them is
made on identical tensors:

  eff_rank, top1      spectral: how low-rank is the delta          (Hurtado's E_1)
  align_U, align_V    cross-tensor agreement of leading directions (ours; one-sided)
  within_L, across_L  the decomposition that splits the two ablation families
  kill                is the recovered subspace DEAD in the new weights
  alpha, resid        best-fit ablation strength, and what it fails to explain

Errors are per-derivative and non-fatal: shape mismatches and missing safetensors are
recorded and skipped, because the candidate list is weak supervision from repo names.
"""

import argparse, json, re, traceback
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from huggingface_hub import HfApi

from d1_deltastruct import shard_groups, drop, RESID_WRITE, RESULTS, key_map
from mech import resid_scan

_API = HfApi()


def compatible(rid, d_model, max_gb=12.0, n_shards=None):
    """Cheap pre-download screen.

    base_model tags on the Hub are frequently wrong: a 43GB model was listed as a
    child of a 0.5B base. Downloading it only to discard every tensor on a shape
    mismatch costs the whole disk, so check config.json (a few KB) and the declared
    file sizes first.
    """
    try:
        cfg = json.loads(Path(hf_hub_download(rid, "config.json")).read_text())
    except Exception as e:
        return False, f"no config.json ({type(e).__name__})"
    h = cfg.get("hidden_size") or cfg.get("text_config", {}).get("hidden_size")
    if h is not None and d_model is not None and h != d_model:
        return False, f"hidden_size {h} != base {d_model}"
    try:
        info = _API.model_info(rid, files_metadata=True)
        sizes = [f.size or 0 for f in info.siblings
                 if f.rfilename.endswith(".safetensors")]
        gb = sum(sizes) / 1e9
        # the cap bounds what we will actually DOWNLOAD, not the repo size: with
        # --max-shards we fetch only the shards holding the compared keys, so a
        # legitimately large checkpoint is fine while a mismatched one is not
        # (hidden_size above already rejects those).
        est = gb if not n_shards else gb * min(1.0, (n_shards + 1) / max(1, len(sizes)))
        if est > max_gb:
            return False, f"~{est:.1f}GB to fetch exceeds {max_gb}GB cap"
        # size must be consistent with the declared architecture. Sakalti/mont-xl-agi
        # declares hidden_size 896 / 41 layers (~0.5B params, ~1GB) in a 43GB repo;
        # whatever that is, it is not the model the config describes.
        L = cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers")
        V = cfg.get("vocab_size") or cfg.get("text_config", {}).get("vocab_size") or 0
        if h and L:
            implied = (12 * L * h * h + 2 * V * h) * 2 / 1e9   # bytes at fp16
            if gb > 6 * implied:
                return False, (f"{gb:.1f}GB repo vs ~{implied:.1f}GB implied by "
                               f"config (hidden {h}, layers {L})")
        if gb == 0:
            return False, "no safetensors"
    except Exception as e:
        return False, f"model_info failed ({type(e).__name__})"
    return True, None


def load_resid(rid, max_shards=0, shard_skip=0):
    order = shard_groups(rid, rid)
    if shard_skip:
        order = order[shard_skip:]
    if max_shards:
        order = order[:max_shards]
    W = {}
    for (sb, _s, ks) in order:
        p = hf_hub_download(rid, sb)
        A = load_file(p)
        if ks is None:
            ks = sorted(A.keys())
        for k in ks:
            if k in A and any(q in k for q in RESID_WRITE) \
               and A[k].dim() == 2 and min(A[k].shape) <= 4096:
                W[k] = A[k].cpu()
        del A
        drop(p)
    return W


def score(Wb, rid, kmax=32, save_dir=None):
    """Compare one derivative against the resident base."""
    km = key_map(rid)
    # only fetch derivative shards that actually hold a key we are comparing:
    # repos shard differently (4 vs 7 files for the same 8B model), and pulling
    # whole checkpoints instead of the needed shards is the difference between
    # 4GB and 16GB per derivative.
    if km:
        shards = sorted({v for k, v in km.items() if k in Wb})
    else:
        shards = ["model.safetensors"]
    per, u1, v1, curves = [], {}, {}, []
    for s in shards:
        p = hf_hub_download(rid, s)
        A = load_file(p)
        for k, Wh in Wb.items():
            if k not in A:
                continue
            T = A[k]
            if T.dim() != 2 or tuple(T.shape) != tuple(Wh.shape):
                continue
            W = Wh.float()
            Wn = T.float().cpu()
            D = Wn - W
            nd, nw = float(D.norm()), float(W.norm())
            if nd < 1e-8 or nw == 0:
                continue
            U, sv, Vh = torch.linalg.svd(D, full_matrices=False)
            e = sv ** 2
            tot = float(e.sum())
            eff = float(tot ** 2 / float((e ** 2).sum()))
            # participation ratio undercounts the rank of a projection when W has
            # structured singular values (15 for a true rank-16 edit), which leaves
            # a direction out of R and inflates resid. Take the smallest prefix
            # holding 99% of the delta energy instead.
            cum = torch.cumsum(e, 0) / tot
            kk = max(1, min(kmax, int((cum < 0.99).sum().item()) + 1))
            R = U[:, :kk]
            P = R @ (R.T @ W)
            np2 = float((P * P).sum())
            be = float((R.T @ W).norm())
            a = -float((D * P).sum()) / np2 if np2 else 0.0
            # scan k instead of estimating it: a fixed k misses an ablation that
            # was followed by repair finetuning, which is standard practice.
            rb, kb, curve, er, _ab = resid_scan(D, W, U, sv, Vh, kmax)
            curves.append(curve)
            per.append({"key": k, "rel": nd / nw, "top1": float(e[0]) / tot,
                        "resid_best": rb, "k_best": kb, "energy_ratio": er,
                        "eff_rank": eff, "k_used": kk,
                        "kill": float((R.T @ Wn).norm()) / be if be else None,
                        "alpha": a,
                        "resid": float((D + a * P).norm()) / nd})
            u1[k], v1[k] = U[:, 0].contiguous(), Vh[0].contiguous()
        del A
        drop(p)
    if not per:
        return None
    # keep the recovered directions: if R really is the refusal direction, then
    # INDEPENDENT authors using independent tools on the same base must converge on
    # it. That is the validation that a projection edit is the thing we care about,
    # and it needs no prompts and no behavioural testing.
    if save_dir is not None and u1:
        ks_ = sorted(u1)
        save_dir.mkdir(exist_ok=True)
        torch.save({"keys": ks_, "U": torch.stack([u1[k] for k in ks_])},
                   save_dir / f"dir_{re.sub(r'[^A-Za-z0-9]+', '_', rid)}.pt")

    def lay(k):
        m = re.search(r"layers\.(\d+)\.", k)
        return int(m.group(1)) if m else -1

    ks = sorted(u1)
    au, av, win, acr = [], [], [], []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a_, b_ = u1[ks[i]], u1[ks[j]]
            c = abs(float(a_ @ b_))
            au.append(c)
            (win if lay(ks[i]) == lay(ks[j]) else acr).append(c)
            if v1[ks[i]].numel() == v1[ks[j]].numel():
                av.append(abs(float(v1[ks[i]] @ v1[ks[j]])))
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    f = lambda n: med([p[n] for p in per if p.get(n) is not None])

    def quant(n, fr):
        """Per-component ablation strength varies, so the MEDIAN over tensors
        washes out the tensors that were most completely ablated. The low
        quantile is what actually separates: a benign finetune has no tensor
        whose leading delta subspace died."""
        v = sorted(p[n] for p in per if p.get(n) is not None)
        return v[int(fr * (len(v) - 1))] if v else None
    U = torch.stack([u1[k] for k in ks])
    s = torch.linalg.svdvals(U) ** 2
    tot = float(s.sum())
    # abliteration must touch EVERY residual-writing matrix (the projection is
    # applied everywhere by construction); defensive edits are localised to the
    # layers they operate on. RMU moved 3 of 18 where abliterations moved 18.
    return {"id": rid, "n_tensors": len(per), "n_candidates": len(Wb),
            "coverage": len(per) / len(Wb) if Wb else None,
            "rel": f("rel"), "top1": f("top1"), "eff_rank": f("eff_rank"),
            "kill": f("kill"), "alpha": f("alpha"), "resid": f("resid"),
            "resid_best": f("resid_best"), "k_best": f("k_best"),
            "energy_ratio": f("energy_ratio"),
            # median resid-vs-k curve, so any later question about the choice of k
            # is answerable from the saved results instead of a re-download
            "resid_curve": ([float(v) for v in
                             torch.stack([c[:min(len(x) for x in curves)]
                                          for c in curves]).median(0).values]
                            if curves and all(c is not None for c in curves) else None),
            "resid_best_min": quant("resid_best", 0.0),
            "kill_min": quant("kill", 0.0), "kill_p10": quant("kill", 0.1),
            "kill_p90": quant("kill", 0.9),
            "resid_min": quant("resid", 0.0), "resid_p10": quant("resid", 0.1),
            "align_U": med(au), "align_V": med(av),
            "within_L": med(win), "across_L": med(acr),
            "k_dir": float(tot ** 2 / float((s ** 2).sum())) if tot else None,
            # keep a compact per-tensor record so band-restricted statistics can be
            # computed offline. Hurtado band-averages E_1 over the mid-stack layers
            # floor(0.33L)..floor(0.67L); comparing our whole-model median against
            # his band-average is not a fair comparison, and re-downloading to fix
            # that later is not worth it.
            "per_tensor": [{"key": p["key"], "top1": p["top1"],
                            "eff_rank": p["eff_rank"], "kill": p["kill"],
                            "resid": p["resid"], "resid_best": p.get("resid_best"),
                            "k_best": p.get("k_best")} for p in per]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--derivs", required=True, help="file with one repo id per line")
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=12.0,
                    help="cap on estimated download per derivative")
    ap.add_argument("--shard-skip", type=int, default=0,
                    help="skip the first N shards; 8B mid-stack lives in shard 2")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    out_p = RESULTS / f"d6_{a.tag}.json"
    done = {}
    if out_p.exists():
        done = {r["id"]: r for r in json.loads(out_p.read_text())["rows"] if r.get("id")}
        print(f"resuming, {len(done)} already scored")

    print(f"loading base {a.base} ...")
    Wb = load_resid(a.base, a.max_shards, a.shard_skip)
    print(f"base resident: {len(Wb)} residual-writing tensors")

    rows = list(done.values())
    todo = [l.strip() for l in open(a.derivs) if l.strip() and l.strip() not in done]
    d_model = next(iter(Wb.values())).shape[0] if Wb else None
    for i, rid in enumerate(todo, 1):
        ok, why = compatible(rid, d_model, a.max_gb, a.max_shards or None)
        if not ok:
            r = {"id": rid, "error": f"screened out: {why}"}
            rows.append(r)
            out_p.write_text(json.dumps({"base": a.base, "rows": rows}, indent=1))
            print(f"[{i}/{len(todo)}] {rid}  SCREEN {why}")
            continue
        try:
            r = score(Wb, rid, save_dir=RESULTS / 'dirs')
            if r is None:
                r = {"id": rid, "error": "no comparable tensors moved"}
        except Exception as e:
            r = {"id": rid, "error": f"{type(e).__name__}: {e}"[:200]}
            traceback.print_exc(limit=1)
        rows.append(r)
        out_p.write_text(json.dumps({"base": a.base, "rows": rows}, indent=1))
        if "error" in r:
            print(f"[{i}/{len(todo)}] {rid}  SKIP {r['error'][:70]}")
        else:
            # every one of these can legitimately be None (align_V is None when
            # o_proj and down_proj have different input dims, so there are no
            # comparable right singular vectors). A None in an f-string format
            # spec raises and aborts the whole sweep from inside a log line.
            def fmt(x, nd=3):
                return "-" if x is None else f"{x:.{nd}f}"
            # log the SCANNED resid: the fixed-k value reads 0.949 on
            # mlabonne/Qwen3-0.6B-abliterated whose scanned value is 0.098, and
            # printing the former makes a correct detection look like a failure.
            rb = r.get("resid_best")
            print(f"[{i}/{len(todo)}] {rid}  effR={fmt(r.get('eff_rank'), 2)} "
                  f"kill={fmt(r.get('kill'), 4)} "
                  f"resid={fmt(rb if rb is not None else r.get('resid'))}"
                  f"{'' if rb is not None else '(fixed-k)'} "
                  f"k={r.get('k_best')} "
                  f"alU={fmt(r.get('align_U'))} winL={fmt(r.get('within_L'))}")
    print("->", out_p)


if __name__ == "__main__":
    main()
