"""The mechanism statistic, scanned over subspace dimension.

resid(k) = ||D + a* P_k|| / ||D||   with P_k = U_k U_k^T W,  a* = -<D,P_k>/||P_k||^2

Fixing k is the problem. Underestimate and R misses a direction; overestimate and R
drags in directions of W that are not in D. Both inflate resid, and a real ablation
that was followed by repair finetuning ("ablate then heal", which is standard
practice) is missed entirely by a fixed k. So scan k and take the minimum.

The scan is nearly free. Writing a_i = u_i^T W (so P_k = sum_{i<k} u_i a_i) and using
u_i^T D = s_i v_i^T:
    ||P_k||^2   = sum_{i<k} ||a_i||^2            cumulative
    <D, P_k>    = sum_{i<k} s_i (v_i . a_i)      cumulative
and substituting the optimal a* collapses the whole thing to
    resid(k) = sqrt(1 - c_k^2),   c_k = <D,P_k> / (||P_k|| ||D||)
so the curve costs one U^T W matmul plus O(kmax) scalars, not kmax matmuls.
"""

import torch


def resid_scan(D, W, U, s, Vh, kmax=64):
    """Return (resid_best, k_best, resid_curve, energy_ratio, alpha).

    D = W_new - W with its SVD given. alpha is the fitted strength at k_best,
    -<D,P>/||P||^2: an abliteration W - a R R^T W reads alpha = a in (0, 1], while
    ADDING a projection of W reads alpha < 0 at the same resid of 0, so the sign
    separates removal from amplification.
    """
    kmax = min(kmax, U.shape[1])
    A = U[:, :kmax].T @ W                      # a_i = u_i^T W, one matmul
    pn2 = torch.cumsum((A * A).sum(1), 0)      # ||P_k||^2
    dot = torch.cumsum(s[:kmax] * (Vh[:kmax] * A).sum(1), 0)   # <D, P_k>
    dn = float(D.norm())
    if dn == 0:
        return None, None, None
    c = dot / (pn2.clamp_min(1e-30).sqrt() * dn)
    r = (1.0 - c.clamp(-1, 1) ** 2).clamp_min(0.0).sqrt()
    k = int(torch.argmin(r).item())

    # SVD low-rank COMPRESSION has exactly the same algebra: W_new = U_k S_k V_k^T
    # means Delta = -P W with P the BOTTOM singular subspace, so it reads resid 0
    # too. It is distinguishable by WHERE the removed subspace sits in W's spectrum:
    # compression removes the lowest-energy directions, ablation removes a direction
    # of typical energy. energy_ratio is ||R^T W||^2 per direction, relative to what
    # a random subspace of the same size would carry -- ~0.003 for compression,
    # ~1.0 for ablation. It needs no SVD of W: ||R^T W||^2 is already in the scan.
    d = W.shape[0]
    wn2 = float((W * W).sum())
    energy_ratio = (float(pn2[k]) / (k + 1)) / (wn2 / d) if wn2 else None
    alpha = -float(dot[k]) / float(pn2[k]) if float(pn2[k]) > 0 else None
    return float(r[k]), k + 1, r, energy_ratio, alpha
