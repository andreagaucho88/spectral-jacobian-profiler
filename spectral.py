#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral finite-depth susceptibility (Algorithm 2 of the paper).

WHAT IS COMPUTED
----------------
Per prompt, we characterize the Jacobian J = d h_{L,T} / d H_0 (shape d x n,
n = T*d) of the input-to-final-token map without ever forming it.

    (1) LEADING amplification.  Block subspace iteration on the SPD operator
        A = J^T J via exact JVP + VJP, with Rayleigh-Ritz extraction.
        Top-k singular values sigma_1 >= ... >= sigma_k of J, plus the leading
        right singular vector v_lead in input shape.

    (2) BULK amplification.  Unbiased Monte-Carlo estimate of ||J||_F^2 from
        sphere probes:
             E ||J xi||_2^2  =  ||J||_F^2 / n     for xi ~ Unif(S^{n-1}).
        Estimator: n * mean_i ||J xi_i||_2^2   (mean of the SQUARES; the
        naive n * (mean_i ||J xi_i||)^2 is biased LOW by Jensen's inequality).

    (3) ANISOTROPY.  Stable rank r_stable = ||J||_F^2 / sigma_max^2.
        Note: with these definitions the "anisotropy ratio"
              a = sigma_max / (||J||_F / sqrt(n))
        satisfies a^2 = n / r_stable identically; we retain it only as a
        floating-point consistency check.

    (4) OUTPUT-SIDE / COORDINATE-FREE.  Fisher susceptibility on the softmax
        family in logit coordinates F = diag(p) - p p^T:
              KL(p_{z+dz} || p_z) = (1/2) dz^T F dz + O(|dz|^3),
        with the 1/2 factor part of the definition. Leading value
              chi_F^max = (1/2) lambda_max( J_z^T F J_z )
        computed by the same block scheme on the SPD operator
              v -> J_z^T [ F ( J_z v ) ].

    (5) DIAGNOSTICS.  Per-prompt convergence residual of the subspace iteration
        (independence from the category label must be checked before any
        spectral contrast is interpreted); linearity table along v_lead over
        an epsilon grid; consistency check between the spectral and legacy
        (random-direction) estimators via
              lambda_bulk_pred = (1/L) log( sqrt( <||J||_F^2> / n ) ).

DESIGN NOTES (choices that matter for the paper)
------------------------------------------------
- Block subspace iteration, not scalar power iteration. Scalar convergence is
  governed by the gap sigma_2/sigma_1; under the spectral hypothesis the gap
  is itself category-dependent, so a scalar scheme would systematically
  underestimate sigma_max on the classes it is meant to characterize.
- Fixed iteration budget; NO early stopping. A category-dependent stopping
  rule would be a confound analogous to prompt length. The last-sweep Ritz
  residual is stored per prompt so its independence from category can be
  tested.
- Separate RNG streams (derived from a per-prompt seed) for the subspace
  block init, the sphere probes, and the Fisher block init.
- Two tracked states: pre-norm (matches the legacy hidden_states[:-1]
  slicing; final_index=-2) and post-norm (the true residual read by the
  logits; final_index=-1). Their difference isolates the radial component
  removed by the final RMSNorm, to which the logits are blind.

SELF-TEST
---------
    python3 spectral.py            # numpy self-test only (no torch needed)

The runner (runner.py) is what invokes this on a real model per prompt.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:                                       # pragma: no cover
    torch = None


EPS = 1e-12


# ============================================================
# Numerical helpers
# ============================================================

def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v.ravel()) + EPS)


def random_unit_direction(
    shape: Tuple[int, ...], rng: np.random.Generator,
) -> np.ndarray:
    return _unit(rng.standard_normal(shape))


# ============================================================
# (1) Leading and top-k via block subspace iteration
# ============================================================

def psd_block_eigs(
    op: Callable[[np.ndarray], np.ndarray],
    n_dim: int,
    k: int = 8,
    n_iter: int = 20,
    seed: int = 0,
) -> Dict[str, object]:
    """Top-k eigenpairs of a symmetric PSD operator on R^{n_dim} by block
    subspace iteration with Rayleigh--Ritz extraction.

    op: flat vector of length n_dim -> flat vector of length n_dim.
    Fixed iteration budget (no early stopping; see module docstring).
    Returns eigvals (descending, PSD-clipped), eigvecs (n_dim x k Ritz vectors
    of the LAST extraction), the relative change of the Ritz values over the
    last sweep (conv_residual), and n_iter actually run.
    """
    k = int(min(k, n_dim))
    rng = np.random.default_rng(seed)
    V, _ = np.linalg.qr(rng.standard_normal((n_dim, k)))
    ritz_prev: Optional[np.ndarray] = None
    ritz = np.zeros(k)
    V_ritz = V
    conv_residual = np.inf
    for _ in range(int(n_iter)):
        W = np.column_stack([op(V[:, j]) for j in range(k)])   # A V
        A_small = V.T @ W                                       # Rayleigh-Ritz
        A_small = 0.5 * (A_small + A_small.T)
        evals, evecs = np.linalg.eigh(A_small)
        order = np.argsort(evals)[::-1]
        ritz = np.clip(evals[order], 0.0, None)
        rot = evecs[:, order]
        V_ritz = V @ rot
        W_rot = W @ rot
        if ritz_prev is not None:
            conv_residual = float(
                np.max(np.abs(ritz - ritz_prev) / (np.abs(ritz_prev) + EPS))
            )
        ritz_prev = ritz.copy()
        V, _ = np.linalg.qr(W_rot)
    return {
        "eigvals": ritz,
        "eigvecs": V_ritz,
        "conv_residual": float(conv_residual),
        "n_iter": int(n_iter),
    }


def jacobian_topk_singular(
    jvp_final: Callable[[np.ndarray], np.ndarray],
    vjp_final: Callable[[np.ndarray], np.ndarray],
    in_shape: Tuple[int, ...],
    k: int = 8,
    n_iter: int = 20,
    seed: int = 0,
) -> Dict[str, object]:
    """Top-k singular values of J via block iteration on the SPD operator J^T J.
    sigma_i = sqrt(eig_i(J^T J)); leading right singular vector is returned in
    the input shape."""
    n_dim = int(np.prod(in_shape))

    def op(v_flat: np.ndarray) -> np.ndarray:
        jv = jvp_final(v_flat.reshape(in_shape))
        return vjp_final(jv).ravel()

    res = psd_block_eigs(op, n_dim, k=k, n_iter=n_iter, seed=seed)
    sigmas = np.sqrt(np.clip(res["eigvals"], 0.0, None))
    v_lead = _unit(res["eigvecs"][:, 0]).reshape(in_shape)
    return {
        "sigmas": sigmas,
        "v_lead": v_lead,
        "conv_residual": res["conv_residual"],
        "n_iter": res["n_iter"],
    }


# ============================================================
# (2) Bulk: unbiased sphere estimator of ||J||_F^2
# ============================================================

def random_probe_stats(
    jvp_final: Callable[[np.ndarray], np.ndarray],
    in_shape: Tuple[int, ...],
    n_dirs: int = 32,
    seed: int = 0,
) -> Dict[str, float]:
    """Distribution of ||J xi||_2 over xi ~ Unif(S^{n-1}), and the mean of
    the SQUARES (which is what the Frobenius estimator needs)."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_dirs, dtype=float)
    for i in range(int(n_dirs)):
        vals[i] = np.linalg.norm(
            jvp_final(random_unit_direction(in_shape, rng)).ravel()
        )
    v2 = vals ** 2
    return {
        "rand_mean": float(vals.mean()),
        "rand_mean_sq": float(v2.mean()),   # <- Frobenius uses THIS
        "rand_rms": float(np.sqrt(v2.mean())),
        "rand_p50": float(np.quantile(vals, 0.50)),
        "rand_p90": float(np.quantile(vals, 0.90)),
        "rand_p99": float(np.quantile(vals, 0.99)),
        "rand_max": float(vals.max()),
        "n_probes": int(n_dirs),
    }


def frobenius_sq_from_probes(rand_mean_sq: float, n_dim: int) -> float:
    """||J||_F^2 = n * E ||J xi||_2^2 for xi uniform on the sphere."""
    return float(n_dim) * float(rand_mean_sq)


# ============================================================
# (3) Spectrum summaries
# ============================================================

def spectrum_summaries(
    sigmas: np.ndarray, frob_sq: float,
) -> Dict[str, float]:
    """Scalar summaries. stable_rank uses the UNBIASED Frobenius estimate, not
    the truncated top-k sum."""
    s = np.asarray(sigmas, dtype=float)
    s2 = s ** 2
    sigma_max = float(s[0])
    return {
        "sigma_max": sigma_max,
        "sigma_2": float(s[1]) if s.size > 1 else float("nan"),
        "spectral_gap": (float(s[1] / (s[0] + EPS))
                        if s.size > 1 else float("nan")),
        "stable_rank": float(frob_sq / (sigma_max ** 2 + EPS)),
        "participation_ratio_topk":
            float((s2.sum() ** 2) / ((s2 ** 2).sum() + EPS)),
        "topk_mass_fraction": float(s2.sum() / (frob_sq + EPS)),
    }


# ============================================================
# (4) Fisher / KL susceptibility
# ============================================================

def fisher_apply(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    """F u where F = diag(p) - p p^T."""
    return p * u - p * float(np.dot(p, u))


def fisher_quadratic_form(dz: np.ndarray, p: np.ndarray) -> float:
    """1/2 dz^T F dz = second-order KL(p_{z+dz} || p_z). The 1/2 factor is
    part of the definition."""
    m = float(np.dot(p, dz))
    return 0.5 * float(np.dot(p, dz ** 2) - m ** 2)


def kl_from_logits(z_new: np.ndarray, z_old: np.ndarray) -> float:
    """KL(p_new || p_old) via log-sum-exp. Numerically stable at small
    logit differences (probability-space evaluation cancels)."""
    def _lse(z: np.ndarray) -> float:
        m = float(z.max())
        return m + float(np.log(np.exp(z - m).sum()))
    p_new = np.exp(z_new - _lse(z_new))
    return float(np.dot(p_new, z_new - z_old) - (_lse(z_new) - _lse(z_old)))


def fisher_susceptibility_along(
    v: np.ndarray,
    jvp_logits: Callable[[np.ndarray], np.ndarray],
    p: np.ndarray,
) -> float:
    """chi_F(v) = (1/2) (J_z v)^T F (J_z v) for a unit input direction v."""
    dz = jvp_logits(_unit(v))
    return fisher_quadratic_form(dz, p)


def fisher_leading_susceptibility(
    jvp_logits: Callable[[np.ndarray], np.ndarray],
    vjp_logits: Callable[[np.ndarray], np.ndarray],
    p: np.ndarray,
    in_shape: Tuple[int, ...],
    k: int = 4,
    n_iter: int = 20,
    seed: int = 0,
) -> Dict[str, object]:
    """Leading Fisher susceptibility chi_F^max = (1/2) lambda_max(J_z^T F J_z)
    and its direction, via block iteration on the SPD operator
    v -> J_z^T F J_z v."""
    n_dim = int(np.prod(in_shape))

    def op(v_flat: np.ndarray) -> np.ndarray:
        dz = jvp_logits(v_flat.reshape(in_shape))
        return vjp_logits(fisher_apply(dz, p)).ravel()

    res = psd_block_eigs(op, n_dim, k=k, n_iter=n_iter, seed=seed)
    lam = np.clip(res["eigvals"], 0.0, None)
    return {
        "chi_fisher_max": float(0.5 * lam[0]),
        "chi_fisher_2": float(0.5 * lam[1]) if lam.size > 1 else float("nan"),
        "v_lead_fisher": _unit(res["eigvecs"][:, 0]).reshape(in_shape),
        "fisher_conv_residual": res["conv_residual"],
    }


# ============================================================
# (5) Linearity check along the leading direction
# ============================================================

def linearity_check(
    forward_final: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    v_lead: np.ndarray,
    sigma_max: float,
    eps_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
) -> Dict[str, float]:
    """Finite-difference growth along v_lead over eps_grid, against sigma_max.
    Agreement at small eps validates the estimator on the real model
    (sigma_max has no dense ground truth there) AND shows that the legacy
    protocol measures ||J v|| for its chosen v (the bulk value when v is
    random, by Lemma 1 of the paper)."""
    base = forward_final(x0)
    out: Dict[str, float] = {}
    v = _unit(v_lead)
    for eps in eps_grid:
        pert = forward_final(x0 + eps * v)
        growth = float(np.linalg.norm(pert - base) / eps)
        tag = f"{eps:.0e}".replace("-0", "-")
        out[f"fd_lead_growth_eps_{tag}"] = growth
        out[f"fd_lead_relerr_eps_{tag}"] = (
            abs(growth - sigma_max) / (sigma_max + EPS))
    return out


# ============================================================
# Torch / HuggingFace adapter
# ============================================================

def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError(
            "torch is required for the model adapter; the numpy core and the "
            "self-test in __main__ run without it."
        )


def torch_build_callables(
    prompt: str,
    tokenizer,
    model,
    final_index: int = -2,
    apply_chat: bool = True,
) -> Dict[str, object]:
    """Build the JVP/VJP oracles for one prompt.

    final_index = -2 tracks hidden_states[final_index] = the state BEFORE the
    final norm (matches the legacy protocol's hidden_states[:-1] slice).
    final_index = -1 tracks the state AFTER the final norm (the actual input
    to the logit head; the final norm removes the radial component, to which
    the logits are blind).

    Returns dict with forward_final, jvp_final, vjp_final, jvp_logits,
    vjp_logits, x0 (numpy), in_shape, depth (blocks traversed to reach the
    tracked state), probs, entropy, top1_prob.
    """
    _require_torch()
    device = next(model.parameters()).device
    text = prompt
    if apply_chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    attn = enc["attention_mask"]

    with torch.no_grad():
        x0_t = model.get_input_embeddings()(enc["input_ids"]).clone()
    in_shape = tuple(x0_t.shape)
    dtype = x0_t.dtype

    def _final_tensor(embeds: "torch.Tensor") -> "torch.Tensor":
        out = model(
            inputs_embeds=embeds, attention_mask=attn,
            output_hidden_states=True, return_dict=True,
        )
        return out.hidden_states[final_index][0, -1, :]  # [d]

    def _logits_tensor(embeds: "torch.Tensor") -> "torch.Tensor":
        out = model(inputs_embeds=embeds, attention_mask=attn, return_dict=True)
        return out.logits[0, -1, :]  # [|V|]

    with torch.no_grad():
        out0 = model(
            inputs_embeds=x0_t, attention_mask=attn,
            output_hidden_states=True, return_dict=True,
        )
        n_states = len(out0.hidden_states)  # L+1 (embed + L blocks)
        depth = n_states + final_index if final_index < 0 else final_index
        depth = max(int(depth), 1)
        logits0 = out0.logits[0, -1, :].float()
        p_t = torch.softmax(logits0, dim=-1)
        entropy = float(-(p_t * torch.log(p_t + 1e-20)).sum().item())
        top1 = float(p_t.max().item())
        probs = p_t.cpu().numpy().astype(np.float64)

    def forward_final(x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x, dtype=dtype, device=device)
        with torch.no_grad():
            return _final_tensor(xt).detach().float().cpu().numpy()

    def jvp_final(v: np.ndarray) -> np.ndarray:
        vt = torch.as_tensor(v, dtype=dtype, device=device)
        _, jv = torch.autograd.functional.jvp(
            _final_tensor, x0_t, vt, strict=False)
        return jv.detach().float().cpu().numpy()

    def vjp_final(u: np.ndarray) -> np.ndarray:
        ut = torch.as_tensor(u, dtype=dtype, device=device)
        _, ju = torch.autograd.functional.vjp(
            _final_tensor, x0_t, ut, strict=False)
        return ju.detach().float().cpu().numpy()

    def jvp_logits(v: np.ndarray) -> np.ndarray:
        vt = torch.as_tensor(v, dtype=dtype, device=device)
        _, jv = torch.autograd.functional.jvp(
            _logits_tensor, x0_t, vt, strict=False)
        return jv.detach().float().cpu().numpy().astype(np.float64)

    def vjp_logits(u: np.ndarray) -> np.ndarray:
        ut = torch.as_tensor(u, dtype=dtype, device=device)
        _, ju = torch.autograd.functional.vjp(
            _logits_tensor, x0_t, ut, strict=False)
        return ju.detach().float().cpu().numpy()

    return {
        "forward_final": forward_final,
        "jvp_final": jvp_final,
        "vjp_final": vjp_final,
        "jvp_logits": jvp_logits,
        "vjp_logits": vjp_logits,
        "x0": x0_t.detach().float().cpu().numpy(),
        "in_shape": in_shape,
        "depth": depth,
        "probs": probs,
        "entropy": entropy,
        "top1_prob": top1,
    }


# ============================================================
# Per-prompt observable bundle (Algorithm 2)
# ============================================================

def spectral_susceptibility_for_prompt(
    prompt: str,
    tokenizer,
    model,
    k_top: int = 6,
    n_probes: int = 32,
    n_iter: int = 15,
    final_index: int = -2,
    with_fisher: bool = True,
    eps_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
    seed: int = 0,
) -> Dict[str, float]:
    """One prompt -> one row of spectral observables (all keys always present,
    for a clean DataFrame). Distinct RNG streams for subspace / probes / Fisher.
    """
    seed_sub, seed_probe, seed_fisher = 1000 * seed + 1, 1000 * seed + 2, 1000 * seed + 3

    cb = torch_build_callables(prompt, tokenizer, model, final_index=final_index)
    in_shape, depth = cb["in_shape"], cb["depth"]
    n_dim = int(np.prod(in_shape))

    top = jacobian_topk_singular(
        cb["jvp_final"], cb["vjp_final"], in_shape,
        k=k_top, n_iter=n_iter, seed=seed_sub,
    )
    sigmas = top["sigmas"]
    sigma_max = float(sigmas[0])

    probes = random_probe_stats(
        cb["jvp_final"], in_shape, n_dirs=n_probes, seed=seed_probe,
    )
    frob_sq = frobenius_sq_from_probes(probes["rand_mean_sq"], n_dim)

    row: Dict[str, float] = {
        "prompt": prompt,
        "n_dim": n_dim,
        "depth": depth,
        "final_index": final_index,
        "sigma_max": sigma_max,
        "sigma_2": float(sigmas[1]) if sigmas.size > 1 else float("nan"),
        **{k: v for k, v in spectrum_summaries(sigmas, frob_sq).items()
           if k not in ("sigma_max", "sigma_2")},
        "frobenius_sq_est": frob_sq,
        "lambda_max": float(np.log(sigma_max + EPS) / depth),
        # bulk prediction of the legacy lambda_L^{(eps)}:
        # legacy ~ (1/L) log(||J||_F / sqrt(n))  by Lemma 1
        "lambda_bulk_pred": float(
            0.5 * np.log(frob_sq / n_dim + EPS) / depth
        ),
        "conv_residual": float(top["conv_residual"]),
        "n_iter_subspace": int(top["n_iter"]),
        **{k: v for k, v in probes.items()},
        "leading_over_rms": float(sigma_max / (probes["rand_rms"] + EPS)),
        "entropy": float(cb["entropy"]),
        "top1_prob": float(cb["top1_prob"]),
    }
    # identity check: a^2 = n / stable_rank must hold to MC accuracy
    a2 = (sigma_max / (probes["rand_rms"] + EPS)) ** 2
    row["anisotropy_identity_relerr"] = float(
        abs(a2 - n_dim / row["stable_rank"]) / (a2 + EPS)
    )

    row.update(linearity_check(
        cb["forward_final"], cb["x0"], top["v_lead"], sigma_max, eps_grid,
    ))

    if with_fisher:
        fl = fisher_leading_susceptibility(
            cb["jvp_logits"], cb["vjp_logits"], cb["probs"], in_shape,
            k=min(4, k_top), n_iter=n_iter, seed=seed_fisher,
        )
        row["chi_fisher_max"] = fl["chi_fisher_max"]
        row["chi_fisher_2"] = fl["chi_fisher_2"]
        row["fisher_conv_residual"] = float(fl["fisher_conv_residual"])
        row["chi_fisher_along_hidden_lead"] = fisher_susceptibility_along(
            top["v_lead"], cb["jvp_logits"], cb["probs"],
        )
        row["lead_alignment"] = float(abs(np.dot(
            _unit(top["v_lead"]).ravel(),
            _unit(fl["v_lead_fisher"]).ravel(),
        )))
    else:
        row.update({
            "chi_fisher_max": float("nan"),
            "chi_fisher_2": float("nan"),
            "fisher_conv_residual": float("nan"),
            "chi_fisher_along_hidden_lead": float("nan"),
            "lead_alignment": float("nan"),
        })
    return row


# ============================================================
# Numpy self-test (dense ground truth; no torch)
# ============================================================

def _self_test(seed: int = 0) -> bool:
    print("=" * 72)
    print("SPECTRAL CORE SELF-TEST (numpy dense ground truth)")
    print("=" * 72)
    rng = np.random.default_rng(seed)
    ok_all = True

    # anisotropic test matrix: planted spectrum
    n, d, k = 400, 48, 8
    U, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Vt, _ = np.linalg.qr(rng.standard_normal((n, d)))
    planted = np.sort(np.concatenate(
        [[12.0, 5.0, 3.0], rng.uniform(0.1, 1.0, d - 3)]
    ))[::-1]
    J = U @ np.diag(planted) @ Vt.T
    jvp = lambda v: J @ np.asarray(v).ravel()
    vjp = lambda u: J.T @ np.asarray(u).ravel()
    svd = np.linalg.svd(J, compute_uv=False)

    # (1) block iteration recovers well-separated head to machine precision;
    #     the near-degenerate tail (uniform in [0.1, 1.0]) may need more iter
    top = jacobian_topk_singular(jvp, vjp, (n,), k=k, n_iter=60, seed=seed)
    rel = np.abs(top["sigmas"] - svd[:k]) / svd[:k]
    ok = rel[:3].max() < 1e-8 and rel.max() < 1e-4
    ok_all &= ok
    print(f"\n(1) top-{k} sigma via block iter: head(3) rel err {rel[:3].max():.2e}, "
          f"tail rel err {rel.max():.2e}   [{'PASS' if ok else 'FAIL'}]")
    print(f"    last-sweep conv_residual = {top['conv_residual']:.2e}")

    # (2) Frobenius sphere estimator: unbiased
    probes = random_probe_stats(jvp, (n,), n_dirs=4000, seed=seed + 1)
    frob_sq_hat = frobenius_sq_from_probes(probes["rand_mean_sq"], n)
    frob_sq_true = float((svd ** 2).sum())
    rel_f = abs(frob_sq_hat - frob_sq_true) / frob_sq_true
    frob_sq_jensen = n * probes["rand_mean"] ** 2  # WRONG estimator
    bias_jensen = frob_sq_jensen / frob_sq_true - 1
    ok = rel_f < 0.05
    ok_all &= ok
    print(f"\n(2) ||J||_F^2 sphere estimator: true={frob_sq_true:.2f}  "
          f"est={frob_sq_hat:.2f}  rel err {rel_f:.2%}   "
          f"[{'PASS' if ok else 'FAIL'}]")
    print(f"    Jensen-biased variant (mean-of-norms)^2 = {frob_sq_jensen:.2f}  "
          f"({bias_jensen:+.1%}; always low)")

    # (3) stable rank + identity check
    sm = spectrum_summaries(top["sigmas"], frob_sq_hat)
    sr_true = frob_sq_true / svd[0] ** 2
    a2 = (sm["sigma_max"] / probes["rand_rms"]) ** 2
    id_rel = abs(a2 - n / sm["stable_rank"]) / a2
    ok = abs(sm["stable_rank"] - sr_true) / sr_true < 0.06 and id_rel < 1e-10
    ok_all &= ok
    print(f"\n(3) stable rank: true={sr_true:.3f}  est={sm['stable_rank']:.3f}   "
          f"identity a^2 = n/r_stable rel err {id_rel:.1e}   "
          f"[{'PASS' if ok else 'FAIL'}]")

    # (4) empirical concentration: random probes underestimate sigma_max
    print(f"\n(4) concentration: sigma_max={svd[0]:.3f}  rand rms={probes['rand_rms']:.3f}  "
          f"p99={probes['rand_p99']:.3f}  max/{probes['n_probes']}={probes['rand_max']:.3f}  "
          f"lead/rms={svd[0]/probes['rand_rms']:.1f}x")

    # (5) Fisher quadratic form vs stable KL from logits
    V_vocab = 200
    z = rng.standard_normal(V_vocab)
    p = np.exp(z - z.max()); p /= p.sum()
    dz_dir = rng.standard_normal(V_vocab)
    for eps in (1e-3, 1e-4):
        dz = eps * dz_dir
        kl = kl_from_logits(z + dz, z)
        quad = fisher_quadratic_form(dz, p)
        rel = abs(kl - quad) / (kl + EPS)
        if eps == 1e-4:
            ok = rel < 1e-3
            ok_all &= ok
            print(f"\n(5) 1/2 dz^T F dz vs stable KL at eps=1e-4: rel err {rel:.2e}   "
                  f"[{'PASS' if ok else 'FAIL'}]  (dropping the 1/2 breaks this)")

    # (6) leading Fisher via block iter vs dense eig
    Jz = rng.standard_normal((V_vocab, n)) / np.sqrt(n)
    jvp_z = lambda v: Jz @ np.asarray(v).ravel()
    vjp_z = lambda u: Jz.T @ np.asarray(u).ravel()
    fl = fisher_leading_susceptibility(
        jvp_z, vjp_z, p, (n,), k=4, n_iter=80, seed=seed + 2,
    )
    F = np.diag(p) - np.outer(p, p)
    M = 0.5 * (Jz.T @ F @ Jz + (Jz.T @ F @ Jz).T)
    lam_true = float(np.linalg.eigvalsh(M)[-1])
    rel_chi = abs(fl["chi_fisher_max"] - 0.5 * lam_true) / (0.5 * lam_true)
    ok = rel_chi < 1e-5
    ok_all &= ok
    print(f"\n(6) chi_F^max: block iter={fl['chi_fisher_max']:.6f}  "
          f"dense={0.5*lam_true:.6f}  rel err {rel_chi:.2e}   "
          f"[{'PASS' if ok else 'FAIL'}]")

    # (7) linearity check on a mildly nonlinear map
    W1 = rng.standard_normal((d, n)) / np.sqrt(n)
    fwd = lambda x: np.tanh(W1 @ np.asarray(x).ravel())
    x0 = rng.standard_normal(n)
    D = np.diag(1.0 - np.tanh(W1 @ x0) ** 2)
    Jx = D @ W1
    jvp_x = lambda v: Jx @ np.asarray(v).ravel()
    vjp_x = lambda u: Jx.T @ np.asarray(u).ravel()
    topx = jacobian_topk_singular(
        jvp_x, vjp_x, (n,), k=4, n_iter=60, seed=seed + 3,
    )
    lin = linearity_check(
        fwd, x0, topx["v_lead"], float(topx["sigmas"][0]),
        eps_grid=(1e-2, 1e-4, 1e-6),
    )
    ok = lin["fd_lead_relerr_eps_1e-6"] < 1e-4
    ok_all &= ok
    print(f"\n(7) linearity along v_lead: rel err vs sigma_max "
          f"{lin['fd_lead_relerr_eps_1e-2']:.2e} (eps=1e-2), "
          f"{lin['fd_lead_relerr_eps_1e-6']:.2e} (eps=1e-6)   "
          f"[{'PASS' if ok else 'FAIL'}]")

    print("\nSELF-TEST:", "ALL PASS" if ok_all else "FAIL")
    return ok_all


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
