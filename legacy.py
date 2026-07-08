#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legacy Algorithm 1: random-direction finite-depth response.

For every prompt, propagate the initial embedding sequence H_0 and a perturbed
H_0 + epsilon * Xi (Xi random, unit Frobenius norm over the T x d sequence)
through the same model. At every block record the Euclidean separation of the
final-token states,
    delta_l = || h_l' - h_l ||_2,
and from it the summary statistics:
    lambda_L^{(eps)} = (1/L) log(delta_L / epsilon)
    g_l              = log(delta_{l+1} / delta_l)
    expansive_frac   = (1/L) |{ l : g_l > 0 }|

We also record the output entropy H(p) and top-1 probability p_max from the
unperturbed pass.

By Lemma 1 of the paper this statistic estimates a BULK Jacobian quantity
(rms singular value), which concentrates in high dimension and is nearly
prompt-invariant by construction. It is kept in the suite because:
    (i)  it reproduces the original result and grounds the direct comparison
         with the spectral estimator (Lemma 1 predicts lambda_L^{(eps)} from
         the Frobenius estimate via lambda_bulk_pred; see spectral.py);
    (ii) it exposes prompt-level output-side observables (entropy, p_max),
         which do not require any Jacobian access.

CLI
    python3 legacy.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --prompts-json lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json \
        --n-prompts 200 --n-directions 8 --epsilon 1e-3 \
        --out-dir results_legacy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import CATEGORY_ORDER, load_prompts_json, slug_of_model, template_signature

try:
    import torch
except ModuleNotFoundError:                                       # pragma: no cover
    torch = None


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("legacy.py requires torch + transformers.")


# ============================================================
# Per-prompt legacy measurement
# ============================================================

def legacy_metric_for_prompt(
    prompt: str,
    tokenizer,
    model,
    epsilon: float = 1e-3,
    n_directions: int = 8,
    apply_chat: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    """One prompt -> {label metadata, ftle_final (= lambda_L^{(eps)}),
    expansive_frac, entropy, top1_prob, token_length, n_layers}.

    Averages over n_directions random unit-Frobenius Xi. Unit of analysis is
    the prompt: the returned row is one scalar per metric per prompt.
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
    token_length = int(enc["input_ids"].shape[-1])

    with torch.no_grad():
        x0 = model.get_input_embeddings()(enc["input_ids"]).clone()
        out0 = model(
            inputs_embeds=x0, attention_mask=attn,
            output_hidden_states=True, return_dict=True,
        )
        hs0 = out0.hidden_states                    # tuple length L+1, [1, T, d]
        # tracked coordinate: final-token hidden state at each block output;
        # we drop the last tuple entry to match the earlier draft (pre-norm)
        traj0 = torch.stack([h[0, -1, :] for h in hs0[:-1]], dim=0).float()  # [K, d]
        # output-side observables
        logits0 = out0.logits[0, -1, :].float()
        p_t = torch.softmax(logits0, dim=-1)
        entropy = float(-(p_t * torch.log(p_t + 1e-20)).sum().item())
        top1 = float(p_t.max().item())

    K = traj0.shape[0]  # number of tracked states = L (pre-norm)
    lambdas = np.empty(n_directions, dtype=float)
    exp_fracs = np.empty(n_directions, dtype=float)
    rng = np.random.default_rng(seed)

    for k in range(int(n_directions)):
        xi = torch.as_tensor(
            rng.standard_normal(x0.shape).astype(np.float32),
            dtype=x0.dtype, device=device,
        )
        xi = xi / (xi.norm() + 1e-12)               # unit Frobenius
        with torch.no_grad():
            outp = model(
                inputs_embeds=x0 + epsilon * xi, attention_mask=attn,
                output_hidden_states=True, return_dict=True,
            )
            hsp = outp.hidden_states
            trajp = torch.stack([h[0, -1, :] for h in hsp[:-1]], dim=0).float()
        deltas = (trajp - traj0).norm(dim=-1).cpu().numpy()   # [K]
        deltas = np.clip(deltas, 1e-30, None)
        L_layers = max(K - 1, 1)  # blocks traversed to reach the tracked final
        lambdas[k] = float(np.log(deltas[-1] / epsilon) / L_layers)
        g = np.log(deltas[1:] / deltas[:-1])
        exp_fracs[k] = float((g > 0).mean())

    return {
        "prompt": prompt,
        "template_id": template_signature(prompt),
        "token_length": token_length,
        "n_layers": int(max(K - 1, 1)),
        "ftle_final": float(np.mean(lambdas)),      # <- the classical column name
        "ftle_final_std": float(np.std(lambdas, ddof=1)) if n_directions > 1 else 0.0,
        "expansive_frac": float(np.mean(exp_fracs)),
        "entropy": entropy,
        "top1_prob": top1,
    }


# ============================================================
# Runner
# ============================================================

def run_legacy_experiment(
    prompts_by_category: Dict[str, List[str]],
    tokenizer,
    model,
    n_prompts: int = 200,
    n_directions: int = 8,
    epsilon: float = 1e-3,
    out_dir: str = "results_legacy",
    seed: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in CATEGORY_ORDER:
        prompts = prompts_by_category.get(label, [])[: n_prompts]
        for i, prompt in enumerate(prompts):
            try:
                r = legacy_metric_for_prompt(
                    prompt, tokenizer, model,
                    epsilon=epsilon, n_directions=n_directions, seed=seed + i,
                )
            except Exception as e:                                  # pragma: no cover
                r = {"prompt": prompt, "error": repr(e)}
            r["label"] = label
            r["prompt_idx"] = i
            rows.append(r)
            if verbose:
                if "error" in r:
                    print(f"[legacy] {label} {i:04d}  ERROR  {r['error']}")
                else:
                    print(f"[legacy] {label} {i:04d}  "
                          f"ftle={r['ftle_final']:.5f}  H={r['entropy']:.3f}  "
                          f"p1={r['top1_prob']:.3f}  T={r['token_length']}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "legacy_per_prompt.csv", index=False)

    if "ftle_final" in df.columns:
        summary = (
            df.groupby("label")[["ftle_final", "expansive_frac", "entropy", "top1_prob"]]
            .agg(["mean", "std"]).round(5)
        )
        summary.to_csv(out / "legacy_summary_by_category.csv")
        if verbose:
            print("\nLEGACY SUMMARY BY CATEGORY")
            print(summary.to_string())
    if verbose:
        print(f"\nSaved under: {out.resolve()}")
    return df


# ============================================================
# CLI
# ============================================================

def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--out-dir", default="results_legacy")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--n-directions", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=1e-3)
    ap.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                    default="float32")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default="auto",
                    help="'mps' is NOT recommended: torch.autograd.functional "
                         "double-backward is unstable on MPS for RMSNorm-based "
                         "models. Legacy Algorithm 1 does not use it, so mps "
                         "works here, but keep cpu on Apple Silicon for "
                         "consistency with the spectral runner.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "cpu")
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(device)

    prompts = load_prompts_json(args.prompts_json)
    slug = slug_of_model(args.model)
    run_legacy_experiment(
        prompts, tokenizer, model,
        n_prompts=args.n_prompts, n_directions=args.n_directions,
        epsilon=args.epsilon, seed=args.seed,
        out_dir=str(Path(args.out_dir) / slug),
    )


if __name__ == "__main__":
    _cli()
