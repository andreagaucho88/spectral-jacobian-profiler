#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral sweep runner (Algorithm 2).

For every prompt in a length-matched JSON set, compute the full per-prompt
spectral row: sigma_max, top-k, unbiased ||J||_F^2, stable rank, chi_F,
linearity check along v_lead, convergence residuals, plus output entropy
and top-1 probability. Both tracked states are exercised by default
(pre-norm final_index=-2, post-norm final_index=-1) to isolate the radial
component removed by the final RMSNorm.

Output tree:
    <out-dir>/<model_slug>/final_index_<-2|-1>/spectral_per_prompt.csv
                                              /spectral_summary_by_category.csv

DEFAULTS ARE CALIBRATED FOR A MAC M4 PRO (24 GB RAM, CPU float32).
See docs/COST_MODEL below in the code for how to scale k / m / q / n_prompts.

CLI
    python3 runner.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --prompts-json lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json \
        --out-dir results_spectral \
        --n-prompts 60 --k-top 6 --n-probes 32 --n-iter 15 \
        --final-indices -2 -1
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from common import CATEGORY_ORDER, load_prompts_json, slug_of_model
from spectral import spectral_susceptibility_for_prompt

try:
    import torch
except ModuleNotFoundError:                                       # pragma: no cover
    torch = None


# =========================================================================
# COST_MODEL  (per-prompt spectral cost on Mac M4 Pro, CPU float32,
# Qwen2.5-0.5B or SmolLM2-360M, prompt lengths T ~ 15-30):
#
#   subspace iter cost = k * n_iter forward-over-reverse JVPs
#                      + k * n_iter reverse VJPs                 (~2 grad each)
#   probe cost         = m JVPs
#   Fisher cost        = k_fisher * n_iter * (JVP + VJP)   [k_fisher=4 default]
#   linearity check    = 3 forward passes
#
# With defaults k=6, m=32, n_iter=15, final_index in {-2,-1}:
#   ~ 6*15 + 32 + 4*15 = 182 autograd calls per final_index per prompt
#   Wall-clock estimate: ~35-60 s / prompt / final_index on M4 Pro CPU.
#   For 4 categories x 60 prompts x 2 final_indices = 480 rows,
#   ballpark 5-10 h per model. Halve n_iter or n_probes to shorten.
#
# Memory: KV cache is off (we run inputs_embeds forward passes). Peak RAM
# per forward pass ~2-3 GB for 0.5B float32 at T <= 32.
# =========================================================================


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("runner.py requires torch + transformers.")


def _device_from_arg(name: str) -> str:
    _require_torch()
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        return "cuda"
    if name == "mps":
        # torch.autograd.functional.jvp double-backward is currently unstable
        # on MPS for models with RMSNorm; documented and refused here.
        raise RuntimeError(
            "MPS is not supported for the spectral runner: "
            "torch.autograd.functional.jvp double-backward is unstable on MPS "
            "for RMSNorm-based models. Use --device cpu on Apple Silicon "
            "(float32 recommended)."
        )
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unknown device: {name}")


# -------------------------------------------------------------------------
# Checkpointing helpers
# -------------------------------------------------------------------------
# The sweep is crash-resumable. Each completed prompt row is flushed to
# spectral_per_prompt.csv immediately (atomic write), a per-directory
# run_config.json fingerprints the numerical settings, and a per-model
# run_manifest.json tracks completed/error counts. On restart we resume by
# (final_index, label, prompt_idx): rows already present are skipped, so a
# killed run continues where it stopped. A file whose run_config.json does
# NOT match the current settings (e.g. a different n_iter) is set aside
# rather than appended to, so incompatible rows never mix.

_RESUME_KEYS = ("k_top", "n_probes", "n_iter", "with_fisher", "seed")


def _is_error_val(v) -> bool:
    """True iff v is a real error message. Empty-string errors round-trip
    through CSV as NaN, so treat None / '' / 'nan' as success."""
    if v is None:
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _atomic_write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _load_resume_rows(
    csv_path: Path, cfg_path: Path, config_now: dict, fi: int,
) -> Tuple[List[dict], Set[Tuple[str, int]]]:
    """Return (prior_rows, done_set). If the on-disk config is incompatible
    with config_now, the stale CSV is moved aside and ([], set()) returned."""
    if not csv_path.exists():
        return [], set()
    compatible = False
    if cfg_path.exists():
        try:
            prev = json.load(open(cfg_path))
            compatible = (all(prev.get(k) == config_now[k] for k in _RESUME_KEYS)
                          and int(prev.get("final_index", 999)) == int(fi))
        except Exception:
            compatible = False
    if not compatible:
        ts = time.strftime("%Y%m%d_%H%M%S")
        aside = csv_path.with_name(f"spectral_per_prompt.superseded_{ts}.csv")
        csv_path.rename(aside)
        print(f"[resume] fi={fi}: existing CSV config mismatch -> set aside as "
              f"{aside.name}; starting fresh", flush=True)
        return [], set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return [], set()
    done: Set[Tuple[str, int]] = set()
    if {"label", "prompt_idx"}.issubset(df.columns):
        sub = df.dropna(subset=["label", "prompt_idx"])
        done = {(str(a), int(b)) for a, b in zip(sub["label"], sub["prompt_idx"])}
    print(f"[resume] fi={fi}: found {len(done)} completed rows -> resuming",
          flush=True)
    return df.to_dict("records"), done


def _progress_from_rows(rows: List[dict]) -> Dict[str, int]:
    prog: Dict[str, int] = {}
    for r in rows:
        lab = str(r.get("label", "?"))
        prog[lab] = prog.get(lab, 0) + 1
        prog["_total"] = prog.get("_total", 0) + 1
        if _is_error_val(r.get("error")):
            prog["_errors"] = prog.get("_errors", 0) + 1
    return prog


def run_spectral_sweep(
    prompts_by_category: Dict[str, List[str]],
    tokenizer,
    model,
    n_prompts: int = 60,
    k_top: int = 6,
    n_probes: int = 32,
    n_iter: int = 15,
    final_indices: Sequence[int] = (-2, -1),
    with_fisher: bool = True,
    out_dir: str = "results_spectral",
    seed: int = 0,
    verbose: bool = True,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    hardware: Optional[str] = None,
    resume: bool = True,
) -> Dict[int, pd.DataFrame]:
    """One sweep per final_index, CRASH-RESUMABLE. Each completed prompt row is
    flushed to spectral_per_prompt.csv immediately; on restart, rows already
    present (keyed by final_index, label, prompt_idx) are skipped. Failures are
    stored as error rows and do not stop the sweep. A run_manifest.json at the
    model directory records config and completed/error counts."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    frames: Dict[int, pd.DataFrame] = {}

    config = {
        "model": model_name, "dtype": dtype, "device": device,
        "hardware": hardware or f"{platform.platform()} | {os.cpu_count()} cpus",
        "n_prompts": n_prompts, "k_top": k_top, "n_probes": n_probes,
        "n_iter": n_iter, "final_indices": list(final_indices),
        "with_fisher": bool(with_fisher), "seed": seed,
    }
    manifest_path = out / "run_manifest.json"
    progress: Dict[str, Dict[str, int]] = {}

    def flush_manifest(status: str = "running") -> None:
        _atomic_write_json({
            "config": config,
            "status": status,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "progress": progress,
        }, manifest_path)

    for fi in final_indices:
        fi_dir = out / f"final_index_{fi}"; fi_dir.mkdir(parents=True, exist_ok=True)
        csv_path = fi_dir / "spectral_per_prompt.csv"
        cfg_path = fi_dir / "run_config.json"
        fi_config = {**{k: config[k] for k in _RESUME_KEYS}, "final_index": int(fi)}

        rows: List[dict] = []
        done: Set[Tuple[str, int]] = set()
        if resume:
            rows, done = _load_resume_rows(csv_path, cfg_path, fi_config, fi)
        _atomic_write_json(fi_config, cfg_path)

        key = f"final_index_{fi}"
        progress[key] = _progress_from_rows(rows)
        flush_manifest()

        t0 = time.time()
        for label in CATEGORY_ORDER:
            prompts = prompts_by_category.get(label, [])[: n_prompts]
            for i, prompt in enumerate(prompts):
                if (label, i) in done:
                    continue
                t_p = time.time()
                try:
                    r = spectral_susceptibility_for_prompt(
                        prompt, tokenizer, model,
                        k_top=k_top, n_probes=n_probes, n_iter=n_iter,
                        final_index=fi, with_fisher=with_fisher,
                        seed=seed + i,
                    )
                    r["error"] = ""
                    ok = True
                except Exception as e:                              # pragma: no cover
                    r = {"prompt": prompt, "final_index": fi, "error": repr(e)}
                    ok = False
                r["label"] = label
                r["prompt_idx"] = i
                r["wall_s"] = time.time() - t_p
                rows.append(r)
                done.add((label, i))

                # --- flush this row immediately (atomic) ---
                _atomic_write_csv(pd.DataFrame(rows), csv_path)
                prog = progress[key]
                prog[label] = prog.get(label, 0) + 1
                prog["_total"] = prog.get("_total", 0) + 1
                if not ok:
                    prog["_errors"] = prog.get("_errors", 0) + 1
                flush_manifest()

                if verbose:
                    if not ok:
                        print(f"[spectral fi={fi}] {label} {i:04d}  "
                              f"ERROR  {r['error']}", flush=True)
                    else:
                        print(f"[spectral fi={fi}] {label} {i:04d}  "
                              f"sigma_max={r['sigma_max']:.3g}  "
                              f"r_stable={r['stable_rank']:.1f}  "
                              f"chi_F={r.get('chi_fisher_max', float('nan')):.3g}  "
                              f"conv={r['conv_residual']:.1e}  "
                              f"{r['wall_s']:.1f}s", flush=True)
                gc.collect()

        df = pd.DataFrame(rows)
        frames[fi] = df
        err_mask = df["error"].map(_is_error_val) if "error" in df else pd.Series(False, index=df.index)
        if verbose:
            print(f"[spectral fi={fi}] total wall {(time.time()-t0)/60:.1f} min "
                  f"({len(df)} rows, {int(err_mask.sum())} errors)", flush=True)

        # per-category summary (means and std) for a quick eyeball; drop error rows
        good = df[~err_mask]
        num_cols = [c for c in [
            "sigma_max", "lambda_max", "lambda_bulk_pred", "stable_rank",
            "spectral_gap", "participation_ratio_topk", "leading_over_rms",
            "conv_residual", "entropy", "top1_prob",
            "chi_fisher_max", "chi_fisher_along_hidden_lead", "lead_alignment",
        ] if c in good.columns]
        if num_cols and len(good):
            summ = good.groupby("label")[num_cols].agg(["mean", "std"]).round(5)
            summ.to_csv(fi_dir / "spectral_summary_by_category.csv")
            if verbose:
                print(f"\nSPECTRAL SUMMARY  (final_index={fi})")
                print(summ.to_string())

    flush_manifest(status="done")
    return frames


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--out-dir", default="results_spectral")
    # sweep dimensions
    ap.add_argument("--n-prompts", type=int, default=60,
                    help="max prompts per category (default 60 for M4 Pro)")
    ap.add_argument("--k-top", type=int, default=6,
                    help="block size / top-k singular values (default 6)")
    ap.add_argument("--n-probes", type=int, default=32,
                    help="sphere probes for the unbiased ||J||_F^2 (default 32)")
    ap.add_argument("--n-iter", type=int, default=15,
                    help="subspace iteration budget (fixed; no early stopping)")
    ap.add_argument("--final-indices", type=int, nargs="+", default=[-2, -1],
                    help="tracked-state indices; -2=pre-norm (legacy), -1=post-norm")
    ap.add_argument("--no-fisher", action="store_true")
    # torch/runtime
    ap.add_argument("--dtype", choices=("float32", "float16", "bfloat16"),
                    default="float32",
                    help="float32 recommended: JVP double-backward needs it "
                         "for stability on small models.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default="auto",
                    help="MPS refuses to run (see docstring). Use cpu on M-series.")
    ap.add_argument("--num-threads", type=int, default=0,
                    help="torch.set_num_threads (0 = leave default)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing checkpoint rows and recompute from scratch")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _require_torch()
    device = _device_from_arg(args.device)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
    model.eval().to(device)
    model.config.use_cache = False

    prompts = load_prompts_json(args.prompts_json)
    slug = slug_of_model(args.model)
    print(f"model             : {args.model}")
    print(f"device / dtype    : {device} / {args.dtype}")
    print("attention impl    : eager")
    print(f"prompts           : {sum(len(v) for v in prompts.values())} "
          f"({args.n_prompts} per category taken)")
    print(f"k_top / probes / iter : {args.k_top} / {args.n_probes} / {args.n_iter}")
    print(f"final_indices     : {args.final_indices}")

    run_spectral_sweep(
        prompts, tokenizer, model,
        n_prompts=args.n_prompts, k_top=args.k_top, n_probes=args.n_probes,
        n_iter=args.n_iter, final_indices=list(args.final_indices),
        with_fisher=not args.no_fisher, seed=args.seed,
        out_dir=str(Path(args.out_dir) / slug),
        model_name=args.model, dtype=args.dtype, device=device,
        hardware=f"{platform.platform()} | {os.cpu_count()} logical cpus | torch {torch.__version__}",
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    _cli()
