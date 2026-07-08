#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint statistical analysis of the legacy and spectral sweeps.

Inputs (any subset)
    --legacy-csv    legacy_per_prompt.csv     from legacy.py
    --spectral-csv  spectral_per_prompt.csv   from runner.py

The frames are already at prompt granularity (one row per prompt), which is
the unit of analysis: no per-direction, per-layer, or per-token averaging
is treated as an independent observation.

Six analyses are produced (each writes one CSV under --out-dir):

    (A) length_by_category.csv
        Length distribution per category. Verifies the length-matched
        construction gave identical histograms.

    (B) length_metric_correlation.csv
        Pearson r of every metric vs token_length, overall and per category.
        A large residual correlation is a red flag that the "length matched"
        pool is not actually matched.

    (C) ols_length_adjusted_contrasts.csv           (needs statsmodels)
        For each metric: OLS raw vs OLS length-adjusted, cluster-robust SEs
        clustered on template_id, and the shrinkage of every category
        contrast when length is partialled out (should be ~0 on matched sets).

    (D) pairwise_effects__<metric>.csv
        For every metric passed via --metrics: all six category pairs with
        mean_a, mean_b, diff, Cohen's d, Cliff's delta, permutation p, and
        Holm-adjusted p (family = six pairs of this metric).

    (E) bulk_consistency.csv                         (needs both --legacy-csv
                                                     and --spectral-csv)
        Merges the two frames on prompt and compares the legacy scalar
        ftle_final with lambda_bulk_pred = (1/L) log( sqrt( <||J||_F^2> / n )).
        Lemma 1 predicts equality up to O(n^{-1/2}) concentration. Reports
        median relative error, Pearson r, and per-category means.

    (F) joint_axes.csv                                (needs --spectral-csv)
        Per-category means of the two axes -- output uncertainty
        (entropy, p_max) and hidden-state response (sigma_max, stable rank,
        chi_F, and the bulk-equivalent lambda_bulk_pred) -- and the
        contrast against a reference category. This is the table that
        exhibits the factual "inversion" from the paper.

CLI
    python3 analysis.py \
        --legacy-csv    results_legacy/<model_slug>/legacy_per_prompt.csv \
        --spectral-csv  results_spectral/<model_slug>/final_index_-2/spectral_per_prompt.csv \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --out-dir results_analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (
    CATEGORY_ORDER, DEFAULT_REFERENCE, make_length_fn,
    pairwise_report, slug_of_model, template_signature,
)

try:
    import statsmodels.formula.api as smf
except ModuleNotFoundError:                                       # pragma: no cover
    smf = None


DEFAULT_LEGACY_METRICS = ["ftle_final", "expansive_frac", "entropy", "top1_prob"]
DEFAULT_SPECTRAL_METRICS = [
    "sigma_max", "lambda_max", "lambda_bulk_pred", "stable_rank",
    "spectral_gap", "chi_fisher_max", "chi_fisher_along_hidden_lead",
    "lead_alignment", "entropy", "top1_prob",
]


# ============================================================
# Annotation helpers
# ============================================================

def _annotate(
    df: pd.DataFrame, model_name: Optional[str], apply_chat: bool = True,
) -> pd.DataFrame:
    """Ensure token_length and template_id exist; add them if not."""
    df = df.copy()
    if "template_id" not in df.columns:
        df["template_id"] = df["prompt"].astype(str).map(template_signature)
    if "token_length" not in df.columns:
        length_fn, kind = make_length_fn(model_name, apply_chat=apply_chat)
        df["token_length"] = df["prompt"].astype(str).map(length_fn)
        df.attrs["length_kind"] = kind
    else:
        df.attrs["length_kind"] = "provided"
    return df


def _existing_metrics(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    return [m for m in wanted if m in df.columns]


# ============================================================
# (A) length distribution
# ============================================================

def length_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("label")["token_length"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reindex(CATEGORY_ORDER).reset_index()
    )


# ============================================================
# (B) length-metric correlation
# ============================================================

def length_metric_correlation(
    df: pd.DataFrame, metrics: List[str],
) -> pd.DataFrame:
    def _corr(a: pd.Series, b: pd.Series) -> float:
        a = a.dropna().to_numpy(); b = b.dropna().to_numpy()
        if len(a) != len(b) or len(a) < 3:
            return float("nan")
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(np.corrcoef(a, b)[0, 1])

    rows = []
    for metric in metrics:
        overall = df[[metric, "token_length"]].dropna()
        rows.append({"metric": metric, "scope": "overall",
                     "n": int(len(overall)),
                     "pearson_r": _corr(overall[metric], overall["token_length"])})
        for label in CATEGORY_ORDER:
            sub = df[df["label"] == label][[metric, "token_length"]].dropna()
            rows.append({"metric": metric, "scope": label,
                         "n": int(len(sub)),
                         "pearson_r": _corr(sub[metric], sub["token_length"])})
    return pd.DataFrame(rows)


# ============================================================
# (C) OLS length adjustment, cluster-robust
# ============================================================

def _coef_table(fit, ref: str) -> pd.DataFrame:
    import re
    params = fit.params; ci = fit.conf_int(); pv = fit.pvalues
    rows = []
    for name in params.index:
        if name.startswith("C(label"):
            m = re.search(r"\[T\.([^\]]+)\]", name)
            label = m.group(1) if m else name
            rows.append({
                "contrast": f"{label} - {ref}",
                "coef": float(params[name]),
                "ci_low": float(ci.loc[name, 0]),
                "ci_high": float(ci.loc[name, 1]),
                "p": float(pv[name]),
            })
    return pd.DataFrame(rows)


def ols_length_adjusted_all(
    df: pd.DataFrame, metrics: List[str], ref: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    if smf is None:
        print("[analysis] statsmodels not installed -- OLS length adjustment "
              "skipped.")
        return pd.DataFrame()
    work = df.copy()
    work["Lc"] = work["token_length"] - work["token_length"].mean()
    work["Lc2"] = work["Lc"] ** 2
    out_rows = []
    for metric in metrics:
        w = work.dropna(subset=[metric]).copy()
        if w["label"].nunique() < 2:
            continue
        w["y"] = w[metric].astype(float)
        formula_raw = f"y ~ C(label, Treatment('{ref}'))"
        formula_adj = f"y ~ C(label, Treatment('{ref}')) + Lc + Lc2"
        cov_kw = {"groups": w["template_id"]}
        try:
            raw = smf.ols(formula_raw, w).fit(cov_type="cluster", cov_kwds=cov_kw)
            adj = smf.ols(formula_adj, w).fit(cov_type="cluster", cov_kwds=cov_kw)
        except Exception as e:                                    # pragma: no cover
            print(f"[analysis] OLS failed for {metric}: {e}")
            continue
        raw_tab = _coef_table(raw, ref).rename(
            columns={"coef": "raw_coef", "ci_low": "raw_ci_low",
                     "ci_high": "raw_ci_high", "p": "raw_p"})
        adj_tab = _coef_table(adj, ref).rename(
            columns={"coef": "adj_coef", "ci_low": "adj_ci_low",
                     "ci_high": "adj_ci_high", "p": "adj_p"})
        merged = raw_tab.merge(adj_tab, on="contrast", how="outer")
        merged.insert(0, "metric", metric)
        merged["shrinkage_pct"] = 100.0 * (
            1.0 - merged["adj_coef"].abs()
            / merged["raw_coef"].abs().replace(0, np.nan)
        )
        merged["n_templates"] = w["template_id"].nunique()
        out_rows.append(merged)
    return pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()


# ============================================================
# (D) pairwise effects with Holm
# ============================================================

def pairwise_all_metrics(
    df: pd.DataFrame, metrics: List[str], seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    return {m: pairwise_report(df, m, seed=seed) for m in metrics if m in df.columns}


# ============================================================
# (E) bulk consistency (Lemma 1)
# ============================================================

def bulk_consistency(
    legacy: pd.DataFrame, spectral: pd.DataFrame,
) -> pd.DataFrame:
    """Merge on prompt and compare ftle_final with lambda_bulk_pred.

    Lemma 1 says lambda_L^{(eps)} ~ (1/L) log(||J||_F / sqrt(n)), which is
    exactly lambda_bulk_pred. Agreement in mean, and Pearson r > 0.9, are
    the empirical fingerprint that Algorithm 1 was measuring the bulk."""
    required = {"prompt", "lambda_bulk_pred", "n_dim", "depth"}
    if not required.issubset(spectral.columns):
        missing = ", ".join(sorted(required - set(spectral.columns)))
        print(f"\nBULK CONSISTENCY skipped: spectral CSV lacks columns: {missing}")
        return pd.DataFrame()
    merged = legacy[["prompt", "label", "ftle_final"]].merge(
        spectral[["prompt", "lambda_bulk_pred", "n_dim", "depth"]],
        on="prompt", how="inner",
    )
    if merged.empty:
        return merged
    merged["abs_err"] = (merged["ftle_final"] - merged["lambda_bulk_pred"]).abs()
    merged["rel_err"] = merged["abs_err"] / merged["ftle_final"].abs().replace(0, np.nan)

    rows = []
    for scope, g in [("overall", merged)] + [
        (c, merged[merged["label"] == c]) for c in CATEGORY_ORDER
    ]:
        if len(g) < 2 or g["ftle_final"].std() == 0 or g["lambda_bulk_pred"].std() == 0:
            r = float("nan")
        else:
            r = float(np.corrcoef(g["ftle_final"], g["lambda_bulk_pred"])[0, 1])
        rows.append({
            "scope": scope,
            "n": int(len(g)),
            "mean_ftle_final": float(g["ftle_final"].mean()) if len(g) else float("nan"),
            "mean_lambda_bulk_pred": float(g["lambda_bulk_pred"].mean()) if len(g) else float("nan"),
            "median_rel_err": float(g["rel_err"].median()) if len(g) else float("nan"),
            "pearson_r": r,
        })
    return pd.DataFrame(rows)


# ============================================================
# (F) joint axes: output uncertainty vs hidden-state response
# ============================================================

def joint_axes(
    df: pd.DataFrame, reference: str = DEFAULT_REFERENCE,
) -> pd.DataFrame:
    """Per-category mean of the joint observables, and signed contrast against
    the reference. On length-matched data this table is what shows the factual
    inversion (most confident at the output, most expansive internally)."""
    keys = [c for c in [
        "entropy", "top1_prob",
        "sigma_max", "stable_rank", "spectral_gap",
        "chi_fisher_max", "lambda_bulk_pred", "lambda_max",
    ] if c in df.columns]
    means = df.groupby("label")[keys].mean().reindex(CATEGORY_ORDER)
    ref_row = means.loc[reference]
    diffs = means.sub(ref_row, axis=1)
    means.columns = [f"mean_{c}" for c in means.columns]
    diffs.columns = [f"diff_{c}" for c in diffs.columns]
    return pd.concat([means, diffs], axis=1).reset_index()


# ============================================================
# Driver
# ============================================================

def run_analysis(
    legacy_csv: Optional[str],
    spectral_csv: Optional[str],
    model_name: Optional[str],
    reference: str,
    metrics: Optional[List[str]],
    out_dir: str,
    seed: int = 0,
) -> None:
    if legacy_csv is None and spectral_csv is None:
        raise ValueError("Provide at least one of --legacy-csv / --spectral-csv.")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    legacy = spectral = None
    if legacy_csv:
        legacy = pd.read_csv(legacy_csv)
        legacy = legacy[legacy["label"].isin(CATEGORY_ORDER)].copy()
        legacy = _annotate(legacy, model_name)
    if spectral_csv:
        spectral = pd.read_csv(spectral_csv)
        spectral = spectral[spectral["label"].isin(CATEGORY_ORDER)].copy()
        spectral = _annotate(spectral, model_name)

    print("=" * 72)
    print("JOINT ANALYSIS")
    print(f"  legacy    : {legacy_csv or '(none)'}"
          + (f"   -> {len(legacy)} rows" if legacy is not None else ""))
    print(f"  spectral  : {spectral_csv or '(none)'}"
          + (f"   -> {len(spectral)} rows" if spectral is not None else ""))
    print(f"  model     : {model_name or '(none)'}")
    print(f"  reference : {reference}")
    print("=" * 72)

    # combine for the length distribution: whichever frame has token_length
    frame_for_length = spectral if spectral is not None else legacy
    len_tab = length_by_category(frame_for_length)
    len_tab.to_csv(out / "length_by_category.csv", index=False)
    print("\nLENGTH BY CATEGORY")
    print(len_tab.round(2).to_string(index=False))

    # (B) length-metric correlation (both frames if both present)
    corr_rows = []
    if legacy is not None:
        m = _existing_metrics(legacy, metrics or DEFAULT_LEGACY_METRICS)
        c = length_metric_correlation(legacy, m); c["source"] = "legacy"
        corr_rows.append(c)
    if spectral is not None:
        m = _existing_metrics(spectral, metrics or DEFAULT_SPECTRAL_METRICS)
        c = length_metric_correlation(spectral, m); c["source"] = "spectral"
        corr_rows.append(c)
    if corr_rows:
        corr = pd.concat(corr_rows, ignore_index=True)
        corr.to_csv(out / "length_metric_correlation.csv", index=False)
        print("\nLENGTH-METRIC CORRELATION (large residual r = length still leaks in)")
        print(corr.round(3).to_string(index=False))

    # (C) OLS length adjustment
    ols_rows = []
    if legacy is not None:
        m = _existing_metrics(legacy, metrics or DEFAULT_LEGACY_METRICS)
        o = ols_length_adjusted_all(legacy, m, ref=reference)
        if not o.empty:
            o["source"] = "legacy"; ols_rows.append(o)
    if spectral is not None:
        m = _existing_metrics(spectral, metrics or DEFAULT_SPECTRAL_METRICS)
        o = ols_length_adjusted_all(spectral, m, ref=reference)
        if not o.empty:
            o["source"] = "spectral"; ols_rows.append(o)
    if ols_rows:
        ols = pd.concat(ols_rows, ignore_index=True)
        ols.to_csv(out / "ols_length_adjusted_contrasts.csv", index=False)
        print(f"\nOLS LENGTH-ADJUSTED CONTRASTS (vs {reference}, cluster-robust)")
        cols = ["source", "metric", "contrast", "raw_coef", "raw_p",
                "adj_coef", "adj_p", "shrinkage_pct", "n_templates"]
        print(ols[cols].round(4).to_string(index=False))

    # (D) pairwise effects per metric
    if legacy is not None:
        m = _existing_metrics(legacy, metrics or DEFAULT_LEGACY_METRICS)
        for metric, tab in pairwise_all_metrics(legacy, m, seed=seed).items():
            tab.to_csv(out / f"pairwise_effects__legacy__{metric}.csv", index=False)
            print(f"\nPAIRWISE  legacy / {metric}")
            print(tab[["label_a", "label_b", "diff", "cohens_d",
                       "cliffs_delta", "perm_p", "perm_p_holm"]]
                  .round(4).to_string(index=False))
    if spectral is not None:
        m = _existing_metrics(spectral, metrics or DEFAULT_SPECTRAL_METRICS)
        for metric, tab in pairwise_all_metrics(spectral, m, seed=seed).items():
            tab.to_csv(out / f"pairwise_effects__spectral__{metric}.csv", index=False)
            print(f"\nPAIRWISE  spectral / {metric}")
            print(tab[["label_a", "label_b", "diff", "cohens_d",
                       "cliffs_delta", "perm_p", "perm_p_holm"]]
                  .round(4).to_string(index=False))

    # (E) bulk consistency
    if legacy is not None and spectral is not None:
        bc = bulk_consistency(legacy, spectral)
        if not bc.empty:
            bc.to_csv(out / "bulk_consistency.csv", index=False)
            print("\nBULK CONSISTENCY  (Lemma 1: ftle_final ~ lambda_bulk_pred)")
            print(bc.round(5).to_string(index=False))

    # (F) joint axes
    if spectral is not None:
        joint = joint_axes(spectral, reference=reference)
        joint.to_csv(out / "joint_axes.csv", index=False)
        print(f"\nJOINT AXES  (reference = {reference})")
        show = [c for c in joint.columns
                if c == "label"
                or c.startswith(("mean_entropy", "mean_top1", "mean_sigma_max",
                                  "mean_stable_rank", "mean_chi_fisher_max",
                                  "diff_entropy", "diff_top1_prob",
                                  "diff_sigma_max", "diff_stable_rank",
                                  "diff_chi_fisher_max"))]
        print(joint[show].round(4).to_string(index=False))

    print(f"\n[analysis] all outputs under: {out.resolve()}")


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--legacy-csv", default=None)
    ap.add_argument("--spectral-csv", default=None)
    ap.add_argument("--model", default=None,
                    help="HF model name (used to compute token lengths and "
                         "template ids if the CSVs don't carry them).")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    choices=list(CATEGORY_ORDER))
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="metric column names to analyse; default: legacy "
                         "and spectral defaults, intersected with what's "
                         "actually in the CSV.")
    ap.add_argument("--out-dir", default="results_analysis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_analysis(
        legacy_csv=args.legacy_csv,
        spectral_csv=args.spectral_csv,
        model_name=args.model,
        reference=args.reference,
        metrics=args.metrics,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
