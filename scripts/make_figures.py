#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication figures for the hardened paper. Reads only completed CSVs; writes
figures/*.pdf. Colour scheme is fixed and colourblind-safe (Okabe-Ito) and
consistent across every figure. Scale-sanity (fig 5) is produced only if
results_spectral_scalecheck/ exists."""
from __future__ import annotations
import os, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"; FIG.mkdir(exist_ok=True)
QSLUG = "Qwen__Qwen2.5-0.5B-Instruct"

# Okabe-Ito colourblind-safe palette, fixed per category everywhere.
CATS = ["factual", "coding", "reasoning", "hallucination_prone"]
CLABEL = {"factual": "factual", "coding": "coding", "reasoning": "arithmetic",
          "hallucination_prone": "imposs.-entity"}
COL = {"factual": "#0072B2", "coding": "#E69F00",
       "reasoning": "#009E73", "hallucination_prone": "#D55E00"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.25,
    "font.family": "DejaVu Sans",
})

def _good(df):
    return df[df["error"].astype(str).str.strip().isin(["", "nan"])] if "error" in df else df

def _legend(ax):
    h = [plt.Line2D([0], [0], marker="o", ls="", mfc=COL[c], mec="none", ms=7) for c in CATS]
    ax.legend(h, [CLABEL[c] for c in CATS], title="category", loc="best", fontsize=8, title_fontsize=8)

spec2 = _good(pd.read_csv(ROOT/f"results_spectral/{QSLUG}/final_index_-2/spectral_per_prompt.csv"))
spec1p = ROOT/f"results_spectral/{QSLUG}/final_index_-1/spectral_per_prompt.csv"
spec1 = _good(pd.read_csv(spec1p)) if spec1p.exists() else None
legacy = pd.read_csv(ROOT/f"results_legacy/{QSLUG}/legacy_per_prompt.csv")

# ---------------------------------------------------------------- Fig 1
def fig1():
    panels = [("Pre-norm  (final-index $-2$)", spec2)]
    if spec1 is not None:
        panels.append(("Post-norm  (final-index $-1$)", spec1))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2*len(panels), 4.2), squeeze=False)
    for ax, (title, df) in zip(axes[0], panels):
        for c in CATS:
            d = df[df["label"] == c]
            ax.scatter(d["entropy"], d["sigma_max"], s=26, c=COL[c], alpha=0.8,
                       edgecolors="white", linewidths=0.4, label=CLABEL[c])
        ax.set_xlabel("output entropy  $H(p)$  (nats)  — output-uncertainty axis")
        ax.set_ylabel(r"$\sigma_{\max}$  — hidden amplification axis")
        ax.set_title(title)
    _legend(axes[0][0])
    fig.suptitle("Two-axis dissociation: output uncertainty vs. hidden amplification (Qwen2.5-0.5B, n=40/cat)",
                 fontsize=11, y=1.02)
    fig.savefig(FIG/"scatter_uncertainty_vs_susceptibility.pdf"); plt.close(fig)

# ---- two-model helpers ----
MODELS = [("Qwen2.5-0.5B", "Qwen__Qwen2.5-0.5B-Instruct"),
          ("SmolLM2-360M", "HuggingFaceTB__SmolLM2-360M-Instruct")]
def _load(slug, fi):
    p = ROOT/f"results_spectral/{slug}/final_index_{fi}/spectral_per_prompt.csv"
    return _good(pd.read_csv(p)) if p.exists() else None
def _load_legacy(slug):
    p = ROOT/f"results_legacy/{slug}/legacy_per_prompt.csv"
    return pd.read_csv(p) if p.exists() else None

# ---------------------------------------------------------------- Fig 2 (two-model)
def fig2():
    metrics = [("sigma_max", r"$\sigma_{\max}$ (hidden)"),
               ("stable_rank", r"stable rank (anisotropy)"),
               ("chi_fisher_max", r"$\chi_F^{\max}$ (output)")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    for row, (mname, mslug) in enumerate(MODELS):
        df = _load(mslug, -2)
        for col, (m, lab) in enumerate(metrics):
            ax = axes[row][col]
            data = [df[df["label"] == c][m].values for c in CATS]
            vp = ax.violinplot(data, showmeans=False, showextrema=False)
            for bb, c in zip(vp["bodies"], CATS):
                bb.set_facecolor(COL[c]); bb.set_alpha(0.35); bb.set_edgecolor(COL[c])
            bp = ax.boxplot(data, widths=0.25, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black"))
            for patch, c in zip(bp["boxes"], CATS):
                patch.set_facecolor(COL[c]); patch.set_alpha(0.8)
            ax.set_xticks(range(1, 5))
            if row == 1:
                ax.set_xticklabels([CLABEL[c] for c in CATS], rotation=20, ha="right")
            else:
                ax.set_xticklabels([])
            ax.set_title(f"{mname}: {lab}", fontsize=9)
    fig.suptitle("Per-category spectral metrics, pre-norm (n=40/cat): Qwen (top), SmolLM2 (bottom)", y=1.005)
    fig.tight_layout()
    fig.savefig(FIG/"box_spectral_metrics.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3 (two-model)
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    out = {}
    for ax, (mname, mslug) in zip(axes, MODELS):
        leg = _load_legacy(mslug); spec = _load(mslug, -2)
        m = pd.merge(leg[["prompt", "label", "ftle_final"]],
                     spec[["prompt", "label", "lambda_bulk_pred"]], on=["prompt", "label"], how="inner")
        x, y = m["ftle_final"].values, m["lambda_bulk_pred"].values
        relerr = float(np.median(np.abs(y - x) / np.abs(x))); r = float(np.corrcoef(x, y)[0, 1])
        out[mname] = (round(relerr*100, 1), round(r, 2), len(m))
        for c in CATS:
            d = m[m["label"] == c]
            ax.scatter(d["ftle_final"], d["lambda_bulk_pred"], s=24, c=COL[c], alpha=0.85,
                       edgecolors="white", linewidths=0.4, label=CLABEL[c])
        lo = float(min(x.min(), y.min())); hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="$y=x$")
        ax.set_xlabel(r"Alg.\,1  $\lambda_L^{(\epsilon)}$  (ftle_final)")
        ax.set_ylabel(r"Alg.\,2 exact bulk  $\lambda_{\mathrm{bulk}}$")
        ax.set_title(f"{mname}: rel. err. {relerr*100:.1f}%, r={r:.2f} (n={len(m)})")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Bulk identity on both models", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG/"bulk_identity.pdf"); plt.close(fig)
    return out

# ---------------------------------------------------------------- Fig 4 (two-model)
def fig4():
    def stat(df, c):
        v = df[df["label"] == c]["sigma_max"].values
        return v.mean(), v.std(ddof=1)/np.sqrt(len(v))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ok = False
    for ax, (mname, mslug) in zip(axes, MODELS):
        pre = _load(mslug, -2); post = _load(mslug, -1)
        if pre is None or post is None:
            ax.set_visible(False); continue
        ok = True
        width = 0.38; xs = np.arange(len(CATS))
        p_ = [stat(pre, c) for c in CATS]; q_ = [stat(post, c) for c in CATS]
        ax.bar(xs - width/2, [a[0] for a in p_], width, yerr=[a[1] for a in p_],
               color=[COL[c] for c in CATS], alpha=0.95, capsize=3, label="pre-norm")
        ax.bar(xs + width/2, [a[0] for a in q_], width, yerr=[a[1] for a in q_],
               color=[COL[c] for c in CATS], alpha=0.5, hatch="//", capsize=3, label="post-norm")
        ax.set_xticks(xs); ax.set_xticklabels([CLABEL[c] for c in CATS], rotation=20, ha="right")
        ax.set_ylabel(r"$\sigma_{\max}$ (mean $\pm$ SEM)"); ax.set_title(mname)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pre- vs. post-norm hidden amplification (n=40/cat): factual leads only pre-norm on both models", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG/"pre_post_norm_shift.pdf"); plt.close(fig)
    return ok

# ---------------------------------------------------------------- Fig 5 (scale sanity, optional)
def fig5():
    rows = []
    base = ROOT/"results_spectral_scalecheck"
    models = [("Qwen2.5-0.5B", ROOT/f"results_spectral/{QSLUG}/final_index_-2/spectral_per_prompt.csv"),
              ("Qwen2.5-1.5B", base/"Qwen__Qwen2.5-1.5B-Instruct/final_index_-2/spectral_per_prompt.csv"),
              ("Qwen2.5-3B",   base/"Qwen__Qwen2.5-3B-Instruct/final_index_-2/spectral_per_prompt.csv")]
    def complete(p):
        if not Path(p).exists(): return False
        d = _good(pd.read_csv(p))
        return all((d["label"] == c).sum() >= 8 for c in CATS)   # all 4 cats, n>=8
    avail = [(name, p) for name, p in models if complete(p)]
    # require BOTH scale-check models complete, else remove any stale premature figure
    if len(avail) < 3:
        (FIG/"scale_sanity.pdf").unlink(missing_ok=True)
        return False
    # normalise each model's category means by that model's cross-category mean, so the
    # WITHIN-model ordering is legible even though absolute scales differ across models.
    dfs = {name: _good(pd.read_csv(p)) for name, p in avail}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, met, lab in zip(axes, ["sigma_max", "chi_fisher_max"],
                            [r"$\sigma_{\max}$ (hidden amplification)", r"$\chi_F^{\max}$ (output susceptibility)"]):
        xs = np.arange(len(avail)); w = 0.2
        catmeans = {name: np.mean([dfs[name][dfs[name]["label"] == c][met].mean() for c in CATS]) for name, _ in avail}
        for j, c in enumerate(CATS):
            vals = [dfs[name][dfs[name]["label"] == c][met].mean() / catmeans[name] for name, _ in avail]
            ax.bar(xs + (j-1.5)*w, vals, w, color=COL[c], alpha=0.9, label=CLABEL[c])
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in avail])
        ax.set_ylabel("mean / model-average"); ax.set_title(lab)
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Scale sanity check (pre-norm), per-model normalised: factual leads $\\sigma_{\\max}$ ONLY at 0.5B; "
                 "coding leads at 1.5B/3B", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG/"scale_sanity.pdf"); plt.close(fig)
    return True

if __name__ == "__main__":
    fig1(); print("fig1 scatter_uncertainty_vs_susceptibility.pdf")
    fig2(); print("fig2 box_spectral_metrics.pdf (two-model)")
    print("fig3 bulk_identity.pdf (two-model):", fig3())
    ok4 = fig4(); print(f"fig4 pre_post_norm_shift.pdf  ({'ok (two-model)' if ok4 else 'skipped'})")
    ok5 = fig5(); print(f"fig5 scale_sanity.pdf  ({'ok' if ok5 else 'skipped: scale-check not ready'})")
    print("figures in", FIG)
