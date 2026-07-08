#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two new figures for the two-axis reframing: the sigma_max-vs-entropy independence
(the solid core) and the chi_F-vs-entropy confound. Reads existing CSVs only."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT/"paper"/"figures"; FIG.mkdir(exist_ok=True)
QSLUG = "Qwen__Qwen2.5-0.5B-Instruct"
CATS = ["factual","coding","reasoning","hallucination_prone"]
CL = {"factual":"factual","coding":"coding","reasoning":"reasoning","hallucination_prone":"halluc.-prone"}
COL = {"factual":"#0072B2","coding":"#E69F00","reasoning":"#009E73","hallucination_prone":"#D55E00"}
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight","font.size":10,
    "axes.titlesize":11,"axes.labelsize":10,"axes.spines.top":False,"axes.spines.right":False,
    "legend.frameon":False,"axes.grid":True,"grid.alpha":0.25,"font.family":"DejaVu Sans"})

df = pd.read_csv(ROOT/f"results_spectral/{QSLUG}/final_index_-2/spectral_per_prompt.csv")
df = df[df["error"].astype(str).str.strip().isin(["","nan"])]

# ---- Fig: two_axis_sigma_entropy (the core dissociation across THREE architectures) ----
PANELS = [
    ("Qwen2.5-0.5B (RMSNorm)", "results_spectral/Qwen__Qwen2.5-0.5B-Instruct/final_index_-2/spectral_per_prompt.csv", "cluster CI $[-0.21,0.17]$"),
    ("SmolLM2-360M (RMSNorm)", "results_spectral/HuggingFaceTB__SmolLM2-360M-Instruct/final_index_-2/spectral_per_prompt.csv", "cluster CI $[-0.49,-0.04]$"),
    ("GPT-2 124M (LayerNorm)", "results_spectral_gpt2/gpt2/final_index_-2/spectral_per_prompt.csv", "$n=16$/cat"),
]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
for ax, (name, path, ci) in zip(axes, PANELS):
    dd = pd.read_csv(ROOT/path); dd = dd[dd["error"].astype(str).str.strip().isin(["","nan"])]
    rr = stats.pearsonr(dd["sigma_max"], dd["entropy"]).statistic
    for c in CATS:
        d = dd[dd.label==c]
        ax.scatter(d["entropy"], d["sigma_max"], s=26, c=COL[c], alpha=0.82,
                   edgecolors="white", linewidths=0.4, label=CL[c])
    ax.set_xlabel("output entropy  $H(p)$  (nats)")
    ax.set_ylabel(r"$\sigma_{\max}$  (hidden leading amplification)")
    ax.set_title(f"{name}\npre-norm: Pearson r={rr:+.2f}  ({ci})")
axes[0].legend(fontsize=8, title="category", title_fontsize=8, loc="upper right")
fig.suptitle("Hidden amplification $\\sigma_{\\max}$ vs. output-entropy LEVEL across three architectures "
             "($R^2 \\leq 0.10$, sign unstable) — but the like-for-like sensitivity comparison is coupled (r=0.45, see text)",
             y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(FIG/"two_axis_sigma_entropy.pdf"); plt.close(fig)
print("two_axis_sigma_entropy.pdf  (two-panel, Qwen + SmolLM2)")

# ---- Fig: chiF_entropy_confound (two-model scatter + Qwen residual collapse) ----
SM = pd.read_csv(ROOT/"results_spectral/HuggingFaceTB__SmolLM2-360M-Instruct/final_index_-2/spectral_per_prompt.csv")
SM = SM[SM["error"].astype(str).str.strip().isin(["","nan"])]
fig, (a0,a1,a2) = plt.subplots(1,3, figsize=(15.5,4.4))
for ax,(name,dd) in zip((a0,a1),[("Qwen2.5-0.5B",df),("SmolLM2-360M",SM)]):
    rr = stats.pearsonr(dd["chi_fisher_max"], dd["entropy"]).statistic
    for c in CATS:
        d = dd[dd.label==c]
        ax.scatter(d["entropy"], d["chi_fisher_max"], s=24, c=COL[c], alpha=0.82,
                   edgecolors="white", linewidths=0.4, label=CL[c])
    b0,b1 = np.polyfit(dd["entropy"], dd["chi_fisher_max"], 1)
    xs = np.linspace(dd["entropy"].min(), dd["entropy"].max(), 50)
    ax.plot(xs, b0*xs+b1, "k--", lw=1, alpha=0.7, label="OLS fit")
    ax.set_xlabel("output entropy  $H(p)$  (nats)"); ax.set_ylabel(r"$\chi_F^{\max}$")
    ax.set_title(f"{name}:  Pearson r={rr:+.2f}")
a0.legend(fontsize=7, loc="upper left")
# panel 3: Qwen raw vs residual category means (z-scored) -> factual/hallucination collapse
X = np.column_stack([np.ones(len(df)), df["entropy"], df["top1_prob"]])
beta,*_ = np.linalg.lstsq(X, df["chi_fisher_max"].values, rcond=None)
df2 = df.assign(chiF_resid=df["chi_fisher_max"].values - X@beta)
xs2 = np.arange(len(CATS)); w=0.38
raw = df2.groupby("label")["chi_fisher_max"].mean().reindex(CATS)
res = df2.groupby("label")["chiF_resid"].mean().reindex(CATS)
rawz = (raw-raw.mean())/raw.std(); resz = (res-res.mean())/res.std()
a2.bar(xs2-w/2, rawz.values, w, color=[COL[c] for c in CATS], alpha=0.95, label="raw $\\chi_F$")
a2.bar(xs2+w/2, resz.values, w, color=[COL[c] for c in CATS], alpha=0.5, hatch="//", label="residual (entropy removed)")
a2.axhline(0, color="k", lw=0.6)
a2.set_xticks(xs2); a2.set_xticklabels([CL[c] for c in CATS], rotation=20, ha="right")
a2.set_ylabel("category mean (z-scored)")
a2.set_title("Qwen: factual$\\leftrightarrow$hallucination collapses\nafter removing entropy (d: $-0.98\\to-0.19$)")
a2.legend(fontsize=7)
fig.suptitle(r"$\chi_F$ is entropy-related and not a stable axis: its entropy correlation even flips sign across models", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(FIG/"chiF_entropy_confound.pdf"); plt.close(fig)
print("chiF_entropy_confound.pdf (two-model scatter + Qwen collapse)")
