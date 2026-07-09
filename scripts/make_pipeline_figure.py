#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schematic roadmap figure: the forward pipeline
   H0 -> h_{L,T} (pre-norm) -> h_{L,T} (post-norm) -> logits z -> p
and where each measured quantity lives. Self-contained matplotlib; writes paper/figures/pipeline_schematic.pdf."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT/"paper"/"figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"savefig.dpi":300,"savefig.bbox":"tight","font.family":"DejaVu Sans","font.size":10})

# colorblind-safe palette (consistent with the other figures)
C_HID="#0072B2"; C_OUT="#D55E00"; C_JAC="#009E73"; C_FISH="#CC79A7"; C_GRAD="#E69F00"
fig, ax = plt.subplots(figsize=(12.6, 5.4)); ax.axis("off")
ax.set_xlim(0,12.6); ax.set_ylim(0,5.4)

# --- forward-pass nodes (x-centres) ---
nodes = [
    (1.15, "$H_0$", "input\nembeddings\n$\\mathbb{R}^{T\\times d}$", C_HID),
    (4.05, "$h_{L,T}$", "pre-(final-)norm\nhidden state\n(read-out $-2$)", C_HID),
    (6.85, "$h_{L,T}$", "post-(final-)norm\nhidden state\n(read-out $-1$)", C_HID),
    (9.35, "$z$", "logits", C_OUT),
    (11.5, "$p$", "output\ndistribution", C_OUT),
]
yc = 4.35; bw, bh = 1.6, 1.0
cx = {}
for x, sym, sub, col in nodes:
    ax.add_patch(FancyBboxPatch((x-bw/2, yc-bh/2), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.10",
                                linewidth=1.6, edgecolor=col, facecolor=col+"22"))
    ax.text(x, yc+0.17, sym, ha="center", va="center", fontsize=15, color=col, fontweight="bold")
    ax.text(x, yc-0.27, sub, ha="center", va="center", fontsize=7.4, color="#333333")
    cx[sym+sub[:4]] = x
X = [n[0] for n in nodes]

# --- transformation arrows between nodes ---
edges = [(X[0],X[1],"Transformer\n($L$ blocks)"),(X[1],X[2],"final\nRMSNorm"),
         (X[2],X[3],"lm_head"),(X[3],X[4],"softmax")]
for x0,x1,lab in edges:
    ax.add_patch(FancyArrowPatch((x0+bw/2, yc),(x1-bw/2, yc), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.4, color="#555555"))
    ax.text((x0+x1)/2, yc+0.72, lab, ha="center", va="center", fontsize=7.6, color="#555555", style="italic")

# --- output-level observables (above p) ---
ax.annotate("", xy=(X[4], yc+0.55), xytext=(X[4], yc+1.15),
            arrowprops=dict(arrowstyle="-", color=C_OUT, lw=1.1))
ax.text(X[4], yc+1.33, "entropy $H(p)$,\n top-1 $p_{\\max}$\n(output level)", ha="center", va="bottom",
        fontsize=8.2, color=C_OUT, fontweight="bold")

def span(x0, x1, y, color, label, sub):
    ax.plot([x0,x0],[y,y+0.16], color=color, lw=1.4)
    ax.plot([x1,x1],[y,y+0.16], color=color, lw=1.4)
    ax.plot([x0,x1],[y,y], color=color, lw=1.7)
    ax.text((x0+x1)/2, y-0.17, label, ha="center", va="top", fontsize=8.6, color=color, fontweight="bold")
    ax.text((x0+x1)/2, y-0.52, sub, ha="center", va="top", fontsize=7.0, color="#444444")

# --- measured-quantity spans (below the flow), increasing depth ---
span(X[0], X[1], 3.42, C_JAC,
     "$J=\\partial h_{L,T}/\\partial H_0$ :  $\\sigma_{\\max}$,  $\\|J\\|_F/\\lambda_{\\mathrm{bulk}}$,  stable rank",
     "Algorithm 2 (pre-norm) + Algorithm 1 sampled-direction $\\lambda_L^{(\\epsilon)}$")
span(X[0], X[2], 2.42, C_HID,
     "$\\partial h_{L,T}^{\\,\\mathrm{post}}/\\partial H_0$ :  $\\sigma_{\\max}$ (post-norm)",
     "Algorithm 2 (post-norm) — the final normalization changes the hidden metric (App. C)")
span(X[0], X[4], 1.42, C_FISH,
     "$\\chi_F$ :  Fisher/KL output susceptibility",
     "how an input perturbation moves the output distribution $p$")
span(X[0], X[4], 0.42, C_GRAD,
     "$\\nabla_{H_0}H$ :  entropy input-gradient  (one backward)",
     "the like-for-like partner of $\\sigma_{\\max}$ — the corrected comparison (Sec. 4.4)")

fig.savefig(FIG/"pipeline_schematic.pdf")
print("wrote", FIG/"pipeline_schematic.pdf")
