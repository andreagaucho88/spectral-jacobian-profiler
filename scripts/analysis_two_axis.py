#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-analysis for the two-axis reframing. Reads only existing CSVs; writes a JSON
of results. No model runs; does not touch the live sweep."""
from __future__ import annotations
import json, numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
from common import template_signature

ROOT = Path(__file__).resolve().parent.parent
QSLUG = "Qwen__Qwen2.5-0.5B-Instruct"
CATS = ["factual", "coding", "reasoning", "hallucination_prone"]
RNG = np.random.default_rng(0)
NPERM, NBOOT = 20000, 10000

def good(df): return df[df["error"].astype(str).str.strip().isin(["", "nan"])] if "error" in df else df
def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b); na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    return float((a.mean()-b.mean())/sp), float(sp)
def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sum(a[:,None]>b[None,:])-np.sum(a[:,None]<b[None,:]))/(len(a)*len(b)))

spec = good(pd.read_csv(ROOT/f"results_spectral/{QSLUG}/final_index_-2/spectral_per_prompt.csv")).copy()
spec["tmpl"] = spec["label"] + "||" + spec["prompt"].map(template_signature)
R = {}

# ============ A. CONFOUND: chi_F & sigma_max vs entropy ============
A = {}
for y in ["chi_fisher_max", "sigma_max", "chi_fisher_along_hidden_lead"]:
    pe = stats.pearsonr(spec[y], spec["entropy"]); sp_ = stats.spearmanr(spec[y], spec["entropy"])
    pt = stats.pearsonr(spec[y], spec["top1_prob"])
    A[y] = {"pearson_entropy": round(float(pe.statistic), 3), "spearman_entropy": round(float(sp_.statistic), 3),
            "pearson_top1": round(float(pt.statistic), 3), "r2_entropy": round(float(pe.statistic)**2, 3)}
# residualize chi_F on entropy+top1, test category structure survival
X = np.column_stack([np.ones(len(spec)), spec["entropy"], spec["top1_prob"]])
beta, *_ = np.linalg.lstsq(X, spec["chi_fisher_max"].values, rcond=None)
spec["chiF_resid"] = spec["chi_fisher_max"].values - X @ beta
res_means = {c: round(float(spec[spec.label==c]["chiF_resid"].mean()), 1) for c in CATS}
fa, ha = spec[spec.label=="factual"], spec[spec.label=="hallucination_prone"]
d_raw, _ = cohen_d(fa["chi_fisher_max"], ha["chi_fisher_max"])
d_res, _ = cohen_d(fa["chiF_resid"], ha["chiF_resid"])
# perm p on residual
obs = abs(fa["chiF_resid"].mean()-ha["chiF_resid"].mean()); pool = np.concatenate([fa["chiF_resid"], ha["chiF_resid"]]); n1 = len(fa)
cnt = sum(abs(np.mean((p:=RNG.permutation(pool))[:n1])-np.mean(p[n1:])) >= obs for _ in range(NPERM))
A["chiF_residual"] = {"category_mean_resid": res_means, "fac_vs_hal_d_raw": round(d_raw, 2),
                      "fac_vs_hal_d_resid": round(d_res, 2), "fac_vs_hal_resid_perm_p": round(cnt/NPERM, 4)}
R["A_confound"] = A

# ============ B. CLUSTER INFERENCE ============
tmpl_counts = {c: int(spec[spec.label==c]["tmpl"].nunique()) for c in CATS}
def template_groups(cat):  # list of arrays (one per template) of the metric values
    d = spec[spec.label==cat]; return [d[d.tmpl==t] for t in d.tmpl.unique()]
def cluster_boot_diff(cat_a, cat_b, metric):
    ga = [g[metric].values for g in template_groups(cat_a)]; gb = [g[metric].values for g in template_groups(cat_b)]
    diffs = []
    for _ in range(NBOOT):
        sa = np.concatenate([ga[i] for i in RNG.integers(0, len(ga), len(ga))])
        sb = np.concatenate([gb[i] for i in RNG.integers(0, len(gb), len(gb))])
        diffs.append(sa.mean()-sb.mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
def cluster_perm_p(cat_a, cat_b, metric):
    ta = template_groups(cat_a); tb = template_groups(cat_b)
    vals = [g[metric].values for g in ta+tb]; na = len(ta)
    allp = np.concatenate(vals); split = [len(v) for v in vals]
    obs = np.concatenate(vals[:na]).mean() - np.concatenate(vals[na:]).mean()
    cnt = 0
    idx = np.arange(len(vals))
    for _ in range(NPERM):
        perm = RNG.permutation(idx)
        A_ = np.concatenate([vals[i] for i in perm[:na]]); B_ = np.concatenate([vals[i] for i in perm[na:]])
        if abs(A_.mean()-B_.mean()) >= abs(obs): cnt += 1
    return round(cnt/NPERM, 4)
B = {"templates_per_category": tmpl_counts}
for (a, b) in [("factual","coding"), ("factual","reasoning"), ("factual","hallucination_prone"), ("coding","reasoning")]:
    dd, _ = cohen_d(spec[spec.label==a]["sigma_max"], spec[spec.label==b]["sigma_max"])
    lo, hi = cluster_boot_diff(a, b, "sigma_max")
    B[f"sigma_max__{a}_vs_{b}"] = {"cohen_d": round(dd, 2), "cluster_boot95_diff": [round(lo,1), round(hi,1)],
                                   "cluster_boot_excludes_0": bool(lo>0 or hi<0), "cluster_perm_p": cluster_perm_p(a, b, "sigma_max")}
# chi_F cluster p for the surviving legs
for (a, b) in [("factual","hallucination_prone"), ("factual","reasoning")]:
    B[f"chi_fisher_max__{a}_vs_{b}_cluster_perm_p"] = cluster_perm_p(a, b, "chi_fisher_max")
# two-axis correlation r(sigma_max, entropy): cluster bootstrap over ALL templates
alltmpl = [spec[spec.tmpl==t] for t in spec.tmpl.unique()]
rs = []
for _ in range(NBOOT):
    s = pd.concat([alltmpl[i] for i in RNG.integers(0, len(alltmpl), len(alltmpl))])
    if s["sigma_max"].std() > 0: rs.append(stats.pearsonr(s["sigma_max"], s["entropy"]).statistic)
B["r_sigma_entropy"] = {"point": round(float(stats.pearsonr(spec["sigma_max"], spec["entropy"]).statistic), 3),
                        "cluster_boot95": [round(float(np.percentile(rs,2.5)),3), round(float(np.percentile(rs,97.5)),3)]}
R["B_cluster"] = B

# ============ C. TOST EQUIVALENCE ============
def tost(a, b, bound_d):
    a, b = np.asarray(a), np.asarray(b); na, nb = len(a), len(b)
    d, sp = cohen_d(a, b); se = sp*np.sqrt(1/na+1/nb); diff = a.mean()-b.mean(); bound = bound_d*sp; dfree = na+nb-2
    t_lo = (diff-(-bound))/se; p_lo = 1-stats.t.cdf(t_lo, dfree)   # H0: diff <= -bound
    t_hi = (diff-bound)/se; p_hi = stats.t.cdf(t_hi, dfree)        # H0: diff >= bound
    p_tost = max(p_lo, p_hi)
    # 90% CI for diff (TOST equivalent to 90% CI within bounds)
    tcrit = stats.t.ppf(0.95, dfree)
    ci = (diff-tcrit*se, diff+tcrit*se)
    return {"cohen_d": round(d,2), "bound_d": bound_d, "p_tost": round(float(p_tost),4),
            "equivalent_at_bound": bool(p_tost<0.05), "diff_90ci": [round(float(ci[0]),1), round(float(ci[1]),1)],
            "bound_raw": round(float(bound),1)}
C = {}
for (a, b) in [("factual","hallucination_prone"), ("coding","reasoning")]:
    for bd in [0.5, 0.35]:
        C[f"sigma_max__{a}_vs_{b}__bound{bd}"] = tost(spec[spec.label==a]["sigma_max"], spec[spec.label==b]["sigma_max"], bd)
R["C_tost"] = C

# ============ D. BULK IDENTITY exact ============
leg = pd.read_csv(ROOT/f"results_legacy/{QSLUG}/legacy_per_prompt.csv")
m = pd.merge(leg[["prompt","label","ftle_final"]], spec[["prompt","label","lambda_bulk_pred"]], on=["prompt","label"])
r_bulk = float(stats.pearsonr(m["ftle_final"], m["lambda_bulk_pred"]).statistic)
rel = float(np.median(np.abs(m["lambda_bulk_pred"]-m["ftle_final"])/np.abs(m["ftle_final"])))
R["D_bulk"] = {"pearson_r": round(r_bulk,4), "pearson_r_2dp": round(r_bulk,2), "median_rel_err_pct": round(rel*100,2), "n": len(m)}

# ============ E. OUTPUT UNCERTAINTY both models (length-matched legacy) ============
E = {}
for m_ in ["Qwen__Qwen2.5-0.5B-Instruct", "HuggingFaceTB__SmolLM2-360M-Instruct"]:
    L = pd.read_csv(ROOT/f"results_legacy/{m_}/legacy_per_prompt.csv")
    hal = L[L.label=="hallucination_prone"]
    row = {"entropy_means": {c: round(float(L[L.label==c]["entropy"].mean()),2) for c in CATS},
           "top1_means": {c: round(float(L[L.label==c]["top1_prob"].mean()),2) for c in CATS}}
    ds = []
    for c in ["factual","coding","reasoning"]:
        de, _ = cohen_d(hal["entropy"], L[L.label==c]["entropy"]); ds.append(abs(de))
    row["halluc_vs_others_entropy_absd"] = [round(x,2) for x in ds]
    row["n_per_cat"] = int(len(L)//4)
    E[m_] = row
R["E_output_uncertainty"] = E

print(json.dumps(R, indent=2))
(ROOT/"results_analysis").mkdir(exist_ok=True)
json.dump(R, open(ROOT/"results_analysis/two_axis_numbers.json","w"), indent=2)
print("\nSAVED -> results_analysis/two_axis_numbers.json")
