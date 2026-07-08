# χ_F vs output entropy — confound analysis

**Question.** Is the Fisher/KL output susceptibility χ_F an observable independent of output entropy
H(p), or is it (partly) entropy re-expressed? Because F = diag(p) − ppᵀ is built from the same p
whose entropy is observable #1, this must be tested, not assumed.

**Data.** Qwen2.5-0.5B, pre-norm, n=160 (40/category), completed run. All numbers reproducible via
`analysis_two_axis.py` → `results_analysis/two_axis_numbers.json`.

## 1. Correlation with the output side
| Metric | Pearson vs H(p) | Spearman vs H(p) | R² (entropy) | Pearson vs top-1 |
|---|---|---|---|---|
| **χ_F^max** | **+0.49** | **+0.56** | **0.24** | −0.40 |
| χ_F along hidden lead | +0.42 | +0.59 | 0.17 | −0.33 |
| **σ_max (hidden)** | **−0.015** | +0.03 | **0.00** | +0.15 |

χ_F shares ~24% of its variance with entropy; σ_max shares **none** (r ≈ 0). So the *hidden* axis is
independent of output uncertainty, but the *χ_F* axis is not.

## 2. Does the χ_F category ordering survive conditioning on entropy?
We regress χ_F^max on (entropy, top-1) and test the category structure on the **residuals**.

| Contrast | raw Cohen d | residual Cohen d | residual perm p |
|---|---|---|---|
| factual vs hallucination-prone | **−0.98** | **−0.19** | **0.46** |

The factual↔hallucination χ_F separation — the leg that supported the "factual inversion" on the
output side — **does not survive** removing entropy (d collapses −0.98 → −0.19, n.s.). Category mean
residuals: factual −117, coding −632, reasoning +560, hallucination +189: χ_F retains *some*
independent structure (a coding↔reasoning contrast), but **not** the factual-inversion axis.

## 3. Conclusion
- **χ_F is NOT an independent fourth observable.** It is an output-side perturbative susceptibility
  that **partially tracks entropy** (r≈0.5); its headline factual/hallucination ordering is largely
  entropy-driven.
- **χ_F remains useful**: it connects an *input perturbation* to *output-distribution movement*
  (a bridge the entropy of a static distribution does not provide), and it carries residual,
  non-entropy structure (coding vs reasoning). It should be reported as **output-side / entropy-related**,
  not as an independent axis.
- **The solid dissociation is on the hidden axis**: σ_max ⟂ entropy (r = −0.015). Output uncertainty
  and hidden leading amplification are distinct prompt-conditioned observables; χ_F is a related
  output-side measure, not a third independent one.

**Effect on paper claims:** the central claim is narrowed from "four distinct observables" to a
**two-axis dissociation** (output uncertainty vs hidden leading amplification), with χ_F reframed as
an output-side, entropy-related susceptibility.
