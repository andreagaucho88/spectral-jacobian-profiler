# Hardening review response

How `main_arxiv_hardened.tex` addresses each critical review point. All numbers are from the
completed Qwen2.5-0.5B runs (n=40/cat, both read-out sites); SmolLM2 Algorithm-1 results are the
paper's original sweep. The running spectral sweep and the scale-check were **not** interrupted.

| # | Review point | What changed in the hardened paper |
|---|---|---|
| 1 | Sub-billion "toy" models | Added an **intermediate-scale sanity-check** section (§Intermediate-scale sanity check, `sec:scale`) on Qwen2.5-1.5B/3B with a minimal config, plus **three pre-registered readings** in the Discussion (agree / disagree / infeasible). Every claim is scoped to "sub-billion" or "Qwen pre-norm"; the abstract and limitations state the scale ceiling explicitly. |
| 2 | Algorithm 2 too expensive for serving | Reframed throughout as **offline model profiling / diagnostic**, never a runtime guardrail. New Methods subsection *Computational cost, and why Algorithm 2 is offline* (`sec:methods-cost`) quantifies ~180 double-backward products/prompt and states serving-time use is precluded; the abstract, intro (*Scope and positioning*), discussion, and limitations all repeat this. |
| 3 | "Coordinate-free" overclaim | χ_F is now called **"intrinsic to the output distribution" / "hidden-representation independent on the output side"**, never "coordinate-free" absolutely. Added the precise caveat that the response functional is intrinsic to the predictive distribution *but still induced by a perturbation fixed in input-embedding coordinates*. All ~11 prior uses were revised; the one remaining "coordinate-free" is the sentence that explicitly says χ_F is *not* coordinate-free in an absolute sense. |
| 4 | Pre/post-norm as definitive localization | Softened to **"consistent with a radial component removed by RMSNorm"**; explicitly "we do not claim a definitive decomposition, as the two read-out sites also differ in the last block's transformation." Table caption and figure caption use the same hedge. |
| 5 | "Pending run" language | **Removed entirely.** Table `tab:alg2` now lists only the **completed** Qwen pre/post rows (with n/cat); SmolLM2 spectral is described as an offline follow-up, not a pending table row. `grep "pending"` → none. |
| 6 | Hallucination-prone ≠ verified hallucination | The category is called a **proxy** at first use (abstract, intro, related work, limitations). The paper states it does not generate or evaluate answers and makes **no hallucination-detection claim**. Phrase "impossible/fictional/hallucination-prone prompts" used consistently. |
| 7 | Missing related work | Added **semantic entropy** (Kuhn et al. 2023 ICLR; Farquhar et al. 2024 *Nature*), **semantic-entropy probes** (Kossen et al. 2024), and **hidden-state hallucination detection** (Azaria & Mitchell 2023 EMNLP Findings; Chen et al. 2024 *INSIDE*/EigenScore ICLR). New Related-Work paragraph positions our offline input-perturbation geometry as complementary to those runtime/answer-level detectors. All five references verified (not invented). |
| 8 | Plots | Added 4 publication figures (5th auto-includes when the scale-check completes): `scatter_uncertainty_vs_susceptibility`, `box_spectral_metrics`, `bulk_identity`, `pre_post_norm_shift`, `scale_sanity`. Colourblind-safe (Okabe–Ito), consistent per-category colours, referenced in the Results. |
| 9 | Stronger limitations | Rewrote Limitations into 8 explicit items: model scale, templated/synthetic prompts, proxy-not-verified, representation dependence, not-a-Lyapunov-exponent, deployment cost, convergence/null results, one-set-for-both-axes. |
| 10 | Keep central claim precise | Central claim is now the exact sentence: **"output uncertainty, hidden amplification, bulk response, and output-distribution susceptibility are distinct prompt-conditioned observables."** Repeated verbatim in abstract and discussion. |

## Claims weakened (honesty upgrades)
- "factual is the most internally expansive" → **"factual has the highest *pre-norm* hidden amplification; post-norm it does not lead"** (radial caveat). σ_max factual-vs-hallucination is a **tie** (Holm p=0.65), so factual is *co*-leading pre-norm.
- "coordinate-free inversion" → **"output-distribution-intrinsic inversion"**, and only doubly-significant vs reasoning/hallucination (factual-vs-coding on χ_F does not survive Holm, d=−0.26).
- "we characterize hallucination" → **"hallucination-prone is a prompt proxy; we do not detect hallucinations."**
- Mechanistic "factual is output-orthogonal" → **dropped** (does not survive at n=40).

## Claims strengthened (by the completed n=40 runs)
- Two-tier σ_max structure {factual, hallucination-prone} ≫ {reasoning, coding} confirmed on the **exact leading direction** (cross-tier |d|≈1.1, Holm p=0.0006), neutralizing the "random direction understates leading growth" objection.
- **New** significant result: factual has the **lowest stable rank** (most anisotropic Jacobian), significant vs all three (Holm p<0.032) — null at n=20, resolved at n=40.
- Bulk identity (Lemma 1) rel err 3.3%, r=0.80, n=160 (Fig. `bulk_identity`).
- Convergence caveat **resolved**: at n_iter=15 only 5% of prompts exceed 1e-2 residual (was 35% at n_iter=8).

## Figures added (figures/)
`scatter_uncertainty_vs_susceptibility.pdf`, `box_spectral_metrics.pdf`, `bulk_identity.pdf`,
`pre_post_norm_shift.pdf` — from completed Qwen data. `scale_sanity.pdf` is produced by the
chained scale-check and auto-included via `\IfFileExists`.

## What remains before arXiv submission
1. Scale-check completion (chained after the primary sweep) → populate `scale_sanity.pdf`, pick the
   applicable pre-registered reading, add a scale row to the results.
2. SmolLM2 Algorithm-2 sweep completion → optionally add a SmolLM2 row to `tab:alg2`.
3. A final adversarial number-audit of every value in the hardened `.tex` against the CSVs.
4. Compile locally (no LaTeX toolchain on this machine): `pdflatex → bibtex → pdflatex ×2` with
   `arxiv.sty`, `orcidlink` (optional), and `figures/` present.
