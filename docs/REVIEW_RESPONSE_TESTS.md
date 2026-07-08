# Reviewer response — decisive tests (existing data, forward passes only)

Run: `make_confound_tests.py` (Qwen2.5-0.5B, the exact n=40/cat run prompts, n=160).

## Point 4 — is pre-norm σ_max just ‖h_L‖ inflation or prompt surprise?
| Correlate | Pearson r | R² | reading |
|---|---|---|---|
| σ_max vs ‖h_pre‖ (final-token pre-norm) | **+0.38** | 0.15 | moderate, **not** the r>0.8 feared |
| σ_max vs ‖h_post‖ | +0.40 | 0.16 | same |
| σ_max vs prompt perplexity (LM surprise of prefix) | **−0.01** | 0.00 | **refuted** |
| factual>coding σ_max, raw | d=**1.12** | — | — |
| factual>coding σ_max, after removing ‖h_pre‖ | d=**0.80** | — | **survives** |

→ σ_max is **not reducible** to hidden-state norm (r=0.38) nor to prompt surprise (r=−0.01); the
factual>coding contrast survives controlling for ‖h_L‖ (d 1.12→0.80). Point 4 largely **addressed**.

## Point 5 — does the σ_max⊥entropy dissociation exist on the POST-norm (readout) state?
| Model | pre-norm r | post-norm r |
|---|---|---|
| Qwen | −0.02 | **−0.06** |
| SmolLM2 | −0.29 | **+0.06** |

→ The dissociation holds on the state the logits actually see (|r| ≤ 0.06 post-norm), so it is **not**
about a discarded radial component. What IS pre-norm-specific is the *factual-highest ordering*,
which should therefore leave the headline. Point 5 **defused for the dissociation**, valid for the
ordering.

## Point 10 — trivial linear-probe baseline (5-fold CV, 4-way)
| Features | accuracy |
|---|---|
| linear probe on h_pre (896-dim) | **0.99** |
| linear probe on h_post (896-dim) | 0.99 |
| probe on 3 scalars [‖h_pre‖, ‖h_post‖, ppl] | 0.66 |
| chance (majority) | 0.25 |

→ **Confirmed**: categories are trivially linearly decodable from h_L. The spectral observables are
**not** classifiers and must not be sold as separating categories; the value is the correlation-
structure dissociation, which a probe does not provide. Must be reported and the framing adjusted.

## Point 3 — is the output-uncertainty result an artifact of single-token / template contamination?
Mean entropy over the first K=8 *generated* (greedy) tokens vs the single first-token entropy
(Qwen, n=40/cat).

| Metric | outlier | halluc-vs-others \|d\| (fac / cod / rea) |
|---|---|---|
| single-token H | hallucination-prone | 2.5 / 1.0 / 2.9 |
| multi-token H (K=8) | **hallucination-prone (survives)** | 1.5 / 0.9 / 2.1 |

corr(single, multi) = 0.65. → The hallucination-prone entropy outlier **survives** a more robust,
generation-based entropy; single-token overstates the effect (as the reviewer suspected) but the
qualitative finding holds. The output axis should still be framed as a near-manipulation-check
(high-entropy by dataset construction), now with multi-token support.

## Point 9 — LayerNorm architecture (GPT-2, 124M, base/non-instruct, n=16/cat)
Convention verified: `lm_head(hidden_states[-1]) == logits` on both Qwen and GPT-2, so final\_index
−1 is the true post-norm readout state on both.

| Site | σ_max means (fac/cod/rea/hal) | top | r(σ_max, entropy) |
|---|---|---|---|
| pre-norm | 181 / 226 / 180 / 109 | coding | **+0.09** |
| post-norm | 151 / 172 / 124 / 103 | coding | **+0.05** |

Two findings:
1. **The σ_max⊥entropy dissociation replicates on a third architecture** (classic LayerNorm, a base
   non-instruct model): r ≈ 0 both pre- and post-norm. The central claim is not RMSNorm/Qwen-specific.
2. **No clean radial reshuffle**: coding leads at both sites (as on Qwen-1.5B/3B), so there is no
   factual-lead to lose; the pre→post reshuffle observed on Qwen-0.5B cannot be tested here and the
   radial reading stays Qwen-specific. Factual-highest is confirmed absent on GPT-2, consistent with
   1.5B/3B — the factual-corner is specific to the two sub-billion instruct models.

---
# Second hostile review — decisive checks (Qwen n=160, existing + one backward)

## Point 1 (THE deep one) — the direct coupling test
The dissociation compared a *sensitivity* (σ_max) to a *level* (entropy H). The like-for-like
comparison uses the entropy's input-sensitivity ‖∇_{H₀}H(p)‖ (one backward).

| correlation | r |
|---|---|
| σ_max vs entropy H (the paper's "dissociation") | −0.02 |
| **‖∇H‖ vs σ_max (like-for-like)** | **+0.45 (R²=0.20)** |
| ‖∇H‖ vs entropy H | +0.38 |
| χ_F vs ‖∇H‖ | +0.49 |

→ **Confirmed and damaging.** Hidden amplification and the output-entropy input-sensitivity are
*moderately correlated*, not independent. The near-zero σ_max–entropy(value) correlation is an
artifact of comparing a sensitivity to a level. The "distinct/independent axes" framing does **not**
survive the correct test. Blindare-il-null (reviewer option b) is not viable.

## Point 2 — softening
Within-category r all weak (fac +0.10, cod +0.11, rea +0.35, hal −0.07): no strong Simpson. But
pooled r **excluding reasoning = −0.11** (should be the reported number, reasoning is quarantined),
and **distance correlation = 0.20** (vs |Pearson|=0.015): weak *nonlinear* dependence Pearson hides.
"Independent" → "weakly (and partly nonlinearly) correlated."

## Point 5 — Algorithm 2 redundancy
**r(σ_max, λ_bulk) = +0.80** per prompt (stable rank ≈ 4 ⇒ σ_max² ≈ ‖J‖²/4). Leading and bulk are
largely redundant for the categorical contrast; Algorithm 2's added value for the main claim is
overstated.

## Point 7 — subsampling confound: REFUTED
The first-40/category are perfectly length-matched (identical T histograms, Kruskal–Wallis p=1.000).
The main sweep's length control is not reopened.

## Verdict
The correct test (point 1) undermines the central scientific claim. The defensible path is a
**methods/toolkit paper** (reviewer option a): matrix-free spectral profiler + rigorous inference,
with the coupling result reported honestly as a case study — not a "dissociation" findings paper.
