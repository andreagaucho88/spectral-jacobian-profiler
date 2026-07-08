# Equivalence testing (TOST) for the "tie" claims

**Why.** The paper's "factual co-leading" / "two-tier" language treats non-significant contrasts as
*ties*. A non-significant difference is **not** evidence of equivalence (Gelman–Stern). We run
two one-sided tests (TOST): the two groups are declared statistically equivalent only if the 90% CI
of the difference lies entirely within an equivalence margin. Margins are set in standardized units
(Cohen d), at ±0.5 SD (medium, half the ~1.1 cross-tier effect we care about) and a stricter ±0.35 SD.
Data: Qwen2.5-0.5B pre-norm, n=40/category.

| Contrast (σ_max) | Cohen d | margin | p(TOST) | 90% CI of diff | equivalent? |
|---|---|---|---|---|---|
| factual vs hallucination-prone | +0.22 | ±0.50 SD (±110) | 0.29 | [−33, +131] | **NO** |
| factual vs hallucination-prone | +0.22 | ±0.35 SD (±77) | 0.29 | [−33, +131] | **NO** |
| coding vs reasoning | −0.16 | ±0.50 SD (±42) | 0.065 | [−44, +18] | **NO** |
| coding vs reasoning | −0.16 | ±0.35 SD (±29) | 0.196 | [−44, +18] | **NO** |

## Interpretation
- **None of the "ties" is a demonstrated equivalence.** For factual vs hallucination-prone the 90% CI
  ([−33, +131]) is wide and includes both zero *and* effects larger than half the cross-tier gap — so
  the data are **underpowered**: we can claim neither a difference (permutation n.s.) nor equivalence
  (TOST n.s.).
- The n=20→n=40 coding↔reasoning sign flip, previously offered as "confirmation of a tie", is equally
  consistent with an underpowered real effect. It is **not** evidence of equivalence.

## Effect on paper claims
Replace "factual is co-leading (tie with hallucination-prone)" and "coding ≈ reasoning" with the
honest wording: **"statistically indistinguishable at n=40, and underpowered to establish equivalence
(TOST not significant at ±0.5 SD)."** The paper should not assert equivalence anywhere; where it needs
a within-tier statement, it should say the contrast is unresolved.
