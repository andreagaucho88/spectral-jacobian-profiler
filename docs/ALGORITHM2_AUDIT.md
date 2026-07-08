# Algorithm‑2 (spectral / exact‑Jacobian) audit against `main_d.tex`

**Purpose.** Run the estimator's self‑test, re‑derive every spectral result directly
from the CSVs, adversarially stress each interpretation, and state — with scientific
precision — what the Algorithm‑2 evidence does and does not say about the paper's claims.

**Method.** Numerical self‑test (`python3 spectral.py`, all checks reproduced); independent
re‑derivation of every mean / effect size / Holm‑adjusted permutation *p* from
`results_spectral/…/final_index_-2/` and `results_analysis/…/pre_norm/`; a 4‑cluster audit
with a per‑claim adversarial refutation pass (43 agents, 0 errors); and a manual
verification of the two most consequential caveats (convergence residuals, bulk‑metric
ordering) against `spectral_per_prompt.csv`.

---

## 0. Scope banner — applies to every spectral result below

The entire spectral evidence base is:

- **one model** — Qwen2.5‑0.5B‑Instruct;
- **one read‑out site** — `final_index = −2` (**pre‑norm**);
- **n = 20 prompts / category** (80 rows), length‑matched to 34–36 tokens.

There is **no SmolLM2 spectral run** and **no post‑norm (`final_index = −1`) run** (those
directories are empty scaffolds). The significance bar throughout is the **Holm‑adjusted
permutation p**. Nothing here is "established"; the strongest attainable verdict is
**"confirmed on Qwen pre‑norm at n = 20."** This matches — and does not exceed — the paper's
own Limitations/Future‑Work section, which asks for exactly this exact‑JVP validation and
flags cross‑model generality as open.

---

## 1. Numerical validity — the estimator is correct; the *deployed* estimates are converged to ~1e‑2

`python3 spectral.py` → **all 7 checks PASS**, every number reproduced:

| Check | Result | Guarantees |
|---|---|---|
| Block subspace iter vs dense SVD | head(3) rel err **7.1e‑16**, tail 4.9e‑6 | σ_max / λ_max recovery is exact where ground truth exists |
| Unbiased Frobenius sphere estimator | 2.25 % MC error; Jensen "(mean‖·‖)²" variant −22 % | stable_rank uses the unbiased form |
| Stable‑rank identity a²=n/r_stable | rel err **6.8e‑15** | anisotropy is a redundancy check, not an independent DoF |
| Fisher ½·dzᵀF dz vs stable KL | rel err **8.1e‑6** | the ½ factor (χ_F) is correct |
| χ_F block‑iter vs dense | rel err **1.2e‑15** | leading Fisher operator is exact |
| Linearity along v_lead | rel err 4.8e‑9 (small ε) | finite‑difference ↔ σ_max tie is real |

**Honest limit.** The self‑test proves *implementation correctness on a well‑separated
synthetic matrix at heavy settings*. It does **not** prove the deployed Qwen estimates are
machine‑precise. The production run used `n_iter_subspace = 8` and `n_probes = 32`:
σ_max `conv_residual` has **median 6.0e‑3, 35 % of prompts > 1e‑2** (max 5.5e‑2), and
`fisher_conv_residual` reaches 11.9 %. So σ_max/λ_max are converged to **~1e‑2, not 1e‑16**.
Consequence for interpretation: the **~21 % cross‑tier σ_max gap dwarfs this noise (safe)**,
but the **within‑tier factual≈hallucination difference (0.4 %) is at/below the noise floor —
which *reinforces* reading it as a statistical tie, not a real ordering.** (The `run.sh`
default is `n_iter = 15`; re‑running the sweep at ≥15 iters would tighten the residuals and
is the cheapest robustness upgrade.)

---

## 2. Verdict per paper claim (C1–C5) under Algorithm‑2 evidence

| Claim | Verdict | Key numbers (Qwen, pre‑norm, n=20) |
|---|---|---|
| **C1** output uncertainty (H(p), p_max) isolates hallucination | **Not tested by Algorithm 2** — the spectral corpus does not carry that axis; C1 remains a legacy‑metric result. | — |
| **C2** hidden‑state response is prompt‑conditioned, not a length artifact; factual largest at matched length | **Confirmed, refined.** Confirmed on the *exact leading direction*: identical length support (mean 35.0, sd 0.73), overall σ_max–length r = 0.063, OLS length shrinkage ~1e‑13. **Refinement:** "factual largest" is a *two‑tier* statement — factual (σ_max 828.1) is significantly above coding (690.5; d=1.02, Holm 0.0093) and reasoning (680.7; d=1.17, Holm 0.0045) but **ties** hallucination (824.7; d=0.022, Holm 1.0). Factual is *co‑leading*, not a sole maximum — consistent with the paper's own Qwen "d=0.09, n.s." |
| **C3** factual inversion; two axes distinct | **Confirmed and extended (see §3a).** On one 80‑prompt set factual has the highest hidden σ_max *and* the lowest coordinate‑free output response χ_F^max (804.6). Sign‑flip doubly Holm‑significant vs coding/reasoning. **Framing fix:** only the output axis is coordinate‑free; σ_max is representation‑dependent. Inversion is *one‑sided* vs hallucination (hidden tie). |
| **C4** two‑tier {factual, halluc high} vs {reasoning, coding low}; fine detail model‑dependent | **Confirmed — the strongest, most robust spectral result.** On σ_max all 4 cross‑tier contrasts survive Holm (\|d\|≈1.02–1.26; Holm 0.0018–0.0108); both within‑tier contrasts null (Holm 1.0). λ_max reproduces it. The factual↔halluc sign even flips between σ_max and λ_max (a mean‑of‑log artifact) — vindicating the paper's "fine detail is model‑dependent." |
| **C5** single random direction ≈ bulk Jacobian statistic (Lemma 1) | **Confirmed (central tendency).** ftle_final vs exact λ_bulk_pred: pooled means 0.0920 vs 0.0935 (1.7 %), median per‑prompt rel err 2.8 %, per‑category rel err 1.55–3.30 %. Validates that the sampled pipeline realizes the bulk statistic → the flat sampled contrast is a *prediction*, not a null. **Caveats:** systematic Jensen offset (λ_bulk > ftle in 61 % of prompts); per‑prompt Pearson **r = 0.80, not ~0.9** (README overstates); rank tracking is central‑tendency only. |

**One contradiction to surface (C4 fine detail).** The *exact bulk* metric λ_bulk_pred ranks
**hallucination (0.0982) numerically above factual (0.0955)** — inverting the legacy
ftle_final order (factual highest). But both orderings are **non‑significant**
(bulk factual‑vs‑halluc Holm 0.195; legacy Holm 0.196). This **undercuts any "factual is the
strict bulk maximum" phrasing without establishing hallucination‑as‑max**; the robust
two‑tier structure survives in the bulk metric (cross‑tier Holm ≤ 0.041). Note σ_max (the
leading value) still ranks factual numerically first. Net: at matched length on Qwen,
**factual is co‑max of hidden response on the leading and legacy metrics, but not on the
exact bulk metric** — exactly the model‑dependent fine structure the paper declines to claim.

---

## 3. Two results that go *beyond* the paper

### (a) A within‑experiment, partly coordinate‑free "factual inversion"

On the same 80 prompts, factual has the **highest** leading hidden amplification σ_max (828.1)
and the **lowest** maximal coordinate‑free output (Fisher/KL) response
χ_F^max (804.6 < coding 1340.9 < reasoning 1613.2 < hallucination 2450.8). All three χ_F^max
contrasts vs factual survive Holm (coding 0.039, reasoning 0.008, hallucination 0.0012;
\|Cliff δ\| 0.54–0.73) and are robust to length adjustment.

*Why this matters:* it realizes the paper's Future‑Work item ("correlating the hidden‑state
response with the change it induces in the final logits") **on a single matched prompt set**,
removing the paper's own "two different sets for the two axes" limitation, and it makes the
output axis **coordinate‑free** (KL is reparameterization‑invariant).

*Precise strength / limits:*
- **Partly**, not fully, coordinate‑free: χ_F is invariant, but σ_max is representation‑dependent.
- **Doubly significant only vs coding and reasoning.** Vs hallucination the inversion is
  *one‑sided*: same hidden‑stretch magnitude (σ_max tie, p=0.945), **~3× larger output
  response** (χ_F ratio 3.05, Holm 0.0012).
- Output ordering only partly resolved: coding‑vs‑reasoning is within noise (Holm 0.216).
- High dispersion (χ_F^max CV 42–88 %), single model, pre‑norm, n=20.

### (b) A mechanistic account — does the hidden‑stretch direction move the logits?

For hallucination the hidden leading direction moves the output far more than for factual:
**χ_F‑along‑lead 1646 vs 294** (≈5.6× mean, ~11× median; d=−1.38, Holm 0.0006; only 1/20
factual prompts exceed the hallucination median), with higher hidden↔output alignment
(**lead_alignment 0.618 vs 0.386**; d=−0.93, Holm 0.020).

*Precise strength / limits:*
- Not purely an alignment effect — hallucination Jacobians are globally more output‑sensitive
  (χ_F^max ~3×); the alignment‑specific component is a more modest ~1.6×.
- **"Factual is uniquely output‑orthogonal" is NOT supported.** On lead_alignment factual
  (0.386) is *not* the lowest — reasoning is (0.192), and factual is *significantly higher*
  than reasoning (d=+0.97, Holm 0.013). On χ_F‑along‑lead factual is the numerical minimum but
  ties reasoning (Holm 0.731). The **only** ordering robust on *both* mechanistic metrics is
  **hallucination > factual**.
- lead_alignment is a direction overlap in [0,1] (means 0.19–0.62; above high‑dim chance ~1/√d
  but far from 1): the leading hidden direction is a *substantial sub‑dominant* logit‑mover,
  not the dominant one. v_lead lives in the pre‑norm basis → not fully coordinate‑free.

---

## 4. What is NOT supported (must not be asserted)

- **stable_rank (anisotropy):** 0 / 6 contrasts survive Holm (min p=0.41). No category
  structure — a failure‑to‑reject at n=20 (factual‑coding d=−0.60 is medium but unresolved),
  not evidence of equal anisotropy.
- **spectral_gap:** 3 / 6 survive (factual separates from both high‑gap classes; hallucination
  from neither) — a *partial* discriminant, not a clean two‑tier separator. Factual has the
  *smallest* top‑spectrum gap (flatter leading spectrum) — a new, single‑model observation.
- **χ_F^max "hallucination is THE max":** Holm‑robust on only **one** of three output metrics
  (χ_F‑along‑lead); on χ_F^max the halluc‑vs‑reasoning gap (Holm 0.080) and on lead_alignment
  the halluc lead (Holm 0.13) are within noise.
- **Cross‑model / post‑norm / representation invariance:** untested. σ_max is
  representation‑dependent by construction; not shown to flip under another read‑out, but not
  shown invariant.

**Quantitative honesty items.**
- Bulk correlation is **r = 0.80, not ~0.9** (README wording is optimistic); Lemma 1 is an
  *absolute‑agreement* claim (rel err 1.5–3.2 % within every category), which holds — the
  modest r is variance‑range restriction, not a failure.
- λ_max magnitudes are tiny by construction (~0.008/layer cross‑tier) but compound over depth
  23 to the ~21 % σ_max gap — "minuscule" is not a fair reading.
- **Reasoning is the least‑clean sample:** within‑category length correlation persists even in
  the 34–36‑token band (σ_max r=0.62). It does **not** confound the between‑category ranking
  (length identically distributed; OLS shrinkage ~1e‑13) but pins contrasts to the narrow
  token regime and flags reasoning specifically.

---

## 5. Bottom line

The Algorithm‑2 (exact‑Jacobian, leading‑direction) evidence **strengthens the paper's core
thesis on Qwen pre‑norm** without overturning any careful claim:

1. The prompt‑conditioned **two‑tier** hidden‑state structure {factual, hallucination} ≫
   {coding, reasoning} is **present in the exact leading singular direction** — so the paper's
   contrast is not an artifact of averaging over a random direction (it neutralizes the
   paper's own "random direction understates leading growth" limitation).
2. Lemma 1 is **borne out**: the sampled pipeline realizes the bulk Jacobian statistic
   (rel err 2.8 %).
3. The distinct‑axes claim is **extended to a coordinate‑free output measure** (Fisher/KL) on
   a single matched set, closing the paper's own future‑work loop — with the honest refinement
   that the inversion is *two‑sided vs coding/reasoning* and *one‑sided vs hallucination*
   (same hidden magnitude, ~3× output response), and that factual is *co‑*max, not sole max.

All of this is **confined to Qwen pre‑norm at n=20**. The decisive next steps — cheap relative
to their evidential value — are: (i) re‑run the sweep at `n_iter ≥ 15` to drop convergence
residuals below 1e‑2; (ii) run the SmolLM2 spectral sweep; (iii) run the post‑norm
(`final_index = −1`) sweep to test representation‑dependence. Only then can the two‑tier
structure and the Fisher inversion be called *established* rather than *confirmed on one model*.
