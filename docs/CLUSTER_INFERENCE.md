# Cluster-aware inference (template_id)

**Why.** Templated prompts within a category are near-duplicates. A prompt-level permutation test
treats them as independent, overstating significance. We re-test with the **template** as the
cluster: cluster bootstrap (resample templates with replacement) and block permutation (permute
whole templates between groups). Template id = category + masked template signature
(`common.template_signature`). Data: Qwen2.5-0.5B pre-norm, n=40/category.

## Templates per category (the key exposure)
| Category | prompts | **distinct templates** | prompts/template |
|---|---|---|---|
| factual | 40 | 39 | ~1.0 |
| coding | 40 | 40 | 1.0 |
| hallucination-prone | 40 | 28 | ~1.4 |
| **reasoning** | 40 | **5** | **~8** |

**Reasoning has only 5 templates.** Its effective sample size is closer to 5 than 40, so any contrast
involving reasoning is far weaker under cluster inference than prompt-level tests suggest.

## σ_max contrasts: prompt-level vs cluster-robust
| Contrast | Cohen d | cluster-boot 95% CI (mean diff) | cluster perm p | cluster-robust? |
|---|---|---|---|---|
| **factual − coding** | +1.12 | **[+133, +297]** (excludes 0) | **0.000** | **YES** |
| factual − reasoning | +1.08 | [+103, +301] (excludes 0) | **0.109** | **NO** (5 templates) |
| factual − hallucination | +0.22 | [−44, +154] (incl. 0) | 0.348 | tie |
| coding − reasoning | −0.16 | [−76, +58] (incl. 0) | 0.791 | tie |

χ_F cluster perm p: factual−hallucination **0.000** (but see confound doc — entropy-driven);
factual−reasoning **0.129** (not robust, 5 templates).

## The two-axis independence is cluster-robust
r(σ_max, entropy) = −0.015, **cluster-bootstrap 95% CI [−0.21, +0.17]** — tightly around zero even
after accounting for template clustering. The core claim survives.

## Conclusions
- **Cluster-robust:** (i) the two-axis independence (σ_max ⟂ entropy), and (ii) factual > coding on
  pre-norm σ_max (d=1.12, cluster perm p=0.000). These are the load-bearing claims and they hold.
- **NOT cluster-robust:** factual > reasoning on σ_max (cluster perm p=0.11) and any
  reasoning-involving separation — because reasoning has only 5 templates. The "two-tier" structure
  therefore weakens to **"factual is robustly above coding; the difference from the low-diversity
  reasoning class is large but not cluster-robust."**
- **Ties** (factual≈hallucination, coding≈reasoning) remain non-significant under clustering too, but
  see `EQUIVALENCE_TESTS.md`: non-significance is not equivalence.
- **Disclosure for the paper:** report templates/category and use cluster permutation (or the
  cluster-robust OLS p) as the headline bar, not prompt-level permutation alone.
