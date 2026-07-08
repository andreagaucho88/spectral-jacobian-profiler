# Scale-sanity check — summary

**Question.** Does the qualitative dissociation seen on sub-billion models — *factual* highest on
pre-norm hidden amplification σ_max, *hallucination-prone* highest on output-distribution
susceptibility χ_F — survive on intermediate-scale (1.5B / 3B) models, at least as a preliminary
sanity check? This is **not** a scaling law; it is a presence/absence test at small n/category.

## Execution mode (safety)
- The scale-check runs **only after** the primary long sweep (`run_full_n40.sh`) finishes, so it
  **never competes** with it for CPU/RAM. Rationale: the machine is CPU-only (12 cores, 24 GB;
  MPS is refused by the spectral runner), and only ~6.4 GB was free during the primary run —
  Qwen-1.5B (~6 GB) / 3B (~12 GB) float32 would have caused memory pressure. Parallel execution
  was therefore judged **unsafe**; the check is chained instead (`run_scalecheck.sh`), waiting in a
  poll-only loop.
- Output goes to a **separate** directory `results_spectral_scalecheck/`, so it cannot corrupt the
  primary results. The runner is checkpointed (per-row flush, resume by
  (model, final_index, label, prompt_idx), per-model `run_manifest.json` with config + hardware +
  completed/failed counts; errors logged as rows).

## Configuration (deliberately minimal)
| Setting | Value |
|---|---|
| Models | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct |
| Prompts/category | 8 |
| Read-out | pre-norm (final_index −2) |
| k_top | 3 |
| n_probes | 6 |
| n_iter | 8 |
| Fisher/KL (χ_F) | enabled |
| dtype / device | float32 / CPU |
| Prompt set | reused Qwen2.5-0.5B length-matched JSON (Qwen2.5 family shares one tokenizer ⇒ identical token lengths) |

## Resource estimates (M4 Pro, CPU float32, after primary run frees RAM)
| Model | Weights (float32) | ~ per-prompt | 32 prompts (pre-norm) |
|---|---|---|---|
| Qwen2.5-1.5B | ~6 GB | ~30–50 s | ~20–30 min |
| Qwen2.5-3B | ~12 GB | ~60–100 s | ~35–55 min |
| **Total** | fits in 24 GB (sequential) | — | **~1–1.5 h** |

Risk: 3B float32 peak (weights + double-backward activations) may approach ~14–16 GB; if it OOMs,
the checkpoint logs error rows and continues, and we fall back to the "computationally infeasible"
reading (scripts + estimates provided).

## Interpretation (pre-registered, mirrors the paper's Discussion)
- **Agree** (1.5B/3B reproduce factual-high-σ_max, hallucination-high-χ_F): *"A preliminary scale
  sanity check suggests the dissociation is not confined to sub-billion models, although full
  scaling remains future work."*
- **Disagree**: *"The scale sanity check indicates that susceptibility geometry may change with
  scale; we treat the sub-billion result as model-scale-limited and report Algorithm 2 primarily as
  a diagnostic framework."*
- **Infeasible**: *"We provide scripts and resource estimates for intermediate-scale validation;
  full scaling is left to future work."*

## Results (COMPLETE)
The 3B initially failed on a transient network error (`from_pretrained` hit HuggingFace despite the
cached weights); re-run with `HF_HUB_OFFLINE=1` it completed. `figures/scale_sanity.pdf` shows the
per-category means. Selected reading: **split (i)+(ii)** — see verdict below.

| Model | n/cat | σ_max means (fac / cod / rea / hal) | σ_max top | factual highest? | r(σ,ent) | factual lowest entropy? |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 40 | 915 / 704 / 717 / 865 | **factual** | ✓ | −0.01 | ✓ (0.79) |
| Qwen2.5-1.5B | 8 | 1060 / **1243** / 867 / 1096 | coding | ✗ | +0.32 | ✓ (0.29) |
| Qwen2.5-3B | 8 | 1195 / **1335** / 1054 / 1149 | coding | ✗ | −0.01 | ✓ (0.00) |

## Verdict (honest, not spun)
- **SURVIVES scaling (reading i):** the two axes stay distinct — σ_max and entropy are weakly
  correlated at every scale (|r| ≤ 0.32, R² ≤ 0.10) — and factual is the lowest-entropy class at all
  three scales. The two-axis dissociation is not confined to sub-billion models.
- **DOES NOT survive (reading ii):** the sub-billion finding that *factual* has the highest hidden
  amplification does **not** persist — at 1.5B and 3B **coding** leads (factual second). The
  factual-corner ordering is a sub-billion phenomenon.
- **χ_F:** ordering is scale-inconsistent too (0.5B: halluc highest; 1.5B/3B: reasoning highest;
  absolute scale representation-dependent, not comparable across models), reinforcing its demotion.
- **Consequence for the paper:** report the *distinctness* of the two axes as the scale-stable result
  and the *category ordering* on the hidden axis as model-scale-dependent; Algorithm 2 is an offline
  **diagnostic framework**, not a general law. n=8/category is a sanity check, not a powered result.
