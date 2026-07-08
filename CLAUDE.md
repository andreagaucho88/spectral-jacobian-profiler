# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a paper on hidden-state dynamics in LLMs: whether a model's
sensitivity to small input perturbations (and output-distribution response)
differs across prompt categories (factual / coding / reasoning /
hallucination-prone), once prompt length is controlled for. Six flat Python
files, no package, no framework — each has a self-contained argparse CLI.
Each module has exactly one job.

## Commands

```bash
# 1. Verify the numerics (~30s, no torch, no model download needed). Treat as a hard gate
#    before trusting any spectral output — run.sh runs this first and stops on failure.
python3 spectral.py

# 2. Build a length-matched prompt set for a model (whitespace-proxy mode omitting
#    --model exercises the matching logic without transformers/tokenizer download).
python3 prompts.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --target-per-category 300 --max-per-template 5 --out-dir lengthmatched_prompts

# 3. Legacy Algorithm 1 sweep (baseline, cheap, no autograd; random-direction FTLE).
python3 legacy.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts-json lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json \
    --n-prompts 200 --n-directions 8 --epsilon 1e-3 --out-dir results_legacy

# 4. Spectral Algorithm 2 sweep (the actual science: sigma_max, stable rank, chi_Fisher).
python3 runner.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts-json lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json \
    --n-prompts 60 --k-top 6 --n-probes 32 --n-iter 15 \
    --final-indices -2 -1 --out-dir results_spectral

# 5. Joint statistical analysis (length control, effect sizes, Lemma 1 check, joint axes).
python3 analysis.py \
    --legacy-csv   results_legacy/Qwen__Qwen2.5-0.5B-Instruct/legacy_per_prompt.csv \
    --spectral-csv results_spectral/Qwen__Qwen2.5-0.5B-Instruct/final_index_-2/spectral_per_prompt.csv \
    --model Qwen/Qwen2.5-0.5B-Instruct --out-dir results_analysis/Qwen__Qwen2.5-0.5B-Instruct/pre_norm

# Or the whole pipeline for both default models, idempotent per stage:
bash run.sh          # NOT `source run.sh` — a failed stage shouldn't kill the shell
```

`run.sh` overrides via env vars: `MODELS`, `N_PROMPTS_LM`, `N_PROMPTS_LEGACY`,
`N_PROMPTS_SPECTRAL`, `N_DIRECTIONS_LEGACY`, `K_TOP`, `N_PROBES`, `N_ITER`.
Each stage only runs if its output file is missing, so re-running `run.sh`
after a partial failure resumes rather than redoing work.

`analysis.py` degrades gracefully when only one of `--legacy-csv` /
`--spectral-csv` is given (bulk-consistency and joint-axes tables are skipped
if the frame they need is absent) — useful for testing one half of the
pipeline in isolation.

There is no test suite in the pytest sense; `python3 spectral.py` running its
seven self-tests against dense linear-algebra ground truth *is* the
correctness check for the numerical core, and the whitespace-proxy mode of
`prompts.py` (Kruskal-Wallis p = 1.000 expected) is the check for the
matching logic.

## Architecture

Dependency graph (`common.py` is the only shared module — nothing else
depends on anything but `common` and, for the torch-touching files, each
other):

```
common.py  ──┬─→ prompts.py
             ├─→ spectral.py ──→ runner.py
             ├─→ legacy.py
             └─→ analysis.py
```

- **`common.py`** — the four analysis categories (`CATEGORY_ORDER =
  ("factual", "coding", "reasoning", "hallucination_prone")`,
  `DEFAULT_REFERENCE = "factual"`), stats primitives (Cohen's d, Cliff's
  delta, bootstrap CI, permutation test, Holm correction), template-signature
  extraction (for clustering prompts by template when computing cluster-robust
  SEs), tokenizer wrapper, JSON I/O for prompt sets.
- **`prompts.py`** — length-matched prompt generator. Over-generates diverse
  candidate pools per category, then selects a subset whose token-length
  histograms are made identical across the four categories (so downstream
  metrics can't just be picking up a length confound).
- **`spectral.py`** — Algorithm 2 numerical core: block subspace iteration
  (Rayleigh-Ritz on `J^T J`) for top-k singular values of the
  input→final-token Jacobian `J = d h_{L,T} / d H_0`, unbiased Frobenius-norm
  estimation from sphere probes, Fisher/KL susceptibility on the softmax
  output, and a linearity check tying the legacy finite-difference metric to
  `sigma_max`. Also has the numpy-only self-test (`python3 spectral.py`, no
  torch call) and the torch/HF model adapter used by `runner.py`.
- **`legacy.py`** — Algorithm 1: random-direction finite-time Lyapunov
  exponent (FTLE), kept as a cheap baseline. Also the source of per-prompt
  entropy and top-1 probability (the "output uncertainty" axis referenced in
  `analysis.py`'s joint-axes table).
- **`runner.py`** — sweep driver for Algorithm 2. Iterates
  prompts × final-index × category and writes one CSV per (model,
  final_index) under `results_spectral/<model-slug>/final_index_<i>/`.
  Deliberately refuses to run on MPS: `torch.autograd.functional.jvp`
  double-backward is unstable on MPS for RMSNorm-based models, so it forces
  CPU on Apple silicon (`legacy.py`'s Algorithm 1 doesn't need autograd and
  has no such restriction).
- **`analysis.py`** — joint statistical analysis consuming the CSVs from
  `legacy.py` and `runner.py`: length-vs-metric residual correlations, an
  OLS length-adjustment with cluster-robust SEs (clustered on template id),
  pairwise category contrasts with Holm-adjusted p-values, the Lemma 1
  bulk-consistency check (`ftle_final` vs `lambda_bulk_pred`), and the
  "joint axes" table (the paper's factual-inversion result: contrasting the
  output-uncertainty axis against the spectral/Fisher axis per category).

Model directories/files are consistently keyed by a slug of the HF model
name with `/` replaced by `__` (e.g. `Qwen/Qwen2.5-0.5B-Instruct` →
`Qwen__Qwen2.5-0.5B-Instruct`), produced by `common.slug_of_model`.

### `final_index` semantics

`runner.py`'s sweep is run at `--final-indices -2 -1`: `-2` is the pre-norm
hidden state (comparable to what `legacy.py` tracks), `-1` is post-norm.
Radial growth is present pre-norm and gone post-norm, so pre-vs-post is how
the paper isolates radial from angular growth (see README's "Radial and
angular growth"). `analysis.py` is run once per final_index; only the `-2`
(pre-norm) run is compared against the legacy CSV, since the legacy protocol
tracks pre-norm by definition.

### Output layout

- `lengthmatched_prompts/lengthmatched_prompts__<slug>.json` / `.csv`
- `results_legacy/<slug>/legacy_per_prompt.csv`
- `results_spectral/<slug>/final_index_-2|-1/spectral_per_prompt.csv`
- `results_analysis/<slug>/pre_norm|post_norm/*.csv` (length_by_category,
  length_metric_correlation, ols_length_adjusted_contrasts, pairwise_effects,
  bulk_consistency, joint_axes)

### Cost model (Mac M4 Pro, 24 GB, CPU float32)

Per-prompt spectral cost with defaults (`k_top=6, n_probes=32, n_iter=15`) is
~180 autograd calls per prompt per `final_index`, ~35-60s per prompt per
index for a 0.5B model at `T <= 32`. Default full sweep (60 prompts × 4
categories × 2 final indices) is 5-10h per model. Halving `n_iter` roughly
halves wall-clock and typically keeps the top 2-3 singular values within 1%;
halving `n_probes` doubles Monte-Carlo noise on `stable_rank` (O(1/sqrt(m))).
The legacy sweep is much cheaper: ~1s per prompt per direction, no autograd.

### Repository layout (reorganized for GitHub release)

The six core files + `run*.sh` stay **flat at the repo root** (this is
deliberate — see the README). Auxiliary material was moved:

- `scripts/` — the `make_*.py` paper-control scripts and `analysis_two_axis.py`.
  Each computes its repo root as `Path(__file__).resolve().parent.parent`, so
  they are run from the root as `python3 scripts/<name>.py` and remain
  path-portable. Figures are written to `paper/figures/`.
- `paper/` — `main.tex` (the current paper; self-contained via embedded
  `references.bib`), `arxiv.sty`, `figures/`, and `archive/` (older `.tex`
  drafts).
- `docs/` — the reviewer-facing `.md` analysis writeups (moved out of
  `results_analysis/`, which keeps only CSV/JSON).
- `archive/` — the stale predecessors below. This dir, `paper/archive/`,
  `*.log`, `backup_n20_*/`, and `.claude/` are **git-ignored** (kept locally,
  not published).

### Stale files — do not use as reference

- **`archive/run_1.sh`** and **`archive/spectral_susceptibility.py`** are an
  earlier iteration of this pipeline. `run_1.sh` invokes filenames that don't
  exist in this repo (`lengthmatched_prompt_generator.py`,
  `experiment_validation_runner_new.py`, `robustness_length_control.py`) and
  writes to `results_lengthmatched/`/`robustness_out/` instead of the current
  `results_legacy/` / `results_spectral/` / `results_analysis/` layout.
  `spectral_susceptibility.py` is a self-contained predecessor of the
  `spectral.py` + `runner.py` split (same Algorithm 2 core, older
  docstring/framing, combined self-test+sweep CLI). Treat `run.sh` and the
  six flat core files as the current source of truth; don't extend or fix bugs
  in the archived files.
