# Spectral Jacobian Profiler

**Matrix-free spectral profiling of prompt-conditioned input→output Jacobians in Transformer language models.**

This repository is the reference implementation and full artifact for the paper
*Matrix-Free Spectral Profiling of Prompt-Conditioned Input→Output Jacobians in
Transformer Language Models* (`paper/main.tex`). It provides a small, dependency-light
toolkit that, for a single prompt, recovers spectral summaries of the exact
input→final-token Jacobian **J = ∂h<sub>L,T</sub> / ∂H₀** without ever materializing it —
using only Jacobian/vector products (double-backward autograd) — together with a rigorous,
length-controlled, cluster-aware statistical protocol for comparing those summaries across
prompt categories.

The core is **six flat Python files, no package, no framework**; each has a self-contained
`argparse` CLI and exactly one job.

---

## What it measures

For each prompt, from JVP/VJP oracles alone:

| Observable | Meaning |
|---|---|
| `sigma_max` | Leading singular value of `J` (top input→hidden amplification), via block subspace iteration (Rayleigh–Ritz on `JᵀJ`). |
| `‖J‖_F`, `lambda_bulk` | Unbiased Frobenius norm (Hutchinson sphere estimator) and the direction-averaged bulk response. |
| `stable_rank` | `‖J‖_F² / σ_max²` — Jacobian anisotropy. |
| `chi_fisher_max` | Fisher/KL output susceptibility: how much an input perturbation moves the **output distribution** (representation-independent on the output side). |
| `lambda_ftle` | Cheap sampled-direction finite-depth response (Algorithm 1 baseline; no autograd). |
| `entropy`, `top1_prob` | Output-distribution level statistics. |

A numpy-only **self-test** validates the numerical core against dense linear-algebra ground
truth (`python3 spectral.py`, no torch, no model download).

## What the case study found (and what it did *not*)

The paper is deliberately framed as a **methods contribution with a cautionary case study**,
not a "dissociation" findings paper. The reproducible chain:

- The leading amplification `sigma_max` is only **weakly related to the output-entropy
  *level*** (`R² ≤ 0.10` at every model and scale tested, sign unstable) — but that compares a
  hidden-state *sensitivity* to an output *level*, which is the wrong comparison.
- The **like-for-like** comparison against the entropy's *input sensitivity* `‖∇_{H₀}H‖`
  gives `r = 0.45`, which is **mostly shared Jacobian magnitude**: partialling out the bulk
  norm drops it to a **partial correlation of 0.10–0.30** across three models (residual 95% CI
  **includes zero on two of three**).
- A **collinearity-free geometric check** (`cos∠(∇_{H₀}H, v_lead)` = 0.41) against a
  **position-matched null** — random directions carrying ∇H's per-position energy profile
  (null = 0.008; ~53× against it, 97.5% of prompts exceed their own null) — shows the two input
  directions are **partially aligned** (`cos² = 0.24`): a genuine within-position directional
  overlap, **not** a positional-support artifact, but modest.
- `sigma_max` is **strongly correlated with the bulk norm** for the categorical contrast
  (`r = 0.80`, i.e. 64% shared variance), and a trivial linear probe separates the categories
  at **99%** — so the spectral observables are *characterizations of response geometry, not
  classifiers*.

**Lesson:** naive single-direction / level summaries of Jacobian geometry are easy to
over-read; the correct like-for-like, magnitude-controlled comparisons must be made. The
profiler is the contribution; the case study is how to read it without over-claiming.

---

## Repository layout

```
.
├── common.py          # categories, stats primitives, template signatures, tokenizer wrapper, I/O
├── prompts.py         # length-matched prompt generator (identical token-length histograms/category)
├── spectral.py        # Algorithm 2 numerical core + torch/HF adapter + numpy self-test
├── legacy.py          # Algorithm 1 (sampled-direction FTLE baseline); also entropy / top1_prob
├── runner.py          # checkpointed/resumable Algorithm 2 sweep driver (one CSV per model×final_index)
├── analysis.py        # joint stats: length control, cluster-robust OLS, effect sizes, bulk check
├── run.sh             # end-to-end pipeline (self-test → prompts → legacy → spectral → analysis)
├── run_full_n40.sh    # detached full n=40 sweep for both primary models
├── run_scalecheck.sh  # 1.5B/3B intermediate-scale sanity check (separate output dir)
│
├── scripts/           # auxiliary paper controls (each reads/writes the result dirs below)
│   ├── make_figures.py          make_figures_twoaxis.py     # → paper/figures/*.pdf
│   ├── make_coupling_extra.py   make_gradH_test.py          # partial-corr / bulk-driver / ‖∇H‖ tests
│   ├── make_alignment_test.py                               # cos∠(∇H, v_lead) geometric alignment
│   ├── make_confound_tests.py   make_multitoken_entropy.py  # norm/ppl controls, linear probe, multi-token H
│   ├── make_coupling_tests.py                               # (superseded by make_coupling_extra.py)
│   └── analysis_two_axis.py                                 # comprehensive single-model analysis
│
├── paper/
│   ├── main.tex       # the paper (self-contained: embedded references.bib via filecontents)
│   ├── arxiv.sty      # "A Preprint" style
│   └── figures/       # generated PDFs (by scripts/make_figures*.py)
│
├── docs/              # per-analysis reviewer-facing writeups (.md)
│
├── lengthmatched_prompts/          # per-model length-matched prompt sets (.json / .csv)
├── results_legacy/                 # Algorithm 1 per-prompt CSVs, per model
├── results_spectral/               # Algorithm 2 per-prompt CSVs, per model × final_index
├── results_spectral_gpt2/          # GPT-2 (LayerNorm) profiling
├── results_spectral_scalecheck/    # Qwen 1.5B / 3B sanity check
├── results_analysis/               # joint-analysis CSVs/JSON + per-model subdirs
│
├── requirements.txt   LICENSE
```

Every model directory/file is keyed by a slug of the HF model name with `/` → `__`
(e.g. `Qwen/Qwen2.5-0.5B-Instruct` → `Qwen__Qwen2.5-0.5B-Instruct`).

`final_index` convention: `-2` = **pre-norm** hidden state (matches the Algorithm-1 tracking),
`-1` = **post-norm** (the actual input to the logit head). Radial growth is present pre-norm
and removed post-norm, so pre-vs-post isolates radial from angular growth.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Models are pulled from the Hugging Face Hub on first use (cached thereafter). The scripts set
`HF_HUB_OFFLINE=1` where a cached model suffices.

## Quick start

```bash
# 1. Validate the numerics (~30 s, no torch, no model download). Hard gate before trusting output.
python3 spectral.py

# 2. Build a length-matched prompt set for a model.
python3 prompts.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --target-per-category 300 --max-per-template 5 --out-dir lengthmatched_prompts

# 3. Algorithm 1 baseline sweep (cheap, no autograd).
python3 legacy.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts-json lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json \
    --n-prompts 200 --n-directions 8 --epsilon 1e-3 --out-dir results_legacy

# 4. Algorithm 2 spectral sweep (the science: sigma_max, stable rank, chi_Fisher).
python3 runner.py --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts-json lengthmatched_prompts/lengthmatched_prompts__Qwen__Qwen2.5-0.5B-Instruct.json \
    --n-prompts 60 --k-top 6 --n-probes 32 --n-iter 15 \
    --final-indices -2 -1 --out-dir results_spectral

# 5. Joint statistical analysis (length control, effect sizes, bulk consistency).
python3 analysis.py \
    --legacy-csv   results_legacy/Qwen__Qwen2.5-0.5B-Instruct/legacy_per_prompt.csv \
    --spectral-csv results_spectral/Qwen__Qwen2.5-0.5B-Instruct/final_index_-2/spectral_per_prompt.csv \
    --model Qwen/Qwen2.5-0.5B-Instruct --out-dir results_analysis/Qwen__Qwen2.5-0.5B-Instruct/pre_norm

# Or the whole pipeline for both default models (idempotent per stage):
bash run.sh
```

`run.sh` is overridable via env vars: `MODELS`, `N_PROMPTS_LM`, `N_PROMPTS_LEGACY`,
`N_PROMPTS_SPECTRAL`, `N_DIRECTIONS_LEGACY`, `K_TOP`, `N_PROBES`, `N_ITER`. Each stage runs
only if its output is missing, so re-running resumes rather than redoing work. `runner.py` is
checkpointed (per-row flush; resume by `(model, final_index, category, prompt_idx)`; a
`run_manifest.json` per run).

### Reproducing the paper's control tests and figures

```bash
python3 scripts/make_coupling_extra.py    # partial correlation r(‖∇H‖,σ|bulk) + bulk-driver, 3 models
python3 scripts/make_alignment_test.py    # cos∠(∇H, v_lead) geometric alignment (Qwen, CPU)
python3 scripts/make_confound_tests.py    # σ_max vs ‖h‖ / perplexity; linear probe
python3 scripts/make_figures.py           # → paper/figures/*.pdf
python3 scripts/make_figures_twoaxis.py
```

---

## Cost model (Mac M4 Pro, 24 GB, CPU float32)

`torch.autograd.functional.jvp`'s double-backward is unstable on MPS for RMSNorm models, so
`runner.py` **forces CPU on Apple silicon** (Algorithm 1 needs no autograd and has no such
restriction). With defaults (`k_top=6, n_probes=32, n_iter=15`), Algorithm 2 costs ~180
autograd calls per prompt per `final_index`, ≈35–110 s per prompt per index for a 0.5B model at
`T ≤ 32`. A full sweep (60 prompts × 4 categories × 2 indices) is ~5–10 h per model. Halving
`n_iter` ≈ halves wall-clock and usually keeps the top 2–3 singular values within 1%; halving
`n_probes` doubles the Monte-Carlo noise on `stable_rank` (O(1/√m)). The legacy sweep is ~1 s
per prompt per direction.

## Self-tests

- `python3 spectral.py` — seven checks against dense ground truth: block iteration recovers
  well-separated singular values to machine precision; the sphere estimator of `‖J‖_F²` is
  unbiased within Monte-Carlo error; the anisotropy identity `a² = n / stable_rank` holds to
  floating point; the ½ factor of the Fisher form matches a stable log-sum-exp KL; the
  linearity check along `v_lead` agrees with `sigma_max` as ε→0. Prints `SELF-TEST: ALL PASS`.
- `python3 prompts.py` (whitespace-proxy mode, no `--model`) exercises the length-matching
  logic without transformers; the Kruskal–Wallis check returns `p = 1.000` by construction.
- `python3 analysis.py` degrades gracefully when only one of `--legacy-csv` / `--spectral-csv`
  is provided.

## Paper

`paper/main.tex` is self-contained (the bibliography is embedded via `filecontents*`, and
`arxiv.sty` ships with it). Build with any TeX distribution:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Citation

```bibtex
@misc{gaudiello2026spectral,
  title  = {Matrix-Free Spectral Profiling of Prompt-Conditioned
            Input-to-Output Jacobians in Transformer Language Models},
  author = {Gaudiello, Andrea},
  year   = {2026},
  note   = {\url{https://github.com/andreagaucho88/spectral-jacobian-profiler}}
}
```

## License

MIT — see [LICENSE](LICENSE).
