#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite-wide primitives.

Everything here is deliberately zero-dependency beyond numpy / pandas (and,
opt-in, transformers for the real tokenizer). Nothing in this file uses torch
or opens a model. Every module in the suite imports from here; nothing here
imports from any other module in the suite.

Contents
--------
Categories & I/O
    CATEGORY_ORDER, CATEGORY_PAIRS, DEFAULT_REFERENCE
    load_prompts_json / save_prompts_json
    slug_of_model
Prompt cleanup
    template_signature   -- collapse near-duplicate templated prompts
    make_length_fn       -- real tokenizer (with chat template) or whitespace
Statistics
    cohens_d, cliffs_delta         -- effect sizes
    bootstrap_mean_ci              -- 95 pct BCa via percentile bootstrap
    permutation_p_value            -- two-sided, equal-mean null
    holm_correct                   -- family-wise correction (Holm-Bonferroni)
    pairwise_report                -- all six pairs, one metric, a tidy frame
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:                                       # pragma: no cover
    AutoTokenizer = None


# ============================================================
# Categories
# ============================================================

CATEGORY_ORDER: Tuple[str, ...] = (
    "factual", "coding", "reasoning", "hallucination_prone",
)
CATEGORY_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    (a, b) for i, a in enumerate(CATEGORY_ORDER) for b in CATEGORY_ORDER[i + 1:]
)
DEFAULT_REFERENCE = "factual"


# ============================================================
# I/O
# ============================================================

def slug_of_model(name: Optional[str]) -> str:
    """File-system-safe slug: 'Qwen/Qwen2.5-0.5B-Instruct' -> 'Qwen__Qwen2.5-0.5B-Instruct'."""
    if not name:
        return "whitespace_proxy"
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


def save_prompts_json(prompts: Dict[str, List[str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    return path


def load_prompts_json(path: str | Path) -> Dict[str, List[str]]:
    with open(path) as f:
        data = json.load(f)
    unknown = set(data) - set(CATEGORY_ORDER)
    if unknown:
        raise ValueError(f"Unknown category keys in {path}: {sorted(unknown)}")
    return {k: list(map(str, data[k])) for k in CATEGORY_ORDER if k in data}


# ============================================================
# Prompt cleanup
# ============================================================

def template_signature(prompt: str) -> str:
    """Collapse a templated prompt to its template family.

    Numbers -> '#', mid-sentence capitalized words -> '<E>'. So
        "what is the capital city of France?" and
        "what is the capital city of Italy?"
    map to the same signature. Used to (i) cluster near-duplicate prompts for
    cluster-robust inference and (ii) diversify the length-matched selection.
    """
    s = re.sub(r"\d+", "#", prompt.strip())
    toks = s.split()
    out: List[str] = []
    for i, t in enumerate(toks):
        core = re.sub(r"[^A-Za-z]", "", t)
        if i > 0 and core[:1].isupper() and len(core) >= 2:
            out.append(re.sub(r"[A-Za-z]+", "<E>", t))
        else:
            out.append(t)
    return re.sub(r"\s+", " ", " ".join(out).lower())


def make_length_fn(
    model_name: Optional[str], apply_chat: bool = True,
) -> Tuple[Callable[[str], int], str]:
    """Return (length_fn, kind).

    With a real HF tokenizer we count chat-templated ids: the same tokens the
    model actually sees. Without it, whitespace tokens as a proxy so the
    matching logic can still be exercised anywhere.
    """
    if model_name and AutoTokenizer is not None:
        tok = AutoTokenizer.from_pretrained(model_name)

        def _len(prompt: str) -> int:
            text = prompt
            if apply_chat and getattr(tok, "chat_template", None):
                text = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            return int(len(tok(text).input_ids))
        return _len, "tokenizer"

    def _heuristic(prompt: str) -> int:
        return int(len(prompt.split()))
    return _heuristic, "whitespace_proxy"


# ============================================================
# Statistics
# ============================================================

def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def cohens_d(a, b) -> float:
    """Standardized mean difference with pooled variance. Sign: mean(a) - mean(b)."""
    a, b = _as1d(a), _as1d(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def cliffs_delta(a, b) -> float:
    """Rank-based effect size in [-1, 1]. Distribution-free, robust to outliers.
    Sign: proportion of a > b minus a < b."""
    a, b = _as1d(a), _as1d(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(np.concatenate([a, b])))  # 0..n-1 ranks with ties handled by argsort
    # A tie-safe O((n+m) log(n+m)) implementation without scipy:
    combined = np.concatenate([a, b])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    r_a = ranks[: len(a)].sum()
    n, m = len(a), len(b)
    # U statistic; delta = 2 U / (n m) - 1
    U = r_a - n * (n + 1) / 2.0
    return float(2.0 * U / (n * m) - 1.0)


def bootstrap_mean_ci(
    x, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns (mean, ci_lo, ci_hi)."""
    x = _as1d(x)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def permutation_p_value(
    a, b, n_permutations: int = 10000, seed: int = 42,
) -> float:
    """Two-sided permutation test for the difference of means."""
    a, b = _as1d(a), _as1d(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(int(n_permutations)):
        rng.shuffle(pooled)
        if abs(pooled[:n].mean() - pooled[n:].mean()) >= observed:
            count += 1
    return float((count + 1) / (n_permutations + 1))


def holm_correct(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order, each clipped to [0, 1]."""
    pvals = list(pvals)
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # step-down monotonicity
        adj[i] = min(running, 1.0)
    return adj


def pairwise_report(
    df: pd.DataFrame,
    metric: str,
    label_col: str = "label",
    n_boot: int = 5000,
    n_perm: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """One tidy frame with, for each of the six category pairs on `metric`:
    mean_a, mean_b, diff, cohens_d, cliffs_delta, permutation p, Holm-adjusted p.
    Adjusted p-values are computed across the six pairs for THIS metric only,
    which is what we want (per-metric family)."""
    rows = []
    pvals = []
    for a, b in CATEGORY_PAIRS:
        xa = df.loc[df[label_col] == a, metric].to_numpy(dtype=float)
        xb = df.loc[df[label_col] == b, metric].to_numpy(dtype=float)
        xa = xa[np.isfinite(xa)]
        xb = xb[np.isfinite(xb)]
        p = permutation_p_value(xa, xb, n_permutations=n_perm, seed=seed)
        pvals.append(p)
        rows.append({
            "metric": metric, "label_a": a, "label_b": b,
            "n_a": int(len(xa)), "n_b": int(len(xb)),
            "mean_a": float(np.mean(xa)) if len(xa) else float("nan"),
            "mean_b": float(np.mean(xb)) if len(xb) else float("nan"),
            "diff": (float(np.mean(xa) - np.mean(xb))
                     if len(xa) and len(xb) else float("nan")),
            "cohens_d": cohens_d(xa, xb),
            "cliffs_delta": cliffs_delta(xa, xb),
            "perm_p": p,
        })
    adj = holm_correct(pvals)
    for r, p_adj in zip(rows, adj):
        r["perm_p_holm"] = p_adj
    return pd.DataFrame(rows)
