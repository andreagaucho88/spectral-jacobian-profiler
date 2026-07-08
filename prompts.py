#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Length-matched prompt generator.

The four prompt categories differ systematically in token length in any naive
generation scheme (short factual recall, long code stubs, medium reasoning
word problems), which confounds every finite-depth response comparison in the
first paper draft: the input perturbation has unit Frobenius norm over the
whole T x d embedding sequence, so its per-token magnitude scales as ~1/sqrt(T),
and the categories had almost no common length support, ruling out regression
adjustment.

This module addresses that by construction. For each target model it:

  1. over-generates a large, DIVERSE pool per category, with terse, medium,
     and verbose phrasings of each template family (so every category spans a
     broad length range while every prompt stays a faithful category member;
     a factual question is still factual recall, etc.);
  2. tokenizes every candidate with the target model's tokenizer AND its
     chat template (the exact ids the model will see at inference);
  3. selects a subset with IDENTICAL token-length histograms across the four
     categories -- exact per-length-bin matching: at every length L, take the
     same number of prompts from each category -- with a per-template cap so
     the matched set does not collapse onto a handful of templates.

Output: JSON {label: [prompts]} plus a CSV with (prompt, label, token_length,
template_id). The JSON is what runner.py loads.

CLI
    python3 prompts.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --target-per-category 400 --max-per-template 5 \
        --out-dir lengthmatched_prompts
    python3 prompts.py                       # whitespace proxy, dry-run
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common import (
    CATEGORY_ORDER, make_length_fn, save_prompts_json,
    slug_of_model, template_signature,
)

try:
    from scipy.stats import kruskal
except ModuleNotFoundError:                                       # pragma: no cover
    kruskal = None


# ============================================================
# Candidate pools (verbatim vocabulary from the earlier draft; the
# combinations are what give us the length range within each category)
# ============================================================

_COUNTRIES = [
    "France", "Italy", "Germany", "Spain", "Japan", "Canada", "Portugal", "Greece",
    "Norway", "Sweden", "Finland", "Poland", "Austria", "Switzerland", "Belgium",
    "Denmark", "Ireland", "Hungary", "Romania", "Croatia", "Brazil", "Argentina",
    "Mexico", "India", "China", "Egypt", "Kenya", "Chile", "Peru", "Singapore",
    "Morocco", "Turkey", "Iceland", "Estonia", "Latvia", "Slovakia", "Slovenia",
    "Bulgaria", "Serbia", "Ukraine", "Lithuania", "Thailand", "Vietnam", "Malaysia",
    "Indonesia", "Colombia", "Ecuador", "Bolivia", "Uruguay", "Paraguay", "Ghana",
    "Nigeria", "Tunisia", "Algeria", "Jordan", "Lebanon", "Nepal", "Bangladesh",
]
_ELEMENTS = [
    "oxygen", "hydrogen", "carbon", "nitrogen", "sodium", "iron", "gold", "silver",
    "helium", "neon", "calcium", "potassium", "copper", "zinc", "sulfur", "chlorine",
    "magnesium", "nickel", "lead", "tin",
]
_PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
_TERMS = [
    "photosynthesis", "gravity", "evaporation", "osmosis", "natural selection",
    "magnetism", "plate tectonics", "inertia", "diffusion", "refraction",
    "combustion", "respiration", "erosion", "condensation", "fermentation",
]
_AUTHORS = {
    "Hamlet": "Shakespeare", "Ulysses": "Joyce", "1984": "Orwell",
    "The Trial": "Kafka", "Faust": "Goethe", "The Odyssey": "Homer",
    "War and Peace": "Tolstoy", "The Stranger": "Camus", "Don Quixote": "Cervantes",
}


def _factual_candidates() -> List[str]:
    out: List[str] = []
    fact_frames = {
        "capital": [
            "Capital of {c}?",
            "What is the capital of {c}?",
            "What is the capital city of {c}?",
            "Name the capital city of the country called {c}.",
            "State the official capital city of the nation of {c} in one word.",
        ],
        "currency": [
            "Currency of {c}?",
            "What currency does {c} use?",
            "What is the official currency of the country of {c}?",
            "Name the official national currency used in the country called {c}.",
        ],
        "continent": [
            "Continent of {c}?",
            "On which continent is {c}?",
            "On which continent is the country of {c} located?",
            "Which continent is the country called {c} part of, in one word?",
        ],
        "language": [
            "Main language of {c}?",
            "What language is spoken in {c}?",
            "What is the primary official language of the country of {c}?",
            "Name the main official language used in the country called {c}.",
        ],
    }
    for c in _COUNTRIES:
        for frames in fact_frames.values():
            for fr in frames:
                out.append(fr.format(c=c))
        out += [
            f"Largest city of {c}?",
            f"What is the largest city by population in the country of {c}?",
            f"Name the most populous city located within the country called {c}.",
            f"Which single city is the most populous one in the country of {c}, by resident population?",
            f"Hemisphere of {c}?",
            f"Is the country of {c} located mainly in the northern or the southern hemisphere?",
            f"State whether the country called {c} lies mainly north or south of the equator.",
            f"Does the country of {c} have a coastline, or is it entirely landlocked? Answer briefly.",
        ]
    for e in _ELEMENTS:
        out += [
            f"Chemical symbol for {e}?",
            f"Symbol of the element {e}?",
            f"What is the chemical symbol of {e}?",
            f"What is the standard chemical symbol of the element {e}?",
            f"Give the periodic-table symbol for the chemical element called {e}.",
        ]
    for p in _PLANETS:
        out += [
            f"Is {p} a planet?",
            f"What colour is {p}, briefly?",
            f"What is the planet {p} mostly made of, briefly?",
            f"Which position from the Sun does the planet {p} occupy in the Solar System?",
        ]
    for t in _TERMS:
        out += [
            f"Define {t}.",
            f"What is {t}?",
            f"Define {t} in one sentence.",
            f"Give a one-sentence definition of the scientific term {t}.",
            f"In one short sentence, explain what the term {t} means in natural science.",
        ]
    for book in _AUTHORS:
        out += [
            f"Who wrote {book}?",
            f"Author of {book}?",
            f"Who is the author of the book {book}?",
            f"Name the writer who is the author of the literary work {book}.",
        ]
    return out


def _reasoning_candidates() -> List[str]:
    out: List[str] = []
    rng = np.random.default_rng(0)
    for _ in range(260):
        a, b = int(rng.integers(11, 99)), int(rng.integers(11, 99))
        out += [
            f"{a} + {b}?",
            f"Compute {a} + {b}.",
            f"What is {a} plus {b}?",
            f"Add the numbers {a} and {b} and give the total.",
            f"Work out the sum of the two numbers {a} and {b} and state the result.",
        ]
    for _ in range(220):
        r, bl = int(rng.integers(2, 20)), int(rng.integers(2, 20))
        out += [
            f"{r} red, {bl} blue; P(red)?",
            f"P(red) from {r} red and {bl} blue balls?",
            f"A box has {r} red and {bl} blue balls; probability of red?",
            f"A box contains {r} red balls and {bl} blue balls. What is the probability of drawing a red ball?",
            f"A box contains {r} red balls and {bl} blue balls. If one ball is drawn at random, what is the probability that it is red?",
        ]
    for _ in range(200):
        w, d = int(rng.integers(2, 40)), int(rng.integers(2, 30))
        out += [
            f"{w} workers, {d} days: worker-days?",
            f"Worker-days for {w} workers over {d} days?",
            f"If {w} workers finish a job in {d} days, how many worker-days are needed?",
            f"If {w} workers can finish a job in {d} days, how many worker-days does the whole job require?",
        ]
    for _ in range(200):
        base = int(rng.integers(20, 400))
        pct = int(rng.choice([5, 10, 15, 20, 25, 30, 40]))
        out += [
            f"{pct}% of {base}?",
            f"What is {pct}% of {base}?",
            f"Compute {pct} percent of the number {base}.",
            f"A price of {base} euros changes by {pct} percent. What is the amount of the change?",
        ]
    for _ in range(160):
        h = int(rng.integers(1, 13))
        m = int(rng.choice([5, 10, 15, 20, 30, 40, 45, 50]))
        out += [
            f"{h:02d}:{m:02d} + 95 min?",
            f"A meeting starts at {h:02d}:{m:02d} and lasts 95 minutes; end time?",
            f"A meeting starts at {h:02d}:{m:02d} and runs for 95 minutes. At what clock time does it end?",
        ]
    for _ in range(160):
        n = int(rng.integers(2, 60))
        out += [
            f"Thursday + {n} days = ?",
            f"If today is Thursday, what day is it in {n} days?",
            f"If today is a Thursday, which day of the week will it be {n} days from now?",
        ]
    for _ in range(160):
        a, b = int(rng.integers(2, 9)), int(rng.integers(3, 11))
        c, d = int(rng.integers(2, 9)), int(rng.integers(3, 11))
        out += [
            f"Larger: {a}/{b} or {c}/{d}?",
            f"Which fraction is larger, {a}/{b} or {c}/{d}?",
            f"Determine which of the two fractions {a}/{b} and {c}/{d} has the greater value.",
        ]
    for _ in range(150):
        s, h = int(rng.integers(30, 120)), int(rng.integers(2, 9))
        out += [
            f"{s} km/h for {h} h: distance?",
            f"A car travels at {s} km/h for {h} hours; how far does it go?",
            f"A car drives at a constant speed of {s} km/h for {h} hours. What total distance does it cover?",
        ]
    for _ in range(150):
        now, ago = int(rng.integers(18, 70)), int(rng.integers(3, 15))
        out += [
            f"Age {ago} yrs ago if now {now}?",
            f"If someone is {now} now, how old were they {ago} years ago?",
            f"A person is {now} years old today. How old were they exactly {ago} years ago?",
        ]
    for _ in range(150):
        n, m = int(rng.integers(10, 99)), int(rng.integers(3, 9))
        out += [
            f"{n} mod {m}?",
            f"What is the remainder of {n} divided by {m}?",
            f"When the number {n} is divided by {m}, what is the remainder?",
        ]
    for _ in range(150):
        a, d = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        out += [
            f"Next: {a},{a+d},{a+2*d},?",
            f"What is the next term in {a}, {a+d}, {a+2*d}, ...?",
            f"Give the next number in the arithmetic sequence {a}, {a+d}, {a+2*d}, and so on.",
        ]
    for _ in range(150):
        km = int(rng.integers(2, 40))
        out += [
            f"{km} km in metres?",
            f"How many metres are there in {km} kilometres?",
            f"Convert a distance of {km} kilometres into metres and state the result.",
        ]
    for _ in range(150):
        p, r = int(rng.integers(100, 900)), int(rng.choice([2, 3, 4, 5, 6]))
        out += [
            f"Simple interest on {p} at {r}% for 1 yr?",
            f"What is one year of simple interest on {p} euros at {r} percent?",
            f"Find the simple interest earned on {p} euros over one year at an annual rate of {r} percent.",
        ]
    for _ in range(150):
        a, b, c = int(rng.integers(10, 40)), int(rng.integers(10, 40)), int(rng.integers(10, 40))
        out += [
            f"Average of {a}, {b}, {c}?",
            f"What is the average of the numbers {a}, {b} and {c}?",
            f"Compute the arithmetic mean of the three numbers {a}, {b} and {c}.",
        ]
    for _ in range(150):
        L, W = int(rng.integers(3, 20)), int(rng.integers(3, 20))
        out += [
            f"Area of {L} by {W} rectangle?",
            f"What is the area of a rectangle that is {L} by {W}?",
            f"A rectangle measures {L} units by {W} units. What is its total area?",
        ]
    for _ in range(150):
        total = int(rng.integers(10, 60))
        r_a, r_b = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        out += [
            f"Split {total} in {r_a}:{r_b}; larger share?",
            f"Divide {total} in the ratio {r_a} to {r_b}; what is the larger share?",
            f"If {total} is divided in the ratio {r_a} to {r_b}, how much is the larger of the two parts?",
        ]
    return out


def _coding_candidates() -> List[str]:
    tasks_short = [
        "reverse a string", "check a palindrome", "find the maximum", "sum a list",
        "count vowels", "flatten a list", "remove duplicates", "sort numbers",
        "find the minimum", "merge two lists", "title-case a string", "fizzbuzz",
        "count words", "swap two variables", "compute a factorial", "check primality",
        "round to two decimals", "trim whitespace", "split on commas", "join with dashes",
        "find the average", "square each element", "filter even numbers", "reverse a list",
    ]
    tasks_long = [
        "validate balanced brackets", "merge overlapping intervals",
        "compute a rolling average", "deduplicate records by key",
        "find the missing number in a sequence", "implement binary search",
        "compute the Levenshtein distance", "build a small inverted index",
        "topologically sort a list of tasks", "detect outliers in a numeric list",
        "group anagrams together", "find the longest common prefix",
        "rotate a matrix in place", "compute a running median",
        "parse a simple query string", "implement a least-recently-used cache",
        "find connected components in a graph", "encode run-length sequences",
        "validate an email address", "compute the moving maximum",
    ]
    langs = ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "Kotlin", "Ruby", "C++"]
    short_frames = [
        "{lang}: {t}.",
        "Write {lang} code to {t}.",
        "In {lang}, write a function to {t}.",
        "In {lang}, write a short function that will {t} and return the result.",
        "Using {lang}, implement a small function to {t}.",
    ]
    long_frames = [
        "{lang}: {t}.",
        "Write {lang} code to {t}.",
        "In {lang}, write a function to {t}.",
        "In {lang}, write a function to {t}; handle edge cases and add a short test.",
        "In {lang}, implement a function that will {t}; handle invalid input and return structured output.",
        "Using {lang}, write a well-tested function to {t} and document its time complexity.",
    ]
    out: List[str] = []
    for lang in langs:
        for t in tasks_short:
            out += [fr.format(lang=lang, t=t) for fr in short_frames]
        for t in tasks_long:
            out += [fr.format(lang=lang, t=t) for fr in long_frames]
    return out


def _hallucination_candidates() -> List[str]:
    fic = [
        "Atlantis", "Elbonia", "Gondor", "Narnia", "Neverland", "Westeros", "Mordor",
        "Camelot", "Eldoria", "Zephyria", "the Moon Kingdom", "the Martian Republic",
        "the Galactic Federation", "the Republic of Oz", "the planet Vulcan",
        "the Sapphire Empire", "the Lunar Commonwealth", "the Aether Republic",
    ]
    impossible = [
        "official GDP", "central bank rate", "ISO country code", "airport code",
        "UN seat number", "constitutional article", "official tax code", "public debt",
        "inflation rate", "internet domain", "official census population", "customs tariff",
        "stock exchange ticker", "national railway operator",
    ]
    out: List[str] = []
    for f in fic:
        out += [
            f"Capital of {f}?",
            f"What is the capital of {f}?",
            f"What is the capital city of the realm of {f}?",
        ]
        for im in impossible:
            out += [
                f"{im} of {f}?",
                f"What is the {im} of {f}?",
                f"Provide the verified real-world {im} of {f}.",
                f"Provide the official, verified real-world {im} of the nation of {f}.",
                f"Which official government source confirms the {im} of the country of {f}?",
            ]
    rng = np.random.default_rng(0)
    for _ in range(300):
        y = int(rng.integers(1500, 2024))
        f = str(rng.choice(fic))
        out += [
            f"Who ruled {f} in {y}?",
            f"Who governed the realm of {f} in the year {y}?",
            f"Which treaty formally recognized the nation of {f} in the year {y}?",
        ]
    for _ in range(200):
        y = int(rng.integers(1500, 2024))
        out += [
            f"Nobel Prize in Alchemy {y} winner?",
            f"Who won the Nobel Prize in Alchemy in {y}?",
            f"Who won the Olympic gold medal in zero-gravity fencing in the year {y}?",
        ]
    return out


def build_candidate_pools() -> Dict[str, List[str]]:
    pools = {
        "factual": _factual_candidates(),
        "coding": _coding_candidates(),
        "reasoning": _reasoning_candidates(),
        "hallucination_prone": _hallucination_candidates(),
    }
    out: Dict[str, List[str]] = {}
    for k, v in pools.items():
        seen = set()
        uniq = []
        for p in v:
            if p not in seen:
                uniq.append(p); seen.add(p)
        out[k] = uniq
    return out


# ============================================================
# Exact per-length-bin matching
# ============================================================

def _diverse_pick(
    cands: List[str], k: int, max_per_template: int, rng: np.random.Generator,
) -> List[str]:
    """Take k prompts from cands under a per-template cap, round-robin over
    templates to maximize the number of distinct signatures used."""
    if k <= 0 or not cands:
        return []
    by_tmpl: Dict[str, List[str]] = {}
    for p in cands:
        by_tmpl.setdefault(template_signature(p), []).append(p)
    for t in by_tmpl:
        rng.shuffle(by_tmpl[t])
    tmpls = list(by_tmpl); rng.shuffle(tmpls)
    taken: List[str] = []
    used: Counter = Counter()
    while len(taken) < k:
        progressed = False
        for t in tmpls:
            if len(taken) >= k:
                break
            if used[t] >= max_per_template or used[t] >= len(by_tmpl[t]):
                continue
            taken.append(by_tmpl[t][used[t]]); used[t] += 1; progressed = True
        if not progressed:
            break
    return taken


def _diverse_capacity(cands: List[str], max_per_template: int) -> int:
    by_tmpl = Counter(template_signature(p) for p in cands)
    return int(sum(min(n, max_per_template) for n in by_tmpl.values()))


def match_by_length(
    pools: Dict[str, List[str]],
    length_fn: Callable[[str], int],
    target_per_category: int = 300,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    max_per_template: int = 5,
    seed: int = 0,
) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """Select a subset with identical token-length histograms across categories.

    Diverse capacity per (category, length) is the count of prompts drawable
    under the per-template cap. At every length, we can take at most
    m(L) = min over categories of diverse capacity; then a greedy fill across L
    reaches the requested target-per-category by taking as many as possible from
    the highest-capacity bins first. Returns matched {label: [prompts]} and a
    long-form diagnostics frame with (prompt, label, token_length, template_id).
    """
    rng = np.random.default_rng(seed)

    records = []
    for label, prompts in pools.items():
        for p in prompts:
            records.append({"label": label, "prompt": p,
                            "token_length": int(length_fn(p))})
    pool = pd.DataFrame(records)
    if min_len is not None:
        pool = pool[pool["token_length"] >= min_len]
    if max_len is not None:
        pool = pool[pool["token_length"] <= max_len]

    cand_index: Dict[Tuple[str, int], List[str]] = {}
    for (label, L), g in pool.groupby(["label", "token_length"]):
        cand_index[(label, int(L))] = g["prompt"].tolist()

    lengths = sorted(pool["token_length"].unique())
    capacity: Dict[int, int] = {}
    for L in lengths:
        caps = [
            _diverse_capacity(cand_index.get((label, L), []), max_per_template)
            for label in CATEGORY_ORDER
        ]
        c = min(caps)
        if c > 0:
            capacity[L] = c
    if not capacity:
        raise ValueError("No token length is populated by all four categories; "
                         "widen the pools or relax --max-per-template.")

    # decide per-length quota: fill highest-capacity bins first up to target.
    total_cap = sum(capacity.values())
    if total_cap < target_per_category:
        # take everything we can
        per_length = dict(capacity)
    else:
        # water-filling on capacity: sort bins by capacity descending, take
        # min(capacity, remaining/n_active) at each pass, capped by capacity.
        per_length = {L: 0 for L in capacity}
        remaining = target_per_category
        avail_bins = list(capacity.items())
        while remaining > 0 and avail_bins:
            share = max(1, remaining // len(avail_bins))
            new_avail = []
            for L, c_L in avail_bins:
                take = min(share, c_L - per_length[L], remaining)
                if take <= 0:
                    continue
                per_length[L] += take
                remaining -= take
                if per_length[L] < c_L:
                    new_avail.append((L, c_L))
                if remaining <= 0:
                    break
            if not new_avail:
                break
            avail_bins = new_avail

    matched: Dict[str, List[str]] = {label: [] for label in CATEGORY_ORDER}
    diag_rows = []
    for L, k in per_length.items():
        if k <= 0:
            continue
        for label in CATEGORY_ORDER:
            picks = _diverse_pick(cand_index.get((label, L), []), k,
                                  max_per_template, rng)
            matched[label].extend(picks)
            for p in picks:
                diag_rows.append({"label": label, "prompt": p,
                                  "token_length": int(L),
                                  "template_id": template_signature(p)})
    diag = pd.DataFrame(diag_rows)
    return matched, diag


def report_match(diag: pd.DataFrame) -> None:
    print("\nMATCHED LENGTH DISTRIBUTION")
    stats = (diag.groupby("label")["token_length"]
             .agg(["count", "mean", "std", "min", "max"])
             .reindex(CATEGORY_ORDER).round(2))
    print(stats.to_string())
    if kruskal is not None:
        groups = [diag[diag["label"] == c]["token_length"].to_numpy()
                  for c in CATEGORY_ORDER]
        if all(len(g) > 0 for g in groups):
            h, p = kruskal(*groups)
            print(f"\nKruskal-Wallis across categories: H = {h:.3f}, p = {p:.3f} "
                  "(large p = length distributions indistinguishable)")
    div = diag.groupby("label")["template_id"].nunique().reindex(CATEGORY_ORDER)
    n = diag.groupby("label")["prompt"].size().reindex(CATEGORY_ORDER)
    print("\nDIVERSITY  (distinct templates / prompts per category)")
    for c in CATEGORY_ORDER:
        print(f"  {c:20} {int(n[c]):5d} prompts  "
              f"{int(div[c]):5d} distinct templates  "
              f"({100*div[c]/max(n[c],1):.0f}% unique)")


# ============================================================
# CLI
# ============================================================

def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", default=None,
                    help="HF model name for real tokenization (recommended)")
    ap.add_argument("--target-per-category", type=int, default=300)
    ap.add_argument("--max-per-template", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--out-dir", default="lengthmatched_prompts")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    length_fn, kind = make_length_fn(args.model)
    print(f"Length source     : {kind}")
    print(f"Model             : {args.model or '(none)'}")
    print(f"Target/category   : {args.target_per_category}")
    print(f"Max/template/bin  : {args.max_per_template}")

    pools = build_candidate_pools()
    print("\nCANDIDATE POOL SIZES")
    for k in CATEGORY_ORDER:
        print(f"  {k:20} {len(pools[k]):6d}")

    matched, diag = match_by_length(
        pools, length_fn,
        target_per_category=args.target_per_category,
        min_len=args.min_len, max_len=args.max_len,
        max_per_template=args.max_per_template, seed=args.seed,
    )
    report_match(diag)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    slug = slug_of_model(args.model)
    json_path = save_prompts_json(matched, out / f"lengthmatched_prompts__{slug}.json")
    csv_path = out / f"lengthmatched_prompts__{slug}.csv"
    diag.to_csv(csv_path, index=False)
    print(f"\nSaved:\n  {json_path.resolve()}\n  {csv_path.resolve()}")


if __name__ == "__main__":
    _cli()
