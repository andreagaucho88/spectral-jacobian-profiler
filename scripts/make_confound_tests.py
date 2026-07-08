#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer points 4 & 10: is pre-norm sigma_max just ||h_L|| inflation / prompt
surprise, and does a trivial linear probe on h_L already separate the categories?
Forward passes only (no autograd). Qwen2.5-0.5B, the exact run prompts."""
from __future__ import annotations
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"; SLUG = "Qwen__Qwen2.5-0.5B-Instruct"
CATS = ["factual", "coding", "reasoning", "hallucination_prone"]
N = 40

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").eval()
model.config.use_cache = False

def templated_ids(prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) \
        if getattr(tok, "chat_template", None) else prompt
    return tok(text, return_tensors="pt")

@torch.no_grad()
def features(prompt):
    enc = templated_ids(prompt)
    out = model(**enc, output_hidden_states=True, return_dict=True)
    h_pre = out.hidden_states[-2][0, -1, :].float()   # pre-norm final-token (matches sigma_max run)
    h_post = out.hidden_states[-1][0, -1, :].float()  # post-norm (readout state)
    # prompt "surprise": LM perplexity of the RAW prompt (no template), teacher forced
    r = tok(prompt, return_tensors="pt")
    lo = model(**r, labels=r["input_ids"], return_dict=True).loss.item()
    return float(h_pre.norm()), float(h_post.norm()), float(np.exp(lo)), h_pre.numpy(), h_post.numpy()

prompts = json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
rows = []; Hpre = []; Hpost = []; ylab = []
for c in CATS:
    for p in prompts[c][:N]:
        hpre_n, hpost_n, ppl, hpre, hpost = features(p)
        rows.append({"prompt": p, "label": c, "h_pre_norm": hpre_n, "h_post_norm": hpost_n, "prompt_ppl": ppl})
        Hpre.append(hpre); Hpost.append(hpost); ylab.append(c)
feat = pd.DataFrame(rows)
Hpre = np.array(Hpre); Hpost = np.array(Hpost); y = np.array(ylab)

spec = pd.read_csv(ROOT/f"results_spectral/{SLUG}/final_index_-2/spectral_per_prompt.csv")
spec = spec[spec["error"].astype(str).str.strip().isin(["", "nan"])]
m = pd.merge(feat, spec[["prompt", "label", "sigma_max"]], on=["prompt", "label"], how="inner")
print(f"merged n={len(m)}")

print("\n=== POINT 4: is sigma_max just norm / prompt-surprise? ===")
for col in ["h_pre_norm", "h_post_norm", "prompt_ppl"]:
    r = stats.pearsonr(m["sigma_max"], m[col]); rs = stats.spearmanr(m["sigma_max"], m[col])
    print(f"  r(sigma_max, {col:12s}) = {r.statistic:+.3f}  (Spearman {rs.statistic:+.3f}, R^2={r.statistic**2:.2f})")
# partial: does factual>coding sigma_max survive controlling for h_pre_norm?
import numpy as np
X = np.column_stack([np.ones(len(m)), m["h_pre_norm"]])
beta,*_ = np.linalg.lstsq(X, m["sigma_max"].values, rcond=None)
m = m.assign(sig_resid=m["sigma_max"].values - X@beta)
def d(a,b):
    a,b=np.asarray(a),np.asarray(b);sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2));return (a.mean()-b.mean())/sp
fc_raw = d(m[m.label=="factual"]["sigma_max"], m[m.label=="coding"]["sigma_max"])
fc_res = d(m[m.label=="factual"]["sig_resid"], m[m.label=="coding"]["sig_resid"])
print(f"  factual>coding sigma_max: d_raw={fc_raw:+.2f}  d_after_removing_||h_pre||={fc_res:+.2f}")

print("\n=== POINT 10: linear probe baseline (5-fold CV accuracy, 4-way) ===")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    def cv(Hf, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        return cross_val_score(clf, Hf, y, cv=5).mean()
    accp = cv(Hpre, y); acps = cv(Hpost, y)
    accn = cv(np.column_stack([feat["h_pre_norm"], feat["h_post_norm"], feat["prompt_ppl"]]), y)
    chance = max(np.bincount([CATS.index(c) for c in y]))/len(y)
    print(f"  linear probe on h_pre  (d={Hpre.shape[1]}): acc={accp:.2f}")
    print(f"  linear probe on h_post (d={Hpost.shape[1]}): acc={acps:.2f}")
    print(f"  probe on 3 scalars [||h_pre||,||h_post||,ppl]: acc={accn:.2f}")
    print(f"  chance (majority) = {chance:.2f}")
except ImportError:
    print("  sklearn non disponibile; salto il probe")

feat.merge(spec[["prompt","label","sigma_max"]],on=["prompt","label"]).to_csv(ROOT/"results_analysis/confound_features.csv", index=False)
print("\nsaved results_analysis/confound_features.csv")
