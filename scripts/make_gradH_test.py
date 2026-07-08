#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer point 1: the direct coupling test. Compute the input-gradient norm of the
output entropy ||d H(p) / d H_0|| per prompt (one backward) and correlate it with
sigma_max. If they are uncorrelated, sigma_max (top singular value of the hidden
Jacobian) genuinely carries different information than the entropy's input sensitivity;
if correlated, the null is trivial. Qwen2.5-0.5B, exact run prompts."""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent.parent
MODEL="Qwen/Qwen2.5-0.5B-Instruct"; SLUG="Qwen__Qwen2.5-0.5B-Instruct"; CATS=["factual","coding","reasoning","hallucination_prone"]; N=40
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").eval()
model.config.use_cache=False

def gradH(prompt):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True) \
        if getattr(tok,"chat_template",None) else prompt
    ids=tok(text,return_tensors="pt")["input_ids"]
    emb=model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
    out=model(inputs_embeds=emb, return_dict=True)
    z=out.logits[0,-1].float()
    logp=torch.log_softmax(z,dim=-1); p=logp.exp()
    H=-(p*logp).sum()
    g,=torch.autograd.grad(H, emb)
    return float(H.detach()), float(g.norm())

prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
rows=[]
for c in CATS:
    for pr in prompts[c][:N]:
        H,gn=gradH(pr); rows.append({"prompt":pr,"label":c,"H":H,"gradH_norm":gn})
d=pd.DataFrame(rows)
spec=pd.read_csv(ROOT/f"results_spectral/{SLUG}/final_index_-2/spectral_per_prompt.csv")
spec=spec[spec["error"].astype(str).str.strip().isin(["","nan"])]
m=pd.merge(d, spec[["prompt","label","sigma_max","chi_fisher_max"]], on=["prompt","label"])
print(f"merged n={len(m)}\n")
print("=== the direct coupling test ===")
print(f"  r(gradH_norm, entropy H)          = {stats.pearsonr(m['gradH_norm'],m['H']).statistic:+.3f}   (does entropy sensitivity track entropy?)")
print(f"  r(gradH_norm, sigma_max)          = {stats.pearsonr(m['gradH_norm'],m['sigma_max']).statistic:+.3f}   (KEY: output-sensitivity vs hidden-amplification)")
print(f"  r(chi_fisher_max, gradH_norm)     = {stats.pearsonr(m['chi_fisher_max'],m['gradH_norm']).statistic:+.3f}")
print(f"  r(sigma_max, entropy) [reference] = {stats.pearsonr(m['sigma_max'],m['H']).statistic:+.3f}")
print("\n  gradH_norm means by category:", {c:round(m[m.label==c]['gradH_norm'].mean(),1) for c in CATS})
m.to_csv(ROOT/"results_analysis/gradH_test.csv",index=False); print("\nsaved results_analysis/gradH_test.csv")
