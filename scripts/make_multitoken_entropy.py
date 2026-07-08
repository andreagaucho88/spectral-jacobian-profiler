#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer point 3: single-token entropy is a weak, template-contaminated proxy.
Compute mean entropy over the first k GENERATED tokens (greedy) and check whether
the hallucination-prone outlier result survives. Qwen2.5-0.5B, length-matched set."""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL="Qwen/Qwen2.5-0.5B-Instruct"; SLUG="Qwen__Qwen2.5-0.5B-Instruct"
CATS=["factual","coding","reasoning","hallucination_prone"]; N=40; K=8
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").eval()

@torch.no_grad()
def entropies(prompt):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True) \
        if getattr(tok,"chat_template",None) else prompt
    ids=tok(text,return_tensors="pt")["input_ids"]
    Hs=[]; past=None; cur=ids
    for _ in range(K):
        out=model(input_ids=cur, past_key_values=past, use_cache=True, return_dict=True)
        past=out.past_key_values
        p=torch.softmax(out.logits[0,-1].float(),dim=-1)
        Hs.append(float(-(p*torch.log(p+1e-20)).sum()))
        cur=p.argmax().view(1,1)   # greedy
    return Hs[0], float(np.mean(Hs))

prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
rows=[]
for c in CATS:
    for pr in prompts[c][:N]:
        h1,hk=entropies(pr); rows.append({"label":c,"H_single":h1,"H_multi":hk})
d=pd.DataFrame(rows)

def cohend(a,b):
    a,b=np.asarray(a),np.asarray(b);sp=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2));return (a.mean()-b.mean())/sp
print(f"Qwen n={len(d)} ({N}/cat), K={K} generated tokens\n")
for col in ["H_single","H_multi"]:
    print(f"=== {col} ===")
    means={c:round(d[d.label==c][col].mean(),2) for c in CATS}
    print("  means:",means, " | outlier(max entropy)=",max(means,key=means.get))
    hal=d[d.label=="hallucination_prone"][col]
    print("  halluc vs others |d|:", {c:round(abs(cohend(hal,d[d.label==c][col])),2) for c in ["factual","coding","reasoning"]})
r=np.corrcoef(d["H_single"],d["H_multi"])[0,1]
print(f"\ncorr(H_single, H_multi) = {r:.2f}")
d.to_csv(ROOT/"results_analysis/multitoken_entropy.csv",index=False)
print("saved results_analysis/multitoken_entropy.csv")
