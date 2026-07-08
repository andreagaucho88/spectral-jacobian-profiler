#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Delta-review: is r(||dH||, sigma_max)=0.45 directional coupling or just shared
magnitude (partial out lambda_bulk)? Replicate ||dH|| on SmolLM2 and GPT-2. And
is dCor(sigma,entropy)=0.20 a within-category effect or centroid separation?"""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent.parent
CATS=["factual","coding","reasoning","hallucination_prone"]; N=40; RNG=np.random.default_rng(0)
MODELS=[("Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct","Qwen__Qwen2.5-0.5B-Instruct","results_spectral"),
        ("SmolLM2-360M","HuggingFaceTB/SmolLM2-360M-Instruct","HuggingFaceTB__SmolLM2-360M-Instruct","results_spectral"),
        ("GPT-2-124M","gpt2","gpt2","results_spectral_gpt2")]

def dcor(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    a=np.abs(x[:,None]-x[None,:]);b=np.abs(y[:,None]-y[None,:])
    A=a-a.mean(0)-a.mean(1)[:,None]+a.mean();B=b-b.mean(0)-b.mean(1)[:,None]+b.mean()
    dv=np.sqrt((A*A).mean()*(B*B).mean())
    return float(np.sqrt((A*B).mean())/dv) if dv>0 else 0.0
def partial_r(a,b,c):  # r(a,b | c)
    rab=stats.pearsonr(a,b).statistic; rac=stats.pearsonr(a,c).statistic; rbc=stats.pearsonr(b,c).statistic
    return (rab-rac*rbc)/np.sqrt((1-rac**2)*(1-rbc**2))

for name,mid,slug,base in MODELS:
    tok=AutoTokenizer.from_pretrained(mid)
    model=AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.config.use_cache=False
    def gradH(prompt):
        text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True) \
            if getattr(tok,"chat_template",None) else prompt
        ids=tok(text,return_tensors="pt")["input_ids"]
        emb=model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
        z=model(inputs_embeds=emb,return_dict=True).logits[0,-1].float()
        lp=torch.log_softmax(z,-1); H=-(lp.exp()*lp).sum()
        g,=torch.autograd.grad(H,emb); return float(g.norm())
    prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{slug}.json"))
    rows=[]
    for c in CATS:
        for pr in prompts[c][:N]:
            rows.append({"prompt":pr,"label":c,"gradH":gradH(pr)})
    g=pd.DataFrame(rows)
    sp=pd.read_csv(ROOT/f"{base}/{slug}/final_index_-2/spectral_per_prompt.csv")
    sp=sp[sp["error"].astype(str).str.strip().isin(["","nan"])]
    m=pd.merge(g, sp[["prompt","label","sigma_max","lambda_bulk_pred","entropy"]], on=["prompt","label"])
    r=stats.pearsonr(m["gradH"],m["sigma_max"]).statistic
    pr=partial_r(m["gradH"].values,m["sigma_max"].values,m["lambda_bulk_pred"].values)
    print(f"\n=== {name} (n={len(m)}) ===")
    print(f"  r(gradH, sigma_max)                 = {r:+.3f}")
    print(f"  PARTIAL r(gradH, sigma_max | bulk)  = {pr:+.3f}   <- se ~0: e' solo magnitudine")
    print(f"  r(gradH, bulk)={stats.pearsonr(m['gradH'],m['lambda_bulk_pred']).statistic:+.2f}  r(sigma,bulk)={stats.pearsonr(m['sigma_max'],m['lambda_bulk_pred']).statistic:+.2f}")
    # within-category dCor(sigma,entropy) + permutation p on pooled
    wc=[dcor(m[m.label==c]["sigma_max"],m[m.label==c]["entropy"]) for c in CATS]
    pooled=dcor(m["sigma_max"],m["entropy"])
    perm=[dcor(m["sigma_max"],RNG.permutation(m["entropy"].values)) for _ in range(2000)]
    pval=(np.sum(np.array(perm)>=pooled)+1)/(len(perm)+1)
    print(f"  dCor(sigma,entropy) pooled={pooled:.2f} (perm p={pval:.3f}) | within-cat mean={np.mean(wc):.2f} {[round(x,2) for x in wc]}")
