#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera-ready point 2: pre-empt the collinearity objection to the partial correlation.
(a) r(||dH||, lambda_bulk): if high, "big Jacobians are big everywhere" is shown from the
    RIGHT side (the entropy sensitivity is itself magnitude-driven), so the 0.45->0.10-0.30
    drop is a real shared-magnitude fact, not just partialling arithmetic.
(b) bootstrap CI on each partial correlation, especially GPT-2 at n=64.
Also saves per-prompt merged data per model and reports the CORRECT within-category dCor."""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, numpy as np, pandas as pd, torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results_analysis"; OUT.mkdir(exist_ok=True)
CATS=["factual","coding","reasoning","hallucination_prone"]; N=40; RNG=np.random.default_rng(0)
MODELS=[("Qwen2.5-0.5B","Qwen/Qwen2.5-0.5B-Instruct","Qwen__Qwen2.5-0.5B-Instruct","results_spectral"),
        ("SmolLM2-360M","HuggingFaceTB/SmolLM2-360M-Instruct","HuggingFaceTB__SmolLM2-360M-Instruct","results_spectral"),
        ("GPT-2-124M","gpt2","gpt2","results_spectral_gpt2")]

def partial_r(a,b,c):  # r(a,b | c)
    rab=stats.pearsonr(a,b).statistic; rac=stats.pearsonr(a,c).statistic; rbc=stats.pearsonr(b,c).statistic
    return (rab-rac*rbc)/np.sqrt((1-rac**2)*(1-rbc**2))

def boot_ci_partial(a,b,c,n_boot=4000):
    n=len(a); idx=np.arange(n); vals=[]
    for _ in range(n_boot):
        s=RNG.choice(idx,n,replace=True)
        try: vals.append(partial_r(a[s],b[s],c[s]))
        except Exception: pass
    lo,hi=np.percentile(vals,[2.5,97.5]); return lo,hi

def dcor(x,y):  # CORRECT normalization: dCor = sqrt(dCov2)/sqrt(sqrt(dVx)*sqrt(dVy))
    x=np.asarray(x,float);y=np.asarray(y,float)
    a=np.abs(x[:,None]-x[None,:]);b=np.abs(y[:,None]-y[None,:])
    A=a-a.mean(0)-a.mean(1)[:,None]+a.mean();B=b-b.mean(0)-b.mean(1)[:,None]+b.mean()
    dcov2=(A*B).mean();dvx=(A*A).mean();dvy=(B*B).mean()
    den=np.sqrt(np.sqrt(dvx)*np.sqrt(dvy))
    return float(np.sqrt(max(dcov2,0.0))/den) if den>0 else 0.0

summary={}
for name,mid,slug,base in MODELS:
    tok=AutoTokenizer.from_pretrained(mid)
    model=AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.config.use_cache=False
    has_chat = getattr(tok,"chat_template",None) is not None
    def gradH(prompt):
        text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True) if has_chat else prompt
        ids=tok(text,return_tensors="pt")["input_ids"]
        emb=model.get_input_embeddings()(ids).detach().clone().requires_grad_(True)
        z=model(inputs_embeds=emb,return_dict=True).logits[0,-1].float()
        lp=torch.log_softmax(z,-1); H=-(lp.exp()*lp).sum()
        g,=torch.autograd.grad(H,emb); return float(g.norm())
    prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{slug}.json"))
    rows=[{"prompt":pr,"label":c,"gradH":gradH(pr)} for c in CATS for pr in prompts[c][:N]]
    g=pd.DataFrame(rows)
    sp=pd.read_csv(ROOT/f"{base}/{slug}/final_index_-2/spectral_per_prompt.csv")
    sp=sp[sp["error"].astype(str).str.strip().isin(["","nan"])]
    m=pd.merge(g, sp[["prompt","label","sigma_max","lambda_bulk_pred","entropy"]], on=["prompt","label"])
    m.to_csv(OUT/f"coupling_{slug}.csv",index=False)
    a,b,c=m["gradH"].values,m["sigma_max"].values,m["lambda_bulk_pred"].values
    r_gs=stats.pearsonr(a,b).statistic; r_gb=stats.pearsonr(a,c).statistic; r_sb=stats.pearsonr(b,c).statistic
    pr=partial_r(a,b,c); lo,hi=boot_ci_partial(a,b,c)
    wc=[dcor(m[m.label==cc]["sigma_max"],m[m.label==cc]["entropy"]) for cc in CATS]
    pooled=dcor(m["sigma_max"],m["entropy"])
    summary[name]={"n":len(m),"chat_template":has_chat,
        "r_gradH_sigma":round(r_gs,3),"r_gradH_bulk":round(r_gb,3),"r_sigma_bulk":round(r_sb,3),
        "partial_gradH_sigma_given_bulk":round(pr,3),"partial_CI95":[round(lo,3),round(hi,3)],
        "dcor_sigma_entropy_pooled":round(pooled,3),"dcor_within_cat_mean":round(float(np.mean(wc)),3)}
    print(f"\n=== {name} (n={len(m)}, chat_template={has_chat}) ===")
    print(f"  r(gradH, sigma)={r_gs:+.3f}  r(gradH, BULK)={r_gb:+.3f}  <-(a) right-side magnitude  r(sigma,bulk)={r_sb:+.3f}")
    print(f"  PARTIAL r(gradH,sigma|bulk)={pr:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  (width {hi-lo:.3f})")
    print(f"  dCor(sigma,entropy) pooled={pooled:.3f}  within-cat mean={np.mean(wc):.3f}")

json.dump(summary, open(OUT/"coupling_extra_summary.json","w"), indent=2)
print("\nsaved results_analysis/coupling_extra_summary.json and coupling_<slug>.csv")
