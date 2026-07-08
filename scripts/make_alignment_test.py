#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera-ready point 2(b): the collinearity-IMMUNE alignment measure.
cos_angle( grad_{H0} H , v_lead ), where v_lead is the leading right singular vector of the
hidden Jacobian J = d h_{L,T}/d H0 (Algorithm 2) and grad_{H0}H is the entropy input-gradient
(one backward). Both live in the same input-embedding space (T x d). If |cos| sits at the
random-vector baseline ~ sqrt(2/(pi N)), then even the r=0.45 like-for-like coupling carries NO
directional alignment -- it is purely shared magnitude, and no partialling arithmetic is involved.
CPU only (autograd double-backward is unstable on MPS for RMSNorm). Qwen2.5-0.5B, the run prompts."""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, csv, numpy as np, torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spectral as S

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results_analysis"; OUT.mkdir(exist_ok=True)
MODEL="Qwen/Qwen2.5-0.5B-Instruct"; SLUG="Qwen__Qwen2.5-0.5B-Instruct"
CATS=["factual","coding","reasoning","hallucination_prone"]; N=40
K_TOP=1; N_ITER=12; CSV=OUT/"alignment_cos.csv"

torch.set_num_threads(max(1, (os.cpu_count() or 4)-2))
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").eval().to("cpu")
model.config.use_cache=False
has_chat=getattr(tok,"chat_template",None) is not None

def gradH_vec(prompt):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True) if has_chat else prompt
    enc=tok(text,return_tensors="pt")
    emb=model.get_input_embeddings()(enc["input_ids"]).detach().clone().requires_grad_(True)
    z=model(inputs_embeds=emb,attention_mask=enc["attention_mask"],return_dict=True).logits[0,-1].float()
    lp=torch.log_softmax(z,-1); H=-(lp.exp()*lp).sum()
    g,=torch.autograd.grad(H,emb)
    return g.detach().float().cpu().numpy()  # [1,T,d]

done=set()
if CSV.exists():
    import pandas as pd
    for _,r in pd.read_csv(CSV).iterrows(): done.add((r["label"],r["prompt"]))
write_header = not CSV.exists()
f=open(CSV,"a",newline=""); w=csv.writer(f)
if write_header: w.writerow(["label","prompt","T","n_dim","cos_abs","rand_baseline","sigma_max","gradH_norm"]); f.flush()

prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
for c in CATS:
    for pr in prompts[c][:N]:
        if (c,pr) in done: continue
        try:
            cb=S.torch_build_callables(pr,tok,model,final_index=-2)
            top=S.jacobian_topk_singular(cb["jvp_final"],cb["vjp_final"],cb["in_shape"],k=K_TOP,n_iter=N_ITER,seed=1)
            v=np.asarray(top["v_lead"],float).ravel()
            g=gradH_vec(pr).ravel()
            nd=v.size
            cos=abs(float(np.dot(v,g)/((np.linalg.norm(v)*np.linalg.norm(g))+1e-30)))
            base=float(np.sqrt(2.0/(np.pi*nd)))
            w.writerow([c,pr,cb["in_shape"][1],nd,f"{cos:.5f}",f"{base:.5f}",
                        f"{float(top['sigmas'][0]):.4f}",f"{float(np.linalg.norm(g)):.4f}"]); f.flush()
            print(f"[{c[:4]}] T={cb['in_shape'][1]:3d} |cos|={cos:.4f} baseline={base:.4f} ratio={cos/base:5.1f}x", flush=True)
        except Exception as e:
            print(f"[{c[:4]}] ERROR {type(e).__name__}: {e}", flush=True)
f.close()

import pandas as pd
d=pd.read_csv(CSV)
print(f"\n=== alignment summary (Qwen, n={len(d)}) ===")
print(f"  mean |cos(gradH, v_lead)| = {d['cos_abs'].mean():.4f}  (median {d['cos_abs'].median():.4f})")
print(f"  mean random baseline      = {d['rand_baseline'].mean():.4f}")
print(f"  mean ratio to baseline    = {(d['cos_abs']/d['rand_baseline']).mean():.2f}x")
print(f"  by category:", {c: round(d[d.label==c]['cos_abs'].mean(),4) for c in CATS})
json.dump({"n":int(len(d)),"mean_cos_abs":round(float(d['cos_abs'].mean()),4),
           "median_cos_abs":round(float(d['cos_abs'].median()),4),
           "mean_baseline":round(float(d['rand_baseline'].mean()),4),
           "mean_ratio":round(float((d['cos_abs']/d['rand_baseline']).mean()),2)},
          open(OUT/"alignment_summary.json","w"),indent=2)
print("saved results_analysis/alignment_summary.json")
