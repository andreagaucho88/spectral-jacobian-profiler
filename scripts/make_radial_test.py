#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review item 6: turn the radial reading from 'consistent with' into a MEASUREMENT.
At the pre-norm site, with the radial direction hhat = h_{L,T}/||h_{L,T}|| (output space):
  (a) |<u1, hhat>|  -- alignment of the leading LEFT singular vector u1 = J v_lead/sigma_max with the
      radial direction: how much the most-amplified output direction is radial (norm-changing).
  (b) rho = ||hhat^T J||^2 / ||J||_F^2  -- radial FRACTION of the bulk (hhat^T J = grad_{H0}||h_{L,T}||,
      one VJP), vs the tangential remainder 1 - rho.
PRE-SPECIFIED PREDICTION (before running): factual has higher |<u1,hhat>| and rho pre-norm than
coding (its excess pre-norm amplification lies along the radial direction, removed by the final norm).
CPU only, Qwen2.5-0.5B, the n=40/category exact-Jacobian prompts."""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, csv, numpy as np, torch
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
import spectral as S

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results_analysis"; OUT.mkdir(exist_ok=True)
MODEL="Qwen/Qwen2.5-0.5B-Instruct"; SLUG="Qwen__Qwen2.5-0.5B-Instruct"
CATS=["factual","coding","reasoning","hallucination_prone"]; N=40
K_TOP=1; N_ITER=12; N_PROBE=12; CSV=OUT/"radial_test.csv"
RNG=np.random.default_rng(0)
torch.set_num_threads(max(1,(os.cpu_count() or 4)-2))
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,attn_implementation="eager").eval().to("cpu")
model.config.use_cache=False

done=set()
if CSV.exists():
    import pandas as pd
    for _,r in pd.read_csv(CSV).iterrows(): done.add((r["label"],r["prompt"]))
f=open(CSV,"a",newline=""); w=csv.writer(f)
if not done: w.writerow(["label","prompt","sigma_max","cos_u1_hhat","rho_radial","frob_sq"]); f.flush()

prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
for c in CATS:
    for pr in prompts[c][:N]:
        if (c,pr) in done: continue
        try:
            cb=S.torch_build_callables(pr,tok,model,final_index=-2)
            x0=cb["x0"]; jvp=cb["jvp_final"]; vjp=cb["vjp_final"]; in_shape=cb["in_shape"]
            h=cb["forward_final"](x0).astype(float)                 # pre-norm h_{L,T} (d,)
            hhat=h/(np.linalg.norm(h)+1e-30)
            top=S.jacobian_topk_singular(jvp,vjp,in_shape,k=K_TOP,n_iter=N_ITER,seed=1)
            v=np.asarray(top["v_lead"],float); smax=float(top["sigmas"][0])
            u1=jvp(v).astype(float); u1/= (np.linalg.norm(u1)+1e-30)  # leading LEFT singular vec (d,)
            cos_u1=abs(float(np.dot(u1,hhat)))
            radial_row=vjp(hhat).astype(float)                       # hhat^T J = grad||h||, input shape
            radial_sq=float(np.sum(radial_row**2))
            # ||J||_F^2 via unit-sphere probes: E||J xi||^2 = ||J||_F^2 / N
            n_dim=int(np.prod(in_shape))
            fs=[]
            for j in range(N_PROBE):
                xi=RNG.standard_normal(in_shape); xi/=np.linalg.norm(xi.ravel())
                fs.append(float(np.sum(jvp(xi).astype(float)**2)))
            frob_sq=n_dim*float(np.mean(fs))
            rho=radial_sq/(frob_sq+1e-30)
            w.writerow([c,pr,f"{smax:.4f}",f"{cos_u1:.5f}",f"{rho:.5f}",f"{frob_sq:.4f}"]); f.flush()
            print(f"[{c[:4]}] sigma={smax:7.1f} |<u1,hhat>|={cos_u1:.3f} rho_radial={rho:.3f}",flush=True)
        except Exception as e:
            print(f"[{c[:4]}] ERROR {type(e).__name__}: {e}",flush=True)
f.close()

import pandas as pd
from scipy import stats
d=pd.read_csv(CSV)
def d_cohen(a,b):
    na,nb=len(a),len(b); sp=np.sqrt(((na-1)*a.var(ddof=1)+(nb-1)*b.var(ddof=1))/(na+nb-2))
    return (a.mean()-b.mean())/(sp+1e-30)
summ={"n":int(len(d))}
for m in ["cos_u1_hhat","rho_radial"]:
    by={c:round(float(d[d.label==c][m].mean()),3) for c in CATS}
    fa=d[d.label=="factual"][m]; co=d[d.label=="coding"][m]
    dd=float(d_cohen(fa,co)); p=float(stats.mannwhitneyu(fa,co,alternative="greater").pvalue)
    summ[m]={"by_cat":by,"factual_vs_coding_d":round(dd,2),"MWU_p_fac>cod":round(p,4)}
    print(f"\n{m}: by category {by}\n  factual vs coding: d={dd:+.2f}, MWU p(fac>cod)={p:.4f}")
json.dump(summ,open(OUT/"radial_test_summary.json","w"),indent=2)
print("\nsaved results_analysis/radial_test_summary.json")
