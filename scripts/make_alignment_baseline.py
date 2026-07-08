#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review point 5a: the POSITION-MATCHED null for |cos(grad_H0 H, v_lead)|.
Both grad_H0 H and v_lead concentrate their energy on the last token positions, so the
fully-random baseline sqrt(2/(pi N)) understates chance alignment (effective dim << Td).
Correct null: random directions r with the SAME per-position energy profile as grad H (random
direction *within* each position), compared to v_lead. If |cos(grad H, v_lead)| > this
position-matched baseline, there is genuine within-position directional alignment beyond shared
positional support. CPU only, Qwen2.5-0.5B, the exact run prompts (recomputes v_lead + grad H)."""
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
K_TOP=1; N_ITER=12; R_DRAWS=400; CSV=OUT/"alignment_baseline.csv"
RNG=np.random.default_rng(0)

torch.set_num_threads(max(1,(os.cpu_count() or 4)-2))
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,attn_implementation="eager").eval().to("cpu")
model.config.use_cache=False
has_chat=getattr(tok,"chat_template",None) is not None

def gradH_vec(prompt):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True) if has_chat else prompt
    enc=tok(text,return_tensors="pt")
    emb=model.get_input_embeddings()(enc["input_ids"]).detach().clone().requires_grad_(True)
    z=model(inputs_embeds=emb,attention_mask=enc["attention_mask"],return_dict=True).logits[0,-1].float()
    lp=torch.log_softmax(z,-1); H=-(lp.exp()*lp).sum()
    g,=torch.autograd.grad(H,emb)
    return g.detach().float().cpu().numpy()[0]  # [T,d]

def pos_matched_baseline(g_td, v_td):
    """mean |cos(r, v)| over R random directions r with g's per-position norm profile."""
    T,d=g_td.shape
    gp=np.linalg.norm(g_td,axis=1)                          # [T] per-position energy of grad H
    v=v_td.ravel(); vn=np.linalg.norm(v)+1e-30
    U=RNG.standard_normal((R_DRAWS,T,d))
    U/=(np.linalg.norm(U,axis=2,keepdims=True)+1e-30)       # unit within each position
    R=U*gp[None,:,None]                                     # scale by grad H's per-position energy
    Rf=R.reshape(R_DRAWS,-1)
    cos=np.abs(Rf@v)/((np.linalg.norm(Rf,axis=1)+1e-30)*vn)
    # analytic check: sqrt(2/pi)*sqrt(sum_t gp^2 vp^2 / d)/(||g|| ||v||)
    vp=np.linalg.norm(v_td,axis=1)
    ana=np.sqrt(2/np.pi)*np.sqrt(np.sum(gp**2*vp**2)/d)/((np.linalg.norm(g_td)+1e-30)*np.linalg.norm(v_td)+1e-30)
    return float(cos.mean()), float(ana)

done=set()
if CSV.exists():
    import pandas as pd
    for _,r in pd.read_csv(CSV).iterrows(): done.add((r["label"],r["prompt"]))
f=open(CSV,"a",newline=""); w=csv.writer(f)
if not done: w.writerow(["label","prompt","T","cos_abs","rand_baseline","pos_baseline_mc","pos_baseline_ana"]); f.flush()

prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
for c in CATS:
    for pr in prompts[c][:N]:
        if (c,pr) in done: continue
        try:
            cb=S.torch_build_callables(pr,tok,model,final_index=-2)
            top=S.jacobian_topk_singular(cb["jvp_final"],cb["vjp_final"],cb["in_shape"],k=K_TOP,n_iter=N_ITER,seed=1)
            T,d=cb["in_shape"][1],cb["in_shape"][2]
            v_td=np.asarray(top["v_lead"],float).reshape(T,d)
            g_td=gradH_vec(pr)
            cos=abs(float(np.dot(v_td.ravel(),g_td.ravel())/((np.linalg.norm(v_td)*np.linalg.norm(g_td))+1e-30)))
            rand=float(np.sqrt(2.0/(np.pi*T*d)))
            pmc,pana=pos_matched_baseline(g_td,v_td)
            w.writerow([c,pr,T,f"{cos:.5f}",f"{rand:.5f}",f"{pmc:.5f}",f"{pana:.5f}"]); f.flush()
            print(f"[{c[:4]}] |cos|={cos:.3f} rand={rand:.4f} pos-null={pmc:.3f} ratio_pos={cos/max(pmc,1e-9):4.1f}x",flush=True)
        except Exception as e:
            print(f"[{c[:4]}] ERROR {type(e).__name__}: {e}",flush=True)
f.close()

import pandas as pd
d=pd.read_csv(CSV)
summ={"n":int(len(d)),"mean_cos":round(float(d.cos_abs.mean()),3),"median_cos":round(float(d.cos_abs.median()),3),
      "mean_rand_baseline":round(float(d.rand_baseline.mean()),4),
      "mean_pos_baseline_mc":round(float(d.pos_baseline_mc.mean()),3),
      "mean_pos_baseline_ana":round(float(d.pos_baseline_ana.mean()),3),
      "ratio_vs_rand":round(float((d.cos_abs/d.rand_baseline).mean()),1),
      "ratio_vs_pos":round(float((d.cos_abs/d.pos_baseline_mc).mean()),2),
      "frac_cos_above_pos":round(float((d.cos_abs>d.pos_baseline_mc).mean()),3)}
json.dump(summ,open(OUT/"alignment_baseline_summary.json","w"),indent=2)
print("\n=== POSITION-MATCHED baseline summary ===")
for k,val in summ.items(): print(f"  {k}: {val}")
print("saved results_analysis/alignment_baseline_summary.json")
