#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Review point 1: template-GROUPED linear probe (no near-duplicate leakage across folds).
Extracts the d=896 pre-norm final-token hidden state for the 160 Qwen prompts (saved to an .npz for
reproducibility), joins template_id (for grouping), the four spectral scalars, and (||h_pre||,
||h_post||, ppl). Runs each of the three probes under 5-fold StratifiedKFold (ungrouped) and
StratifiedGroupKFold on template_id (grouped); the scaler is fit inside each training fold (no
standardization leakage). Reports six accuracies + per-class recall for the grouped hidden probe.

PRE-SPECIFIED CRITERION (before running): grouped hidden-state accuracy >=90% -> variant A;
70-90% -> intermediate; <70% -> variant B."""
from __future__ import annotations
import os; os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import json, numpy as np, pandas as pd, torch
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import recall_score

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"results_analysis"; OUT.mkdir(exist_ok=True)
MODEL="Qwen/Qwen2.5-0.5B-Instruct"; SLUG="Qwen__Qwen2.5-0.5B-Instruct"
CATS=["factual","coding","reasoning","hallucination_prone"]; N=40; NPZ=OUT/"probe_hidden_states.npz"
torch.set_num_threads(max(1,(os.cpu_count() or 4)-2))

# ---- extract (or reload) the d=896 hidden states + norms + ppl ----
lm=pd.read_csv(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.csv")[["label","prompt","template_id"]]
if NPZ.exists():
    z=np.load(NPZ, allow_pickle=True); Xh=z["Xh"]; labels=list(z["labels"]); prompts=list(z["prompts"])
    tpl=list(z["template_id"]); hpre=z["hpre_norm"]; hpost=z["hpost_norm"]; ppl=z["ppl"]
    print(f"reloaded {NPZ.name}: {Xh.shape}")
else:
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,attn_implementation="eager").eval().to("cpu")
    model.config.use_cache=False
    prompts=json.load(open(ROOT/f"lengthmatched_prompts/lengthmatched_prompts__{SLUG}.json"))
    Xh=[]; labels=[]; plist=[]; tpl=[]; hpre=[]; hpost=[]; ppl=[]
    tmap={(r.label,r.prompt):r.template_id for r in lm.itertuples()}
    for c in CATS:
        for pr in prompts[c][:N]:
            text=tok.apply_chat_template([{"role":"user","content":pr}],tokenize=False,add_generation_prompt=True)
            ids=tok(text,return_tensors="pt")["input_ids"]
            with torch.no_grad():
                out=model(ids,output_hidden_states=True,return_dict=True)
            hpre_v=out.hidden_states[-2][0,-1,:].float().numpy()
            hpost_v=out.hidden_states[-1][0,-1,:].float().numpy()
            # prompt perplexity (mean NLL over the prompt tokens)
            lg=out.logits[0,:-1,:].float(); tgt=ids[0,1:]
            nll=torch.nn.functional.cross_entropy(lg,tgt).item()
            Xh.append(hpre_v); labels.append(c); plist.append(pr); tpl.append(tmap.get((c,pr),f"{c}_?"))
            hpre.append(float(np.linalg.norm(hpre_v))); hpost.append(float(np.linalg.norm(hpost_v))); ppl.append(float(np.exp(nll)))
    Xh=np.array(Xh); prompts=plist; hpre=np.array(hpre); hpost=np.array(hpost); ppl=np.array(ppl)
    np.savez_compressed(NPZ, Xh=Xh, labels=np.array(labels), prompts=np.array(prompts,dtype=object),
                        template_id=np.array(tpl,dtype=object), hpre_norm=hpre, hpost_norm=hpost, ppl=ppl)
    print(f"extracted + saved {NPZ.name}: {Xh.shape}, {len(set(tpl))} templates")

y=np.array(labels); groups=np.array([str(x) for x in tpl])
# ---- join spectral scalars ----
sp=pd.read_csv(ROOT/f"results_spectral/{SLUG}/final_index_-2/spectral_per_prompt.csv")
sp=sp[sp["error"].astype(str).str.strip().isin(["","nan"])]
spmap=sp.set_index(["label","prompt"])[["sigma_max","lambda_bulk_pred","stable_rank","chi_fisher_max"]]
Xspec=np.array([spmap.loc[(labels[i],prompts[i])].values for i in range(len(labels))],dtype=float)
Xnorm=np.column_stack([hpre,hpost,ppl])

print(f"\nn={len(y)}, templates/cat:", {c:len(set(groups[y==c])) for c in CATS})
def probe(X, grouped):
    pipe=make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=5000))
    cv=(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=0) if grouped
        else StratifiedKFold(n_splits=5,shuffle=True,random_state=0))
    kw={"groups":groups} if grouped else {}
    return cross_val_score(pipe,X,y,cv=cv,**kw).mean(), (pipe,cv,kw)

feats=[("hidden d=896",Xh),("4 spectral scalars",Xspec),("3 norm/ppl scalars",Xnorm)]
res={}
for name,X in feats:
    ung,_=probe(X,False); grp,(pipe,cv,kw)=probe(X,True)
    res[name]={"ungrouped":round(float(ung),3),"grouped":round(float(grp),3)}
    print(f"  {name:20s}: ungrouped={ung:.3f}  GROUPED={grp:.3f}")
    if name=="hidden d=896":
        yp=cross_val_predict(pipe,X,y,cv=cv,**kw)
        rec={c:round(float(recall_score(y,yp,labels=[c],average=None)[0]),2) for c in CATS}
        res["hidden_grouped_recall_per_class"]=rec; print(f"    grouped per-class recall: {rec}")
g=res["hidden d=896"]["grouped"]
verdict="A (holds, >=90%)" if g>=0.90 else ("intermediate (70-90%)" if g>=0.70 else "B (<70%)")
res["hidden_grouped"]=g; res["verdict"]=verdict; print(f"\nPRE-SPECIFIED VERDICT: grouped hidden={g:.3f} -> variant {verdict}")
json.dump(res,open(OUT/"grouped_probe_summary.json","w"),indent=2); print("saved grouped_probe_summary.json")
