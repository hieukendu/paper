"""Exact five-fold threshold audit for a dedicated binary-expert prediction file."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
from final_best_tuned_exact_oof import evaluate_label
from vipragsent.utils.io import read_jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument("--label",required=True);p.add_argument("--predictions",type=Path,required=True);p.add_argument("--name",required=True);p.add_argument("--candidate-dir",type=Path,default=ROOT/"answer/final_best_tuned_candidates");p.add_argument("--dev",type=Path,default=ROOT/"data/processed/vipragsent_dev.jsonl");p.add_argument("--output-dir",type=Path);p.add_argument("--register",action="store_true");a=p.parse_args()
 rows=list(read_jsonl(a.dev));ids=[str(r["id"]) for r in rows]; values={str(r["id"]):float(r["probability"]) for r in read_jsonl(a.predictions)}
 if set(values)!=set(ids):raise ValueError("development ID mismatch")
 folds_doc=json.loads((a.candidate_dir/"configs/development_folds.json").read_text());folds=[int(folds_doc["assignments"][rid]) for rid in ids]
 y=np.asarray([int(r["labels"][a.label]) for r in rows]); result=evaluate_label(y,np.asarray([values[rid] for rid in ids]),folds,a.label,fp_limit=1 if a.label=="idiom_figurative" else None)
 out=a.output_dir or a.candidate_dir/"next_cycle"/"oof";out.mkdir(parents=True,exist_ok=True);(out/f"{a.name}.json").write_text(json.dumps(result,indent=2)+"\n")
 registry=a.candidate_dir/"experiment_registry.csv";fields=["branch","stage","status","candidate","target_label","alpha","selection_score","oof_f1","fold_mean","fold_std","full_dev_threshold","notes"]
 row={"branch":"new_target_representation","stage":"A","status":"evaluated","candidate":a.name,"target_label":a.label,"alpha":"","selection_score":result["selection_score"],"oof_f1":result["oof_binary_macro_f1"],"fold_mean":result["fold_mean"],"fold_std":result["fold_std"],"full_dev_threshold":result["full_dev_threshold"],"notes":"dedicated binary expert; exact nested OOF threshold; train-only checkpoint selection"}
 if a.register:
  existing=list(csv.DictReader(registry.open())) if registry.exists() else []
  with registry.open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(existing);w.writerow(row)
  j=registry.with_suffix(".json");rowsj=json.loads(j.read_text()) if j.exists() else [];rowsj.append(row);j.write_text(json.dumps(rowsj,indent=2)+"\n")
 print(json.dumps({k:result[k] for k in ("label","oof_binary_macro_f1","fold_mean","fold_std","selection_score","full_dev_threshold","oof_fp","oof_fn")},indent=2))
if __name__=="__main__":main()
