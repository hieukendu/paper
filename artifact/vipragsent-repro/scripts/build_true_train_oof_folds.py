"""Create deterministic five-fold train manifests for genuine cross-fitted experts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
from final_best_tuned_exact_oof import stable_metadata_folds
from vipragsent.utils.io import read_jsonl
def main():
 p=argparse.ArgumentParser();p.add_argument("--train",type=Path,default=ROOT/"data/processed/vipragsent_train.jsonl");p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 rows=list(read_jsonl(a.train));folds,diag=stable_metadata_folds(rows);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({"folds":5,"assignments":{str(row["id"]):int(fold) for row,fold in zip(rows,folds)},**diag},indent=2)+"\n")
 for fold in range(5):
  path=a.output.parent/f"train_fold_{fold}_heldout.jsonl"
  with path.open("w") as h:
   for row,item in zip(rows,folds):
    if item==fold:h.write(json.dumps(row,ensure_ascii=False)+"\n")
 print(json.dumps({"records":len(rows),"output":str(a.output),"fold_sizes":diag["fold_sizes"]},indent=2))
if __name__=="__main__":main()
