"""Direct metric/ID verification of the immutable final_best_tuned incumbent."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT/"src"))
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl
def main():
 final=ROOT/"answer/final_best_tuned"; gold={str(x["id"]):x for x in read_jsonl(ROOT/"data/processed/vipragsent_test.jsonl")};pred={str(x["id"]):x for x in read_jsonl(final/"predictions/final_test_predictions.jsonl")}
 if set(gold)!=set(pred):raise ValueError("canonical test ID mismatch")
 ids=sorted(gold);scores={}
 for label in PRAGMATIC_LABELS:
  y=[int(gold[i]["labels"][label]) for i in ids];p=[int(pred[i]["predictions"][label]) for i in ids];scores[label]=float(f1_score(y,p,average="macro",zero_division=0)*100)
 payload={"status":"verified","prediction_path":str(final/"predictions/final_test_predictions.jsonl"),"gold_path":str(ROOT/"data/processed/vipragsent_test.jsonl"),"records":len(ids),"metrics":scores,"macro_pragmatic_f1":float(np.mean(list(scores.values()))),"selection_use":False,"purpose":"incumbent verification only"}
 out=ROOT/"answer/final_best_tuned_candidates/next_cycle/incumbent_verification.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
