"""ID-aligned uniform/weighted ensemble for dedicated binary probability files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",action="append",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--weights",default="");a=p.parse_args();rows=[]
 for path in a.input:rows.append({str(x["id"]):x for x in read_jsonl(path)})
 ids=list(rows[0]);
 if any(set(source)!=set(ids) for source in rows):raise ValueError("ID mismatch")
 weights=[float(x) for x in a.weights.split(",")] if a.weights else [1/len(rows)]*len(rows)
 if len(weights)!=len(rows) or min(weights)<0 or abs(sum(weights)-1)>1e-8:raise ValueError("invalid weights")
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w") as h:
  for rid in ids:
   base=rows[0][rid];h.write(json.dumps({"id":rid,"probability":sum(w*source[rid]["probability"] for w,source in zip(weights,rows)),"label":base["label"],"system":"uniform_target_expert_ensemble","seeds":[source[rid].get("seed") for source in rows]})+"\n")
 print(json.dumps({"records":len(ids),"weights":weights,"output":str(a.output)},indent=2))
if __name__=="__main__":main()
