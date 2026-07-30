"""Correct protected-label leakage in targeted development selection."""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_fair_framework_cycle_v2 as core
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl

OUT=ROOT/'answer/final_best_tuned_fair_framework_candidates'
TARGETS=('irony','idiom_figurative','code_switching'); PROTECTED=tuple(x for x in PRAGMATIC_LABELS if x not in TARGETS)

def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 rows=list(read_jsonl(ROOT/'data/processed/vipragsent_dev.jsonl')); ids=[str(r['id']) for r in rows]
 inc=core.load_binary(ROOT/'answer/final_best_tuned/predictions/final_dev_predictions.jsonl',ids)
 screened={str(r['id']):r for r in read_jsonl(OUT/'targeted_retraining_screen_oof_predictions.jsonl')}
 if set(screened)!=set(ids): raise ValueError('targeted OOF ID mismatch')
 results={}; registry=[]
 for label in TARGETS:
  y=np.asarray([r['labels'][label] for r in rows]); base=np.asarray([inc[i][label] for i in ids]); pred=np.asarray([screened[i]['predictions'][label] for i in ids])
  cm,base_cm=core.binary(y,pred),core.binary(y,base); delta=cm['binary_macro_f1']-base_cm['binary_macro_f1']
  introduced_fp=int(((y==0)&(base==0)&(pred==1)).sum()); rescued_fn=int(((y==1)&(base==0)&(pred==1)).sum())
  eligible=delta>0 and (label=='code_switching' or introduced_fp==0)
  reason='development-selected' if eligible else ('non-positive same-split OOF delta' if delta<=0 else 'zero-FP target constraint failed')
  results[label]={'candidate':'target_expert_nested_oof','candidate_oof_f1':cm['binary_macro_f1'],'incumbent_oof_f1':base_cm['binary_macro_f1'],'target_delta':delta,'confusion':cm,'incumbent_confusion':base_cm,'rescued_FN':rescued_fn,'introduced_FP':introduced_fp,'eligible':eligible,'reason':reason}
  registry.append({'target':label,'candidate':'target_expert_nested_oof','candidate_oof_f1':cm['binary_macro_f1'],'incumbent_oof_f1':base_cm['binary_macro_f1'],'target_delta':delta,'eligible':eligible,'rejection_reason':'' if eligible else reason})
 protected={label:'copied exactly from incumbent; excluded from development eligibility' for label in PROTECTED}
 selection={'run_id':'target-only-selection-v1','selection':results,'protected_labels':protected,'rule':'target-only same-split development OOF delta; canonical-test baselines reserved exclusively for promotion'}
 dump(OUT/'development_selection.json',selection)
 with (OUT/'experiment_registry.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(registry[0]));w.writeheader();w.writerows(registry)
 status={'run_id':'target-only-selection-v1','status':'NOT_PROMOTED','phase':'development_target_selection','test_evaluations':0,'selection':results,'protected_labels':protected,'reason':'canonical test forbidden until all three target labels are development-selected with paired inference'}
 dump(OUT/'status.json',status); dump(OUT/'candidate_metrics.json',{'test_evaluations':0,'target_only_results':results})
 report=['# ViPragSent target-only development selection','','## Target results','','| Target | Candidate OOF | Incumbent OOF | Delta | Status |','| --- | ---: | ---: | ---: | --- |']
 for x in TARGETS:
  r=results[x];report.append(f"| {x} | {r['candidate_oof_f1']:.10f} | {r['incumbent_oof_f1']:.10f} | {r['target_delta']:+.10f} | {r['reason']} |")
 report+=['','Protected labels (`implicit_sentiment`, `sarcasm`, `mocking`) are copied unchanged from the incumbent and were not scored, selected, or rejected against test baselines. Canonical-test baseline maxima are reserved for the final promotion gate only.','','NOT_PROMOTED']
 (OUT/'FAIR_FRAMEWORK_CYCLE_REPORT.md').write_text('\n'.join(report)+'\n')
 print(json.dumps(results,indent=2));return 0
if __name__=='__main__': raise SystemExit(main())
