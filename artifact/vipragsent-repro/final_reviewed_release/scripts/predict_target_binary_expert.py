"""Inference-only companion for a frozen dedicated binary expert."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT/"src"))
from train_target_binary_expert import Expert, Records, collator, load_encoder
from transformers import AutoTokenizer

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 state=torch.load(a.checkpoint,map_location="cpu",weights_only=False);cfg=state["args"];device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 tokenizer=AutoTokenizer.from_pretrained(cfg["model_id"],use_fast=True);model=Expert(load_encoder(cfg["model_id"],Path(cfg["base_state"]),Path(cfg["tapt_checkpoint"]) if cfg.get("tapt_checkpoint") else None),cfg["pooling"],cfg["head"],cfg["dropout"],cfg["token_aux"]).to(device);model.load_state_dict(state["model"]);model.eval()
 # Records reads labels for training, so construct a label-free inference set.
 rows=[]
 from vipragsent.utils.io import read_jsonl
 for row in read_jsonl(a.data): rows.append({"id":str(row["id"]),"text":str(row["text"]),"label":0.,"weight":1.})
 class D:
  def __len__(self):return len(rows)
  def __getitem__(self,i):return rows[i]
 loader=DataLoader(D(),batch_size=32,collate_fn=collator(tokenizer,cfg["token_aux"],cfg["max_length"]),num_workers=2);a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w") as h:
  with torch.no_grad():
   for batch in loader:
    data={k:v.to(device) for k,v in batch.items() if torch.is_tensor(v) and k not in {"labels","weights","token_labels"}};logit,_=model(**data)
    for rid,value in zip(batch["ids"],torch.sigmoid(logit).cpu().tolist()):h.write(json.dumps({"id":rid,"probability":value,"label":cfg["label"],"system":cfg["system"],"seed":cfg["seed"]})+"\n")
 print(json.dumps({"status":"ok","records":len(rows),"output":str(a.output),"test_labels_read":False},indent=2))
if __name__=="__main__":main()
