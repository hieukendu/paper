"""Dedicated train-only binary experts for irony, idiom, and code switching.

The script uses a train-hash validation partition for checkpoint stopping; the
development split is scored only after the checkpoint is frozen.  It supports a
phrase-sensitive attention representation and a deterministic token-language
auxiliary head for code-switching without consulting development/test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl

LABELS = ("irony", "idiom_figurative", "code_switching")
TOKEN_CLASSES = ("vietnamese", "english", "proper_name", "brand_product", "acronym", "other")
BRANDS = {"facebook", "tiktok", "youtube", "zalo", "google", "iphone", "samsung", "shopee", "lazada", "grab", "app", "ios", "android"}
ENGLISH = {"the", "a", "an", "and", "or", "but", "with", "for", "from", "to", "of", "in", "on", "at", "is", "are", "was", "were", "be", "have", "has", "i", "you", "we", "they", "love", "like", "hate", "good", "bad", "nice", "cool", "thanks", "thank", "please", "sorry", "hello", "bye", "ok", "okay", "wow", "why", "what", "when", "where", "how", "my", "your", "this", "that", "it", "not", "so", "very", "really", "work", "job", "money", "game", "music", "movie", "food", "happy", "sad", "new", "old", "best", "worst", "lol", "omg"}


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_encoder(model_id: str, base_state: Path, tapt_checkpoint: Path | None):
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_id))
    state = torch.load(base_state, map_location="cpu", weights_only=True)
    state = {key.removeprefix("roberta."): value for key, value in state.items() if key.startswith("roberta.") and key != "roberta.embeddings.position_ids"} or state
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    unsafe = [key for key in missing if key != "embeddings.position_ids" and not key.startswith("pooler.")]
    if unsafe or unexpected:
        raise RuntimeError(f"bad base encoder load missing={unsafe[:5]} unexpected={unexpected[:5]}")
    if tapt_checkpoint:
        state = torch.load(tapt_checkpoint, map_location="cpu", weights_only=True)["model"]
        state = {key.removeprefix("roberta."): value for key, value in state.items() if key.startswith("roberta.") and key != "roberta.embeddings.position_ids"}
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        unsafe = [key for key in missing if key != "embeddings.position_ids" and not key.startswith("pooler.")]
        if unsafe or unexpected:
            raise RuntimeError(f"bad TAPT encoder load missing={unsafe[:5]} unexpected={unexpected[:5]}")
    return encoder


def token_class(token: str) -> int:
    raw = token.strip(".,!?;:()[]{}\"'`“”")
    lower = raw.lower()
    if not raw or re.match(r"^(https?://|www\.|@|#|\d)", raw): return 5
    if lower in BRANDS: return 3
    if raw.isupper() and 2 <= len(raw) <= 10 and raw.isalpha(): return 4
    if raw[:1].isupper() and raw[1:].islower() and raw.isalpha(): return 2
    if lower in ENGLISH: return 1
    return 0


def hard_weight(record: dict, label: str, hp: float, hn: float) -> float:
    labels, text = record["labels"], str(record.get("text") or "")
    positive = int(labels[label]) == 1
    if label == "irony":
        hard = bool(labels.get("sarcasm") or labels.get("mocking"))
    elif label == "idiom_figurative":
        hard = positive and (bool(labels.get("sarcasm")) or bool(labels.get("irony")) or len(text.split()) > 14)
    else:
        tokens = text.split(); latin = sum(bool(re.fullmatch(r"[A-Za-z]+", token.strip(".,!?"))) for token in tokens)
        hard = (not positive and any(token_class(token) in {2, 3, 4, 5} for token in tokens)) or (positive and latin <= 2)
    return (hp if positive else hn) if hard else 1.0


class Records(Dataset):
    def __init__(self, path: Path, label: str, hp: float, hn: float, partition: str | None = None, exclude_ids: set[str] | None = None, extra_hard_ids: set[str] | None = None, extra_hard_weight: float = 1.0):
        rows = list(read_jsonl(path)); self.rows = []
        for row in rows:
            if exclude_ids and str(row["id"]) in exclude_ids: continue
            split = hashlib.sha256(str(row["id"]).encode()).digest()[0] % 10 == 0
            if partition == "holdout" and not split: continue
            if partition == "train" and split: continue
            record_id = str(row["id"])
            weight = hard_weight(row, label, hp, hn)
            # These IDs are derived only from predictions made while the row
            # was held out of the corresponding OOF expert training fold.
            if extra_hard_ids and record_id in extra_hard_ids:
                weight *= extra_hard_weight
            self.rows.append({"id": record_id, "text": str(row["text"]), "label": float(row["labels"][label]), "weight": weight})
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Expert(nn.Module):
    def __init__(self, encoder, pooling: str, head: str, dropout: float, token_aux: bool):
        super().__init__(); self.encoder, self.pooling, self.token_aux = encoder, pooling, token_aux
        hidden = encoder.config.hidden_size
        dim = hidden * {"cls": 1, "clsmean": 2, "clsmeanmax": 3, "attention": 2}[pooling]
        self.query = nn.Parameter(torch.empty(hidden)) if pooling == "attention" else None
        if self.query is not None: nn.init.normal_(self.query, std=.02)
        self.token_head = nn.Linear(hidden, len(TOKEN_CLASSES)) if token_aux else None
        stats = 4 if token_aux else 0
        if head == "linear": self.head = nn.Linear(dim + stats, 1)
        else:
            self.head_in = nn.Linear(dim + stats, hidden); self.head_block = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)); self.head_out = nn.Linear(hidden, 1)
        self.head_type = head; self.dropout = nn.Dropout(dropout)
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**encoder_kwargs).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype); mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        maximum = hidden.masked_fill(~attention_mask.unsqueeze(-1).bool(), torch.finfo(hidden.dtype).min).max(1).values
        if self.pooling == "cls": pooled = hidden[:, 0]
        elif self.pooling == "clsmean": pooled = torch.cat((hidden[:, 0], mean), -1)
        elif self.pooling == "clsmeanmax": pooled = torch.cat((hidden[:, 0], mean, maximum), -1)
        else:
            scores = (hidden * self.query).sum(-1).masked_fill(~attention_mask.bool(), -1e4); attn = scores.softmax(-1).unsqueeze(-1); pooled = torch.cat((hidden[:, 0], (hidden * attn).sum(1)), -1)
        token_logits = self.token_head(hidden) if self.token_aux else None
        if token_logits is not None:
            english = token_logits.softmax(-1)[..., 1] * attention_mask
            count = attention_mask.sum(1).clamp_min(1)
            transitions = ((english[:, 1:] > .5) != (english[:, :-1] > .5)).float() * attention_mask[:, 1:]
            pooled = torch.cat((pooled, english.max(1).values[:, None], (english.sum(1) / count)[:, None], (english > .5).sum(1, keepdim=True) / count[:, None], transitions.sum(1, keepdim=True) / count[:, None]), -1)
        pooled = self.dropout(pooled)
        if self.head_type == "linear": logit = self.head(pooled).squeeze(-1)
        else:
            base = self.head_in(pooled); logit = self.head_out(base + self.head_block(base)).squeeze(-1)
        return logit, token_logits


def collator(tokenizer, token_aux, max_length):
    def call(rows):
        enc = tokenizer([row["text"] for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        result = {**enc, "ids": [row["id"] for row in rows], "labels": torch.tensor([row["label"] for row in rows]), "weights": torch.tensor([row["weight"] for row in rows])}
        if token_aux:
            pseudo = torch.full_like(enc["input_ids"], -100)
            for i, row in enumerate(rows):
                try:
                    word_ids = enc.word_ids(batch_index=i) if getattr(enc, "word_ids", None) else None
                except ValueError:
                    # PhoBERT's tokenizer in this environment is a slow
                    # SentencePiece implementation.  It has no offset map;
                    # retain only deterministic non-special pseudo labels.
                    word_ids = None
                words = row["text"].split()
                if word_ids is not None:
                    for j, word_idx in enumerate(word_ids):
                            if word_idx is not None and word_idx < len(words): pseudo[i, j] = token_class(words[word_idx])
                else:
                    active = enc["attention_mask"][i].bool()
                    pseudo[i, active] = 0
                    pseudo[i, 0] = -100
            result["token_labels"] = pseudo
        return result
    return call


def predict(model, loader, device):
    model.eval(); ids=[]; probabilities=[]
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch["ids"]); data={key: value.to(device) for key, value in batch.items() if torch.is_tensor(value) and key not in {"labels", "weights", "token_labels"}}
            logit, _ = model(**data); probabilities.extend(torch.sigmoid(logit).cpu().tolist())
    return ids, probabilities


def asymmetric_loss(logits, labels, *, gamma_positive: float, gamma_negative: float, clip: float, positive_weight: float, negative_weight: float):
    """Standard asymmetric binary loss; no undocumented positive downweight."""
    p = torch.sigmoid(logits)
    p_neg = (1 - p + clip).clamp(max=1.0) if clip else 1 - p
    positive = -labels * torch.log(p.clamp_min(1e-8)) * (1 - p).pow(gamma_positive) * positive_weight
    negative = -(1 - labels) * torch.log(p_neg.clamp_min(1e-8)) * p.pow(gamma_negative) * negative_weight
    return positive + negative


def symmetric_bernoulli_kl(first, second):
    first = first.clamp(1e-6, 1 - 1e-6); second = second.clamp(1e-6, 1 - 1e-6)
    pq = first * (first.log() - second.log()) + (1 - first) * ((1 - first).log() - (1 - second).log())
    qp = second * (second.log() - first.log()) + (1 - second) * ((1 - second).log() - (1 - first).log())
    return .5 * (pq + qp).mean()


def exact_threshold(y, probabilities, fp_limit):
    values = sorted(set(float(value) for value in probabilities))
    candidates = [0.0, *values, *[(a + b) / 2 for a, b in zip(values, values[1:])], 1.0]
    scored = []
    for threshold in candidates:
        prediction = probabilities >= threshold; fp = int(((y == 0) & prediction).sum())
        scored.append((float(f1_score(y, prediction, average="macro", zero_division=0)), fp, threshold))
    feasible = [item for item in scored if fp <= fp_limit] or scored
    best = max(item[0] for item in feasible); winners = [item for item in feasible if abs(item[0] - best) < 1e-12]
    return min(winners, key=lambda item: abs(item[2] - .5))[2]


def checkpoint_metrics(model, loader, device, label, fp_limit):
    model.eval(); ids=[]; gold=[]; probabilities=[]
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch["ids"]); gold.extend(batch["labels"].tolist())
            data={key:value.to(device) for key,value in batch.items() if torch.is_tensor(value) and key not in {"labels", "weights", "token_labels"}}
            logits,_=model(**data); probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    y=torch.tensor(gold).numpy().astype(int); p=torch.tensor(probabilities).numpy()
    # The holdout itself uses digest byte 0 modulo 10; use byte 1 here so the
    # calibration/selection partition is independent rather than degenerate.
    cal=np.asarray([(hashlib.sha256(str(record_id).encode()).digest()[1] & 1) == 0 for record_id in ids])
    if not cal.any() or cal.all(): raise RuntimeError("checkpoint calibration split empty")
    threshold=exact_threshold(y[cal],p[cal],fp_limit)
    selected=~cal; pred=p[selected]>=threshold; ys=y[selected]
    fp=int(((ys==0)&pred).sum()); fn=int(((ys==1)&~pred).sum())
    return {"threshold":float(threshold),"binary_macro_f1":float(f1_score(ys,pred,average="macro",zero_division=0)),"pr_auc":float(average_precision_score(ys,p[selected])),"positive_precision":float(precision_score(ys,pred,zero_division=0)),"positive_recall":float(recall_score(ys,pred,zero_division=0)),"fp":fp,"fn":fn,"tp":int(((ys==1)&pred).sum()),"tn":int(((ys==0)&~pred).sum()),"calibration_records":int(cal.sum()),"selection_records":int(selected.sum()),"feasible":fp<=fp_limit}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=LABELS, required=True); parser.add_argument("--model-id", required=True); parser.add_argument("--base-state", type=Path, required=True); parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl"); parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--tapt-checkpoint", type=Path); parser.add_argument("--system", required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1.5e-5); parser.add_argument("--epochs", type=int, default=6); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--num-workers",type=int,default=0,help="0 avoids forked encoder-memory copies on constrained MIG partitions"); parser.add_argument("--grad-accum",type=int,default=1); parser.add_argument("--freeze-bottom-layers",type=int,default=0); parser.add_argument("--bf16",action="store_true"); parser.add_argument("--dropout", type=float, default=.1); parser.add_argument("--pooling", choices=("cls", "clsmean", "clsmeanmax", "attention"), default="clsmean"); parser.add_argument("--head", choices=("linear", "residual"), default="linear")
    parser.add_argument("--loss", choices=("bce", "asymmetric"), default="bce"); parser.add_argument("--gamma-positive", type=float, default=0.0); parser.add_argument("--gamma-negative", type=float, default=2.0); parser.add_argument("--asl-clip", type=float, default=0.0); parser.add_argument("--positive-class-weight", type=float, default=1.0); parser.add_argument("--negative-class-weight", type=float, default=1.0); parser.add_argument("--hard-positive-weight", type=float, default=1.0); parser.add_argument("--hard-negative-weight", type=float, default=1.0); parser.add_argument("--oof-hard-ids",type=Path,help="JSON list of train IDs found in an OOF error audit"); parser.add_argument("--oof-hard-weight",type=float,default=1.0); parser.add_argument("--token-aux", action="store_true"); parser.add_argument("--rdrop-weight", type=float, default=0.0); parser.add_argument("--ema-decay", type=float, default=0.0); parser.add_argument("--head-lr", type=float); parser.add_argument("--train-folds",type=Path); parser.add_argument("--exclude-fold",type=int); parser.add_argument("--max-length", type=int, default=128)
    args=parser.parse_args(); seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.token_aux and args.label != "code_switching": raise SystemExit("token auxiliary is code-switching only")
    tokenizer=AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    excluded=set()
    if args.train_folds and args.exclude_fold is not None:
        assignment=json.loads(args.train_folds.read_text())["assignments"]
        excluded={record_id for record_id,fold in assignment.items() if int(fold)==args.exclude_fold}
    extra_hard_ids=set(json.loads(args.oof_hard_ids.read_text())) if args.oof_hard_ids else set()
    train=Records(args.train,args.label,args.hard_positive_weight,args.hard_negative_weight,"train",excluded,extra_hard_ids,args.oof_hard_weight); holdout=Records(args.train,args.label,args.hard_positive_weight,args.hard_negative_weight,"holdout",excluded,extra_hard_ids,args.oof_hard_weight); dev=Records(args.dev,args.label,args.hard_positive_weight,args.hard_negative_weight,None)
    make=collator(tokenizer,args.token_aux,args.max_length); train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=make,num_workers=args.num_workers); hold_loader=DataLoader(holdout,batch_size=args.batch_size,collate_fn=make,num_workers=args.num_workers); dev_loader=DataLoader(dev,batch_size=args.batch_size,collate_fn=make,num_workers=args.num_workers)
    model=Expert(load_encoder(args.model_id,args.base_state,args.tapt_checkpoint),args.pooling,args.head,args.dropout,args.token_aux).to(device)
    if args.freeze_bottom_layers:
        for name,parameter in model.named_parameters():
            match=re.match(r"encoder\.encoder\.layer\.(\d+)\.",name)
            if name.startswith("encoder.embeddings.") or (match and int(match.group(1)) < args.freeze_bottom_layers): parameter.requires_grad=False
    head_parameters=[parameter for name,parameter in model.named_parameters() if not name.startswith("encoder.") and parameter.requires_grad]
    encoder_parameters=[parameter for name,parameter in model.named_parameters() if name.startswith("encoder.") and parameter.requires_grad]
    if args.grad_accum < 1: raise SystemExit("--grad-accum must be positive")
    updates=max(1,(len(train_loader)+args.grad_accum-1)//args.grad_accum)*args.epochs
    opt=torch.optim.AdamW([{"params":encoder_parameters,"lr":args.lr},{"params":head_parameters,"lr":args.head_lr or args.lr}],weight_decay=.01); sch=get_linear_schedule_with_warmup(opt,int(.06*updates),updates)
    args.output_dir.mkdir(parents=True,exist_ok=True); best_rank=(-1,-1.,-1.);stale=0;hist=[];top=[];started=time.monotonic();ema={name:value.detach().clone() for name,value in model.state_dict().items()} if args.ema_decay else None;fp_limit=1 if args.label in {"irony","idiom_figurative"} else 10**9
    for epoch in range(1,args.epochs+1):
        model.train();total=0.0;opt.zero_grad(set_to_none=True)
        for step,batch in enumerate(train_loader,1):
            data={key:value.to(device) for key,value in batch.items() if torch.is_tensor(value)}; inputs={key:value for key,value in data.items() if key not in {"labels","weights","token_labels"}}
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.bf16 and device.type=="cuda"):
                logit,tok=model(**inputs)
                per=F.binary_cross_entropy_with_logits(logit,data["labels"],reduction="none") if args.loss=="bce" else asymmetric_loss(logit,data["labels"],gamma_positive=args.gamma_positive,gamma_negative=args.gamma_negative,clip=args.asl_clip,positive_weight=args.positive_class_weight,negative_weight=args.negative_class_weight)
                loss=(per*data["weights"]).mean()
                if tok is not None: loss=loss+.20*F.cross_entropy(tok.flatten(0,1),data["token_labels"].flatten(),ignore_index=-100)
                if args.rdrop_weight:
                    logit2,_=model(**inputs); loss=loss+args.rdrop_weight*symmetric_bernoulli_kl(torch.sigmoid(logit),torch.sigmoid(logit2))
                loss=loss/args.grad_accum
            loss.backward();total+=float(loss.detach())*args.grad_accum
            if step%args.grad_accum==0 or step==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step();opt.zero_grad(set_to_none=True)
            if ema:
                with torch.no_grad():
                    for name,value in model.state_dict().items(): ema[name].mul_(args.ema_decay).add_(value.detach(),alpha=1-args.ema_decay)
        original=None
        if ema:
            original={name:value.detach().clone() for name,value in model.state_dict().items()};model.load_state_dict(ema,strict=True)
        metrics=checkpoint_metrics(model,hold_loader,device,args.label,fp_limit)
        if original: model.load_state_dict(original,strict=True)
        rank=(int(metrics["feasible"]),metrics["binary_macro_f1"],metrics["pr_auc"]);hist.append({"epoch":epoch,"train_loss":total/max(len(train_loader),1),"checkpoint_selection":metrics,"elapsed_seconds":time.monotonic()-started})
        snapshot={"model":ema if ema else model.state_dict(),"args":vars(args),"checkpoint_selection":metrics,"epoch":epoch}
        path=args.output_dir/f"checkpoint_epoch_{epoch}.pt";torch.save(snapshot,path);top.append((rank,path,metrics));top.sort(key=lambda item:item[0],reverse=True);top=top[:3]
        if rank>best_rank: best_rank=rank;stale=0;torch.save(snapshot,args.output_dir/"best.pt")
        else:
            stale+=1
            if stale>=2:break
    state=torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False);model.load_state_dict(state["model"]);ids,probs=predict(model,dev_loader,device)
    pred_path=args.output_dir/"dev_probabilities.jsonl"
    with pred_path.open("w") as handle:
        for rid,p in zip(ids,probs): handle.write(json.dumps({"id":rid,"probability":p,"label":args.label,"system":args.system,"seed":args.seed})+"\n")
    manifest={"status":"ok","system":args.system,"target_label":args.label,"seed":args.seed,"model_id":args.model_id,"base_state":str(args.base_state),"tapt_checkpoint":str(args.tapt_checkpoint) if args.tapt_checkpoint else None,"train_records":len(train),"train_holdout_records":len(holdout),"excluded_train_fold":args.exclude_fold,"dev_records":len(dev),"test_labels_read":False,"test_predictions_created":False,"epochs_completed":len(hist),"checkpoint_selection":"train-holdout cross-fitted calibration/selection split; target macro-F1 objective", "best_checkpoint_selection":state["checkpoint_selection"],"top_checkpoints":[{"path":str(path),"rank":rank,"metrics":metrics} for rank,path,metrics in top],"elapsed_seconds":time.monotonic()-started,"config":{key:value for key,value in vars(args).items() if key not in {"base_state","train","dev","output_dir","tapt_checkpoint"}},"checkpoint":str(args.output_dir/"best.pt"),"dev_probabilities":str(pred_path)}
    # ``train_folds`` is a Path as well; use a JSON default so the manifest
    # remains a complete, machine-readable record of the exact invocation.
    (args.output_dir/"history.json").write_text(json.dumps(hist,indent=2,default=str)+"\n");(args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2,default=str)+"\n");print(json.dumps(manifest,indent=2,default=str))


if __name__=="__main__":main()
