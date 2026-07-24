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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

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
    def __init__(self, path: Path, label: str, hp: float, hn: float, partition: str | None = None):
        rows = list(read_jsonl(path)); self.rows = []
        for row in rows:
            split = hashlib.sha256(str(row["id"]).encode()).digest()[0] % 10 == 0
            if partition == "holdout" and not split: continue
            if partition == "train" and split: continue
            self.rows.append({"id": str(row["id"]), "text": str(row["text"]), "label": float(row["labels"][label]), "weight": hard_weight(row, label, hp, hn)})
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=LABELS, required=True); parser.add_argument("--model-id", required=True); parser.add_argument("--base-state", type=Path, required=True); parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl"); parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--tapt-checkpoint", type=Path); parser.add_argument("--system", required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1.5e-5); parser.add_argument("--epochs", type=int, default=6); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--dropout", type=float, default=.1); parser.add_argument("--pooling", choices=("cls", "clsmean", "clsmeanmax", "attention"), default="clsmean"); parser.add_argument("--head", choices=("linear", "residual"), default="linear")
    parser.add_argument("--loss", choices=("bce", "asymmetric"), default="bce"); parser.add_argument("--gamma-negative", type=float, default=2.0); parser.add_argument("--hard-positive-weight", type=float, default=1.0); parser.add_argument("--hard-negative-weight", type=float, default=1.0); parser.add_argument("--token-aux", action="store_true"); parser.add_argument("--rdrop-weight", type=float, default=0.0); parser.add_argument("--max-length", type=int, default=128)
    args=parser.parse_args(); seed_all(args.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.token_aux and args.label != "code_switching": raise SystemExit("token auxiliary is code-switching only")
    tokenizer=AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    train=Records(args.train,args.label,args.hard_positive_weight,args.hard_negative_weight,"train"); holdout=Records(args.train,args.label,args.hard_positive_weight,args.hard_negative_weight,"holdout"); dev=Records(args.dev,args.label,args.hard_positive_weight,args.hard_negative_weight,None)
    make=collator(tokenizer,args.token_aux,args.max_length); train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,collate_fn=make,num_workers=2); hold_loader=DataLoader(holdout,batch_size=args.batch_size,collate_fn=make,num_workers=2); dev_loader=DataLoader(dev,batch_size=args.batch_size,collate_fn=make,num_workers=2)
    model=Expert(load_encoder(args.model_id,args.base_state,args.tapt_checkpoint),args.pooling,args.head,args.dropout,args.token_aux).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01); sch=get_linear_schedule_with_warmup(opt,int(.06*len(train_loader)*args.epochs),len(train_loader)*args.epochs)
    args.output_dir.mkdir(parents=True,exist_ok=True); best=float("inf");stale=0;hist=[];started=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train();total=0.0
        for batch in train_loader:
            data={key:value.to(device) for key,value in batch.items() if torch.is_tensor(value)}; logit,tok=model(data["input_ids"],data["attention_mask"])
            per=F.binary_cross_entropy_with_logits(logit,data["labels"],reduction="none")
            if args.loss=="asymmetric": per=per*torch.where(data["labels"]>0,.25,torch.sigmoid(logit).detach().pow(args.gamma_negative))
            loss=(per*data["weights"]).mean()
            if tok is not None: loss=loss+.20*F.cross_entropy(tok.flatten(0,1),data["token_labels"].flatten(),ignore_index=-100)
            if args.rdrop_weight:
                logit2,_=model(data["input_ids"],data["attention_mask"]); p=torch.sigmoid(logit);q=torch.sigmoid(logit2); kl=(p*(p.clamp_min(1e-6).log()-q.clamp_min(1e-6).log())+(1-p)*((1-p).clamp_min(1e-6).log()-(1-q).clamp_min(1e-6).log())).mean(); loss=loss+args.rdrop_weight*kl
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step();opt.zero_grad(set_to_none=True);total+=float(loss.detach())
        model.eval(); losses=[]
        with torch.no_grad():
            for batch in hold_loader:
                data={key:value.to(device) for key,value in batch.items() if torch.is_tensor(value)}; logit,tok=model(data["input_ids"],data["attention_mask"]); losses.append(float(F.binary_cross_entropy_with_logits(logit,data["labels"])))
        value=sum(losses)/max(len(losses),1);hist.append({"epoch":epoch,"train_loss":total/max(len(train_loader),1),"train_holdout_bce":value,"elapsed_seconds":time.monotonic()-started})
        if value<best: best=value;stale=0;torch.save({"model":model.state_dict(),"args":vars(args),"best_train_holdout_bce":best},args.output_dir/"best.pt")
        else:
            stale+=1
            if stale>=2:break
    state=torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False);model.load_state_dict(state["model"]);ids,probs=predict(model,dev_loader,device)
    pred_path=args.output_dir/"dev_probabilities.jsonl"
    with pred_path.open("w") as handle:
        for rid,p in zip(ids,probs): handle.write(json.dumps({"id":rid,"probability":p,"label":args.label,"system":args.system,"seed":args.seed})+"\n")
    manifest={"status":"ok","system":args.system,"target_label":args.label,"seed":args.seed,"model_id":args.model_id,"base_state":str(args.base_state),"tapt_checkpoint":str(args.tapt_checkpoint) if args.tapt_checkpoint else None,"train_records":len(train),"train_holdout_records":len(holdout),"dev_records":len(dev),"test_labels_read":False,"test_predictions_created":False,"epochs_completed":len(hist),"best_train_holdout_bce":best,"elapsed_seconds":time.monotonic()-started,"config":{key:value for key,value in vars(args).items() if key not in {"base_state","train","dev","output_dir","tapt_checkpoint"}},"checkpoint":str(args.output_dir/"best.pt"),"dev_probabilities":str(pred_path)}
    (args.output_dir/"history.json").write_text(json.dumps(hist,indent=2)+"\n");(args.output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n");print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
