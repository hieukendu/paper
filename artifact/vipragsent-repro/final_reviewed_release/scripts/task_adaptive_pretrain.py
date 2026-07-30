"""Small, auditable task-adaptive MLM pretraining for local ViPragSent encoders.

Only text fields are read. Labels and all test records are deliberately outside
the input interface, so the resulting checkpoint is safe to use as a TAPT
initialization for downstream target experts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from vipragsent.utils.io import read_jsonl


class Texts(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def load_model(model_id: str, state_path: Path):
    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_config(config)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Legacy RoBERTa archives store a shared ``lm_head.bias`` but omit the
    # tied decoder aliases.  ``tie_weights`` deterministically restores them.
    model.tie_weights()
    unsafe_missing = [key for key in missing if key not in {"roberta.embeddings.position_ids", "roberta.pooler.dense.weight", "roberta.pooler.dense.bias", "lm_head.decoder.weight", "lm_head.decoder.bias"}]
    if unsafe_missing:
        raise RuntimeError(f"incomplete MLM initialization: {unsafe_missing[:10]}")
    return model, {"missing": missing, "unexpected": unexpected}


def mask_batch(encoded, tokenizer, probability: float):
    inputs = encoded["input_ids"].clone(); labels = inputs.clone()
    special = torch.zeros_like(inputs, dtype=torch.bool)
    for i, sequence in enumerate(inputs.tolist()):
        special[i] = torch.tensor(tokenizer.get_special_tokens_mask(sequence, already_has_special_tokens=True), dtype=torch.bool)
    mask = torch.bernoulli(torch.full(labels.shape, probability)).bool() & ~special & encoded["attention_mask"].bool()
    labels[~mask] = -100
    replace = torch.bernoulli(torch.full(labels.shape, .8)).bool() & mask
    inputs[replace] = tokenizer.mask_token_id
    random_mask = torch.bernoulli(torch.full(labels.shape, .5)).bool() & mask & ~replace
    inputs[random_mask] = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)[random_mask]
    encoded["input_ids"] = inputs; encoded["labels"] = labels
    return encoded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-state", required=True, type=Path)
    parser.add_argument("--corpus", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--mask-probability", type=float, default=.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    source_rows = []
    seen = set()
    for path in args.corpus:
        for row in read_jsonl(path):
            text = str(row.get("text") or "").strip()
            record_id = str(row.get("id"))
            if text and record_id not in seen:
                source_rows.append({"id": record_id, "text": text}); seen.add(record_id)
    if len(source_rows) < 100:
        raise SystemExit("TAPT corpus is unexpectedly small")
    train_rows, heldout = [], []
    for row in source_rows:
        (heldout if hashlib.sha256(row["id"].encode()).digest()[0] % 10 == 0 else train_rows).append(row)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    model, loading = load_model(args.model_id, args.base_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    def collate(rows):
        return tokenizer([row["text"] for row in rows], padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
    train_loader = DataLoader(Texts(train_rows), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    holdout_loader = DataLoader(Texts(heldout), batch_size=args.batch_size, collate_fn=collate, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(.06 * len(train_loader) * args.epochs), len(train_loader) * args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best, stale, history, started = float("inf"), 0, [], time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for batch in train_loader:
            batch = mask_batch(batch, tokenizer, args.mask_probability)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); total += float(loss.detach())
        model.eval(); losses = []
        with torch.no_grad():
            for batch in holdout_loader:
                batch = mask_batch(batch, tokenizer, args.mask_probability)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    losses.append(float(model(**batch).loss))
        holdout_loss = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch, "train_mlm_loss": total / max(len(train_loader), 1), "heldout_mlm_loss": holdout_loss, "elapsed_seconds": time.monotonic() - started})
        if holdout_loss < best:
            best = holdout_loss; stale = 0
            torch.save({"model": model.state_dict(), "config": model.config.to_dict(), "tapt": {"corpus_paths": [str(path) for path in args.corpus], "labels_read": False, "test_text_read": False, "seed": args.seed}}, args.output_dir / "best_mlm.pt")
        else:
            stale += 1
            if stale >= 2: break
    manifest = {"status": "ok", "method": "task_adaptive_masked_language_modeling", "model_id": args.model_id, "base_state": str(args.base_state), "corpus_paths": [str(path) for path in args.corpus], "records": len(source_rows), "train_records": len(train_rows), "heldout_records": len(heldout), "labels_read": False, "test_text_read": False, "loading": loading, "epochs_completed": len(history), "best_heldout_mlm_loss": best, "elapsed_seconds": time.monotonic() - started, "hyperparameters": {"lr": args.lr, "mask_probability": args.mask_probability, "weight_decay": .01, "max_length": args.max_length, "seed": args.seed}}
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "tapt_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
