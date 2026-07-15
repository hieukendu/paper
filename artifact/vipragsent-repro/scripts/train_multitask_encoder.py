from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The school image ships an ONNX build generated against an older protobuf.
# Transformers imports torchvision/ONNX while loading text encoders, although
# this experiment never uses ONNX.  This compatibility fallback affects only
# that import path and keeps model training on the compiled PyTorch backend.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl

POLARITIES = ["negative", "neutral", "positive"]
EMOTIONS = ["enjoyment", "sadness", "anger", "disgust", "fear", "surprise", "other"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JsonlDataset(Dataset):
    def __init__(self, path: Path, rationales: dict[str, str] | None = None):
        self.records = list(read_jsonl(path))
        self.rationales = rationales or {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        labels = record["labels"]
        return {
            "id": record["id"],
            "text": record["text"],
            "pragmatic": [float(labels[name]) for name in PRAGMATIC_LABELS],
            "polarity": POLARITIES.index(labels["polarity"]),
            "emotion": EMOTIONS.index(labels["emotion"]),
            "rationale": self.rationales.get(record["id"], ""),
        }


class JsonlPredictionDataset(Dataset):
    """Dataset used for inference over records whose labels may be incomplete."""

    def __init__(self, path: Path):
        self.records = list(read_jsonl(path))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {"id": record["id"], "text": record["text"]}


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        encoded = self.tokenizer(
            [row["text"] for row in rows], padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        batch = {**encoded, "ids": [row["id"] for row in rows]}
        if "pragmatic" not in rows[0]:
            return batch
        rationales = self.tokenizer(
            [row["rationale"] or row["text"] for row in rows], padding=True,
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        batch.update(
            {
                "rationale_input_ids": rationales["input_ids"],
                "rationale_attention_mask": rationales["attention_mask"],
                "pragmatic": torch.tensor([row["pragmatic"] for row in rows]),
                "polarity": torch.tensor([row["polarity"] for row in rows]),
                "emotion": torch.tensor([row["emotion"] for row in rows]),
            }
        )
        return batch


class MultiTaskModel(nn.Module):
    def __init__(self, model_id: str, *, uncertainty: bool, rationale_aux: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.pragmatic = nn.Linear(hidden, len(PRAGMATIC_LABELS))
        self.polarity = nn.Linear(hidden, len(POLARITIES))
        self.emotion = nn.Linear(hidden, len(EMOTIONS))
        self.rationale_projection = nn.Linear(hidden, hidden) if rationale_aux else None
        self.log_vars = nn.Parameter(torch.zeros(4)) if uncertainty else None

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return self.pragmatic(pooled), self.polarity(pooled), self.emotion(pooled), pooled


def load_rationales(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    return {str(row["id"]): str(row.get("rationale") or "") for row in read_jsonl(path)}


def task_loss(model, batch, device, *, beta, use_emotion, use_polarity, use_rationale):
    pragmatic, polarity, emotion, pooled = model(batch["input_ids"], batch["attention_mask"])
    losses = [F.binary_cross_entropy_with_logits(pragmatic, batch["pragmatic"])]
    losses.append(F.cross_entropy(polarity, batch["polarity"]) if use_polarity else pragmatic.sum() * 0)
    losses.append(F.cross_entropy(emotion, batch["emotion"]) if use_emotion else pragmatic.sum() * 0)
    if use_rationale:
        with torch.no_grad():
            embeddings = model.encoder.get_input_embeddings()(batch["rationale_input_ids"])
            mask = batch["rationale_attention_mask"].unsqueeze(-1)
            target = (embeddings * mask).sum(1) / mask.sum(1).clamp_min(1)
        rationale_loss = F.mse_loss(model.rationale_projection(pooled), target)
    else:
        rationale_loss = pragmatic.sum() * 0
    weighted = [losses[0], losses[1], losses[2], beta * rationale_loss]
    if model.log_vars is None:
        total = sum(weighted)
    else:
        total = sum(torch.exp(-model.log_vars[i]) * loss + model.log_vars[i] for i, loss in enumerate(weighted))
    return total, (pragmatic, polarity, emotion), [x.detach().item() for x in weighted]


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); gold = []; pred = []
    for batch in loader:
        batch = move(batch, device)
        logits, _, _, _ = model(batch["input_ids"], batch["attention_mask"])
        gold.extend(batch["pragmatic"].cpu().int().tolist())
        pred.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
    values = []
    for i in range(len(PRAGMATIC_LABELS)):
        values.append(binary_macro_f1([row[i] for row in gold], [row[i] for row in pred]))
    return sum(values) / len(values), dict(zip(PRAGMATIC_LABELS, values))


@torch.no_grad()
def write_predictions(model, loader, device, output: Path, system: str, seed: int):
    model.eval(); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for batch in loader:
            batch = move(batch, device)
            prag, pol, emo, _ = model(batch["input_ids"], batch["attention_mask"])
            prag_prob = torch.sigmoid(prag).cpu(); pol_prob = torch.softmax(pol, -1).cpu(); emo_prob = torch.softmax(emo, -1).cpu()
            for i, record_id in enumerate(batch["ids"]):
                predictions = {name: int(prag_prob[i, j] >= 0.5) for j, name in enumerate(PRAGMATIC_LABELS)}
                predictions["polarity"] = POLARITIES[int(pol_prob[i].argmax())]
                predictions["emotion"] = EMOTIONS[int(emo_prob[i].argmax())]
                probabilities = {name: float(prag_prob[i, j]) for j, name in enumerate(PRAGMATIC_LABELS)}
                probabilities["polarity"] = {name: float(pol_prob[i, j]) for j, name in enumerate(POLARITIES)}
                sarcasm = float(prag_prob[i, PRAGMATIC_LABELS.index("sarcasm")])
                irony = float(prag_prob[i, PRAGMATIC_LABELS.index("irony")])
                mocking = float(prag_prob[i, PRAGMATIC_LABELS.index("mocking")])
                raw_pragmatic_polarity = {
                    "positive": float(pol_prob[i, POLARITIES.index("positive")]) * (1.0 - irony),
                    "negative": float(pol_prob[i, POLARITIES.index("negative")]) * (1.0 - sarcasm),
                    "neutral": float(pol_prob[i, POLARITIES.index("neutral")]),
                    "ironic-positive": irony * float(pol_prob[i, POLARITIES.index("positive")]),
                    "sarcastic-negative": sarcasm * float(pol_prob[i, POLARITIES.index("negative")]),
                    "mocking": mocking,
                }
                norm = sum(raw_pragmatic_polarity.values()) or 1.0
                probabilities["pragmatic_polarity"] = {key: value / norm for key, value in raw_pragmatic_polarity.items()}
                row = {
                    "id": record_id, "system": system, "seed": seed,
                    "predictions": predictions, "probabilities": probabilities,
                    "logits": {
                        "pragmatic": [float(value) for value in prag[i].detach().cpu()],
                        "polarity": [float(value) for value in pol[i].detach().cpu()],
                        "emotion": [float(value) for value in emo[i].detach().cpu()],
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/vipragsent_train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/processed/vipragsent_dev.jsonl")
    parser.add_argument("--test", type=Path, default=ROOT / "data/processed/vipragsent_test.jsonl")
    parser.add_argument("--rationales", type=Path, default=ROOT / "data/generated/rationales.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--prediction-root", type=Path, default=ROOT / "results/predictions/main_pragmatic")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--no-rationale", action="store_true")
    parser.add_argument("--no-emotion", action="store_true")
    parser.add_argument("--no-polarity", action="store_true")
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="Existing checkpoint for prediction-only mode.")
    parser.add_argument("--predict-data", type=Path, help="JSONL data to score without training.")
    parser.add_argument("--prediction-output", type=Path, help="Explicit prediction JSONL output path.")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.predict_data):
        raise SystemExit("--checkpoint and --predict-data must be supplied together for prediction-only mode.")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    rationales = load_rationales(args.rationales)
    if args.checkpoint:
        collate = Collator(tokenizer, args.max_length)
        prediction_data = DataLoader(
            JsonlPredictionDataset(args.predict_data),
            batch_size=args.batch_size * 2,
            collate_fn=collate,
            num_workers=2,
        )
        model = MultiTaskModel(args.model_id, uncertainty=not args.no_uncertainty, rationale_aux=not args.no_rationale).to(device)
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        output = args.prediction_output or args.prediction_root / args.system / f"{args.seed}.jsonl"
        write_predictions(model, prediction_data, device, output, args.system, args.seed)
        print(json.dumps({"status": "ok", "mode": "predict_only", "checkpoint": str(args.checkpoint), "records": len(prediction_data.dataset), "predictions": str(output)}, indent=2))
        return 0
    train_ids = {str(row["id"]) for row in read_jsonl(args.train)}
    if not args.no_rationale and not train_ids.issubset(rationales):
        raise SystemExit(
            f"Rationale coverage incomplete: {len(train_ids & set(rationales))}/{len(train_ids)}. "
            "Resume scripts/06_generate_rationales.py or pass --no-rationale for the ablation."
        )
    collate = Collator(tokenizer, args.max_length)
    train = DataLoader(JsonlDataset(args.train, rationales), batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    dev = DataLoader(JsonlDataset(args.dev), batch_size=args.batch_size * 2, collate_fn=collate, num_workers=2)
    test = DataLoader(JsonlDataset(args.test), batch_size=args.batch_size * 2, collate_fn=collate, num_workers=2)
    model = MultiTaskModel(args.model_id, uncertainty=not args.no_uncertainty, rationale_aux=not args.no_rationale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.bf16)
    run_dir = args.output_root / args.system / str(args.seed); run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best = -1.0; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        for step, batch in enumerate(train, 1):
            batch = move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=device.type == "cuda"):
                loss, _, _ = task_loss(model, batch, device, beta=args.beta, use_emotion=not args.no_emotion, use_polarity=not args.no_polarity, use_rationale=not args.no_rationale)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward(); running += loss.item()
            if step % args.grad_accum == 0 or step == len(train):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        score, per_label = evaluate(model, dev, device)
        history.append({"epoch": epoch, "train_loss": running, "dev_macro_pragmatic_f1": score, "per_label": per_label, "elapsed_seconds": round(time.monotonic() - started, 3)})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if score > best:
            best = score; stale = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "best_dev": best}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["model"])
    prediction = args.prediction_root / args.system / f"{args.seed}.jsonl"
    write_predictions(model, test, device, prediction, args.system, args.seed)
    manifest = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "system": args.system,
        "seed": args.seed,
        "model_id": args.model_id,
        "device": str(device),
        "bf16": bool(args.bf16),
        "train_records": len(train.dataset),
        "dev_records": len(dev.dataset),
        "test_records": len(test.dataset),
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "best_dev_macro_pragmatic_f1": best,
        "checkpoint": str(run_dir / "best.pt"),
        "predictions": str(prediction),
        "rationales_loaded": len(rationales),
        "flags": {"no_rationale": args.no_rationale, "no_emotion": args.no_emotion, "no_polarity": args.no_polarity, "no_uncertainty": args.no_uncertainty},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
