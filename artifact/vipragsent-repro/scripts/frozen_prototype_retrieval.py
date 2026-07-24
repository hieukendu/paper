#!/usr/bin/env python3
"""Train-prototype retrieval over frozen contextual embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_embedding_knn import embed, load_encoder
from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl

ALPHAS = [value / 10 for value in range(11)]
TEMPERATURES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
THRESHOLDS = [round(value / 100, 2) for value in range(1, 100)]


def gold(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = list(read_jsonl(path)); return rows, {str(row["id"]): row["labels"] for row in rows}


def base_rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in _ensemble_rows(sorted(path.glob("reproduction_*.jsonl")))}


def prototype_scores(train_embeddings: torch.Tensor, train_rows: list[dict], query_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    scores = {}
    for label in PRAGMATIC_LABELS:
        mask = torch.tensor([int(row["labels"][label]) == 1 for row in train_rows])
        positive = F.normalize(train_embeddings[mask].mean(dim=0), dim=0)
        negative = F.normalize(train_embeddings[~mask].mean(dim=0), dim=0)
        scores[label] = torch.stack((query_embeddings @ positive, query_embeddings @ negative), dim=1)
    return scores


def fit(args: argparse.Namespace) -> None:
    tokenizer, model = load_encoder(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    dev_rows, dev_embeddings = embed(args.dev_gold, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    retrieval = prototype_scores(train_embeddings, train_rows, dev_embeddings)
    base = base_rows(args.dev_predictions)
    labels = {}
    for label in PRAGMATIC_LABELS:
        truth = [int(row["labels"][label]) for row in dev_rows]
        model_scores = torch.tensor([float(base[row["id"]]["probabilities"][label]) for row in dev_rows])
        sim = retrieval[label]; diff = sim[:, 0] - sim[:, 1]
        best = (-1.0, None, None, None)
        for temperature in TEMPERATURES:
            proto = torch.sigmoid(diff / temperature)
            for alpha in ALPHAS:
                blend = alpha * proto + (1.0 - alpha) * model_scores
                for threshold in THRESHOLDS:
                    value = binary_macro_f1(truth, [int(score >= threshold) for score in blend.tolist()]) * 100
                    candidate = (value, -abs(alpha - 0.5), -abs(threshold - 0.5), -abs(math.log10(temperature)), temperature, alpha, threshold)
                    best_cmp = (best[0], -abs(best[2] - 0.5) if best[2] is not None else -9, -abs(best[3] - 0.5) if best[3] is not None else -9, -abs(math.log10(best[1])) if best[1] else -9, best[1] or 0, best[2] or 0, best[3] or 0)
                    if candidate > best_cmp:
                        best = (value, temperature, alpha, threshold)
        labels[label] = {"temperature": best[1], "alpha_prototype": best[2], "threshold": best[3], "development_binary_macro_f1": round(best[0], 4)}
    payload = {"method": "train_centroid_prototype_similarity_blended_with_frozen_head", "selection_split": "development", "base": str(args.base), "checkpoint": str(args.checkpoint) if args.checkpoint else None, "pooling": args.pooling, "max_length": args.max_length, "temperatures": TEMPERATURES, "labels": labels, "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: state["development_binary_macro_f1"] for label, state in labels.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer, model = load_encoder(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config["pooling"])
    data_rows, data_embeddings = embed(args.data, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config["pooling"])
    retrieval = prototype_scores(train_embeddings, train_rows, data_embeddings)
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    position = {row["id"]: index for index, row in enumerate(data_rows)}
    for row in rows:
        index = position[row["id"]]
        for label, state in config["labels"].items():
            sim = retrieval[label][index]; proto = torch.sigmoid((sim[0] - sim[1]) / state["temperature"])
            value = state["alpha_prototype"] * float(proto) + (1.0 - state["alpha_prototype"]) * float(row["probabilities"][label])
            row["probabilities"][f"{label}_prototype_blend"] = value
            row["predictions"][label] = int(value >= state["threshold"])
        row["system"] = "vipragsent_frozen_prototype_retrieval"
    write_jsonl(args.output, rows); print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "apply"):
        command = commands.add_parser(name); command.add_argument("--base", type=Path, required=True); command.add_argument("--base-bin", type=Path, required=True); command.add_argument("--checkpoint", type=Path); command.add_argument("--train", type=Path, required=True); command.add_argument("--batch-size", type=int, default=128)
        if name == "fit":
            command.add_argument("--dev-gold", type=Path, required=True); command.add_argument("--dev-predictions", type=Path, required=True); command.add_argument("--max-length", type=int, default=128); command.add_argument("--pooling", choices=["cls", "mean"], default="cls"); command.add_argument("--output", type=Path, required=True); command.set_defaults(func=fit)
        else:
            command.add_argument("--config", type=Path, required=True); command.add_argument("--data", type=Path, required=True); command.add_argument("--predictions", type=Path, required=True); command.add_argument("--base-config", type=Path, required=True); command.add_argument("--output", type=Path, required=True); command.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args); return 0


if __name__ == "__main__":
    raise SystemExit(main())
