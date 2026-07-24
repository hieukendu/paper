#!/usr/bin/env python3
"""Similarity-weighted train-label retrieval over frozen embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_embedding_knn import embed, load_encoder
from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.utils.io import read_jsonl, write_jsonl

K_VALUES = [3, 5, 11, 25, 50, 100, 200, 500]
TEMPERATURES = [0.02, 0.05, 0.1, 0.2, 0.5]
ALPHAS = [value / 10 for value in range(11)]
THRESHOLDS = np.arange(1, 100, dtype=np.float64) / 100


def weighted_neighbors(train_emb: torch.Tensor, query_emb: torch.Tensor, labels: torch.Tensor) -> dict[tuple[int, float], torch.Tensor]:
    maximum = max(K_VALUES); pieces: dict[tuple[int, float], list[torch.Tensor]] = {(k, t): [] for k in K_VALUES for t in TEMPERATURES}
    for start in range(0, len(query_emb), 256):
        sim = query_emb[start : start + 256] @ train_emb.T
        values, indices = sim.topk(maximum, dim=1)
        gathered = labels[indices]
        for k in K_VALUES:
            for temperature in TEMPERATURES:
                weights = torch.softmax(values[:, :k] / temperature, dim=1).unsqueeze(-1)
                pieces[(k, temperature)].append((weights * gathered[:, :k]).sum(dim=1))
    return {key: torch.cat(value) for key, value in pieces.items()}


def metric_grid(truth: np.ndarray, scores: np.ndarray) -> np.ndarray:
    prediction = scores[:, None] >= THRESHOLDS[None, :]
    true = truth[:, None].astype(bool)
    tp = np.sum(prediction & true, axis=0); fp = np.sum(prediction & ~true, axis=0); fn = np.sum(~prediction & true, axis=0)
    tn = np.sum(~prediction & ~true, axis=0)
    f1_positive = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    f1_negative = np.divide(2 * tn, 2 * tn + fp + fn, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    return (f1_positive + f1_negative) * 50


def gold(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = list(read_jsonl(path)); return rows, {str(row["id"]): row["labels"] for row in rows}


def base(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in _ensemble_rows(sorted(path.glob("reproduction_*.jsonl")))}


def fit(args: argparse.Namespace) -> None:
    tokenizer, model = load_encoder(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    dev_rows, dev_embeddings = embed(args.dev_gold, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    train_labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    retrieval = weighted_neighbors(train_embeddings, dev_embeddings, train_labels)
    base_rows = base(args.dev_predictions)
    labels = {}
    for index, label in enumerate(PRAGMATIC_LABELS):
        truth = np.asarray([int(row["labels"][label]) for row in dev_rows])
        model_scores = np.asarray([float(base_rows[row["id"]]["probabilities"][label]) for row in dev_rows])
        best = (-1.0, None, None, None)
        for (k, temperature), output in retrieval.items():
            retrieval_scores = output[:, index].numpy()
            for alpha in ALPHAS:
                values = metric_grid(truth, alpha * retrieval_scores + (1 - alpha) * model_scores)
                threshold_index = int(values.argmax()); candidate = (float(values[threshold_index]), k, temperature, alpha, float(THRESHOLDS[threshold_index]))
                if candidate[0] > best[0]: best = candidate
        labels[label] = {"k": best[1], "temperature": best[2], "alpha_knn": best[3], "threshold": best[4], "development_binary_macro_f1": round(best[0], 4)}
    payload = {"method": "similarity_softmax_weighted_knn_blended_with_frozen_head", "selection_split": "development", "base": str(args.base), "checkpoint": str(args.checkpoint) if args.checkpoint else None, "pooling": args.pooling, "max_length": args.max_length, "k_values": K_VALUES, "temperatures": TEMPERATURES, "labels": labels, "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps({key: value["development_binary_macro_f1"] for key, value in labels.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8")); tokenizer, model = load_encoder(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config["pooling"])
    data_rows, data_embeddings = embed(args.data, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config["pooling"])
    global K_VALUES, TEMPERATURES
    old_k, old_t = K_VALUES, TEMPERATURES; K_VALUES = sorted({value["k"] for value in config["labels"].values()}); TEMPERATURES = sorted({value["temperature"] for value in config["labels"].values()})
    train_labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32); retrieval = weighted_neighbors(train_embeddings, data_embeddings, train_labels); K_VALUES, TEMPERATURES = old_k, old_t
    base_config = json.loads(args.base_config.read_text(encoding="utf-8")); rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"]); position = {row["id"]: i for i, row in enumerate(data_rows)}
    for row in rows:
        for index, label in enumerate(PRAGMATIC_LABELS):
            state = config["labels"][label]; value = state["alpha_knn"] * float(retrieval[(state["k"], state["temperature"])][position[row["id"]], index]) + (1 - state["alpha_knn"]) * float(row["probabilities"][label]); row["probabilities"][f"{label}_weighted_knn"] = value; row["predictions"][label] = int(value >= state["threshold"])
        row["system"] = "vipragsent_frozen_weighted_knn"
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


if __name__ == "__main__": raise SystemExit(main())
