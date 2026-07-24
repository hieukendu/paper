#!/usr/bin/env python3
"""Frozen contextual-embedding k-NN enhancement for ViPragSent.

The script performs non-parametric retrieval over train-set labels.  It uses
``model.eval()``, ``torch.inference_mode()``, and never creates an optimizer or
gradient graph.  Hyperparameters are selected from development gold only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Match the compatibility fallback used by the repository's prediction script:
# text-encoder inference does not use ONNX, but current Transformers imports
# that optional stack while resolving the RoBERTa model class.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from frozen_threshold_ensemble import _ensemble_rows
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl

K_VALUES = [1, 3, 5, 7, 11, 15, 25, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
ALPHAS = [value / 10 for value in range(11)]
THRESHOLDS = [round(value / 100, 2) for value in range(1, 100)]


class TextDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def load_encoder(base: Path, base_bin: Path, checkpoint: Path | None) -> tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    config = AutoConfig.from_pretrained(base)
    model = AutoModel.from_config(config)
    state = torch.load(base_bin, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("roberta."): value
        for key, value in state.items()
        if key.startswith("roberta.") and key != "roberta.embeddings.position_ids"
    }
    incompat = model.load_state_dict(state, strict=False)
    allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    if set(incompat.missing_keys) - allowed_missing or incompat.unexpected_keys:
        raise RuntimeError(f"base model mismatch: {incompat}")
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        trained = {
            key.removeprefix("encoder."): value
            for key, value in payload["model"].items()
            if key.startswith("encoder.") and key != "encoder.embeddings.position_ids"
        }
        incompat = model.load_state_dict(trained, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            raise RuntimeError(f"checkpoint encoder mismatch: {incompat}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def embed(path: Path, *, tokenizer, model, max_length: int, batch_size: int, pooling: str) -> tuple[list[dict], torch.Tensor]:
    dataset = TextDataset(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def collate(rows: list[dict]) -> dict:
        tokens = tokenizer([str(row["text"]) for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {"rows": rows, **tokens}

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=2)
    outputs = []
    rows_out: list[dict] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden_states = model(**inputs).last_hidden_state
                if pooling == "cls":
                    hidden = hidden_states[:, 0]
                else:
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    hidden = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(F.normalize(hidden.float(), dim=-1).cpu())
            rows_out.extend(batch["rows"])
    return rows_out, torch.cat(outputs)


def neighbors(train_embeddings: torch.Tensor, query_embeddings: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
    maximum = max(K_VALUES)
    values: dict[int, list[torch.Tensor]] = {k: [] for k in K_VALUES}
    for start in range(0, len(query_embeddings), 256):
        similarities = query_embeddings[start : start + 256] @ train_embeddings.T
        indices = similarities.topk(maximum, dim=1).indices
        gathered = labels[indices]
        for k in K_VALUES:
            values[k].append(gathered[:, :k].mean(dim=1))
    return {k: torch.cat(chunks) for k, chunks in values.items()}


def base_rows(path: Path) -> list[dict]:
    files = sorted(path.glob("reproduction_*.jsonl"))
    if not files:
        raise ValueError(f"no reproduction predictions found in {path}")
    return _ensemble_rows(files)


def fit(args: argparse.Namespace) -> None:
    tokenizer, model = load_encoder(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    dev_rows, dev_embeddings = embed(args.dev_gold, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size, pooling=args.pooling)
    train_labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    retrieval = neighbors(train_embeddings, dev_embeddings, train_labels)
    base = {row["id"]: row for row in base_rows(args.dev_predictions)}
    if {row["id"] for row in dev_rows} != set(base):
        raise ValueError("development gold/prediction IDs differ")
    selected = {}
    for label_index, label in enumerate(PRAGMATIC_LABELS):
        truth = [int(row["labels"][label]) for row in dev_rows]
        model_scores = torch.tensor([float(base[row["id"]]["probabilities"][label]) for row in dev_rows])
        best = (-1.0, None, None, None)
        for k, scores in retrieval.items():
            retrieval_scores = scores[:, label_index]
            for alpha in ALPHAS:
                blend = alpha * retrieval_scores + (1.0 - alpha) * model_scores
                for threshold in THRESHOLDS:
                    value = binary_macro_f1(truth, [int(score >= threshold) for score in blend.tolist()]) * 100
                    candidate = (value, -abs(alpha - 0.5), -abs(threshold - 0.5), k, alpha, threshold)
                    if candidate > (best[0], -abs(best[2] - 0.5) if best[2] is not None else -9, -abs(best[3] - 0.5) if best[3] is not None else -9, best[1] or 0, best[2] or 0, best[3] or 0):
                        best = (value, k, alpha, threshold)
        selected[label] = {"k": best[1], "alpha_knn": best[2], "threshold": best[3], "development_binary_macro_f1": round(best[0], 4)}
    payload = {
        "method": "frozen_contextual_embedding_knn_blended_with_frozen_head",
        "selection_split": "development", "train_path": str(args.train), "dev_gold_path": str(args.dev_gold),
        "base": str(args.base), "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "max_length": args.max_length, "pooling": args.pooling, "k_values": K_VALUES, "alpha_values": ALPHAS, "threshold_grid": [0.01, 0.99, 0.01],
        "labels": selected,
        "frozen_weight_compliance": {"neural_weight_updates": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer, model = load_encoder(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config.get("pooling", "cls"))
    data_rows, data_embeddings = embed(args.data, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size, pooling=config.get("pooling", "cls"))
    train_labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    needed_k = sorted({rule["k"] for rule in config["labels"].values()})
    global K_VALUES
    previous = K_VALUES; K_VALUES = needed_k
    retrieval = neighbors(train_embeddings, data_embeddings, train_labels)
    K_VALUES = previous
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    by_id = {row["id"]: position for position, row in enumerate(data_rows)}
    for row in rows:
        position = by_id[row["id"]]
        for label_index, label in enumerate(PRAGMATIC_LABELS):
            rule = config["labels"][label]
            value = rule["alpha_knn"] * float(retrieval[rule["k"]][position, label_index]) + (1.0 - rule["alpha_knn"]) * float(row["probabilities"][label])
            row["probabilities"][f"{label}_knn_blend"] = value
            row["predictions"][label] = int(value >= rule["threshold"])
    write_jsonl(args.output, rows)
    print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "apply"):
        command = commands.add_parser(name)
        command.add_argument("--base", type=Path, required=True)
        command.add_argument("--base-bin", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--train", type=Path, required=True)
        command.add_argument("--data" if name == "apply" else "--dev-gold", type=Path, required=True)
        command.add_argument("--predictions" if name == "apply" else "--dev-predictions", type=Path, required=True)
        command.add_argument("--output" if name == "fit" else "--base-config", type=Path, required=True)
        command.add_argument("--config", type=Path) if name == "apply" else None
        if name == "apply":
            command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-length", type=int, default=128)
        command.add_argument("--batch-size", type=int, default=128)
        if name == "fit":
            command.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "fit":
        fit(args)
    else:
        if args.config is None:
            raise SystemExit("--config is required for apply")
        apply(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
