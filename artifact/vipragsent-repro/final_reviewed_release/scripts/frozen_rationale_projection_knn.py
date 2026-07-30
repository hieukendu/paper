#!/usr/bin/env python3
"""k-NN over the existing frozen ViPragSent rationale-projection output.

The projection was already part of the archived multitask checkpoint.  This
script only evaluates it as a fixed representation for train-label retrieval;
it does not generate rationales or modify any checkpoint tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frozen_embedding_knn import ALPHAS, K_VALUES, THRESHOLDS, neighbors
from frozen_threshold_ensemble import _ensemble_rows
from train_multitask_encoder import MultiTaskModel
from vipragsent.data.schema import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1
from vipragsent.utils.io import read_jsonl, write_jsonl


class TextDataset(Dataset):
    def __init__(self, path: Path): self.rows = list(read_jsonl(path))
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int) -> dict: return self.rows[index]


def load_model(base: Path, base_bin: Path, checkpoint: Path):
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(base))
    state = torch.load(base_bin, map_location="cpu", weights_only=True)
    state = {key.removeprefix("roberta."): value for key, value in state.items() if key.startswith("roberta.") and key != "roberta.embeddings.position_ids"}
    incompatible = encoder.load_state_dict(state, strict=False)
    allowed_missing = {"pooler.dense.weight", "pooler.dense.bias"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"base model mismatch: {incompatible}")
    model = MultiTaskModel(str(base), uncertainty=True, rationale_aux=True, encoder=encoder)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(payload["model"])
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    if model.rationale_projection is None:
        raise RuntimeError("archived checkpoint does not expose a rationale projection")
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return tokenizer, model


def embed(path: Path, *, tokenizer, model, max_length: int, batch_size: int) -> tuple[list[dict], torch.Tensor]:
    dataset = TextDataset(path); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    def collate(rows: list[dict]) -> dict:
        encoded = tokenizer([str(row["text"]) for row in rows], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return {"rows": rows, **encoded}
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, num_workers=2)
    vectors, rows_out = [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, _, _, pooled = model(inputs["input_ids"], inputs["attention_mask"])
                projected = model.rationale_projection(pooled)
            vectors.append(F.normalize(projected.float(), dim=-1).cpu()); rows_out.extend(batch["rows"])
    return rows_out, torch.cat(vectors)


def fit(args: argparse.Namespace) -> None:
    tokenizer, model = load_model(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size)
    dev_rows, dev_embeddings = embed(args.dev_gold, tokenizer=tokenizer, model=model, max_length=args.max_length, batch_size=args.batch_size)
    labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    retrieval = neighbors(train_embeddings, dev_embeddings, labels)
    frozen = {row["id"]: row for row in _ensemble_rows(sorted(args.dev_predictions.glob("reproduction_*.jsonl")))}
    selected = {}
    for label_index, label in enumerate(PRAGMATIC_LABELS):
        truth = [int(row["labels"][label]) for row in dev_rows]
        base = torch.tensor([float(frozen[row["id"]]["probabilities"][label]) for row in dev_rows])
        best = (-1.0, None, None, None)
        for k, scores in retrieval.items():
            for alpha in ALPHAS:
                blend = alpha * scores[:, label_index] + (1 - alpha) * base
                for threshold in THRESHOLDS:
                    value = binary_macro_f1(truth, [int(score >= threshold) for score in blend.tolist()]) * 100
                    candidate = (value, -abs(alpha - 0.5), -abs(threshold - 0.5), k, alpha, threshold)
                    if candidate > (best[0], -abs(best[2] - 0.5) if best[2] is not None else -9, -abs(best[3] - 0.5) if best[3] is not None else -9, best[1] or 0, best[2] or 0, best[3] or 0):
                        best = (value, k, alpha, threshold)
        selected[label] = {"k": best[1], "alpha_knn": best[2], "threshold": best[3], "development_binary_macro_f1": round(best[0], 4)}
    payload = {
        "method": "frozen_archived_rationale_projection_knn_blended_with_frozen_head", "selection_split": "development",
        "base": str(args.base), "checkpoint": str(args.checkpoint), "max_length": args.max_length, "k_values": K_VALUES, "alpha_values": ALPHAS, "labels": selected,
        "frozen_weight_compliance": {"neural_weight_updates": False, "rationale_generation": False, "selection_uses_test_labels": False, "optimizer_or_backward_called": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({label: rule["development_binary_macro_f1"] for label, rule in selected.items()}, indent=2))


def apply(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8")); tokenizer, model = load_model(args.base, args.base_bin, args.checkpoint)
    train_rows, train_embeddings = embed(args.train, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size)
    data_rows, data_embeddings = embed(args.data, tokenizer=tokenizer, model=model, max_length=config["max_length"], batch_size=args.batch_size)
    labels = torch.tensor([[int(row["labels"][label]) for label in PRAGMATIC_LABELS] for row in train_rows], dtype=torch.float32)
    import frozen_embedding_knn
    previous = frozen_embedding_knn.K_VALUES[:]; frozen_embedding_knn.K_VALUES = sorted({rule["k"] for rule in config["labels"].values()})
    retrieval = neighbors(train_embeddings, data_embeddings, labels); frozen_embedding_knn.K_VALUES = previous
    base_config = json.loads(args.base_config.read_text(encoding="utf-8"))
    rows = _ensemble_rows(sorted(args.predictions.glob("reproduction_*.jsonl")), thresholds=base_config["thresholds"])
    position = {row["id"]: index for index, row in enumerate(data_rows)}
    for row in rows:
        for label_index, label in enumerate(PRAGMATIC_LABELS):
            rule = config["labels"][label]
            value = rule["alpha_knn"] * float(retrieval[rule["k"]][position[row["id"]], label_index]) + (1 - rule["alpha_knn"]) * float(row["probabilities"][label])
            row["probabilities"][f"{label}_rationale_knn_blend"] = value; row["predictions"][label] = int(value >= rule["threshold"])
        row["system"] = "vipragsent_frozen_rationale_projection_knn"
    write_jsonl(args.output, rows); print(json.dumps({"status": "ok", "records": len(rows), "output": str(args.output)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "apply"):
        command = commands.add_parser(name); command.add_argument("--base", type=Path, required=True); command.add_argument("--base-bin", type=Path, required=True); command.add_argument("--checkpoint", type=Path, required=True); command.add_argument("--train", type=Path, required=True); command.add_argument("--batch-size", type=int, default=128)
        if name == "fit":
            command.add_argument("--dev-gold", type=Path, required=True); command.add_argument("--dev-predictions", type=Path, required=True); command.add_argument("--max-length", type=int, default=128); command.add_argument("--output", type=Path, required=True); command.set_defaults(func=fit)
        else:
            command.add_argument("--config", type=Path, required=True); command.add_argument("--data", type=Path, required=True); command.add_argument("--predictions", type=Path, required=True); command.add_argument("--base-config", type=Path, required=True); command.add_argument("--output", type=Path, required=True); command.set_defaults(func=apply)
    args = parser.parse_args(); args.func(args); return 0


if __name__ == "__main__": raise SystemExit(main())
